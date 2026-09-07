# rivalsmeta.com Crawler — Design

Date: 2026-09-07
Status: approved for planning

## 1. Goal

Build a long-running, resumable crawler that collects Marvel Rivals **competitive** match data from rivalsmeta.com into a local SQLite database, sized and shaped for a regression over match/player outcomes. This is the first of the user's Marvel Rivals side-projects to build a general-purpose data pipeline; it is intentionally scoped to this one site rather than generalized across sources (YAGNI — a second source can be added later behind the same fetcher interface if needed).

Related prior work: `~/projects/marvel-rivals-analytics` is a stale (last touched January 2026), single-player analysis tool built against a *different* third-party API (`marvelrivalsapi.com`, key-gated, currently returning 502 at the origin). It has no crawler/BFS logic to reuse and is not extended by this project.

## 2. Regression requirements this design must satisfy

The user needs, per match, per player: who won, the hero played (including mid-match swaps and per-hero time), the full ban/pick order, the map and mode, the date/time played, and each player's elo (rank score) at the time of the match. It must also actively counteract Marvel Rivals' pick-rate skew (some heroes are rarely played) rather than passively reflecting it, and must not be intrusive to rivalsmeta.com's servers.

## 3. Verified API surface

All endpoints are same-origin on `rivalsmeta.com`, unauthenticated, plain JSON over HTTPS, confirmed working via direct `curl` with a normal desktop User-Agent (no cookies/key/headers required). `cache-control: private, no-store` — nothing is CDN-cached, so every hit reaches their origin directly.

- `GET /api/player/{uid}?season={N}` — profile: `stats`, `heroes_ranked`/`heroes_unranked` (per-hero season aggregates, including `matches` count — useful for coverage bookkeeping without extra requests), `matchups`, `teammates`, `match_history` (most recent 20 only), `visibility` (per-field privacy flags: `overview`, `career_stats`, `match_history`, `timeline`), `player.info` (name, icon, rank blobs).
- `GET /api/player-match-history/{uid}?skip={N}&game_mode_id={M}&hero_id=0&season={N}` — paginated match history for one player, 20 per page, plain JSON array (no total count — end of list is signaled by a page shorter than 20, including empty). This is the mechanism for pulling a player's **full** season history, not just the last 20.
- `GET /api/matches/{match_uid}` — full match detail:
  - `dynamic_fields.ban_pick_info[]`: `round_idx`, `battle_side`, `hero_id`, `is_pick` — the ordered ban/pick sequence.
  - `match_players[]` (all 12): `player_uid`, `nick_name`, `camp`, `cur_hero_id`, `k`/`d`/`a`, `total_hero_damage`, `total_hero_heal`, `total_damage_taken`, `is_win`, `session_hit_rate`, `dynamic_fields.add_score`/`new_score` (rank-score delta and post-match rank score — pre-match score = `new_score - add_score`), `player_heroes[]` (`hero_id`, `k`/`d`/`a`, `play_time` — the mid-match swap breakdown).
  - `match_time_stamp`, `match_play_duration`, `game_mode_id`, `map_id` (called `match_map_id` from the player-side view), `mvp_uid`/`mvp_hero_id`, `svp_uid`/`svp_hero_id`, `replay_id`.
  - **Verified live**: this endpoint returns full `match_players`/`player_heroes` data for participants whose own profile is entirely private (`visibility` all `false`). Profile privacy only gates that player's *own* aggregate profile/history; it never redacts their row inside a match reached via a public co-player.
- `GET /api/hero-leaderboard/{hero_id}?device=1&season=last` — top players **for that specific hero**, confirmed live on a niche pick (Jeff The Land Shark). This is the fix for pick-rate skew: seed every hero from its own leaderboard rather than relying on a hero's natural population share to surface enough mains via BFS.
- Global leaderboard (`/leaderboard`, top 500 only) is not a separate client-visible API call — it's embedded in the page's Nuxt SSR payload (`window.__NUXT__.data`). Each entry includes `rank.rank_game_id`, which is the current season's **numeric id** — this is how the crawler resolves "what season is it right now" without hardcoding or guessing the 9.5 → 19 mapping.
- 404 (`"player not found"`) is a distinct, normal outcome for UIDs rivalsmeta hasn't indexed yet — not an error, not "private."

## 4. Architecture

Three layers, one Python process:

- **Fetcher** (`fetcher.py`): the only thing that calls `rivalsmeta.com`. Owns the adaptive rate limiter, retries/backoff, and User-Agent. Every other module goes through it.
- **Orchestrator** (`crawler.py`): the BFS loop over players — decides what to fetch next, updates state, upserts results.
- **SQLite** (`db.py`): durable state and the queue. A player's row *is* its queue entry (`crawl_status` column) — no separate queue table.

## 5. Seeding & coverage strategy

On startup (and on a periodic reseed, e.g. daily): fetch the global top-500 leaderboard, then one `hero-leaderboard` call per hero (~40 total, one-time cost per reseed). Upsert every returned player into `players` as `pending`, recording which hero (if any) surfaced them as `discovery_hero_id`.

Queue ordering is **not** FIFO: the orchestrator always picks the next pending player via
```sql
SELECT * FROM players WHERE crawl_status='pending'
ORDER BY (SELECT COUNT(*) FROM match_player_heroes WHERE hero_id = players.discovery_hero_id)
LIMIT 1
```
i.e., whichever hero currently has the fewest collected match-rows gets its mains crawled first. This only changes *acquisition order* — once a player is crawled, all of their competitive matches are pulled and all 12 participants' data in each is stored, regardless of hero. No data is ever discarded for being "already enough."

## 6. Crawl algorithm

For the player at the front of the priority queue:

1. If we already know (from a previously-ingested match's `new_score`) that this player is below the Diamond floor, mark `skipped_floor` — no request spent.
2. Otherwise fetch `/api/player/{uid}` once. Handle 404 → `not_indexed` (terminal, no retry). On success, read `visibility`.
3. If `visibility.match_history` is false: store/refresh whatever we already had (name, last-known score from a match) and mark `skipped_private`. Never attempted further from this player — but their matches keep arriving passively whenever a public co-player surfaces them.
4. If public and below the Diamond floor (from the profile's own rank data): mark `skipped_floor`.
5. If public and Diamond+: paginate `player-match-history` (competitive only — `game_mode_id` value TBD, see §10) newest-first, `skip += 20`, **stopping the first time a returned `match_uid` is already in the `matches` table**. This one rule handles both "pull this player's full season history" on first visit and "pull only what's new" on every later revisit (the ongoing/incremental case), with no separate code path.
6. For every new `match_uid`: fetch `/api/matches/{match_uid}`, store `matches` + `match_bans` + all 12 `match_players`/`match_player_heroes` rows unconditionally (see §3's verified privacy behavior), then upsert all 12 `player_uid`s into `players` (creating `pending` rows for any not already known, tagged with their `cur_hero_id` as a fallback `discovery_hero_id` if they weren't seeded from a hero-leaderboard) — only players at/above the Diamond floor (from that match's `new_score`) get expanded further.
7. Mark the player `done`, `last_crawled_at = now`.

## 7. Data model (SQLite)

- `players(uid PK, nick_name, discovery_hero_id, latest_known_score, visibility_json, crawl_status, last_crawled_at, created_at)`
- `matches(match_uid PK, match_time_stamp, match_play_duration, game_mode_id, map_id, season, winner_side, mvp_uid, mvp_hero_id, svp_uid, svp_hero_id, replay_id, fetched_at)`
- `match_bans(match_uid FK, round_idx, battle_side, hero_id, is_pick, PRIMARY KEY(match_uid, round_idx, battle_side, hero_id))`
- `match_players(match_uid FK, player_uid FK, camp, cur_hero_id, k, d, a, total_hero_damage, total_hero_heal, total_damage_taken, is_win, add_score, new_score, session_hit_rate, PRIMARY KEY(match_uid, player_uid))`
- `match_player_heroes(match_uid FK, player_uid FK, hero_id, k, d, a, play_time, PRIMARY KEY(match_uid, player_uid, hero_id))`

All inserts are `INSERT OR REPLACE`/`ON CONFLICT DO UPDATE` keyed by these natural primary keys — every write is idempotent, so re-processing a match or a player is always safe and never duplicates rows.

The regression's core join is `matches ⋈ match_players ⋈ match_bans` — every needed field (winner, hero, elo before/after via `new_score - add_score`, timestamp, ban order) lives in these tables directly.

## 8. Politeness (adaptive rate limiting)

Single sequential stream, no concurrency. A token-bucket delay starts at ~1 req/sec. On a sustained run of fast, clean `200`s, the delay eases down toward a hard floor; any `429`/`403`/`5xx`, or a latency spike, triggers exponential backoff and a cooldown period before it tries to speed up again. Standard desktop User-Agent; no header spoofing, no proxies, no concurrency tricks — the goal is being a good citizen, not evading detection.

## 9. Robustness

- **Idempotent writes** (§7) make crash recovery trivial: on restart, the orchestrator just resumes pulling `pending` players from the queue — no special resume logic, no risk of double-counting.
- **Durable, incremental commits**: SQLite in WAL mode, committing after each match/player rather than batching a long run into one transaction, so a crash or kill mid-run loses at most the in-flight request, never previously-collected data.
- **Graceful shutdown**: catch `SIGINT`/`SIGTERM`, finish the in-flight request, commit, exit cleanly.
- **Error classification**: 404 → terminal, no retry. 5xx / connection errors / timeouts → retry with backoff (bounded attempts), then mark the player `error` for a later pass rather than blocking the queue. 429/403 → treated as a rate-limit signal to the adaptive limiter, not a per-player error.
- **Circuit breaker**: if the error rate over a rolling window stays elevated despite backoff (a sign of an actual block, not transient noise), the crawler pauses entirely and logs loudly, rather than continuing to hammer a site that may be blocking it.
- **Defensive parsing**: every API response is accessed defensively (missing/renamed fields log a warning with the offending `match_uid`/`uid` and are skipped, not a crash) — undocumented third-party APIs can change shape without notice.
- **Data sanity checks**: after storing a match, assert `len(match_players) == 12` and that exactly one `winner_side` is consistent with the `is_win` flags; log (don't crash) on violation, for later inspection.
- **Season handled dynamically**: the current season's numeric id is read from live leaderboard data (`rank.rank_game_id`, §3) at each reseed, never hardcoded — so the 9.5 → 10 rollover is picked up automatically rather than silently mislabeling new matches under the old season.
- **Observability**: a periodic log line (every few minutes) reporting players processed, matches collected, current adaptive delay, error counts by type, and per-hero coverage counts (min/max/median matches collected across heroes) — so a long unattended run is legible without querying the DB, consistent with wanting progress/ETA visibility on long jobs.

## 10. Implementation-time verification tasks (not blocking the plan, but must be resolved early in it)

1. Exact format of the Nuxt SSR payload for `/leaderboard` (plain JSON vs. Nuxt's "devalue" encoding) — determines whether a regex/JSON extraction is sufficient or a headless-browser read of `window.__NUXT__` is needed as a fallback.
2. The numeric `game_mode_id` value that means "Competitive" (have one confirmed sample at `2`; needs a small confirmation pass across a few known ranked vs. quickplay matches, or reading the site's own mode-filter dropdown).
3. The current season's Diamond-rank-score cutoff (available from the site's own rank-tier display; not yet pinned down).

## 11. Non-goals

- PC platform only (`device=1`, rivalsmeta's default on every leaderboard call observed). Console leaderboards/players are out of scope for v1 — flagging this as a default, not a confirmed requirement, since it wasn't explicitly discussed.
- No second data source (`marvelrivalsapi.com` or otherwise) in this phase.
- No proxy rotation, header spoofing, or other detection-evasion — politeness is achieved by being genuinely slow and well-behaved, not by hiding.
- No backfill of past seasons.
- No quickplay/custom-mode data collection.
- No web UI/dashboard — output is the SQLite file, consumed directly from a notebook/pandas for the regression.
