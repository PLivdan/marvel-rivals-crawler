import json
import sys

import db
import fetcher
import ingest
import rivalsmeta

# Hero coverage is computed ONCE as an aggregate CTE and LEFT JOINed, rather
# than as a correlated subquery evaluated per candidate row (which measured
# ~5.2s at only 2000 pending players, and gets far worse at realistic scale).
# Ordering semantics are identical to the correlated form: untagged players
# sort last, then fewest collected hero-rows first, then oldest-created first.
PRIORITY_SQL = """
WITH hero_counts AS (
    SELECT hero_id, COUNT(*) AS c FROM match_player_heroes GROUP BY hero_id
)
SELECT p.uid FROM players p
LEFT JOIN hero_counts h ON h.hero_id = p.discovery_hero_id
WHERE p.crawl_status = 'pending'
ORDER BY
    CASE WHEN p.discovery_hero_id IS NULL THEN 999999999 ELSE COALESCE(h.c, 0) END ASC,
    p.created_at ASC
LIMIT 1
"""


def select_next_player(conn):
    """Read-only peek at the head of the frontier. Safe on its own only when
    nothing else is crawling; concurrent callers must use claim_next_player,
    which builds on this."""
    row = conn.execute(PRIORITY_SQL).fetchone()
    return row[0] if row else None


def claim_next_player(conn):
    """Atomically claim one pending player, or return None if the queue is
    empty right now (which may mean truly empty, or that a concurrent
    worker won the race for the one candidate row this connection saw --
    callers should treat both cases the same way: try again shortly).

    The atomicity comes from nothing more exotic than SQLite's ordinary write
    serialization: one connection's UPDATE commits at a time, and the
    `AND crawl_status='pending'` guard means a connection that lost the race
    for this row simply matches zero rows and reports rowcount 0.

    The claim stamps `claimed_at`, NOT `last_crawled_at`. crawl_player reads
    `last_crawled_at IS NULL` as "this player has never COMPLETED a crawl" and
    uses it to keep paginating past already-known matches; stamping it here
    would make every first crawl look like a revisit and silently strand the
    rest of that player's history (the exact bug the first-crawl resume fix
    exists to prevent)."""
    uid = select_next_player(conn)
    if uid is None:
        return None
    cur = conn.execute(
        "UPDATE players SET crawl_status='claimed', claimed_at=? "
        "WHERE uid=? AND crawl_status='pending'",
        (db.now(), uid),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None
    return uid


def release_player(conn, uid):
    """Put a claimed player straight back on the frontier, untouched.

    Used when a crawl was abandoned for a reason that is not this player's
    fault (a tripped circuit, a site-wide 429/403): the pre-concurrency loop
    simply left them 'pending' to be retried, and this restores exactly that.
    `last_crawled_at` is deliberately not written — the claim never touched it,
    so releasing preserves whatever first-crawl/revisit state the row had."""
    conn.execute(
        "UPDATE players SET crawl_status='pending', claimed_at=NULL "
        "WHERE uid=? AND crawl_status='claimed'",
        (uid,),
    )
    conn.commit()


def crawl_player(conn, client, uid, season):
    row = conn.execute(
        "SELECT latest_known_level FROM players WHERE uid=?", (uid,)
    ).fetchone()
    if row and row[0] is not None and not rivalsmeta.is_diamond_plus(row[0]):
        return _set_status(conn, uid, "skipped_floor")

    try:
        profile = rivalsmeta.get_player(client, uid, season)
    except fetcher.PlayerNotFoundError:
        return _set_status(conn, uid, "not_indexed")

    visibility = profile.get("visibility") or {}
    rank_blob = rivalsmeta.current_season_rank(profile, season) or {}
    level = rank_blob.get("level")
    score = rank_blob.get("rank_score")
    name = profile.get("player", {}).get("info", {}).get("name")

    db.upsert(
        conn,
        "players",
        ["uid"],
        {
            "uid": uid,
            "nick_name": name,
            "latest_known_score": score,
            "latest_known_level": level,
            "visibility_json": json.dumps(visibility),
        },
    )
    conn.commit()

    if not visibility.get("match_history", False):
        return _set_status(conn, uid, "skipped_private")

    if level is not None and not rivalsmeta.is_diamond_plus(level):
        return _set_status(conn, uid, "skipped_floor")

    # Stop-at-first-known is only correct for a REVISIT. On a player's
    # first-ever crawl, an already-known match_uid means a previous attempt
    # was interrupted (crash/kill/FetchError) partway through their history —
    # breaking there would mark them 'done' and permanently lose every older
    # match. A player with last_crawled_at IS NULL has never completed a
    # crawl, so we skip past known matches (free, no extra request) and keep
    # paginating to the real end instead.
    crawled_before_row = conn.execute(
        "SELECT last_crawled_at FROM players WHERE uid=?", (uid,)
    ).fetchone()
    is_revisit = bool(crawled_before_row) and crawled_before_row[0] is not None

    skip = 0
    while True:
        page = rivalsmeta.get_player_match_history_page(client, uid, skip, season)
        if not page:
            break
        hit_known = False
        for entry in page:
            match_uid = entry["match_uid"]
            already_known = conn.execute(
                "SELECT 1 FROM matches WHERE match_uid=?", (match_uid,)
            ).fetchone()
            if already_known:
                if is_revisit:
                    hit_known = True
                    break
                continue
            try:
                detail = rivalsmeta.get_match_detail(client, match_uid)
                # The history entry carries match_map_id, which the
                # match-detail endpoint does not expose at all — pass it
                # through so map_id lands.
                ingest.ingest_match(conn, detail, season, history_entry=entry)
            except fetcher.PlayerNotFoundError:
                # A match that showed up in this player's history but is no
                # longer fetchable (404) — normal attrition, not an error and
                # not a payload problem. Nothing was written, so no rollback.
                print(
                    f"match {match_uid} no longer available (404); skipping",
                    file=sys.stderr,
                )
                continue
            except (KeyError, TypeError, ValueError) as exc:
                # This is an undocumented third-party API that can change
                # shape without notice. One malformed match must not take
                # down the whole run — log it and move on. The rollback drops
                # whatever partial rows that match wrote before failing, so a
                # later match's commit can't flush a half-ingested match.
                conn.rollback()
                print(
                    f"skipping malformed match {match_uid}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
        if hit_known or len(page) < 20:
            break
        skip += 20

    return _set_status(conn, uid, "done")


def requeue_stale_players(
    conn,
    done_revisit_seconds=43200,
    error_retry_seconds=3600,
    claimed_stale_seconds=600,
):
    """Return stale players to the frontier, so the crawl is actually ongoing.

    Without this nothing ever leaves 'done' or 'error', so the incremental
    re-crawl the design is built around could never happen, one transient
    failure abandoned a player permanently, and a worker that died holding a
    claim would strand that player forever.

    The three resets are deliberately NOT symmetric, because crawl_player reads
    `last_crawled_at IS NULL` to tell a first-ever crawl from a revisit:

    - A 'done' player KEEPS last_crawled_at. They completed a full crawl, so
      this is a genuine revisit: stopping at the first already-known match is
      both correct and the cheap way to pull only what's new.
    - An 'error' player has last_crawled_at CLEARED to NULL. They never
      completed a first crawl — they may have ingested none, some, or most of
      their history before failing — so the retry must behave like a fresh
      first crawl (skip past already-known matches and keep paginating).
      Leaving the timestamp set would make the retry stop at the first match
      the failed attempt happened to ingest, silently stranding the rest.
    - A 'claimed' player older than `claimed_stale_seconds` is an orphan: the
      worker that claimed it died (crash, SIGKILL) without ever writing a
      terminal status. Same reasoning as 'error' — the crawl it was in the
      middle of never finished, so last_crawled_at is CLEARED to NULL and the
      retry runs with fresh-crawl semantics. `claimed_at` is cleared too, so
      the row stops looking claimed. The default window (10 minutes) is
      generously longer than any single player's crawl should take even with a
      large match history, so a live worker is never robbed of its player.

    A NULL timestamp never satisfies `< cutoff` in SQL, so rows in an
    unexpected state are left alone rather than requeued.
    """
    now = db.now()

    done_reset = conn.execute(
        "UPDATE players SET crawl_status='pending' "
        "WHERE crawl_status='done' AND last_crawled_at IS NOT NULL AND last_crawled_at < ?",
        (now - done_revisit_seconds,),
    ).rowcount
    error_reset = conn.execute(
        "UPDATE players SET crawl_status='pending', last_crawled_at=NULL "
        "WHERE crawl_status='error' AND last_crawled_at IS NOT NULL AND last_crawled_at < ?",
        (now - error_retry_seconds,),
    ).rowcount
    claimed_reset = conn.execute(
        "UPDATE players SET crawl_status='pending', last_crawled_at=NULL, claimed_at=NULL "
        "WHERE crawl_status='claimed' AND claimed_at IS NOT NULL AND claimed_at < ?",
        (now - claimed_stale_seconds,),
    ).rowcount

    conn.commit()
    return done_reset + error_reset + claimed_reset


def reseed(conn, client, hero_refresh_seconds=86400):
    # Hero leaderboards are seeded BEFORE the general leaderboard on purpose:
    # _seed_player only tags a player's discovery_hero_id on first creation,
    # so if the same player appears on both a hero leaderboard and the
    # general top-500, we want the more useful hero-specific tag to win
    # rather than being pre-empted by a generic (untagged) seed.
    queued = 0

    cutoff = db.now() - hero_refresh_seconds
    stale_heroes = conn.execute(
        "SELECT hero_id FROM heroes WHERE last_seeded_at IS NULL OR last_seeded_at < ?",
        (cutoff,),
    ).fetchall()
    for (hero_id,) in stale_heroes:
        hero_board = rivalsmeta.get_hero_leaderboard(client, hero_id)
        for p in hero_board.get("players", []):
            uid = p.get("player_uid")
            if uid is None:
                continue
            name = p.get("info", {}).get("name")
            queued += _seed_player(conn, uid=int(uid), name=name, discovery_hero_id=hero_id)
        db.upsert(conn, "heroes", ["hero_id"], {"hero_id": hero_id, "last_seeded_at": db.now()})

    board = rivalsmeta.get_global_leaderboard(client)
    for p in board["players"]:
        queued += _seed_player(conn, uid=int(p["uid"]), name=p.get("name"), discovery_hero_id=None)

    conn.commit()
    return queued


def _seed_player(conn, uid, name, discovery_hero_id):
    existing = conn.execute("SELECT uid FROM players WHERE uid=?", (uid,)).fetchone()
    if existing is not None:
        return 0
    db.upsert(
        conn,
        "players",
        ["uid"],
        {
            "uid": uid,
            "nick_name": name,
            "discovery_hero_id": discovery_hero_id,
            "crawl_status": "pending",
        },
    )
    return 1


def _set_status(conn, uid, status):
    # claimed_at is cleared alongside the terminal status so it only ever means
    # "a worker is holding this player right now", which is what the orphaned-
    # claim sweep in requeue_stale_players relies on.
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": uid, "crawl_status": status, "last_crawled_at": db.now(), "claimed_at": None},
    )
    conn.commit()
    return status
