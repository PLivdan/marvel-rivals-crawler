import json
import sys

import db
import fetcher
import ingest
import rivalsmeta

# Priority: untagged players sort last, then fewest collected hero-rows first,
# then oldest-created first. The same order the original ORDER BY over the
# whole frontier produced — but that form sorted every pending row through a
# temp B-tree and re-aggregated hero coverage over match_player_heroes on
# EVERY claim: 2-4.6s at 410k pending / 3.9M hero rows (2026-09-20), growing
# linearly, and with six workers all computing the same deterministic head,
# five lost the claim race each round. This form costs a few dozen index seeks
# on idx_players_pending_priority however large the frontier gets.
#
# The recursive CTE is SQLite's idiom for a loose index scan: each step seeks
# to the next distinct discovery_hero_id that still has a pending player, so
# the hero set comes from the players table itself (a hero tagged on a player
# need not exist in heroes/hero_coverage) without scanning the pending range —
# SELECT DISTINCT over it measured 110ms at 410k rows; this is ~3ms.
#
# Ties on coverage are broken by the oldest pending player across the tied
# heroes, exactly as the single ORDER BY did.
_PRIORITY_TEMPLATE = """
WITH RECURSIVE pending_heroes(hero_id) AS (
    SELECT MIN(discovery_hero_id) FROM players
     WHERE crawl_status = '{status}' AND discovery_hero_id IS NOT NULL
    UNION ALL
    SELECT (SELECT MIN(discovery_hero_id) FROM players
             WHERE crawl_status = '{status}' AND discovery_hero_id > ph.hero_id)
      FROM pending_heroes ph WHERE ph.hero_id IS NOT NULL
),
ranked AS (
    SELECT ph.hero_id,
           COALESCE(hc.n, 0) AS coverage,
           (SELECT MIN(created_at) FROM players
             WHERE crawl_status = '{status}' AND discovery_hero_id = ph.hero_id) AS oldest
      FROM pending_heroes ph
      LEFT JOIN hero_coverage hc ON hc.hero_id = ph.hero_id
     WHERE ph.hero_id IS NOT NULL
)
SELECT uid FROM players
 WHERE crawl_status = '{status}'
   AND discovery_hero_id = (SELECT hero_id FROM ranked ORDER BY coverage, oldest LIMIT 1)
 ORDER BY created_at
 LIMIT 1
"""

# Only reached when no tagged player is pending at all.
_UNTAGGED_TEMPLATE = """
SELECT uid FROM players
 WHERE crawl_status = '{status}' AND discovery_hero_id IS NULL
 ORDER BY created_at
 LIMIT 1
"""


# First crawls are always claimed before revisits. A revisit re-reads a player
# already crawled and stops at the first match played before that crawl, so it
# yields ~1.6 matches per player against ~12 for a first crawl (22 Sep vs 20 Sep).
# Queued revisits used to sit in 'pending' as the oldest rows and so went
# first (2026-09-22 review, finding I3).
QUEUE_STATUSES = ("pending", "revisit")
PRIORITY_SQL = {st: _PRIORITY_TEMPLATE.format(status=st) for st in QUEUE_STATUSES}
UNTAGGED_FALLBACK_SQL = {st: _UNTAGGED_TEMPLATE.format(status=st) for st in QUEUE_STATUSES}


REVISIT_STOP_SLACK_SECONDS = 3600


def select_next_player(conn):
    """Read-only peek at the head of the frontier. Safe on its own only when
    nothing else is crawling; concurrent callers must use claim_next_player,
    which builds on this."""
    head = _select_head(conn)
    return head[0] if head else None


def _select_head(conn):
    """(uid, queue status) of the next player to crawl, or None."""
    for status in QUEUE_STATUSES:
        row = conn.execute(PRIORITY_SQL[status]).fetchone()
        if row is None:
            row = conn.execute(UNTAGGED_FALLBACK_SQL[status]).fetchone()
        if row is not None:
            return row[0], status
    return None


def claim_next_player(conn):
    """Atomically claim one pending player, or return None if the queue is
    empty right now (which may mean truly empty, or that a concurrent
    worker won the race for the one candidate row this connection saw --
    callers should treat both cases the same way: try again shortly).

    The atomicity comes from nothing more exotic than SQLite's ordinary write
    serialization: one connection's UPDATE commits at a time, and the
    `AND crawl_status=<queue status>` guard means a connection that lost the race
    for this row simply matches zero rows and reports rowcount 0.

    The claim stamps `claimed_at`, NOT `last_crawled_at`. crawl_player reads
    `last_crawled_at IS NULL` as "this player has never COMPLETED a crawl" and
    uses it to keep paginating past already-known matches; stamping it here
    would make every first crawl look like a revisit and silently strand the
    rest of that player's history (the exact bug the first-crawl resume fix
    exists to prevent)."""
    head = _select_head(conn)
    if head is None:
        return None
    uid, status = head
    cur = conn.execute(
        "UPDATE players SET crawl_status='claimed', claimed_at=? "
        "WHERE uid=? AND crawl_status=?",
        (db.now(), uid, status),
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
    # A player who has completed a crawl before goes back to the revisit queue,
    # so an interrupted revisit does not jump ahead of first crawls.
    conn.execute(
        "UPDATE players SET claimed_at=NULL, crawl_status=CASE WHEN last_crawled_at IS NULL "
        "THEN 'pending' ELSE 'revisit' END WHERE uid=? AND crawl_status='claimed'",
        (uid,),
    )
    conn.commit()


def crawl_player(conn, client, uid, season):
    row = conn.execute(
        "SELECT latest_known_level FROM players WHERE uid=?", (uid,)
    ).fetchone()
    if row and row[0] is not None and not rivalsmeta.is_diamond_plus(row[0]):
        return set_status(conn, uid, "skipped_floor")

    try:
        profile = rivalsmeta.get_player(client, uid, season)
    except fetcher.PlayerNotFoundError:
        return set_status(conn, uid, "not_indexed")

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
        return set_status(conn, uid, "skipped_private")

    if level is not None and not rivalsmeta.is_diamond_plus(level):
        return set_status(conn, uid, "skipped_floor")

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
    # A revisit stops at the first known match played BEFORE the last completed
    # crawl, not at the first known match. An interrupted revisit (circuit, busy
    # database, shutdown) may have ingested the newest match and released the
    # player with last_crawled_at unchanged; stopping at that match would lose
    # every newer match it had not reached yet (review finding I2, reproduced).
    # The hour of slack covers clock skew between the site and this machine.
    # An entry with no timestamp keeps the old stop-at-first-known behaviour.
    stop_before = crawled_before_row[0] - REVISIT_STOP_SLACK_SECONDS if is_revisit else None

    skip = 0
    while True:
        try:
            page = rivalsmeta.get_player_match_history_page(client, uid, skip, season)
        except fetcher.PlayerNotFoundError:
            # The profile answered but the history 404s: the site no longer
            # indexes this player. Before, this escaped crawl_player, killed
            # the worker pool, and crashed again on the same player after every
            # restart (review finding I1, reproduced).
            return set_status(conn, uid, "not_indexed")
        if not page:
            break
        if not isinstance(page, list):
            raise fetcher.FetchError(
                f"malformed history page for player {uid}: {type(page).__name__}, not a list"
            )
        hit_known = False
        for entry in page:
            match_uid = entry.get("match_uid") if isinstance(entry, dict) else None
            if not match_uid:
                continue
            _record_history_entry(conn, match_uid, uid, entry)
            already_known = conn.execute(
                "SELECT 1 FROM matches WHERE match_uid=?", (match_uid,)
            ).fetchone()
            if already_known:
                if is_revisit:
                    played = entry.get("match_time_stamp")
                    if played is None or played < stop_before:
                        hit_known = True
                        break
                continue
            try:
                detail = rivalsmeta.get_match_detail(client, match_uid)
                # The history entry carries match_map_id, which the
                # match-detail endpoint does not expose at all — pass it
                # through so map_id lands.
                ingest.ingest_match(conn, detail, season, history_entry=entry, source_player_uid=uid)
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

    return set_status(conn, uid, "done")


def _record_history_entry(conn, match_uid, uid, entry):
    """Keep the crawled player's own leaver flag and result from a history entry (committed
    with the page's next write). Known matches are recorded too: no request is involved."""
    mp = entry.get("match_player") if isinstance(entry.get("match_player"), dict) else {}
    esc = mp.get("has_escaped")
    win = mp.get("is_win")
    db.upsert(conn, "history_entries", ["match_uid", "player_uid"], {
        "match_uid": match_uid, "player_uid": uid,
        "has_escaped": None if esc is None else int(bool(esc)),
        "is_win": None if win is None else int(win),
    })


def requeue_stale_players(
    conn,
    done_revisit_seconds=3 * 86400,
    error_retry_seconds=3600,
    claimed_stale_seconds=3600,
    max_revisits=2000,
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
      the row stops looking claimed.

      The window must stay comfortably longer than a real crawl, or this sweep
      steals a live worker's player: idempotent writes make that harmless, but
      it burns requests for nothing, which is precisely what the politeness
      budget cannot spare. A fully-crawled player runs to ~112 matches, and the
      shared limiter can sit at max_delay=8s under sustained strain, so a
      single legitimate crawl can take ~15 minutes — which a 10-minute window
      would wrongly reap. Hence one hour.

    A NULL timestamp never satisfies `< cutoff` in SQL, so rows in an
    unexpected state are left alone rather than requeued.
    """
    now = db.now()

    # Revisits go to their own 'revisit' queue, which is only claimed when no
    # first crawl is waiting, and at most `max_revisits` per cycle, oldest
    # first. Before, every player finished more than 12 h ago went back into
    # 'pending' at every startup, ahead of the players never crawled.
    done_reset = conn.execute(
        "UPDATE players SET crawl_status='revisit' WHERE uid IN ("
        " SELECT uid FROM players WHERE crawl_status='done' AND last_crawled_at IS NOT NULL"
        " AND last_crawled_at < ? ORDER BY last_crawled_at LIMIT ?)",
        (now - done_revisit_seconds, max_revisits),
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
        # Commit per hero, NOT once at the end. pysqlite opens the write
        # transaction at the first write and holds SQLite's single write lock
        # until the commit, so a trailing commit would hold that lock across
        # every remaining hero-leaderboard fetch (~42 requests, minutes of
        # wall time once the adaptive limiter has backed off). Every concurrent
        # worker's claim/ingest/status write would then block for busy_timeout
        # and fail with "database is locked". Committing here means no network
        # call is ever made while this connection holds the write lock.
        conn.commit()

    # Safe to fetch now: the loop above left no transaction open.
    board = rivalsmeta.get_global_leaderboard(client)
    for p in board["players"]:
        queued += _seed_player(conn, uid=int(p["uid"]), name=p.get("name"), discovery_hero_id=None)

    # This loop makes no network calls of its own, so one commit after it holds
    # the write lock only for the seeding itself.
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


def set_status(conn, uid, status):
    """The single writer of a terminal crawl status. Public so that every
    caller — including main.py's transient-failure handler, which marks a
    player 'error' — goes through one place, rather than each re-implementing
    the claimed_at invariant inline and risking one of them forgetting it.

    claimed_at is cleared alongside the terminal status so it only ever means
    "a worker is holding this player right now", which is what the orphaned-
    claim sweep in requeue_stale_players relies on."""
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": uid, "crawl_status": status, "last_crawled_at": db.now(), "claimed_at": None},
    )
    conn.commit()
    return status
