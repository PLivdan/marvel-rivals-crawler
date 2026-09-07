import json

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
    row = conn.execute(PRIORITY_SQL).fetchone()
    return row[0] if row else None


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
                hit_known = True
                break
            detail = rivalsmeta.get_match_detail(client, match_uid)
            ingest.ingest_match(conn, detail, season)
        if hit_known or len(page) < 20:
            break
        skip += 20

    return _set_status(conn, uid, "done")


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
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": status, "last_crawled_at": db.now()})
    conn.commit()
    return status
