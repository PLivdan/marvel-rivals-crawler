import json
import pathlib

import db
import ingest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_match():
    return json.loads((FIXTURES / "match_detail.json").read_text())


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def test_ingest_match_stores_all_players_bans_and_hero_segments():
    conn = make_conn()
    match = load_match()
    ingest.ingest_match(conn, match, season=19)

    match_uid = match["match_uid"]
    n_players = conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE match_uid=?", (match_uid,)
    ).fetchone()[0]
    assert n_players == 12

    n_bans = conn.execute(
        "SELECT COUNT(*) FROM match_bans WHERE match_uid=?", (match_uid,)
    ).fetchone()[0]
    assert n_bans == len(match["dynamic_fields"]["ban_pick_info"])

    # uid 130729830 has two hero segments in the fixture (a mid-match swap).
    n_hero_segments = conn.execute(
        "SELECT COUNT(*) FROM match_player_heroes WHERE match_uid=? AND player_uid=130729830",
        (match_uid,),
    ).fetchone()[0]
    assert n_hero_segments == 2


def test_ingest_match_is_idempotent():
    conn = make_conn()
    match = load_match()
    ingest.ingest_match(conn, match, season=19)
    ingest.ingest_match(conn, match, season=19)
    n_players = conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE match_uid=?", (match["match_uid"],)
    ).fetchone()[0]
    assert n_players == 12


def test_ingest_match_creates_pending_players_with_discovery_hero_id():
    conn = make_conn()
    match = load_match()
    new_uids = ingest.ingest_match(conn, match, season=19)
    assert 130729830 in new_uids
    row = conn.execute(
        "SELECT crawl_status, discovery_hero_id FROM players WHERE uid=130729830"
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] == 1041  # cur_hero_id for that player in the fixture


def test_ingest_match_does_not_overwrite_existing_players_discovery_hero_id_or_status():
    conn = make_conn()
    match = load_match()
    # Simulate a player already known from hero-leaderboard seeding.
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 130729830, "discovery_hero_id": 9999, "crawl_status": "done"},
    )
    ingest.ingest_match(conn, match, season=19)
    row = conn.execute(
        "SELECT discovery_hero_id, crawl_status FROM players WHERE uid=130729830"
    ).fetchone()
    assert row == (9999, "done")


def test_ingest_match_records_newly_seen_heroes():
    conn = make_conn()
    match = load_match()
    ingest.ingest_match(conn, match, season=19)
    hero_ids = {row[0] for row in conn.execute("SELECT hero_id FROM heroes").fetchall()}
    ban_hero_ids = {b["hero_id"] for b in match["dynamic_fields"]["ban_pick_info"]}
    assert ban_hero_ids <= hero_ids
