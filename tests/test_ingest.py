import json
import pathlib

import db
import ingest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_match():
    return json.loads((FIXTURES / "match_detail.json").read_text())


def load_history_page():
    return json.loads((FIXTURES / "player_match_history_page.json").read_text())


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


def test_ingest_match_stores_map_id_from_history_entry():
    # /api/matches/{uid} has NO map_id / match_map_id key — the map id only
    # exists on the player-match-history entry that surfaced the match, so
    # production code must pass that entry through or map_id is always NULL.
    conn = make_conn()
    match = load_match()
    # The two captured fixtures are from different matches, so pair the real
    # history entry's real match_map_id with the detail fixture's match_uid.
    history_entry = dict(load_history_page()[0])
    assert history_entry["match_map_id"] == 1245  # real captured value
    history_entry["match_uid"] = match["match_uid"]

    ingest.ingest_match(conn, match, season=19, history_entry=history_entry)

    map_id = conn.execute(
        "SELECT map_id FROM matches WHERE match_uid=?", (match["match_uid"],)
    ).fetchone()[0]
    assert map_id is not None
    assert map_id == 1245


def test_ingest_match_leaves_map_id_null_without_a_history_entry():
    conn = make_conn()
    match = load_match()
    ingest.ingest_match(conn, match, season=19)
    map_id = conn.execute(
        "SELECT map_id FROM matches WHERE match_uid=?", (match["match_uid"],)
    ).fetchone()[0]
    assert map_id is None


def test_ingest_match_records_newly_seen_heroes():
    conn = make_conn()
    match = load_match()
    ingest.ingest_match(conn, match, season=19)
    hero_ids = {row[0] for row in conn.execute("SELECT hero_id FROM heroes").fetchall()}
    ban_hero_ids = {b["hero_id"] for b in match["dynamic_fields"]["ban_pick_info"]}
    assert ban_hero_ids <= hero_ids
