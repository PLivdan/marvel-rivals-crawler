import sqlite3
import db


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def test_init_schema_creates_all_tables():
    conn = make_conn()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "players",
        "heroes",
        "matches",
        "match_bans",
        "match_players",
        "match_player_heroes",
    } <= tables


def test_upsert_insert_then_partial_update_preserves_other_columns():
    conn = make_conn()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "nick_name": "Alice", "discovery_hero_id": 1042, "crawl_status": "pending"},
    )
    # A later partial update (as done when refreshing score/level) must not
    # clobber discovery_hero_id or crawl_status.
    db.upsert(conn, "players", ["uid"], {"uid": 1, "latest_known_score": 5000.0})
    row = conn.execute(
        "SELECT nick_name, discovery_hero_id, crawl_status, latest_known_score FROM players WHERE uid=1"
    ).fetchone()
    assert row == ("Alice", 1042, "pending", 5000.0)


def test_upsert_is_idempotent():
    conn = make_conn()
    row = {"match_uid": "m1", "match_time_stamp": 100, "winner_side": 1}
    db.upsert(conn, "matches", ["match_uid"], row)
    db.upsert(conn, "matches", ["match_uid"], row)
    count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert count == 1


def test_players_created_at_defaults_without_caller_supplying_it():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 7, "nick_name": "Bob"})
    created_at = conn.execute("SELECT created_at FROM players WHERE uid=7").fetchone()[0]
    assert created_at is not None and created_at > 0
