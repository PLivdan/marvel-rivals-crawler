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


def test_connect_creates_a_missing_parent_directory(tmp_path):
    # The documented run command is `--db-path data/rivals.db`, which fails
    # with "unable to open database file" on a fresh checkout without this.
    nested = tmp_path / "data" / "nested" / "rivals.db"
    assert not nested.parent.exists()

    conn = db.connect(str(nested))
    db.init_schema(conn)
    db.upsert(conn, "players", ["uid"], {"uid": 1})
    conn.commit()
    conn.close()

    assert nested.exists()


def test_connect_readonly_reads_without_taking_a_write_lock(tmp_path):
    path = str(tmp_path / "ro.db")
    writer = db.connect(path)
    db.init_schema(writer)
    db.upsert(writer, "players", ["uid"], {"uid": 1, "crawl_status": "done"})
    writer.commit()

    # Opened while the writer connection is still live, as `--status` would be.
    reader = db.connect_readonly(path)
    assert reader.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 1
    try:
        reader.execute("INSERT INTO players (uid) VALUES (2)")
        reader.commit()
        raise AssertionError("read-only connection should refuse writes")
    except sqlite3.OperationalError:
        pass
    reader.close()
    writer.close()


def test_connect_sets_a_busy_timeout_so_concurrent_writers_wait_instead_of_erroring():
    # Without this, a worker whose commit lands while another worker holds the
    # write lock fails immediately with "database is locked" instead of
    # blocking and retrying for a few seconds.
    conn = make_conn()
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS


def test_connect_readonly_also_sets_a_busy_timeout(tmp_path):
    path = str(tmp_path / "ro.db")
    writer = db.connect(path)
    db.init_schema(writer)
    writer.commit()
    writer.close()

    reader = db.connect_readonly(path)
    assert reader.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS
    reader.close()


def test_init_schema_adds_claimed_at_to_a_database_created_before_that_column(tmp_path):
    # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a DB file
    # written by an older build of this crawler would otherwise be missing
    # claimed_at and every claim would fail with "no such column".
    path = str(tmp_path / "legacy.db")
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE players ("
        "uid INTEGER PRIMARY KEY, nick_name TEXT, discovery_hero_id INTEGER, "
        "latest_known_score REAL, latest_known_level INTEGER, visibility_json TEXT, "
        "crawl_status TEXT NOT NULL DEFAULT 'pending', last_crawled_at INTEGER, "
        "created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')))"
    )
    legacy.execute("INSERT INTO players (uid, crawl_status) VALUES (1, 'pending')")
    legacy.commit()
    legacy.close()

    conn = db.connect(path)
    db.init_schema(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(players)").fetchall()}
    assert "claimed_at" in columns
    # The pre-existing row survives the migration, with a NULL claim.
    assert conn.execute("SELECT claimed_at FROM players WHERE uid=1").fetchone()[0] is None
    # And re-running is still safe (the column is added at most once).
    db.init_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 1


def test_players_created_at_defaults_without_caller_supplying_it():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 7, "nick_name": "Bob"})
    created_at = conn.execute("SELECT created_at FROM players WHERE uid=7").fetchone()[0]
    assert created_at is not None and created_at > 0
