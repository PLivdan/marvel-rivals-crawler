import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    uid INTEGER PRIMARY KEY,
    nick_name TEXT,
    discovery_hero_id INTEGER,
    latest_known_score REAL,
    latest_known_level INTEGER,
    visibility_json TEXT,
    crawl_status TEXT NOT NULL DEFAULT 'pending',
    last_crawled_at INTEGER,
    claimed_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS heroes (
    hero_id INTEGER PRIMARY KEY,
    first_seen_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    last_seeded_at INTEGER
);

CREATE TABLE IF NOT EXISTS matches (
    match_uid TEXT PRIMARY KEY,
    match_time_stamp INTEGER,
    match_play_duration REAL,
    game_mode_id INTEGER,
    map_id INTEGER,
    season INTEGER,
    winner_side INTEGER,
    mvp_uid INTEGER,
    mvp_hero_id INTEGER,
    svp_uid INTEGER,
    svp_hero_id INTEGER,
    replay_id INTEGER,
    fetched_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS match_bans (
    match_uid TEXT NOT NULL REFERENCES matches(match_uid),
    round_idx INTEGER,
    battle_side INTEGER,
    hero_id INTEGER,
    is_pick INTEGER,
    PRIMARY KEY (match_uid, round_idx, battle_side, hero_id)
);

CREATE TABLE IF NOT EXISTS match_players (
    match_uid TEXT NOT NULL REFERENCES matches(match_uid),
    player_uid INTEGER NOT NULL,
    camp INTEGER,
    cur_hero_id INTEGER,
    k INTEGER,
    d INTEGER,
    a INTEGER,
    total_hero_damage REAL,
    total_hero_heal REAL,
    total_damage_taken REAL,
    is_win INTEGER,
    add_score REAL,
    new_score REAL,
    level INTEGER,
    new_level INTEGER,
    session_hit_rate REAL,
    PRIMARY KEY (match_uid, player_uid)
);

CREATE TABLE IF NOT EXISTS match_player_heroes (
    match_uid TEXT NOT NULL REFERENCES matches(match_uid),
    player_uid INTEGER NOT NULL,
    hero_id INTEGER NOT NULL,
    k INTEGER,
    d INTEGER,
    a INTEGER,
    play_time REAL,
    PRIMARY KEY (match_uid, player_uid, hero_id)
);

-- Indexes for crawler.select_next_player, which runs once per crawled player
-- and scans the whole pending frontier (a single reseed can queue ~21,000
-- pending players across ~42 heroes). CREATE INDEX IF NOT EXISTS is safe to
-- re-run on every startup, so these self-apply to already-existing DB files.
CREATE INDEX IF NOT EXISTS idx_players_crawl_status ON players(crawl_status);
CREATE INDEX IF NOT EXISTS idx_match_player_heroes_hero_id ON match_player_heroes(hero_id);
"""

# Columns added to a table after this project's first DB files were created.
# CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so unlike the
# CREATE INDEX IF NOT EXISTS statements above, a new column does NOT self-apply
# to an already-existing DB file — it needs an explicit ALTER TABLE.
ADDED_COLUMNS = {
    "players": {
        # Set by crawler.claim_next_player when a worker takes a player, and
        # cleared when that player reaches a terminal status. Deliberately
        # separate from last_crawled_at, which crawl_player reads as "has this
        # player ever COMPLETED a crawl" — stamping that at claim time would
        # make every first crawl look like a revisit.
        "claimed_at": "INTEGER",
    },
}

# Concurrent workers hold one connection each to the same WAL-mode file, so a
# writer can briefly find the write lock held by another worker's commit. This
# makes SQLite block and retry for up to 5s instead of failing the statement
# immediately with "database is locked".
BUSY_TIMEOUT_MS = 5000


def connect(path):
    # The documented run command is `python main.py --db-path data/rivals.db`,
    # which would fail with "unable to open database file" on a fresh checkout
    # where data/ doesn't exist yet.
    if path != ":memory:" and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.row_factory = None
    return conn


def connect_readonly(path):
    """Open an EXISTING database read-only, for out-of-band reads (--status)
    that must not contend with a live crawl process on the same file. Takes no
    write lock and creates nothing: the WAL/foreign-key pragmas connect() sets
    only affect writes, and setting journal_mode on a read-only handle would
    itself require a write. Errors if the database doesn't exist yet — correct,
    since there's nothing to report on until a crawl has run."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    # A pure connection-level setting that needs no write, so it is safe here
    # even though journal_mode/foreign_keys are not: it lets --status wait out
    # a worker's in-flight commit rather than erroring on a locked database.
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.row_factory = None
    return conn


def init_schema(conn):
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    conn.commit()


def _add_missing_columns(conn):
    """Apply ADDED_COLUMNS to a DB file created before those columns existed.

    Keeps init_schema's "safe to re-run on every startup, self-applies to
    already-existing DB files" property (see the CREATE INDEX comment in
    SCHEMA), which CREATE TABLE IF NOT EXISTS alone does not give for columns.
    """
    for table, columns in ADDED_COLUMNS.items():
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in columns.items():
            if name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def upsert(conn, table, pk_cols, row):
    cols = list(row.keys())
    col_list = ",".join(cols)
    placeholders = ",".join("?" for _ in cols)
    update_cols = [c for c in cols if c not in pk_cols]
    if update_cols:
        update_clause = ",".join(f"{c}=excluded.{c}" for c in update_cols)
        conflict_clause = f"ON CONFLICT({','.join(pk_cols)}) DO UPDATE SET {update_clause}"
    else:
        conflict_clause = f"ON CONFLICT({','.join(pk_cols)}) DO NOTHING"
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) {conflict_clause}"
    conn.execute(sql, [row[c] for c in cols])


def now():
    return int(time.time())
