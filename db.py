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
"""


def connect(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = None
    return conn


def init_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


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
