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

-- Reference data scraped from the site's client bundle by refresh_reference.py,
-- NOT from the JSON API and NOT touched by the crawl. hero_id/teamup_id are the
-- same ids the match endpoints emit, so these join straight onto match_players,
-- match_player_heroes and match_bans to turn opaque ids into names.
CREATE TABLE IF NOT EXISTS hero_info (
    hero_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT,
    class INTEGER,
    difficulty INTEGER,
    gender TEXT,
    attack_method TEXT,
    internal_name TEXT,
    real_name TEXT,
    refreshed_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

-- One row per team-up ability. `anchor_hero_id` is the hero who owns the
-- ability; the partner(s) who enable it live in teamup_heroes. end_sub_season
-- IS NULL means the team-up is live in the current season — team-ups are
-- re-worked between seasons, so a match's team-ups are only meaningful
-- alongside matches.season.
CREATE TABLE IF NOT EXISTS teamups (
    teamup_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    anchor_hero_id INTEGER,
    start_sub_season TEXT,
    end_sub_season TEXT,
    description TEXT,
    refreshed_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

-- Membership edge. Includes the anchor itself (is_anchor=1), so the full
-- roster of a team-up is one query with no union against teamups.
CREATE TABLE IF NOT EXISTS teamup_heroes (
    teamup_id INTEGER NOT NULL REFERENCES teamups(teamup_id),
    hero_id INTEGER NOT NULL,
    is_anchor INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (teamup_id, hero_id)
);

CREATE INDEX IF NOT EXISTS idx_teamup_heroes_hero_id ON teamup_heroes(hero_id);

-- Indexes for crawler.select_next_player, which runs once per crawled player
-- and scans the whole pending frontier (a single reseed can queue ~21,000
-- pending players across ~42 heroes). CREATE INDEX IF NOT EXISTS is safe to
-- re-run on every startup, so these self-apply to already-existing DB files.
CREATE INDEX IF NOT EXISTS idx_players_crawl_status ON players(crawl_status);
CREATE INDEX IF NOT EXISTS idx_match_player_heroes_hero_id ON match_player_heroes(hero_id);

-- Running per-hero total of match_player_heroes rows, kept by the trigger
-- below and rebuilt from scratch by init_schema on every startup. The claim
-- priority needs "how much data do we already hold for each hero" on every
-- claim; aggregating it live cost 0.9s at 3.9M rows (2026-09-20, six workers
-- doing it concurrently) and grew with the table. The trigger fires only on
-- a genuine INSERT — db.upsert's ON CONFLICT DO UPDATE path is an UPDATE, so
-- re-ingesting a known row does not inflate the total.
CREATE TABLE IF NOT EXISTS hero_coverage (
    hero_id INTEGER PRIMARY KEY,
    n INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_hero_coverage_insert
AFTER INSERT ON match_player_heroes
BEGIN
    INSERT INTO hero_coverage (hero_id, n) VALUES (NEW.hero_id, 1)
    ON CONFLICT(hero_id) DO UPDATE SET n = n + 1;
END;

-- Lets crawler.select_next_player find the oldest pending player of a given
-- discovery hero, and walk the distinct discovery heroes that have pending
-- players, with index seeks alone. Without it the selection sorted the whole
-- pending frontier through a temp B-tree on every claim: 1-3.7s at 410k
-- pending, and six workers racing for the same deterministic head.
CREATE INDEX IF NOT EXISTS idx_players_pending_priority
    ON players(crawl_status, discovery_hero_id, created_at);

-- The progress line counts players finished in the last ten minutes every 60 s;
-- without this it was a full scan of players (1.5 s at 440k rows, review M3).
CREATE INDEX IF NOT EXISTS idx_players_last_crawled ON players(last_crawled_at);

-- Last request time of the previous run, read at startup to detect a warm restart.
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- Every match-history entry the crawl has read: (match, the crawled player) with that
-- player's leaver flag and result. The history endpoint is the only place `has_escaped`
-- exists, and only for the player whose history it is; recorded for known matches too,
-- at no extra request (2026-09-24 model review: leavers decide matches).
CREATE TABLE IF NOT EXISTS history_entries (
    match_uid TEXT NOT NULL,
    player_uid INTEGER NOT NULL,
    has_escaped INTEGER,
    is_win INTEGER,
    seen_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    PRIMARY KEY (match_uid, player_uid)
);

-- Indexes for reading the data back out, not for the crawl itself. Both
-- match_players and match_player_heroes are keyed by (match_uid, ...), so
-- player_uid is only ever a non-leading column and any per-player lookup
-- degrades to a full table scan (measured: a per-player EXISTS over the
-- players table did not finish in 120s at 1.1M match_player rows).
-- cur_hero_id and match_time_stamp carry the hero-level and patch-window
-- filters the regression is built on. Write cost is negligible here: the
-- crawler is network-bound at ~18 requests/sec, so index maintenance never
-- becomes the bottleneck, while the tables are headed for millions of rows.
CREATE INDEX IF NOT EXISTS idx_match_players_player_uid ON match_players(player_uid);
CREATE INDEX IF NOT EXISTS idx_match_player_heroes_player_uid ON match_player_heroes(player_uid);
CREATE INDEX IF NOT EXISTS idx_match_players_cur_hero_id ON match_players(cur_hero_id);
CREATE INDEX IF NOT EXISTS idx_matches_time_stamp ON matches(match_time_stamp);

-- Covering indexes for the APM feature build, added after profiling the real
-- 483k-match database. Both replace a plan that read far more than it needed:
--
--   attribution_weights' pull+join was a raw TABLE SCAN of match_player_heroes
--   (776 MB) because the PK autoindex (match_uid, player_uid, hero_id) does not
--   carry play_time. Adding it makes the query covering: 16.9s -> 9.1s, and it
--   runs twice per design build. The same index also serves W0's
--   MIN(rowid)-per-player grouping, which is narrower here than on the PK
--   autoindex: 64.1s -> 5.1s. (rowid itself cannot be indexed -- SQLite rejects
--   it with "no such column" -- so this is the only available route.)
--
--   The skill differential groups by (match_uid, camp) while the PK is
--   (match_uid, player_uid), so camp was out of index order and SQLite sorted
--   5.8M rows through a temp B-tree. Covering it removes the sort: 16.7s -> 1.8s.
--
-- Together they cost ~1.1 GB on a 3.4 GB database. That is a deliberate trade:
-- these run on every fit, and the crawler is network-bound so the extra write
-- cost never becomes its bottleneck.
CREATE INDEX IF NOT EXISTS idx_mph_cover_attribution
    ON match_player_heroes(match_uid, player_uid, hero_id, play_time);
CREATE INDEX IF NOT EXISTS idx_mp_cover_skill
    ON match_players(match_uid, camp, new_score, add_score);

-- Model results. Coefficients are meaningless without the run metadata that
-- produced them, so apm_runs records the specification, attribution rule,
-- sample filters and git commit alongside every fit.
CREATE TABLE IF NOT EXISTS apm_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    spec TEXT NOT NULL,
    attribution_rule TEXT NOT NULL,
    forfeit_floor INTEGER,
    n_matches INTEGER,
    n_players INTEGER,
    exclusions TEXT,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS apm_hero_effects (
    run_id INTEGER NOT NULL REFERENCES apm_runs(run_id),
    hero_id INTEGER NOT NULL,
    effect_logodds REAL,
    std_error REAL,
    ci_low REAL,
    ci_high REAL,
    p_adjusted REAL,
    PRIMARY KEY (run_id, hero_id)
);

CREATE TABLE IF NOT EXISTS apm_teamup_effects (
    run_id INTEGER NOT NULL REFERENCES apm_runs(run_id),
    teamup_id INTEGER NOT NULL,
    effect_logodds REAL,
    std_error REAL,
    PRIMARY KEY (run_id, teamup_id)
);
"""

# Columns added to a table after this project's first DB files were created.
# CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so unlike the
# CREATE INDEX IF NOT EXISTS statements above, a new column does NOT self-apply
# to an already-existing DB file — it needs an explicit ALTER TABLE.
ADDED_COLUMNS = {
    "matches": {
        # The crawled player whose match history first surfaced this match. The crawl reaches
        # matches through hero-leaderboard specialists first, so hero effects can be biased by
        # those players' hero-specific skill; this is what makes that measurable
        # (2026-09-24 model review). NULL for matches ingested before it existed.
        "source_player_uid": "INTEGER",
    },
    "players": {
        # Set by crawler.claim_next_player when a worker takes a player, and
        # cleared when that player reaches a terminal status. Deliberately
        # separate from last_crawled_at, which crawl_player reads as "has this
        # player ever COMPLETED a crawl" — stamping that at claim time would
        # make every first crawl look like a revisit.
        "claimed_at": "INTEGER",
        # When latest_known_level/score were observed: the match's time for a level taken from a
        # match, the crawl time for one taken from the profile. A match only replaces the level
        # if it is at least this new. Before, every ingested match overwrote it in whatever order
        # matches arrived, so an old sub-Diamond match could floor-skip a Diamond+ player for
        # good (11,598 such players in the 2026-10-02 season 10 database).
        "level_seen_at": "INTEGER",
    },
}

# Run once, right after ADDED_COLUMNS adds the column it names, to fill it for the rows that
# already exist.
BACKFILLS = {
    ("players", "level_seen_at"): """
        UPDATE players SET (latest_known_level, latest_known_score, level_seen_at) = (
            SELECT m.new_level, m.new_score, x.match_time_stamp
              FROM match_players m JOIN matches x ON x.match_uid = m.match_uid
             WHERE m.player_uid = players.uid AND m.new_level IS NOT NULL
               AND x.match_time_stamp IS NOT NULL
             ORDER BY x.match_time_stamp DESC LIMIT 1)
         WHERE EXISTS (
            SELECT 1 FROM match_players m JOIN matches x ON x.match_uid = m.match_uid
             WHERE m.player_uid = players.uid AND m.new_level IS NOT NULL
               AND x.match_time_stamp IS NOT NULL)
    """,
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
    _rebuild_hero_coverage(conn)
    _migrate_revisits_out_of_pending(conn)
    conn.commit()


def _migrate_revisits_out_of_pending(conn):
    """Queued revisits used to share 'pending' with first crawls. A pending row with
    last_crawled_at set is a revisit (error and orphan retries clear the timestamp),
    so it moves to 'revisit', which is claimed only when no first crawl waits."""
    conn.execute(
        "UPDATE players SET crawl_status='revisit' "
        "WHERE crawl_status='pending' AND last_crawled_at IS NOT NULL"
    )


def get_meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn, key, value):
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
    conn.commit()


def _rebuild_hero_coverage(conn):
    """Recompute hero_coverage from match_player_heroes.

    Runs on every startup, not only when the table is empty: a database that
    predates the table has rows the trigger never saw, and rebuilding
    unconditionally also heals any drift without needing to detect it. One
    GROUP BY over the table — ~1s at 4M rows — once per process, instead of
    once per claim."""
    conn.execute("DELETE FROM hero_coverage")
    conn.execute(
        "INSERT INTO hero_coverage (hero_id, n) "
        "SELECT hero_id, COUNT(*) FROM match_player_heroes GROUP BY hero_id"
    )


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
                if (table, name) in BACKFILLS:
                    conn.execute(BACKFILLS[(table, name)])


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
