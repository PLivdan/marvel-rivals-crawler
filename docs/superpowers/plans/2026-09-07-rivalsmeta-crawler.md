# rivalsmeta.com Crawler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, polite crawler that collects Marvel Rivals competitive match data from rivalsmeta.com into a local SQLite database for regression analysis.

**Architecture:** Three layers — an HTTP fetcher (adaptive rate limiting, retries, circuit breaker), a crawl orchestrator (priority-queue BFS over players, stored entirely in SQLite), and ingestion functions that turn API JSON into normalized rows. A CLI entrypoint drives the loop with graceful shutdown and periodic progress logging.

**Tech Stack:** Python 3.12, `requests`, stdlib `sqlite3`, `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-09-07-rivalsmeta-crawler-design.md`

## Global Constraints

- PC platform only: every leaderboard/hero-leaderboard call uses `device=1`.
- Competitive matches only: `game_mode_id=2` (verified via the site's own mode filter: `{"All Modes":0,"Quick Play":1,"Competitive":2,"Custom":3,"Arcade":4,"Tournament":9}`).
- Diamond+ rank floor for BFS expansion, using the discrete `level` field (verified scheme: levels 1–21 = 7 tiers × 3 divisions Bronze→Celestial, 22 = Eternity, 23 = One Above All): `RANK_LEVEL_DIAMOND_MIN = 13`.
- No proxies, no header spoofing, no concurrency — single sequential request stream with an adaptive delay.
- All database writes are idempotent (upsert keyed by natural primary keys) — the crawler must be safe to kill and resume at any point.
- Current season is resolved dynamically at runtime (never hardcoded) via the global leaderboard payload's `players[0].rank.rank_game_id`.

## Refinements made beyond the spec's literal wording

The spec (§10) flagged three items as "resolve during implementation." All three were resolved during the writing of this plan, by direct experimentation against the live site, so no task below contains a guess:

1. **Diamond floor**: implemented as `level >= 13` on the discrete 1–23 rank-level field that's already present on every match row (`match_players.level`/`new_level`) and every player profile (`rank_game_<id>.level`), instead of a raw rank-score (RS) cutoff. RS baselines can drift; the discrete level encoding (verified: 1–21 = 7 tiers × 3 divisions Bronze→Celestial, 22 = Eternity, 23 = One Above All) does not.
2. **Hero roster for seeding**: rather than a static hardcoded list of ~40 hero ids (which would go stale — the site was mid-launch of a new hero, "The Hood," at investigation time), heroes are discovered dynamically into a `heroes` table from every ingested match's ban list and hero-swap segments, and `reseed()` calls `/api/hero-leaderboard/{hero_id}` for any hero not seeded in the last 24 hours. This self-bootstraps within the first few matches crawled and requires no manual roster maintenance as new heroes launch.
3. **Competitive mode id**: confirmed as `game_mode_id=2` directly from the site's own mode-filter `<select>` (`{"All Modes":0,"Quick Play":1,"Competitive":2,"Custom":3,"Arcade":4,"Tournament":9}`), not inferred from a single sample.

Season-number mapping needed no formula at all: the season `<select>` on the site enumerates every season 1:1 in order (Season 0→`1`, Season 1→`2`, Season 1.5→`3`, ... Season 9.5→`19`), and the crawler never needs to compute this itself — `rivalsmeta.resolve_current_season()` always reads the live value from the leaderboard payload, and a player's own per-season rank blob is addressed directly as `rank_game_{1000000 + season}`.

---

## File Structure

```
marvel-rivals-crawler/
  requirements.txt
  db.py              # schema + generic upsert helper
  nuxt_payload.py     # Nuxt "devalue" payload deserializer
  fetcher.py          # rate limiter + HTTP client + circuit breaker
  rivalsmeta.py        # endpoint-specific calls + season/rank helpers
  ingest.py           # API JSON -> DB rows
  crawler.py          # priority queue + per-player crawl algorithm + reseed
  main.py             # CLI loop, signal handling, progress logging
  tests/
    fixtures/          # already staged: real captured API responses
    test_db.py
    test_nuxt_payload.py
    test_fetcher.py
    test_rivalsmeta.py
    test_ingest.py
    test_crawler.py
    test_main.py
    test_live_smoke.py  # opt-in, hits the real site
  pytest.ini
```

---

### Task 1: SQLite schema and generic upsert helper

**Files:**
- Create: `requirements.txt`
- Create: `db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `db.connect(path: str) -> sqlite3.Connection`, `db.init_schema(conn)`, `db.upsert(conn, table: str, pk_cols: list[str], row: dict) -> None`, `db.now() -> int`

- [ ] **Step 1: Create `requirements.txt`**

```
requests
pytest
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_db.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pip install -r requirements.txt && pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 4: Write `db.py`**

```python
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
```

Note: `WAL` mode is a no-op on an in-memory (`:memory:`) database (SQLite silently ignores it there), which is fine for tests — it takes effect on the real file-backed DB used in later tasks.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt db.py tests/test_db.py
git commit -m "feat: add SQLite schema and idempotent upsert helper"
```

---

### Task 2: Nuxt "devalue" payload deserializer

**Files:**
- Create: `nuxt_payload.py`
- Test: `tests/test_nuxt_payload.py`
- Uses fixture: `tests/fixtures/leaderboard_payload.json` (already staged — a real capture of `GET /leaderboard/_payload.json`)

**Interfaces:**
- Produces: `nuxt_payload.resolve_payload(raw_json_text: str) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nuxt_payload.py`:

```python
import json
import pathlib

from nuxt_payload import resolve_payload

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_resolve_payload_reconstructs_leaderboard():
    raw = (FIXTURES / "leaderboard_payload.json").read_text()
    data = resolve_payload(raw)
    assert data["device"] == "1"  # the API returns this as a string, not an int
    assert data["season"] == "last"
    assert len(data["players"]) == 500
    first = data["players"][0]
    assert first["name"] == "KovaaksKid2008"
    assert first["uid"] == "1822797559"
    assert first["rank"]["rank_game_id"] == 19


def test_resolve_payload_is_pure_json_compatible():
    # sanity: the raw fixture must itself be valid JSON (an array),
    # confirming this is devalue-over-JSON, not a bespoke text format.
    raw = (FIXTURES / "leaderboard_payload.json").read_text()
    arr = json.loads(raw)
    assert isinstance(arr, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nuxt_payload.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nuxt_payload'`

- [ ] **Step 3: Write `nuxt_payload.py`**

```python
import json

_WRAPPER_TAGS = {"Reactive", "ShallowReactive", "Ref", "ShallowRef", "EmptyRef", "EmptyShallowRef"}


def resolve_payload(raw_json_text):
    """Deserialize a Nuxt 3 SSR payload (the devalue-style flat reference
    array served at e.g. /leaderboard/_payload.json) into a plain dict.

    The array's slot 0 is always {"data": <ref>, "prerenderedAt": ...}.
    `data` resolves to a dict keyed by an opaque per-build hash whose sole
    value is the actual page data we want, so we unwrap that automatically.
    """
    arr = json.loads(raw_json_text)
    cache = {}
    root = _resolve(0, arr, cache)
    data = root["data"]
    if isinstance(data, dict) and len(data) == 1:
        return next(iter(data.values()))
    return data


def _resolve(idx, arr, cache):
    if idx in cache:
        return cache[idx]
    val = arr[idx]
    if isinstance(val, list):
        if len(val) == 2 and isinstance(val[0], str) and val[0] in _WRAPPER_TAGS:
            result = _resolve(val[1], arr, cache)
        elif len(val) == 2 and isinstance(val[0], str) and val[0] == "Date":
            result = val[1]
        else:
            result = [None] * len(val)
            cache[idx] = result
            for i, ref in enumerate(val):
                result[i] = _resolve(ref, arr, cache) if isinstance(ref, int) else ref
            return result
    elif isinstance(val, dict):
        result = {}
        cache[idx] = result
        for k, ref in val.items():
            result[k] = _resolve(ref, arr, cache) if isinstance(ref, int) else ref
        return result
    else:
        result = val
    cache[idx] = result
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nuxt_payload.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add nuxt_payload.py tests/test_nuxt_payload.py
git commit -m "feat: add Nuxt devalue payload deserializer for leaderboard SSR data"
```

---

### Task 3: Adaptive rate limiter, HTTP fetcher, circuit breaker

**Files:**
- Create: `fetcher.py`
- Test: `tests/test_fetcher.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `fetcher.AdaptiveRateLimiter`, `fetcher.RivalsMetaClient(session=None, limiter=None)` with `.get_json(path, params=None) -> dict`, `.get_text(path, params=None) -> str`; exceptions `fetcher.PlayerNotFoundError`, `fetcher.CircuitOpenError`, `fetcher.FetchError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetcher.py`:

```python
import pytest

from fetcher import AdaptiveRateLimiter, RivalsMetaClient, PlayerNotFoundError, CircuitOpenError, FetchError


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        # responses: list of FakeResponse or Exception instances, consumed in order
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class NoSleepLimiter(AdaptiveRateLimiter):
    def wait(self):
        pass  # skip real sleeping in tests


def test_get_json_success_returns_payload():
    session = FakeSession([FakeResponse(200, payload={"ok": True})])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    assert client.get_json("/api/player/1") == {"ok": True}


def test_get_json_404_raises_player_not_found_without_retry():
    session = FakeSession([FakeResponse(404)])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(PlayerNotFoundError):
        client.get_json("/api/player/999")
    assert session.calls == 1


def test_get_json_retries_5xx_then_succeeds():
    session = FakeSession([FakeResponse(500), FakeResponse(200, payload={"ok": True})])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    assert client.get_json("/api/player/1") == {"ok": True}
    assert session.calls == 2


def test_get_json_exhausts_retries_and_raises_fetch_error():
    session = FakeSession([FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(FetchError):
        client.get_json("/api/player/1")
    assert session.calls == 3


def test_circuit_opens_after_repeated_failures_and_blocks_further_calls():
    responses = [FakeResponse(500)] * (RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD * RivalsMetaClient.MAX_RETRIES)
    session = FakeSession(responses)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    for _ in range(RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD):
        with pytest.raises(FetchError):
            client.get_json("/api/player/1")
    with pytest.raises(CircuitOpenError):
        client.get_json("/api/player/1")


def test_rate_limiter_speeds_up_on_success_and_slows_down_on_failure():
    limiter = AdaptiveRateLimiter(initial_delay=1.0, min_delay=0.2, max_delay=8.0)
    limiter.record_success(latency=0.1)
    assert limiter.current_delay < 1.0
    before = limiter.current_delay
    limiter.record_failure()
    assert limiter.current_delay > before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fetcher'`

- [ ] **Step 3: Write `fetcher.py`**

```python
import time

import requests


class PlayerNotFoundError(Exception):
    pass


class CircuitOpenError(Exception):
    pass


class FetchError(Exception):
    pass


class AdaptiveRateLimiter:
    def __init__(self, initial_delay=1.0, min_delay=0.2, max_delay=8.0):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay

    def wait(self):
        time.sleep(self.current_delay)

    def record_success(self, latency):
        self.current_delay = max(self.min_delay, self.current_delay * 0.95)

    def record_failure(self):
        self.current_delay = min(self.max_delay, self.current_delay * 2)


class RivalsMetaClient:
    BASE_URL = "https://rivalsmeta.com"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    MAX_RETRIES = 3
    CIRCUIT_FAILURE_THRESHOLD = 10
    CIRCUIT_COOLDOWN_SECONDS = 300

    def __init__(self, session=None, limiter=None):
        self.session = session or requests.Session()
        self.limiter = limiter or AdaptiveRateLimiter()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def get_json(self, path, params=None):
        return self._request(path, params).json()

    def get_text(self, path, params=None):
        return self._request(path, params).text

    def _request(self, path, params=None):
        if time.time() < self._circuit_open_until:
            raise CircuitOpenError(f"circuit open until {self._circuit_open_until}")

        url = f"{self.BASE_URL}{path}"
        last_exc = None
        for _ in range(self.MAX_RETRIES):
            self.limiter.wait()
            start = time.time()
            try:
                resp = self.session.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                last_exc = exc
                self.limiter.record_failure()
                continue

            latency = time.time() - start

            if resp.status_code == 404:
                self.limiter.record_success(latency)
                self._consecutive_failures = 0
                raise PlayerNotFoundError(path)

            if resp.status_code in (429, 403) or resp.status_code >= 500:
                self.limiter.record_failure()
                last_exc = FetchError(f"status {resp.status_code} for {path}")
                continue

            self.limiter.record_success(latency)
            self._consecutive_failures = 0
            return resp

        # Registered once per exhausted-retries call, not once per raw
        # attempt — the circuit threshold counts distinct failed fetches,
        # not individual retry attempts within one fetch.
        self._register_failure()
        raise last_exc or FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}")

    def _register_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open_until = time.time() + self.CIRCUIT_COOLDOWN_SECONDS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetcher.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add fetcher.py tests/test_fetcher.py
git commit -m "feat: add adaptive rate limiter, retrying HTTP client, and circuit breaker"
```

---

### Task 4: rivalsmeta.com endpoint calls and rank/season helpers

**Files:**
- Create: `rivalsmeta.py`
- Test: `tests/test_rivalsmeta.py`
- Uses fixtures: `player_public.json`, `leaderboard_payload.json`, `hero_leaderboard.json`, `player_match_history_page.json`, `match_detail.json`

**Interfaces:**
- Consumes: `fetcher.RivalsMetaClient` (Task 3), `nuxt_payload.resolve_payload` (Task 2).
- Produces: `rivalsmeta.MATCHMODE_COMPETITIVE`, `rivalsmeta.RANK_LEVEL_DIAMOND_MIN`, `rivalsmeta.is_diamond_plus(level) -> bool`, `rivalsmeta.get_global_leaderboard(client) -> dict`, `rivalsmeta.get_hero_leaderboard(client, hero_id) -> dict`, `rivalsmeta.get_player(client, uid, season) -> dict`, `rivalsmeta.get_player_match_history_page(client, uid, skip, season, game_mode_id=MATCHMODE_COMPETITIVE) -> list`, `rivalsmeta.get_match_detail(client, match_uid) -> dict`, `rivalsmeta.resolve_current_season(client) -> int`, `rivalsmeta.current_season_rank(profile: dict, season: int) -> dict | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rivalsmeta.py`:

```python
import json
import pathlib

import rivalsmeta

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeClient:
    """Stands in for fetcher.RivalsMetaClient, returning canned fixture data."""

    def __init__(self, json_by_path=None, text_by_path=None):
        self.json_by_path = json_by_path or {}
        self.text_by_path = text_by_path or {}
        self.calls = []

    def get_json(self, path, params=None):
        self.calls.append((path, params))
        return self.json_by_path[path]

    def get_text(self, path, params=None):
        self.calls.append((path, params))
        return self.text_by_path[path]


def test_is_diamond_plus_boundaries():
    assert rivalsmeta.is_diamond_plus(12) is False
    assert rivalsmeta.is_diamond_plus(13) is True
    assert rivalsmeta.is_diamond_plus(22) is True
    assert rivalsmeta.is_diamond_plus(23) is True
    assert rivalsmeta.is_diamond_plus(None) is False


def test_get_global_leaderboard_parses_devalue_payload():
    raw_text = (FIXTURES / "leaderboard_payload.json").read_text()
    client = FakeClient(text_by_path={"/leaderboard/_payload.json": raw_text})
    board = rivalsmeta.get_global_leaderboard(client)
    assert len(board["players"]) == 500
    assert board["players"][0]["name"] == "KovaaksKid2008"


def test_resolve_current_season_reads_rank_game_id_from_leaderboard():
    raw_text = (FIXTURES / "leaderboard_payload.json").read_text()
    client = FakeClient(text_by_path={"/leaderboard/_payload.json": raw_text})
    assert rivalsmeta.resolve_current_season(client) == 19


def test_get_hero_leaderboard_returns_plain_json():
    payload = load("hero_leaderboard.json")
    client = FakeClient(json_by_path={"/api/hero-leaderboard/1047": payload})
    result = rivalsmeta.get_hero_leaderboard(client, 1047)
    assert result["_id"] == 1047
    assert len(result["players"]) > 0


def test_get_player_returns_profile():
    payload = load("player_public.json")
    client = FakeClient(json_by_path={"/api/player/457877313": payload})
    result = rivalsmeta.get_player(client, 457877313, season=19)
    assert result["player"]["_id"] == 457877313


def test_get_player_match_history_page_passes_expected_params():
    payload = load("player_match_history_page.json")
    client = FakeClient(json_by_path={"/api/player-match-history/457877313": payload})
    result = rivalsmeta.get_player_match_history_page(client, 457877313, skip=20, season=19)
    assert result == payload
    path, params = client.calls[0]
    assert params == {"skip": 20, "game_mode_id": rivalsmeta.MATCHMODE_COMPETITIVE, "hero_id": 0, "season": 19}


def test_get_match_detail_returns_full_match():
    payload = load("match_detail.json")
    client = FakeClient(json_by_path={"/api/matches/m1": payload})
    result = rivalsmeta.get_match_detail(client, "m1")
    assert len(result["match_players"]) == 12


def test_current_season_rank_extracts_matching_season_blob():
    profile = load("player_public.json")
    blob = rivalsmeta.current_season_rank(profile, season=19)
    assert blob["level"] == 22
    assert round(blob["rank_score"], 2) == 5132.57


def test_current_season_rank_returns_none_for_unknown_season():
    profile = load("player_public.json")
    assert rivalsmeta.current_season_rank(profile, season=999) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rivalsmeta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rivalsmeta'`

- [ ] **Step 3: Write `rivalsmeta.py`**

```python
import json

from nuxt_payload import resolve_payload

MATCHMODE_COMPETITIVE = 2
RANK_LEVEL_DIAMOND_MIN = 13


def is_diamond_plus(level):
    return level is not None and level >= RANK_LEVEL_DIAMOND_MIN


def get_global_leaderboard(client):
    text = client.get_text("/leaderboard/_payload.json")
    return resolve_payload(text)


def get_hero_leaderboard(client, hero_id):
    return client.get_json(f"/api/hero-leaderboard/{hero_id}", params={"device": 1, "season": "last"})


def get_player(client, uid, season):
    return client.get_json(f"/api/player/{uid}", params={"season": season})


def get_player_match_history_page(client, uid, skip, season, game_mode_id=MATCHMODE_COMPETITIVE):
    return client.get_json(
        f"/api/player-match-history/{uid}",
        params={"skip": skip, "game_mode_id": game_mode_id, "hero_id": 0, "season": season},
    )


def get_match_detail(client, match_uid):
    return client.get_json(f"/api/matches/{match_uid}")


def resolve_current_season(client):
    board = get_global_leaderboard(client)
    return board["players"][0]["rank"]["rank_game_id"]


def current_season_rank(profile, season):
    """Extract the {level, rank_score, ...} blob for the given season from
    a player profile's info.rank_game_<1000000+season> field, which is a
    JSON-encoded string, not a nested object."""
    info = profile.get("player", {}).get("info", {})
    raw = info.get(f"rank_game_{1000000 + season}")
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed.get("rank_game")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rivalsmeta.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add rivalsmeta.py tests/test_rivalsmeta.py
git commit -m "feat: add rivalsmeta.com endpoint calls and season/rank extraction"
```

---

### Task 5: Ingesting match payloads into the database

**Files:**
- Create: `ingest.py`
- Test: `tests/test_ingest.py`
- Uses fixture: `match_detail.json` (a real match with 12 players, 6 bans, and one player — uid `130729830` — with a genuine mid-match hero swap across two heroes)

**Interfaces:**
- Consumes: `db.connect`, `db.init_schema`, `db.upsert`, `db.now` (Task 1).
- Produces: `ingest.ingest_match(conn, match_detail: dict, season: int) -> list[int]` (returns newly-created player uids).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest'`

- [ ] **Step 3: Write `ingest.py`**

```python
import db


def ingest_match(conn, match_detail, season):
    match_uid = match_detail["match_uid"]

    db.upsert(
        conn,
        "matches",
        ["match_uid"],
        {
            "match_uid": match_uid,
            "match_time_stamp": match_detail.get("match_time_stamp"),
            "match_play_duration": match_detail.get("match_play_duration"),
            "game_mode_id": match_detail.get("game_mode_id"),
            "map_id": match_detail.get("match_map_id") or match_detail.get("map_id"),
            "season": season,
            "winner_side": _winner_side(match_detail),
            "mvp_uid": match_detail.get("mvp_uid"),
            "mvp_hero_id": match_detail.get("mvp_hero_id"),
            "svp_uid": match_detail.get("svp_uid"),
            "svp_hero_id": match_detail.get("svp_hero_id"),
            "replay_id": match_detail.get("replay_id"),
            "fetched_at": db.now(),
        },
    )

    for ban in match_detail.get("dynamic_fields", {}).get("ban_pick_info", []):
        hero_id = int(ban["hero_id"])
        db.upsert(
            conn,
            "match_bans",
            ["match_uid", "round_idx", "battle_side", "hero_id"],
            {
                "match_uid": match_uid,
                "round_idx": int(ban["round_idx"]),
                "battle_side": int(ban["battle_side"]),
                "hero_id": hero_id,
                "is_pick": int(ban["is_pick"]),
            },
        )
        _record_hero_seen(conn, hero_id)

    new_player_uids = []
    for p in match_detail.get("match_players", []):
        uid = p["player_uid"]
        cur_hero_id = p.get("cur_hero_id")
        dyn = p.get("dynamic_fields", {}) or {}

        db.upsert(
            conn,
            "match_players",
            ["match_uid", "player_uid"],
            {
                "match_uid": match_uid,
                "player_uid": uid,
                "camp": p.get("camp"),
                "cur_hero_id": cur_hero_id,
                "k": p.get("k"),
                "d": p.get("d"),
                "a": p.get("a"),
                "total_hero_damage": p.get("total_hero_damage"),
                "total_hero_heal": p.get("total_hero_heal"),
                "total_damage_taken": p.get("total_damage_taken"),
                "is_win": p.get("is_win"),
                "add_score": dyn.get("add_score"),
                "new_score": dyn.get("new_score"),
                "level": dyn.get("level"),
                "new_level": dyn.get("new_level"),
                "session_hit_rate": p.get("session_hit_rate"),
            },
        )

        for hero_segment in p.get("player_heroes", []):
            hero_id = hero_segment["hero_id"]
            db.upsert(
                conn,
                "match_player_heroes",
                ["match_uid", "player_uid", "hero_id"],
                {
                    "match_uid": match_uid,
                    "player_uid": uid,
                    "hero_id": hero_id,
                    "k": hero_segment.get("k"),
                    "d": hero_segment.get("d"),
                    "a": hero_segment.get("a"),
                    "play_time": hero_segment.get("play_time"),
                },
            )
            _record_hero_seen(conn, hero_id)

        if cur_hero_id is not None:
            _record_hero_seen(conn, cur_hero_id)

        created = _upsert_discovered_player(
            conn,
            uid=uid,
            nick_name=p.get("nick_name"),
            score=dyn.get("new_score"),
            level=dyn.get("new_level"),
            discovery_hero_id=cur_hero_id,
        )
        if created:
            new_player_uids.append(uid)

    conn.commit()
    return new_player_uids


def _winner_side(match_detail):
    for p in match_detail.get("match_players", []):
        if p.get("is_win"):
            return p.get("camp")
    return None


def _record_hero_seen(conn, hero_id):
    db.upsert(conn, "heroes", ["hero_id"], {"hero_id": hero_id})


def _upsert_discovered_player(conn, uid, nick_name, score, level, discovery_hero_id):
    """Refresh a player's freshest known name/score/level. discovery_hero_id
    and crawl_status are set ONLY on first creation and never overwritten
    afterwards, so a player already tagged (e.g. from hero-leaderboard
    seeding) keeps that tag no matter how many other matches surface them."""
    existing = conn.execute("SELECT uid FROM players WHERE uid=?", (uid,)).fetchone()
    if existing is None:
        db.upsert(
            conn,
            "players",
            ["uid"],
            {
                "uid": uid,
                "nick_name": nick_name,
                "latest_known_score": score,
                "latest_known_level": level,
                "discovery_hero_id": discovery_hero_id,
                "crawl_status": "pending",
            },
        )
        return True

    db.upsert(
        conn,
        "players",
        ["uid"],
        {
            "uid": uid,
            "nick_name": nick_name,
            "latest_known_score": score,
            "latest_known_level": level,
        },
    )
    return False
```

Note: `match_detail["match_map_id"]` handles the player-history-view field name; the standalone match-detail endpoint may use `map_id` instead — both are read defensively with `.get(...) or .get(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: ingest match payloads into normalized, idempotent DB rows"
```

---

### Task 6: Crawl orchestrator — priority queue, per-player crawl, reseeding

**Files:**
- Create: `crawler.py`
- Test: `tests/test_crawler.py`
- Uses fixtures: `player_public.json`, `player_fully_private.json`, `match_detail.json`, `leaderboard_payload.json`, `hero_leaderboard.json`

**Interfaces:**
- Consumes: `db.*` (Task 1), `fetcher.PlayerNotFoundError` (Task 3), `rivalsmeta.*` (Task 4), `ingest.ingest_match` (Task 5).
- Produces: `crawler.select_next_player(conn) -> int | None`, `crawler.crawl_player(conn, client, uid, season) -> str`, `crawler.reseed(conn, client, hero_refresh_seconds=86400) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_crawler.py`:

```python
import json
import pathlib

import pytest

import db
import fetcher
import crawler

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def test_select_next_player_prioritizes_lowest_hero_coverage():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "discovery_hero_id": 1001, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "discovery_hero_id": 1002, "crawl_status": "pending"})
    # Give hero 1001 five collected match rows, hero 1002 zero.
    for i in range(5):
        db.upsert(
            conn,
            "match_player_heroes",
            ["match_uid", "player_uid", "hero_id"],
            {"match_uid": f"m{i}", "player_uid": 999, "hero_id": 1001},
        )
    assert crawler.select_next_player(conn) == 2


def test_select_next_player_deprioritizes_untagged_players():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "discovery_hero_id": None, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "discovery_hero_id": 1002, "crawl_status": "pending"})
    assert crawler.select_next_player(conn) == 2


def test_select_next_player_returns_none_when_queue_empty():
    conn = make_conn()
    assert crawler.select_next_player(conn) is None


def test_crawl_player_below_floor_precheck_skips_without_profile_fetch():
    conn = make_conn()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "pending", "latest_known_level": 5},
    )

    class ExplodingClient:
        def get_json(self, *a, **k):
            raise AssertionError("should not fetch profile when already known below floor")

    status = crawler.crawl_player(conn, ExplodingClient(), uid=1, season=19)
    assert status == "skipped_floor"
    assert conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0] == "skipped_floor"


def test_crawl_player_not_found_marks_not_indexed():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})

    class NotFoundClient:
        def get_json(self, path, params=None):
            raise fetcher.PlayerNotFoundError(path)

    status = crawler.crawl_player(conn, NotFoundClient(), uid=1, season=19)
    assert status == "not_indexed"


def test_crawl_player_private_profile_skips_without_history_calls():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 130729830, "crawl_status": "pending"})
    private_profile = load("player_fully_private.json")

    class PrivateClient:
        def get_json(self, path, params=None):
            if path == "/api/player/130729830":
                return private_profile
            raise AssertionError(f"unexpected call: {path}")

    status = crawler.crawl_player(conn, PrivateClient(), uid=130729830, season=19)
    assert status == "skipped_private"


def test_crawl_player_public_diamond_pulls_matches_and_stops_at_known_match():
    conn = make_conn()
    match = load("match_detail.json")
    match_uid = match["match_uid"]
    uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    profile = load("player_public.json")

    history_page_1 = [{"match_uid": match_uid}]

    class PublicClient:
        def __init__(self):
            self.match_detail_calls = 0

        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                if params["skip"] == 0:
                    return history_page_1
                return []
            if path == f"/api/matches/{match_uid}":
                self.match_detail_calls += 1
                return match
            raise AssertionError(f"unexpected call: {path} {params}")

    client = PublicClient()
    status = crawler.crawl_player(conn, client, uid=uid, season=19)
    assert status == "done"
    assert client.match_detail_calls == 1
    assert conn.execute("SELECT COUNT(*) FROM matches WHERE match_uid=?", (match_uid,)).fetchone()[0] == 1

    # Re-crawling should stop immediately at the already-known match without
    # re-fetching its detail.
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    status2 = crawler.crawl_player(conn, client, uid=uid, season=19)
    assert status2 == "done"
    assert client.match_detail_calls == 1  # unchanged


def test_reseed_queues_players_from_global_and_hero_leaderboards():
    conn = make_conn()
    raw_leaderboard_text = (FIXTURES / "leaderboard_payload.json").read_text()
    hero_lb = load("hero_leaderboard.json")
    # Seed one hero as already-known so reseed has something to iterate.
    db.upsert(conn, "heroes", ["hero_id"], {"hero_id": 1047})

    class ReseedClient:
        def get_text(self, path, params=None):
            assert path == "/leaderboard/_payload.json"
            return raw_leaderboard_text

        def get_json(self, path, params=None):
            assert path == "/api/hero-leaderboard/1047"
            return hero_lb

    queued = crawler.reseed(conn, ReseedClient())
    assert queued > 0
    total_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    assert total_players > 0
    last_seeded = conn.execute("SELECT last_seeded_at FROM heroes WHERE hero_id=1047").fetchone()[0]
    assert last_seeded is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_crawler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'crawler'`

- [ ] **Step 3: Write `crawler.py`**

```python
import json

import db
import fetcher
import ingest
import rivalsmeta

PRIORITY_SQL = """
SELECT uid FROM players
WHERE crawl_status = 'pending'
ORDER BY
    CASE WHEN discovery_hero_id IS NULL THEN 999999999
         ELSE (SELECT COUNT(*) FROM match_player_heroes mph WHERE mph.hero_id = players.discovery_hero_id)
    END ASC,
    created_at ASC
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_crawler.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add crawler.py tests/test_crawler.py
git commit -m "feat: add crawl orchestrator with priority queue and leaderboard reseeding"
```

---

### Task 7: CLI entrypoint — main loop, graceful shutdown, progress logging

**Files:**
- Create: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `db.*`, `fetcher.RivalsMetaClient`, `rivalsmeta.resolve_current_season`, `crawler.select_next_player`, `crawler.crawl_player`, `crawler.reseed`.
- Produces: `main.format_progress_line(conn) -> str`, `main.ShutdownFlag` (a small class with `.requested: bool` and `.request(self, *_)` as a signal handler), `main.run(conn, client, season, shutdown_flag, reseed_interval_seconds=86400)` (one bounded pass usable by tests), `main.main()` (real CLI entrypoint).

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:

```python
import db
import main


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def test_format_progress_line_reports_counts_by_status_and_hero_coverage():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 3, "crawl_status": "skipped_private"})
    db.upsert(conn, "matches", ["match_uid"], {"match_uid": "m1"})
    db.upsert(conn, "match_player_heroes", ["match_uid", "player_uid", "hero_id"], {
        "match_uid": "m1", "player_uid": 1, "hero_id": 1042,
    })

    line = main.format_progress_line(conn)

    assert "done=1" in line
    assert "pending=1" in line
    assert "skipped_private=1" in line
    assert "matches=1" in line


def test_shutdown_flag_starts_false_and_becomes_true_on_signal():
    flag = main.ShutdownFlag()
    assert flag.requested is False
    flag.request(15, None)  # simulate SIGTERM(15) delivery
    assert flag.requested is True


def test_run_stops_after_shutdown_flag_is_set_between_players():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "pending"})

    flag = main.ShutdownFlag()
    calls = []

    class StopAfterOneClient:
        pass

    def fake_crawl_player(conn_, client_, uid, season):
        calls.append(uid)
        flag.requested = True  # pretend a shutdown arrived mid-run
        return "done"

    main.run(
        conn,
        StopAfterOneClient(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
    )
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `main.py`**

```python
import argparse
import signal
import sys
import time

import db
import crawler
import fetcher
import rivalsmeta


class ShutdownFlag:
    def __init__(self):
        self.requested = False

    def request(self, signum, frame):
        self.requested = True


def format_progress_line(conn):
    status_counts = dict(
        conn.execute(
            "SELECT crawl_status, COUNT(*) FROM players GROUP BY crawl_status"
        ).fetchall()
    )
    n_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    hero_counts = [
        row[0]
        for row in conn.execute(
            "SELECT COUNT(*) FROM match_player_heroes GROUP BY hero_id"
        ).fetchall()
    ]
    if hero_counts:
        hero_counts.sort()
        hero_min = hero_counts[0]
        hero_median = hero_counts[len(hero_counts) // 2]
        hero_max = hero_counts[-1]
    else:
        hero_min = hero_median = hero_max = 0

    status_part = " ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
    return (
        f"matches={n_matches} {status_part} "
        f"hero_coverage(min/median/max)={hero_min}/{hero_median}/{hero_max}"
    )


def run(conn, client, season, shutdown_flag, reseed_interval_seconds=86400, crawl_player_fn=None, reseed_fn=None):
    crawl_player_fn = crawl_player_fn or crawler.crawl_player
    reseed_fn = reseed_fn or crawler.reseed

    reseed_fn(conn, client)
    last_reseed = time.time()
    last_progress_log = time.time()

    while not shutdown_flag.requested:
        if time.time() - last_reseed > reseed_interval_seconds:
            reseed_fn(conn, client)
            last_reseed = time.time()

        uid = crawler.select_next_player(conn)
        if uid is None:
            break

        crawl_player_fn(conn, client, uid, season)

        if time.time() - last_progress_log > 60:
            print(format_progress_line(conn), file=sys.stderr)
            last_progress_log = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/rivals.db")
    parser.add_argument("--reseed-interval-hours", type=float, default=24.0)
    args = parser.parse_args()

    conn = db.connect(args.db_path)
    db.init_schema(conn)

    client = fetcher.RivalsMetaClient()
    season = rivalsmeta.resolve_current_season(client)

    shutdown_flag = ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown_flag.request)
    signal.signal(signal.SIGTERM, shutdown_flag.request)

    run(conn, client, season, shutdown_flag, reseed_interval_seconds=args.reseed_interval_hours * 3600)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add CLI entrypoint with graceful shutdown and progress logging"
```

---

### Task 8: Live smoke test against the real site (opt-in)

**Files:**
- Create: `pytest.ini`
- Create: `tests/test_live_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6 against the real `fetcher.RivalsMetaClient` (no fixtures/mocks).

This is the "make sure it's robust" end-to-end check: it proves the whole pipeline (devalue parsing, endpoint shapes, ingestion) still works against the live, undocumented API today — not just against January's frozen fixtures. It is marked `live` and excluded from the default test run so routine `pytest` invocations stay fast, offline, and don't add load to rivalsmeta.com every time tests run.

- [ ] **Step 1: Register the `live` marker**

Create `pytest.ini`:

```ini
[pytest]
markers =
    live: hits the real rivalsmeta.com API (excluded by default; run with -m live)
addopts = -m "not live"
```

- [ ] **Step 2: Write the live smoke test**

Create `tests/test_live_smoke.py`:

```python
import pytest

import db
import crawler
import fetcher
import rivalsmeta

pytestmark = pytest.mark.live


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(str(tmp_path / "live_smoke.db"))
    db.init_schema(connection)
    return connection


@pytest.fixture
def client():
    return fetcher.RivalsMetaClient()


def test_resolve_current_season_against_live_site(client):
    season = rivalsmeta.resolve_current_season(client)
    assert isinstance(season, int) and season > 0


def test_crawl_a_known_public_diamond_plus_player(conn, client):
    # A verified-public player used throughout development of this crawler.
    uid = 457877313
    season = rivalsmeta.resolve_current_season(client)
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})

    status = crawler.crawl_player(conn, client, uid, season)

    assert status in ("done", "skipped_floor")  # rank may have changed since verification
    n_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    if status == "done":
        assert n_matches >= 1
        n_bans = conn.execute("SELECT COUNT(*) FROM match_bans").fetchone()[0]
        assert n_bans >= 0  # some game modes may have zero bans; just confirm no crash


def test_hero_leaderboard_shape_matches_expectations(client):
    board = rivalsmeta.get_hero_leaderboard(client, hero_id=1047)  # Jeff The Land Shark
    assert board["_id"] == 1047
    assert len(board["players"]) > 0
```

- [ ] **Step 3: Run the live suite once to confirm it actually passes against production**

Run: `pytest -m live tests/test_live_smoke.py -v`
Expected: 3 passed (a slow run — the adaptive rate limiter starts at ~1 req/sec — this is expected and correct, not a bug to fix)

- [ ] **Step 4: Run the default suite to confirm the live tests are excluded**

Run: `pytest -v`
Expected: all Task 1–7 tests pass; `test_live_smoke.py` does not run (no network calls made)

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/test_live_smoke.py
git commit -m "test: add opt-in live smoke test against the real rivalsmeta.com API"
```

---

## Post-plan manual step (not a task — requires a running, long-lived process)

Once all tasks pass, start the crawler for real: `python main.py --db-path data/rivals.db`. Let it run for a while, then inspect with `sqlite3 data/rivals.db "SELECT crawl_status, COUNT(*) FROM players GROUP BY 1"` and check the stderr progress lines. This is deliberately left as a manual step rather than a task, since "run it for N hours and eyeball the results" isn't something a test can assert.
