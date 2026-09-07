import json
import pathlib
import time

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
    # match_player_heroes.match_uid is FK-constrained to matches(match_uid)
    # (db.py schema, PRAGMA foreign_keys=ON), so the parent match row must
    # exist first.
    for i in range(5):
        db.upsert(conn, "matches", ["match_uid"], {"match_uid": f"m{i}"})
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


def test_select_next_player_stays_fast_at_realistic_queue_size():
    # A single reseed can queue ~21,000 pending players across ~42 heroes.
    # The original correlated-subquery form measured ~5.2s at only 2,000
    # pending players; this guards the aggregate+LEFT JOIN rewrite (and the
    # supporting indexes) so that regression can't silently return.
    conn = make_conn()
    n_players = 2000
    n_matches = 400
    n_heroes = 40

    conn.executemany(
        "INSERT INTO players (uid, discovery_hero_id, crawl_status) VALUES (?, ?, 'pending')",
        [(uid, 1000 + (uid % n_heroes)) for uid in range(n_players)],
    )
    conn.executemany(
        "INSERT INTO matches (match_uid) VALUES (?)",
        [(f"perf{i}",) for i in range(n_matches)],
    )
    # 400 matches x 12 players x 5 hero segments = 24,000 match_player_heroes rows.
    conn.executemany(
        "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id) VALUES (?, ?, ?)",
        [
            (f"perf{i}", p, 1000 + ((i * 12 + p) * 7 + s * 3) % n_heroes)
            for i in range(n_matches)
            for p in range(12)
            for s in range(5)
        ],
    )
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM match_player_heroes").fetchone()[0] >= 20000
    assert conn.execute(
        "SELECT COUNT(*) FROM players WHERE crawl_status='pending'"
    ).fetchone()[0] >= 2000

    start = time.time()
    uid = crawler.select_next_player(conn)
    elapsed = time.time() - start

    assert uid is not None
    assert elapsed < 1.0, f"select_next_player took {elapsed:.2f}s at 2000 pending players"


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

    # uid 1772998912 ("MAINTANKSLOP") appears on BOTH the hero-1047
    # leaderboard and the general top-500 leaderboard in these fixtures.
    # Because hero leaderboards are processed first, this player must end
    # up tagged with the hero-specific discovery_hero_id=1047 rather than
    # the untagged (None) tag the general leaderboard would otherwise give
    # them.
    row = conn.execute(
        "SELECT discovery_hero_id FROM players WHERE uid=1772998912"
    ).fetchone()
    assert row[0] == 1047
