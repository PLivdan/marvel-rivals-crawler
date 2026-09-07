import copy
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

    # match_map_id lives only on the history entry, never on the match detail.
    history_page_1 = [{"match_uid": match_uid, "match_map_id": 1245}]

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
    # crawl_player must forward the history entry so map_id is actually stored.
    assert conn.execute(
        "SELECT map_id FROM matches WHERE match_uid=?", (match_uid,)
    ).fetchone()[0] == 1245

    # Re-crawling should stop immediately at the already-known match without
    # re-fetching its detail.
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    status2 = crawler.crawl_player(conn, client, uid=uid, season=19)
    assert status2 == "done"
    assert client.match_detail_calls == 1  # unchanged


def test_crawl_player_resumes_interrupted_first_crawl_past_already_known_matches():
    # Simulates a kill mid-first-crawl: match A was ingested, match B was not,
    # the player is still 'pending' and last_crawled_at is still NULL. On
    # resume the crawler must SKIP A and still fetch B, rather than treating A
    # as "we've caught up" and dropping the rest of the player's history.
    conn = make_conn()
    match = load("match_detail.json")
    already_ingested_uid = "interrupted_match_a"
    uid = 457877313
    profile = load("player_public.json")

    db.upsert(conn, "matches", ["match_uid"], {"match_uid": already_ingested_uid})
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    conn.commit()
    assert conn.execute("SELECT last_crawled_at FROM players WHERE uid=?", (uid,)).fetchone()[0] is None

    unseen_uid = match["match_uid"]
    history_page = [
        {"match_uid": already_ingested_uid, "match_map_id": 1200},
        {"match_uid": unseen_uid, "match_map_id": 1245},
    ]

    class ResumeClient:
        def __init__(self):
            self.detail_calls = []

        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                return history_page if params["skip"] == 0 else []
            if path.startswith("/api/matches/"):
                self.detail_calls.append(path.rsplit("/", 1)[1])
                return match
            raise AssertionError(f"unexpected call: {path} {params}")

    client = ResumeClient()
    status = crawler.crawl_player(conn, client, uid=uid, season=19)

    assert status == "done"
    assert client.detail_calls == [unseen_uid]
    assert conn.execute(
        "SELECT COUNT(*) FROM matches WHERE match_uid=?", (unseen_uid,)
    ).fetchone()[0] == 1


def test_crawl_player_revisit_still_stops_at_first_known_match():
    # The mirror of the test above: once last_crawled_at is set (a genuine
    # revisit), the first already-known match_uid means we've caught up, and
    # pagination must stop there rather than walking the whole history again.
    conn = make_conn()
    match = load("match_detail.json")
    known_uid = "already_have_this_one"
    uid = 457877313
    profile = load("player_public.json")

    db.upsert(conn, "matches", ["match_uid"], {"match_uid": known_uid})
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": uid, "crawl_status": "pending", "last_crawled_at": db.now()},
    )
    conn.commit()

    history_page = [
        {"match_uid": known_uid, "match_map_id": 1200},
        {"match_uid": match["match_uid"], "match_map_id": 1245},
    ]

    class RevisitClient:
        def __init__(self):
            self.detail_calls = []

        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                return history_page if params["skip"] == 0 else []
            if path.startswith("/api/matches/"):
                self.detail_calls.append(path.rsplit("/", 1)[1])
                return match
            raise AssertionError(f"unexpected call: {path} {params}")

    client = RevisitClient()
    status = crawler.crawl_player(conn, client, uid=uid, season=19)

    assert status == "done"
    assert client.detail_calls == []


def test_crawl_player_skips_a_malformed_match_and_keeps_going():
    # The match-detail endpoint is undocumented and can change shape without
    # notice. One bad payload must not crash crawl_player (and with it the
    # whole run loop) — it should be logged, skipped, and leave no partial
    # rows behind, while the sane matches around it still land.
    conn = make_conn()
    match = load("match_detail.json")
    sane_uid = match["match_uid"]
    malformed_uid = "malformed_match_uid"

    malformed = copy.deepcopy(match)
    malformed["match_uid"] = malformed_uid
    del malformed["match_players"][0]["player_uid"]  # shape change mid-ingest

    uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    profile = load("player_public.json")

    history_page = [
        {"match_uid": malformed_uid, "match_map_id": 1200},
        {"match_uid": sane_uid, "match_map_id": 1245},
    ]

    class MixedClient:
        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                return history_page if params["skip"] == 0 else []
            if path == f"/api/matches/{malformed_uid}":
                return malformed
            if path == f"/api/matches/{sane_uid}":
                return match
            raise AssertionError(f"unexpected call: {path} {params}")

    status = crawler.crawl_player(conn, MixedClient(), uid=uid, season=19)

    assert status == "done"  # did not raise, completed normally
    stored = {row[0] for row in conn.execute("SELECT match_uid FROM matches").fetchall()}
    assert malformed_uid not in stored  # no partial row left behind
    assert sane_uid in stored  # the good match around it still landed
    assert conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE match_uid=?", (malformed_uid,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE match_uid=?", (sane_uid,)
    ).fetchone()[0] == 12


def test_crawl_player_skips_a_match_that_404s_and_keeps_going():
    # A match can appear in a player's history and still 404 on the detail
    # endpoint. That raised PlayerNotFoundError, which crawl_player only
    # caught around the *profile* fetch — so it escaped crawl_player, escaped
    # run() (which has no handler for it), and killed the process.
    conn = make_conn()
    match = load("match_detail.json")
    sane_uid = match["match_uid"]
    gone_uid = "match_that_404s"
    uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    profile = load("player_public.json")

    history_page = [
        {"match_uid": gone_uid, "match_map_id": 1200},
        {"match_uid": sane_uid, "match_map_id": 1245},
    ]

    class MissingMatchClient:
        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                return history_page if params["skip"] == 0 else []
            if path == f"/api/matches/{gone_uid}":
                raise fetcher.PlayerNotFoundError(path)
            if path == f"/api/matches/{sane_uid}":
                return match
            raise AssertionError(f"unexpected call: {path} {params}")

    status = crawler.crawl_player(conn, MissingMatchClient(), uid=uid, season=19)

    assert status == "done"  # did not raise
    stored = {row[0] for row in conn.execute("SELECT match_uid FROM matches").fetchall()}
    assert gone_uid not in stored
    assert sane_uid in stored  # the rest of the page still got ingested
    assert conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE match_uid=?", (sane_uid,)
    ).fetchone()[0] == 12


def test_requeue_stale_players_revisits_done_players_keeping_last_crawled_at():
    conn = make_conn()
    now = db.now()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "done", "last_crawled_at": now - 50000},
    )

    n = crawler.requeue_stale_players(conn, done_revisit_seconds=43200)

    assert n == 1
    status, last = conn.execute(
        "SELECT crawl_status, last_crawled_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "pending"
    # A completed crawl is a genuine revisit: keeping the timestamp is what
    # lets crawl_player stop at the first already-known match and pull only
    # what's new.
    assert last == now - 50000


def test_requeue_stale_players_retries_error_players_clearing_last_crawled_at():
    conn = make_conn()
    now = db.now()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "error", "last_crawled_at": now - 7200},
    )

    n = crawler.requeue_stale_players(conn, error_retry_seconds=3600)

    assert n == 1
    status, last = conn.execute(
        "SELECT crawl_status, last_crawled_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "pending"
    # An errored player never completed a first crawl, so the retry must be
    # treated as a fresh one — NULL here is what makes crawl_player skip past
    # partially-ingested matches instead of stopping at the first of them.
    assert last is None


def test_requeue_stale_players_leaves_players_inside_their_windows_alone():
    conn = make_conn()
    now = db.now()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": now - 100})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "error", "last_crawled_at": now - 100})

    n = crawler.requeue_stale_players(conn, done_revisit_seconds=43200, error_retry_seconds=3600)

    assert n == 0
    rows = dict(conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert rows == {1: "done", 2: "error"}


def test_requeue_stale_players_never_touches_other_statuses():
    # Terminal/never-crawlable states must stay out of the frontier no matter
    # how old they are: re-crawling them would burn requests to learn nothing.
    conn = make_conn()
    ancient = db.now() - 10**7
    statuses = {
        1: "pending",
        2: "skipped_private",
        3: "skipped_floor",
        4: "not_indexed",
    }
    for uid, status in statuses.items():
        db.upsert(
            conn,
            "players",
            ["uid"],
            {"uid": uid, "crawl_status": status, "last_crawled_at": ancient},
        )

    n = crawler.requeue_stale_players(conn)

    assert n == 0
    rows = dict(conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert rows == statuses
    # And the timestamps are untouched too.
    assert all(
        row[0] == ancient
        for row in conn.execute("SELECT last_crawled_at FROM players").fetchall()
    )


def test_requeue_stale_players_counts_both_categories():
    conn = make_conn()
    now = db.now()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": now - 50000})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "done", "last_crawled_at": now - 50000})
    db.upsert(conn, "players", ["uid"], {"uid": 3, "crawl_status": "error", "last_crawled_at": now - 7200})
    db.upsert(conn, "players", ["uid"], {"uid": 4, "crawl_status": "done", "last_crawled_at": now})

    assert crawler.requeue_stale_players(conn) == 3


def test_requeued_error_player_is_recrawled_as_a_fresh_first_crawl():
    # The whole point of the last_crawled_at asymmetry, end to end: a player
    # who errored partway through their first crawl must, on retry, skip past
    # the matches that attempt did ingest and keep paginating — not stop dead
    # at the first of them and strand the rest of their history.
    conn = make_conn()
    match = load("match_detail.json")
    partially_ingested_uid = "ingested_before_the_error"
    unseen_uid = match["match_uid"]
    uid = 457877313
    profile = load("player_public.json")

    db.upsert(conn, "matches", ["match_uid"], {"match_uid": partially_ingested_uid})
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": uid, "crawl_status": "error", "last_crawled_at": db.now() - 7200},
    )
    conn.commit()

    assert crawler.requeue_stale_players(conn) == 1
    assert crawler.select_next_player(conn) == uid  # back in the frontier

    history_page = [
        {"match_uid": partially_ingested_uid, "match_map_id": 1200},
        {"match_uid": unseen_uid, "match_map_id": 1245},
    ]

    class RetryClient:
        def __init__(self):
            self.detail_calls = []

        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                return history_page if params["skip"] == 0 else []
            if path.startswith("/api/matches/"):
                self.detail_calls.append(path.rsplit("/", 1)[1])
                return match
            raise AssertionError(f"unexpected call: {path} {params}")

    client = RetryClient()
    assert crawler.crawl_player(conn, client, uid=uid, season=19) == "done"
    assert client.detail_calls == [unseen_uid]


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
