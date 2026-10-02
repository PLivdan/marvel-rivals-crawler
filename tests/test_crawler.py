import copy
import json
import pathlib
import sqlite3
import time

import pytest

import db
import fetcher
import crawler
import ingest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


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


def test_select_next_player_breaks_coverage_ties_by_oldest_player_across_heroes():
    # Two heroes with identical (zero) coverage: the older pending player wins
    # even though they belong to the higher-numbered hero. This pins the
    # original ORDER BY (coverage, created_at) semantics across a rewrite.
    conn = make_conn()
    conn.execute(
        "INSERT INTO players (uid, discovery_hero_id, crawl_status, created_at) VALUES (1, 1001, 'pending', 200)"
    )
    conn.execute(
        "INSERT INTO players (uid, discovery_hero_id, crawl_status, created_at) VALUES (2, 1002, 'pending', 100)"
    )
    conn.commit()
    assert crawler.select_next_player(conn) == 2


def test_select_next_player_cost_does_not_scale_with_the_size_of_the_frontier():
    # Observed live on 2026-09-20: at 410k pending players and 3.9M
    # match_player_heroes rows, the claim query took 2-4.6s and six workers
    # spent most of their time in it (claimed=0-2 of 6, throughput halved).
    # The selection must cost a handful of index seeks, not a sort of the
    # whole pending set: 100k pending across 40 heroes must stay well under
    # the ~0.3s the old form needed for that sort.
    conn = make_conn()
    n_players = 100_000
    n_heroes = 40
    conn.executemany(
        "INSERT INTO players (uid, discovery_hero_id, crawl_status, created_at) VALUES (?, ?, 'pending', ?)",
        [(uid, 1000 + (uid % n_heroes), 1_000_000 + uid) for uid in range(n_players)],
    )
    conn.commit()

    start = time.time()
    for _ in range(10):
        uid = crawler.select_next_player(conn)
    elapsed = (time.time() - start) / 10

    assert uid == 0  # every hero has zero coverage; the oldest player wins
    assert elapsed < 0.05, f"select_next_player took {elapsed:.3f}s per call at 100k pending players"


def test_claim_next_player_returns_none_when_queue_empty():
    conn = make_conn()
    assert crawler.claim_next_player(conn) is None


def test_claim_next_player_marks_the_player_claimed_and_stamps_claimed_at():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})

    before = db.now()
    assert crawler.claim_next_player(conn) == 1

    status, claimed_at = conn.execute(
        "SELECT crawl_status, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "claimed"
    assert claimed_at is not None and claimed_at >= before


def test_claim_next_player_follows_the_same_priority_order_as_select_next_player():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "discovery_hero_id": None, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "discovery_hero_id": 1002, "crawl_status": "pending"})
    # Untagged players sort last, so the hero-tagged one must be claimed first.
    assert crawler.claim_next_player(conn) == 2


def test_two_sequential_claims_hand_out_two_different_players():
    # The plain sequential case: a claimed player is out of the frontier, so a
    # second claim cannot possibly return them again.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "pending"})

    first = crawler.claim_next_player(conn)
    second = crawler.claim_next_player(conn)

    assert {first, second} == {1, 2}
    assert crawler.claim_next_player(conn) is None  # nothing pending left
    statuses = dict(conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert statuses == {1: "claimed", 2: "claimed"}


def test_claim_next_player_does_not_stamp_last_crawled_at():
    # crawl_player reads `last_crawled_at IS NULL` as "has never COMPLETED a
    # crawl". Writing it at claim time would make every first crawl look like a
    # revisit, so the crawler would stop at the first already-known match and
    # permanently strand the rest of that player's history.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})

    assert crawler.claim_next_player(conn) == 1

    assert conn.execute("SELECT last_crawled_at FROM players WHERE uid=1").fetchone()[0] is None


def test_claim_next_player_preserves_an_existing_last_crawled_at_for_revisits():
    conn = make_conn()
    stamped = db.now() - 50000
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "pending", "last_crawled_at": stamped},
    )

    assert crawler.claim_next_player(conn) == 1

    assert conn.execute("SELECT last_crawled_at FROM players WHERE uid=1").fetchone()[0] == stamped


def test_a_claimed_first_time_player_is_still_crawled_as_a_fresh_first_crawl():
    # End-to-end guard for the two tests above: claiming must not disturb the
    # interrupted-first-crawl resume. The player has one already-ingested match
    # and one unseen one; a fresh crawl skips past the known one and still
    # fetches the unseen one, where a revisit would stop dead at the known one.
    conn = make_conn()
    match = load("match_detail.json")
    already_ingested_uid = "ingested_before_the_interruption"
    unseen_uid = match["match_uid"]
    uid = 457877313
    profile = load("player_public.json")

    db.upsert(conn, "matches", ["match_uid"], {"match_uid": already_ingested_uid})
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    conn.commit()

    assert crawler.claim_next_player(conn) == uid

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
    assert crawler.crawl_player(conn, client, uid=uid, season=19) == "done"
    assert client.detail_calls == [unseen_uid]


def test_crawl_player_clears_claimed_at_when_it_reaches_a_terminal_status():
    # claimed_at must mean "a worker is holding this player right now" and
    # nothing else — that is exactly what the orphaned-claim sweep keys on.
    conn = make_conn()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "pending", "latest_known_level": 5},
    )
    assert crawler.claim_next_player(conn) == 1
    assert conn.execute("SELECT claimed_at FROM players WHERE uid=1").fetchone()[0] is not None

    class UnusedClient:
        def get_json(self, *a, **k):
            raise AssertionError("below-floor player needs no request")

    assert crawler.crawl_player(conn, UnusedClient(), uid=1, season=19) == "skipped_floor"
    assert conn.execute("SELECT claimed_at FROM players WHERE uid=1").fetchone()[0] is None


def test_two_connections_racing_for_the_same_player_produce_exactly_one_claim(tmp_path, monkeypatch):
    """The race the claim exists to close, played out by hand.

    Two real connections to the same on-disk WAL database contend for the only
    pending row. Connection A is frozen between its SELECT (it has picked its
    candidate) and its UPDATE (it has not yet written the claim) — precisely
    the window in which the old select-then-crawl pair would have let both
    workers start crawling the same player. B runs a complete claim and commits
    inside that window. A then finishes: its `AND crawl_status='pending'` guard
    matches zero rows, rowcount is 0, and A correctly reports None.
    """
    path = str(tmp_path / "race.db")
    setup = db.connect(path)
    db.init_schema(setup)
    db.upsert(setup, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    setup.commit()
    setup.close()

    conn_a = db.connect(path)
    conn_b = db.connect(path)
    real_select = crawler._select_head
    b_result = []

    def select_then_let_b_claim(conn):
        uid = real_select(conn)
        if conn is conn_a:
            # A has its candidate but has NOT claimed it yet. Restore the real
            # select first so B's claim runs normally, then let B claim and
            # commit inside A's window.
            monkeypatch.setattr(crawler, "_select_head", real_select)
            b_result.append(crawler.claim_next_player(conn_b))
        return uid

    monkeypatch.setattr(crawler, "_select_head", select_then_let_b_claim)
    a_result = crawler.claim_next_player(conn_a)

    assert b_result == [1]  # B, which committed first, won the row
    assert a_result is None  # A lost, and says so rather than double-claiming
    # Exactly one claim exists, not two, not zero.
    assert sorted([a_result] + b_result, key=lambda v: (v is None, v)) == [1, None]
    status, claimed_at = conn_a.execute(
        "SELECT crawl_status, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "claimed"
    assert claimed_at is not None
    conn_a.close()
    conn_b.close()


def test_release_player_returns_a_claim_to_the_frontier_untouched():
    conn = make_conn()
    stamped = db.now() - 50000
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "pending", "last_crawled_at": stamped},
    )
    assert crawler.claim_next_player(conn) == 1

    crawler.release_player(conn, 1)

    status, last, claimed_at = conn.execute(
        "SELECT crawl_status, last_crawled_at, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "revisit"                                # it has completed a crawl before
    assert claimed_at is None
    # Releasing is a no-op on history: this player's revisit/first-crawl state
    # must survive a back-off unchanged.
    assert last == stamped
    assert crawler.claim_next_player(conn) == 1  # and they are claimable again


def test_release_player_leaves_a_player_who_is_not_claimed_alone():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": db.now()})

    crawler.release_player(conn, 1)

    assert conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0] == "done"


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
    del profile["match_history"]  # so the history pages are read, not the profile's list

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
    assert status == "revisit"
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
    # (A floor-skipped player with no Diamond+ level since, here none at all,
    # stays put too. Private players are retried on their own timer.)
    conn = make_conn()
    ancient = db.now() - 10**7
    statuses = {
        1: "pending",
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
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": now - 4 * 86400})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "done", "last_crawled_at": now - 4 * 86400})
    db.upsert(conn, "players", ["uid"], {"uid": 3, "crawl_status": "error", "last_crawled_at": now - 7200})
    db.upsert(conn, "players", ["uid"], {"uid": 4, "crawl_status": "done", "last_crawled_at": now})

    assert crawler.requeue_stale_players(conn) == 3


def test_requeue_stale_players_frees_an_orphaned_claim_as_a_fresh_crawl():
    # A worker that died mid-crawl leaves its player 'claimed' forever. Like an
    # 'error' player they never completed the crawl they were in the middle of,
    # so last_crawled_at is cleared and the retry runs with fresh-crawl
    # semantics rather than stopping at the first match that attempt ingested.
    conn = make_conn()
    now = db.now()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {
            "uid": 1,
            "crawl_status": "claimed",
            "claimed_at": now - 1200,
            "last_crawled_at": now - 90000,
        },
    )

    n = crawler.requeue_stale_players(conn, claimed_stale_seconds=600)

    assert n == 1
    status, last, claimed_at = conn.execute(
        "SELECT crawl_status, last_crawled_at, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "pending"
    assert last is None
    assert claimed_at is None


def test_requeue_stale_players_leaves_a_claim_inside_its_window_alone():
    # The window has to be long enough that a worker legitimately grinding
    # through a big match history is never robbed of the player it is crawling.
    conn = make_conn()
    now = db.now()
    db.upsert(
        conn,
        "players",
        ["uid"],
        {"uid": 1, "crawl_status": "claimed", "claimed_at": now - 60},
    )

    n = crawler.requeue_stale_players(conn, claimed_stale_seconds=600)

    assert n == 0
    status, claimed_at = conn.execute(
        "SELECT crawl_status, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "claimed"
    assert claimed_at == now - 60


def test_requeue_stale_players_counts_orphaned_claims_alongside_the_other_categories():
    conn = make_conn()
    now = db.now()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": now - 4 * 86400})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "error", "last_crawled_at": now - 7200})
    # Older than the 1-hour default orphan window, so it is reaped...
    db.upsert(conn, "players", ["uid"], {"uid": 3, "crawl_status": "claimed", "claimed_at": now - 7200})
    # ...while a claim a worker could still plausibly be working on is not.
    db.upsert(conn, "players", ["uid"], {"uid": 4, "crawl_status": "claimed", "claimed_at": now - 1200})

    assert crawler.requeue_stale_players(conn) == 3
    assert conn.execute("SELECT crawl_status FROM players WHERE uid=4").fetchone()[0] == "claimed"


def test_requeue_stale_players_default_orphan_window_outlasts_a_real_crawl():
    # A fully-crawled player runs to ~112 matches, and the shared limiter can
    # sit at max_delay=8s under sustained strain, so one legitimate crawl can
    # take ~15 minutes. The default window must comfortably exceed that or this
    # sweep steals players out from under live workers — idempotent, so
    # harmless to the data, but it burns requests the politeness budget cannot
    # spare. A half-hour-old claim must therefore survive the default.
    conn = make_conn()
    now = db.now()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "claimed", "claimed_at": now - 1800})

    assert crawler.requeue_stale_players(conn) == 0
    assert conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0] == "claimed"


def test_requeue_stale_players_ignores_a_claimed_row_with_no_claim_timestamp():
    # A NULL timestamp never satisfies `< cutoff`, so a row in an unexpected
    # state is left alone rather than yanked out from under a live worker.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "claimed"})

    assert crawler.requeue_stale_players(conn) == 0
    assert conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0] == "claimed"


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


def test_reseed_never_holds_the_write_lock_across_a_network_call(tmp_path):
    """The root cause of the worst bug this pool could have.

    reseed writes (seeding players, stamping heroes.last_seeded_at) inside a
    loop that makes one network call per hero, ~42 of them, plus the global
    board. pysqlite opens the write transaction at the first write and holds
    SQLite's single write lock until commit — so committing only at the end
    would hold that lock across every one of those fetches: 40s at the default
    delay, minutes once the shared limiter has backed off to max_delay. Every
    concurrent worker's claim, ingest and status write would block for the full
    busy_timeout and then raise "database is locked".

    This probes from a SECOND connection at the exact moment reseed is inside a
    network call, and asserts the database is writable there every time.
    """
    path = str(tmp_path / "reseed_lock.db")
    conn = db.connect(path)
    db.init_schema(conn)
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    for hero_id in (1047, 1048, 1049):
        db.upsert(conn, "heroes", ["hero_id"], {"hero_id": hero_id})
    conn.commit()

    other = db.connect(path)
    other.execute("PRAGMA busy_timeout=50")  # fail fast rather than wait 5s
    raw_leaderboard_text = (FIXTURES / "leaderboard_payload.json").read_text(encoding="utf-8")
    hero_lb = load("hero_leaderboard.json")
    observations = []

    def probe(where):
        # Stands in for a worker trying to claim/ingest while reseed is out on
        # the network.
        try:
            other.execute("UPDATE players SET nick_name='probe' WHERE uid=1")
            other.commit()
            observations.append((where, "writable"))
        except sqlite3.OperationalError:
            other.rollback()
            observations.append((where, "LOCKED"))

    class ProbingClient:
        def get_json(self, path_, params=None):
            probe("hero_leaderboard")
            return hero_lb

        def get_text(self, path_, params=None):
            probe("global_leaderboard")
            return raw_leaderboard_text

    crawler.reseed(conn, ProbingClient())

    # One probe per hero fetch plus one for the global board, and every single
    # one found the database writable.
    assert len(observations) == 4
    assert [w for _, w in observations] == ["writable"] * 4, observations
    conn.close()
    other.close()


def test_crawl_player_never_holds_the_write_lock_across_a_network_call(tmp_path):
    """The reseed bug above, in the per-player crawl loop (2026-09-27).

    _record_history_entry writes a row and leaves it for the page's next
    commit, so the write lock was held across the match-detail fetch that
    followed it, and across the next history page after a run of known
    matches. Each of those fetches first waits its turn in the shared pacer
    (~2.7s at 8 workers / 3 req/s), longer than it takes the other workers'
    5s busy_timeout to run out: 5,106 "database is locked" pauses and 172
    stranded claims in 3.6 active hours of the 2026-09-26 run.

    Page 1 holds one new match then 19 known ones, so both paths are probed:
    the detail fetch after a new match's history row, and the page-2 fetch
    after 19 known matches' history rows."""
    path = str(tmp_path / "crawl_lock.db")
    conn = db.connect(path)
    db.init_schema(conn)
    uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    known = [f"known-{i}" for i in range(19)]
    for match_uid in known:
        db.upsert(conn, "matches", ["match_uid"], {"match_uid": match_uid})
    conn.commit()

    other = db.connect(path)
    other.execute("PRAGMA busy_timeout=50")  # fail fast rather than wait 5s
    profile = load("player_public.json")
    match = load("match_detail.json")
    page_1 = [{"match_uid": match["match_uid"]}] + [{"match_uid": m} for m in known]
    observations = []

    def probe(where):
        # Stands in for another worker claiming or ingesting while this one
        # is out on the network.
        try:
            other.execute("UPDATE players SET nick_name='probe' WHERE uid=?", (uid,))
            other.commit()
            observations.append((where, "writable"))
        except sqlite3.OperationalError:
            other.rollback()
            observations.append((where, "LOCKED"))

    class ProbingClient:
        def get_json(self, path_, params=None):
            if path_ == f"/api/player/{uid}":
                probe("profile")
                return profile
            if path_ == f"/api/player-match-history/{uid}":
                probe(f"history skip={params['skip']}")
                return page_1 if params["skip"] == 0 else []
            if path_ == f"/api/matches/{match['match_uid']}":
                probe("match detail")
                return match
            raise AssertionError(f"unexpected call: {path_} {params}")

    assert crawler.crawl_player(conn, ProbingClient(), uid=uid, season=19) == "done"

    assert [w for w, _ in observations] == [
        "profile", "history skip=0", "match detail", "history skip=20",
    ]
    assert [s for _, s in observations] == ["writable"] * 4, observations
    conn.close()
    other.close()


def test_reseed_queues_players_from_global_and_hero_leaderboards():
    conn = make_conn()
    raw_leaderboard_text = (FIXTURES / "leaderboard_payload.json").read_text(encoding="utf-8")
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


# ---- 2026-09-22 review: per-player failures, revisits -------------------------------------------

class _ScriptedClient:
    """Serves a public profile, the given history pages, and a match detail for any uid."""
    def __init__(self, uid, pages, profile=None, history_exc=None):
        self.uid, self.pages, self.history_exc = uid, pages, history_exc
        self.profile = profile or load("player_public.json")
        self.detail_calls = []

    def get_json(self, path, params=None):
        if path == f"/api/player/{self.uid}":
            return self.profile
        if path == f"/api/player-match-history/{self.uid}":
            if self.history_exc:
                raise self.history_exc
            i = params["skip"] // 20
            return self.pages[i] if i < len(self.pages) else []
        if path.startswith("/api/matches/"):
            muid = path.rsplit("/", 1)[1]
            self.detail_calls.append(muid)
            m = copy.deepcopy(load("match_detail.json")); m["match_uid"] = muid
            return m
        raise AssertionError(f"unexpected call: {path} {params}")


def test_a_404_on_match_history_marks_the_player_not_indexed_instead_of_escaping():
    # Reproduced by the review: this used to escape crawl_player, kill the worker, stop
    # the pool, and crash again on the same player after every restart.
    conn = make_conn(); uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    client = _ScriptedClient(uid, [], history_exc=fetcher.PlayerNotFoundError("history"))
    assert crawler.crawl_player(conn, client, uid=uid, season=20) == "not_indexed"


def test_a_history_page_that_is_not_a_list_is_a_fetch_error_not_a_crash():
    conn = make_conn(); uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    client = _ScriptedClient(uid, [{"error": "unexpected shape"}])
    with pytest.raises(fetcher.FetchError):
        crawler.crawl_player(conn, client, uid=uid, season=20)


def test_history_entries_without_a_match_uid_are_skipped():
    conn = make_conn(); uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    client = _ScriptedClient(uid, [[{"match_map_id": 1}, {"match_uid": "m_ok", "match_map_id": 1245}]])
    assert crawler.crawl_player(conn, client, uid=uid, season=20) == "done"
    assert client.detail_calls == ["m_ok"]


def test_an_interrupted_revisit_does_not_lose_the_newer_matches_it_had_not_reached():
    # Reproduced by the review: history [A, B, C, D]; D is from before the last completed
    # crawl, A was ingested by a revisit that was then interrupted. The next revisit used to
    # stop at A and never fetch B or C. It must stop only at a known match played before
    # the last completed crawl.
    conn = make_conn(); uid = 457877313
    last_done = db.now() - 5 * 86400
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed", "last_crawled_at": last_done})
    for known in ("A", "D"):
        db.upsert(conn, "matches", ["match_uid"], {"match_uid": known})
    conn.commit()
    newer = last_done + 86400
    page = [{"match_uid": "A", "match_time_stamp": newer + 300},
            {"match_uid": "B", "match_time_stamp": newer + 200},
            {"match_uid": "C", "match_time_stamp": newer + 100},
            {"match_uid": "D", "match_time_stamp": last_done - 7200},
            {"match_uid": "E", "match_time_stamp": last_done - 9000}]
    client = _ScriptedClient(uid, [page])
    assert crawler.crawl_player(conn, client, uid=uid, season=20) == "done"
    assert client.detail_calls == ["B", "C"]                  # stopped at D, never reached E


def test_requeue_moves_old_done_players_to_revisit_not_pending():
    conn = make_conn(); now = db.now()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": now - 4 * 86400})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "done", "last_crawled_at": now - 2 * 86400})
    n = crawler.requeue_stale_players(conn)
    assert n == 1                                             # default revisit window is 72 h
    assert dict(conn.execute("SELECT uid, crawl_status FROM players")) == {1: "revisit", 2: "done"}


def test_requeue_caps_revisits_per_cycle_oldest_first():
    conn = make_conn(); now = db.now()
    for uid in range(10):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "done", "last_crawled_at": now - 10 * 86400 + uid})
    crawler.requeue_stale_players(conn, max_revisits=3)
    assert [r[0] for r in conn.execute("SELECT uid FROM players WHERE crawl_status='revisit' ORDER BY uid")] == [0, 1, 2]


def test_first_crawls_are_claimed_before_revisits():
    conn = make_conn()
    conn.execute("INSERT INTO players (uid, discovery_hero_id, crawl_status, created_at, last_crawled_at) VALUES (1, 1001, 'revisit', 1, 1)")
    conn.execute("INSERT INTO players (uid, discovery_hero_id, crawl_status, created_at) VALUES (2, 1001, 'pending', 999)")
    conn.commit()
    assert crawler.claim_next_player(conn) == 2
    assert crawler.claim_next_player(conn) == 1               # revisits only once no first crawl is waiting
    assert crawler.claim_next_player(conn) is None


def test_init_schema_migrates_queued_revisits_out_of_pending(tmp_path):
    path = str(tmp_path / "old.db"); conn = db.connect(path); db.init_schema(conn)
    conn.execute("INSERT INTO players (uid, crawl_status, last_crawled_at) VALUES (1, 'pending', 123)")
    conn.execute("INSERT INTO players (uid, crawl_status) VALUES (2, 'pending')")
    conn.commit(); db.init_schema(conn)
    assert dict(conn.execute("SELECT uid, crawl_status FROM players")) == {1: "revisit", 2: "pending"}


# ---- 2026-09-24: provenance and leaver capture (model review) --------------------------------------
# The review found the crawl selects matches through hero-leaderboard specialists, a bias that can
# only be measured if we know whose history surfaced each match; and leavers decide matches, but
# the flag exists only in each player's own history entries. Both are read at no extra request.

def test_a_new_match_records_the_player_whose_history_surfaced_it():
    conn = make_conn(); uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    client = _ScriptedClient(uid, [[{"match_uid": "m_new", "match_map_id": 1245,
                                     "match_player": {"player_uid": uid, "has_escaped": False}}]])
    crawler.crawl_player(conn, client, uid=uid, season=20)
    assert conn.execute("SELECT source_player_uid FROM matches WHERE match_uid='m_new'").fetchone()[0] == uid


def test_every_history_entry_read_is_recorded_with_its_leaver_flag_even_for_known_matches():
    conn = make_conn(); uid = 457877313
    db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    db.upsert(conn, "matches", ["match_uid"], {"match_uid": "m_known"}); conn.commit()
    page = [{"match_uid": "m_new", "match_player": {"player_uid": uid, "has_escaped": True, "is_win": 0}},
            {"match_uid": "m_known", "match_player": {"player_uid": uid, "has_escaped": False, "is_win": 1}}]
    crawler.crawl_player(conn, _ScriptedClient(uid, [page]), uid=uid, season=20)
    rows = dict(conn.execute("SELECT match_uid, has_escaped FROM history_entries WHERE player_uid=?", (uid,)))
    assert rows == {"m_new": 1, "m_known": 0}


def test_a_match_first_surfaced_by_one_player_keeps_that_source_when_another_reports_it():
    conn = make_conn()
    for uid in (1, 2):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "claimed"})
    entry = lambda uid: [{"match_uid": "m_shared", "match_player": {"player_uid": uid, "has_escaped": False}}]
    crawler.crawl_player(conn, _ScriptedClient(1, [entry(1)]), uid=1, season=20)
    crawler.crawl_player(conn, _ScriptedClient(2, [entry(2)]), uid=2, season=20)
    assert conn.execute("SELECT source_player_uid FROM matches WHERE match_uid='m_shared'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM history_entries WHERE match_uid='m_shared'").fetchone()[0] == 2


# ---- floor and private retries, dated levels, failed match fetches (2026-10-02 review) ----

def _player(conn, uid, **cols):
    db.upsert(conn, "players", ["uid"], {"uid": uid, **cols})


def test_requeue_returns_a_floor_skipped_player_once_their_level_reads_diamond():
    conn = make_conn()
    old = db.now() - 10
    _player(conn, 1, crawl_status="skipped_floor", latest_known_level=14, last_crawled_at=old)
    _player(conn, 2, crawl_status="skipped_floor", latest_known_level=12, last_crawled_at=old)

    assert crawler.requeue_stale_players(conn) == 1

    rows = dict(conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert rows == {1: "pending", 2: "skipped_floor"}
    # A first crawl again, so it paginates their whole history.
    assert conn.execute("SELECT last_crawled_at FROM players WHERE uid=1").fetchone()[0] is None


def test_requeue_retries_private_players_after_a_week_capped_oldest_first():
    conn = make_conn()
    now = db.now()
    _player(conn, 1, crawl_status="skipped_private", last_crawled_at=now - 9 * 86400)
    _player(conn, 2, crawl_status="skipped_private", last_crawled_at=now - 8 * 86400)
    _player(conn, 3, crawl_status="skipped_private", last_crawled_at=now - 86400)

    assert crawler.requeue_stale_players(conn, max_private_retries=1) == 1

    rows = dict(conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert rows == {1: "pending", 2: "skipped_private", 3: "skipped_private"}


def _ingest_with(conn, match, match_uid, ts, uid, new_level):
    m = copy.deepcopy(match)
    m["match_uid"] = match_uid
    m["match_time_stamp"] = ts
    for p in m["match_players"]:
        if p["player_uid"] == uid:
            p.setdefault("dynamic_fields", {})["new_level"] = new_level
    ingest.ingest_match(conn, m, season=20)


def test_an_older_match_never_replaces_a_newer_level():
    conn = make_conn()
    match = load("match_detail.json")
    uid = match["match_players"][0]["player_uid"]
    level = "SELECT latest_known_level, level_seen_at FROM players WHERE uid=?"

    _ingest_with(conn, match, "newer", 2_000, uid, new_level=14)
    _ingest_with(conn, match, "older", 1_000, uid, new_level=11)
    assert conn.execute(level, (uid,)).fetchone() == (14, 2_000)

    _ingest_with(conn, match, "newest", 3_000, uid, new_level=12)
    assert conn.execute(level, (uid,)).fetchone() == (12, 3_000)


def test_a_profile_without_this_seasons_rank_keeps_the_stored_level():
    conn = make_conn()
    uid = 457877313
    _player(conn, uid, crawl_status="pending", latest_known_level=15, level_seen_at=1_000)
    profile = copy.deepcopy(load("player_public.json"))
    info = profile["player"]["info"]
    profile["player"]["info"] = {k: v for k, v in info.items() if not k.startswith("rank_game_")}
    profile["visibility"] = {"match_history": False}

    class ProfileOnly:
        def get_json(self, path, params=None):
            return profile

    assert crawler.crawl_player(conn, ProfileOnly(), uid=uid, season=20) == "skipped_private"
    row = conn.execute("SELECT latest_known_level, level_seen_at FROM players WHERE uid=?", (uid,)).fetchone()
    assert row == (15, 1_000)


def _failing_match_client(uid, profile, history_page, good, failing):
    class Client:
        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            if path == f"/api/player-match-history/{uid}":
                return history_page if params["skip"] == 0 else []
            match_uid = path.rsplit("/", 1)[-1]
            if match_uid in failing:
                raise fetcher.FetchError(f"non-JSON response: {path}")
            if match_uid == good["match_uid"]:
                return good
            raise AssertionError(f"unexpected call: {path} {params}")
    return Client()


def test_crawl_player_skips_a_match_whose_fetch_fails_and_keeps_going():
    conn = make_conn()
    match = load("match_detail.json")
    uid = 457877313
    _player(conn, uid, crawl_status="pending")
    page = [{"match_uid": "bad"}, {"match_uid": match["match_uid"]}]
    client = _failing_match_client(uid, load("player_public.json"), page, match, {"bad"})

    assert crawler.crawl_player(conn, client, uid=uid, season=19) == "done"
    stored = {r[0] for r in conn.execute("SELECT match_uid FROM matches").fetchall()}
    assert stored == {match["match_uid"]}


def test_crawl_player_still_fails_when_many_match_fetches_fail():
    conn = make_conn()
    match = load("match_detail.json")
    uid = 457877313
    _player(conn, uid, crawl_status="pending")
    bad = {f"bad{i}" for i in range(crawler.MAX_MATCH_FETCH_FAILURES)}
    page = [{"match_uid": b} for b in sorted(bad)] + [{"match_uid": match["match_uid"]}]
    client = _failing_match_client(uid, load("player_public.json"), page, match, bad)

    with pytest.raises(fetcher.FetchError):
        crawler.crawl_player(conn, client, uid=uid, season=19)


# ---- the profile's own match history, and private 403s (2026-10-02) ----

def _entry(match_uid, ts, mode=2, season=20):
    return {"match_uid": match_uid, "match_time_stamp": ts, "game_mode_id": mode, "match_season": str(season)}


class _HistoryClient:
    """Serves a profile and match details; records history-page calls."""

    def __init__(self, uid, profile, match, page=None):
        self.uid, self.profile, self.match, self.page = uid, profile, match, page or []
        self.history_calls, self.detail_calls = 0, []

    def get_json(self, path, params=None):
        if path == f"/api/player/{self.uid}":
            return self.profile
        if path == f"/api/player-match-history/{self.uid}":
            self.history_calls += 1
            return self.page if params["skip"] == 0 else []
        if path.startswith("/api/matches/"):
            mu = path.rsplit("/", 1)[1]
            self.detail_calls.append(mu)
            return dict(copy.deepcopy(self.match), match_uid=mu)
        raise AssertionError(f"unexpected call: {path} {params}")


def _profile_with(entries):
    profile = copy.deepcopy(load("player_public.json"))
    profile["match_history"] = entries
    return profile


def test_a_short_profile_history_is_the_whole_season_so_no_history_page_is_fetched():
    conn = make_conn()
    uid = 457877313
    _player(conn, uid, crawl_status="pending")
    entries = [_entry("r1", 3_000), _entry("q1", 2_500, mode=3), _entry("r2", 2_000)]
    client = _HistoryClient(uid, _profile_with(entries), load("match_detail.json"))

    assert crawler.crawl_player(conn, client, uid=uid, season=20) == "done"
    assert client.history_calls == 0
    assert client.detail_calls == ["r1", "r2"]  # ranked only


def test_a_full_profile_history_on_a_first_crawl_still_reads_the_pages():
    conn = make_conn()
    uid = 457877313
    _player(conn, uid, crawl_status="pending")
    entries = [_entry(f"r{i}", 10_000 - i) for i in range(crawler.HISTORY_PAGE_SIZE)]
    client = _HistoryClient(uid, _profile_with(entries), load("match_detail.json"), page=[_entry("p1", 9_000)])

    crawler.crawl_player(conn, client, uid=uid, season=20)
    assert client.history_calls == 1
    assert client.detail_calls == ["p1"]


def test_a_revisit_uses_a_profile_history_that_reaches_back_before_the_last_crawl():
    conn = make_conn()
    uid = 457877313
    last = 100_000
    _player(conn, uid, crawl_status="revisit", last_crawled_at=last)
    stop = last - crawler.REVISIT_STOP_SLACK_SECONDS
    db.upsert(conn, "matches", ["match_uid"], {"match_uid": "known_old"})
    entries = ([_entry("new1", last + 50), _entry("new2", last + 10)]
               + [_entry("known_old", stop - 10)]
               + [_entry(f"older{i}", stop - 100 - i) for i in range(crawler.HISTORY_PAGE_SIZE - 3)])
    client = _HistoryClient(uid, _profile_with(entries), load("match_detail.json"))

    assert crawler.crawl_player(conn, client, uid=uid, season=20) == "done"
    assert client.history_calls == 0
    assert client.detail_calls == ["new1", "new2"]


def test_a_profile_history_from_another_season_is_not_trusted():
    conn = make_conn()
    uid = 457877313
    _player(conn, uid, crawl_status="pending")
    client = _HistoryClient(uid, _profile_with([_entry("old_season", 1_000, season=19)]),
                            load("match_detail.json"), page=[_entry("p1", 9_000)])

    crawler.crawl_player(conn, client, uid=uid, season=20)
    assert client.history_calls == 1
    assert client.detail_calls == ["p1"]


def test_a_history_that_turns_private_marks_the_player_private():
    conn = make_conn()
    uid = 457877313
    _player(conn, uid, crawl_status="pending")
    profile = copy.deepcopy(load("player_public.json"))
    del profile["match_history"]

    class TurnsPrivate:
        def get_json(self, path, params=None):
            if path == f"/api/player/{uid}":
                return profile
            raise fetcher.PrivateError(path)

    assert crawler.crawl_player(conn, TurnsPrivate(), uid=uid, season=19) == "skipped_private"
