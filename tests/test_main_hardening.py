"""main.py after the 2026-09-22 review: halting on a block, per-player failures,
warm restarts, and log lines that carry a time."""
import re
import threading
import time

import pytest

import crawler
import db
import fetcher
import main

from test_main import _pool_db, make_conn


def _run_pool(path, conn, crawl, workers=3, **kw):
    kw.setdefault("reseed_fn", lambda conn_, client_: 0)
    kw.setdefault("requeue_fn", lambda conn_: 0)
    return main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(),
                    reseed_interval_seconds=10**9, crawl_player_fn=crawl, workers=workers,
                    worker_conn_fn=lambda: db.connect(path), worker_idle_sleep_seconds=0.01,
                    coordinator_poll_seconds=0.01, **kw)


def test_a_block_stops_every_worker_and_leaves_players_pending(tmp_path):
    path, conn = _pool_db(tmp_path, 30)
    calls = []
    lock = threading.Lock()

    def crawl(conn_, client_, uid, season):
        with lock:
            calls.append(uid)
            n = len(calls)
        if n >= 3:
            raise fetcher.BlockedError("BLOCKED: status 400 for /api/player/x (body=error)")
        crawler.set_status(conn_, uid, "done")

    outcome = _run_pool(path, conn, crawl, workers=4)
    assert outcome == main.OUTCOME_BLOCKED
    statuses = dict(conn.execute("SELECT crawl_status, COUNT(*) FROM players GROUP BY 1"))
    assert "error" not in statuses                             # a block is nobody's fault
    assert "claimed" not in statuses                           # every claim handed back
    assert len(calls) <= 2 + 4                                 # at most one attempt per worker after the block
    conn.close()


def test_a_block_during_the_startup_reseed_stops_before_any_crawl():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})

    def reseed(conn_, client_):
        raise fetcher.BlockedError("BLOCKED: status 400 for /api/hero-leaderboard/1011")

    def crawl(*a):
        raise AssertionError("must not crawl after a block")

    outcome = main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(),
                       reseed_interval_seconds=10**9, crawl_player_fn=crawl, reseed_fn=reseed,
                       requeue_fn=lambda c: 0)
    assert outcome == main.OUTCOME_BLOCKED


def test_an_unexpected_per_player_exception_marks_that_player_error_and_the_crawl_goes_on():
    conn = make_conn()
    for uid in (1, 2, 3):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})

    def crawl(conn_, client_, uid, season):
        if uid == 2:
            raise KeyError("match_uid")                        # a payload that changed shape
        crawler.set_status(conn_, uid, "done")

    main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(), reseed_interval_seconds=10**9,
             crawl_player_fn=crawl, reseed_fn=lambda c, cl: 0, requeue_fn=lambda c: 0)
    assert dict(conn.execute("SELECT uid, crawl_status FROM players")) == {1: "done", 2: "error", 3: "done"}


def test_a_run_of_unexpected_exceptions_still_aborts_so_a_real_bug_is_not_hidden():
    conn = make_conn()
    for uid in range(1, 20):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})

    def crawl(conn_, client_, uid, season):
        raise RuntimeError("a bug in our own code")

    with pytest.raises(RuntimeError, match="a bug in our own code"):
        main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(), reseed_interval_seconds=10**9,
                 crawl_player_fn=crawl, reseed_fn=lambda c, cl: 0, requeue_fn=lambda c: 0)
    errors = conn.execute("SELECT COUNT(*) FROM players WHERE crawl_status='error'").fetchone()[0]
    assert errors == main.MAX_CONSECUTIVE_UNEXPECTED


def test_keyboard_interrupt_still_releases_the_claim_and_propagates():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})

    def crawl(*a):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(), reseed_interval_seconds=10**9,
                 crawl_player_fn=crawl, reseed_fn=lambda c, cl: 0, requeue_fn=lambda c: 0)
    assert conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0] == "pending"


def test_the_queue_is_not_drained_while_revisits_wait():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "revisit", "last_crawled_at": 1})
    assert not main._queue_is_drained(conn)


def test_a_restart_soon_after_the_last_request_holds_the_slow_start():
    now = 10_000_000
    assert main.warm_restart_hold_seconds(None, now) == 0
    assert main.warm_restart_hold_seconds(now - 7200, now) == 0
    assert main.warm_restart_hold_seconds(now - 120, now) == main.WARM_RESTART_HOLD_SECONDS


def test_log_lines_carry_a_timestamp_and_the_thread(capsys):
    main.log("hello")
    err = capsys.readouterr().err
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[MainThread\] hello$", err.strip())


def test_the_stats_line_reports_status_counts_rate_and_budget():
    class Client:
        def take_status_counts(self):
            return {200: 170, 404: 10}
        def circuit_seconds_remaining(self):
            return 0.0
        class limiter:
            daily_budget = 250_000
            @staticmethod
            def current_rate():
                return 3.0
            @staticmethod
            def requests_in_window():
                return 12_345
    line = main.format_request_stats(Client(), window_seconds=60)
    assert "requests=180" in line and "3.00/s" in line
    assert "200:170" in line and "404:10" in line
    assert "cap=3.00/s" in line and "24h=12,345/250,000" in line


def test_the_workers_start_staggered(tmp_path):
    path, conn = _pool_db(tmp_path, 3)
    slept = []

    def crawl(conn_, client_, uid, season):
        crawler.set_status(conn_, uid, "done")

    main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(), reseed_interval_seconds=10**9,
             crawl_player_fn=crawl, reseed_fn=lambda c, cl: 0, requeue_fn=lambda c: 0, workers=3,
             worker_conn_fn=lambda: db.connect(path), worker_idle_sleep_seconds=0.01,
             coordinator_poll_seconds=0.01, worker_start_stagger_seconds=7.0,
             # Record the stagger waits; really sleep the short idle/poll waits, or idle
             # workers spin through claim attempts and the pool never looks idle.
             sleep_fn=lambda s: slept.append(s) if s >= 1 else time.sleep(s))
    assert sorted(s for s in slept if s >= 7.0) == [7.0, 14.0]


def test_a_season_rollover_stops_the_run_at_the_next_periodic_tick():
    conn = make_conn()
    for uid in range(1, 50):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})

    def crawl(conn_, client_, uid, season):
        crawler.set_status(conn_, uid, "done")

    outcome = main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(),
                       reseed_interval_seconds=-1,           # tick before every player
                       crawl_player_fn=crawl, reseed_fn=lambda c, cl: 0, requeue_fn=lambda c: 0,
                       season_fn=lambda client: 21)
    assert outcome == main.OUTCOME_SEASON_CHANGED
    assert conn.execute("SELECT COUNT(*) FROM players WHERE crawl_status='done'").fetchone()[0] == 0


def test_an_unchanged_season_lets_the_run_continue():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    outcome = main.run(conn, object(), season=20, shutdown_flag=main.ShutdownFlag(),
                       reseed_interval_seconds=-1,
                       crawl_player_fn=lambda c, cl, uid, s: crawler.set_status(c, uid, "done"),
                       reseed_fn=lambda c, cl: 0, requeue_fn=lambda c: 0, season_fn=lambda client: 20)
    assert outcome == main.OUTCOME_DONE
    assert conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0] == "done"
