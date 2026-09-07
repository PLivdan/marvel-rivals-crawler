import threading
import time

import pytest

import db
import fetcher
import main
import rivalsmeta


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


def test_format_progress_line_reports_rate_and_eta_from_recent_throughput():
    conn = make_conn()
    now = db.now()
    for uid in range(5):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "done", "last_crawled_at": now})
    for uid in range(5, 15):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})

    line = main.format_progress_line(conn, window_seconds=60)

    assert "rate=5.0/min" in line
    assert "eta_to_drain_queue=~2m" in line


def test_format_progress_line_eta_unknown_when_no_recent_throughput():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    line = main.format_progress_line(conn, window_seconds=60)
    assert "eta_to_drain_queue=unknown" in line


def test_format_progress_line_eta_shows_queue_empty_when_nothing_pending():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": db.now()})
    line = main.format_progress_line(conn, window_seconds=60)
    assert "eta_to_drain_queue=queue empty" in line


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

    def fake_crawl_player(conn_, client_, uid, season):
        calls.append(uid)
        flag.requested = True  # pretend a shutdown arrived mid-run
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
    )
    assert len(calls) == 1


def test_run_marks_player_error_and_continues_after_fetch_error():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    db.upsert(conn, "players", ["uid"], {"uid": 2, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    calls = []

    def flaky_crawl_player(conn_, client_, uid, season):
        calls.append(uid)
        if uid == 1:
            raise fetcher.FetchError("boom")
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=flaky_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
    )

    assert calls == [1, 2]
    status = conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0]
    assert status == "error"


def test_run_pauses_on_circuit_open_error_then_retries():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    calls = []
    sleeps = []

    def circuit_then_success(conn_, client_, uid, season):
        calls.append(uid)
        if len(calls) == 1:
            raise fetcher.CircuitOpenError("open")
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=circuit_then_success,
        reseed_fn=lambda conn_, client_: 0,
        circuit_cooldown_seconds=0,
        sleep_fn=sleeps.append,
    )

    # The same player being claimed a SECOND time is itself the proof that the
    # first attempt released its claim back to 'pending' — a still-claimed row
    # is invisible to claim_next_player, so calls could only ever be [1].
    assert calls == [1, 1]
    assert sleeps == [0]
    # The fake crawl_player_fn never writes to the DB itself (that's real
    # crawler.crawl_player's job, covered by Task 6's tests) — this asserts
    # the narrower thing this test actually owns: run() must NOT mark a
    # player "error" after a CircuitOpenError the way it does for a
    # FetchError, since a tripped circuit isn't this player's fault. (The row
    # is left 'claimed' here only because the fake never reaches a terminal
    # status; test_run_releases_the_claim_when_the_circuit_opens pins the
    # release down directly.)
    status = conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0]
    assert status != "error"


def test_run_backs_off_on_rate_limited_error_and_leaves_player_pending():
    # A sustained 429/403 is site-wide throttling, not this player's fault:
    # run() must back off like a tripped circuit and leave them 'pending',
    # never mark them 'error' (nothing in this codebase resets 'error').
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    calls = []
    sleeps = []

    def rate_limited_then_success(conn_, client_, uid, season):
        calls.append(uid)
        if len(calls) == 1:
            raise fetcher.RateLimitedError("status 429 for /api/player/1")
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=rate_limited_then_success,
        reseed_fn=lambda conn_, client_: 0,
        circuit_cooldown_seconds=0,
        sleep_fn=sleeps.append,
    )

    # Being claimed a second time proves the back-off released the claim: a row
    # still marked 'claimed' can never be handed out again.
    assert calls == [1, 1]  # retried the same player after backing off
    assert sleeps == [0]
    status = conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0]
    assert status != "error"


def test_run_releases_the_claim_when_the_circuit_opens():
    # The direct form of the assertion the two tests above make indirectly:
    # once a player is claimed, a tripped circuit must hand them straight back
    # to the frontier. Leaving them 'claimed' would take them out of the queue
    # until the orphaned-claim sweep noticed, ten minutes later.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()

    def circuit_open_then_shut_down(conn_, client_, uid, season):
        flag.requested = True  # so the loop exits right after the release
        raise fetcher.CircuitOpenError("open")

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=circuit_open_then_shut_down,
        reseed_fn=lambda conn_, client_: 0,
        circuit_cooldown_seconds=0,
        sleep_fn=lambda s: None,
    )

    status, claimed_at = conn.execute(
        "SELECT crawl_status, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "pending"
    assert claimed_at is None


def test_run_releases_the_claim_when_the_crawl_raises_something_unexpected():
    # Only three exception types are handled by name. Anything else — an
    # unwrapped PlayerNotFoundError from the match-history endpoint, a
    # JSONDecodeError when the site answers with an HTML challenge page (a
    # RequestException, so NOT a FetchError) — used to kill the process, but it
    # left the player 'pending' and instantly retryable. Now they are claimed,
    # so an unhandled exception must hand the claim back on its way out or the
    # player sits out the whole orphan-sweep window for no reason.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()

    class UnexpectedError(Exception):
        pass

    def explode(conn_, client_, uid, season):
        raise UnexpectedError("an HTML challenge page, say")

    with pytest.raises(UnexpectedError):
        main.run(
            conn,
            object(),
            season=19,
            shutdown_flag=flag,
            reseed_interval_seconds=10**9,
            crawl_player_fn=explode,
            reseed_fn=lambda conn_, client_: 0,
        )

    status, claimed_at = conn.execute(
        "SELECT crawl_status, claimed_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "pending"  # not stranded at 'claimed'
    assert claimed_at is None


def test_run_marks_a_fetch_error_through_the_one_terminal_status_writer():
    # crawler.set_status is the single writer of a terminal status, and the
    # only place that clears claimed_at alongside it. Duplicating that upsert
    # inline here is exactly how a future edit forgets the invariant the
    # orphaned-claim sweep depends on.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()

    def always_fetch_error(conn_, client_, uid, season):
        flag.requested = True
        raise fetcher.FetchError("boom")

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=always_fetch_error,
        reseed_fn=lambda conn_, client_: 0,
    )

    status, claimed_at, last = conn.execute(
        "SELECT crawl_status, claimed_at, last_crawled_at FROM players WHERE uid=1"
    ).fetchone()
    assert status == "error"
    assert claimed_at is None  # the claim invariant survived the error path
    assert last is not None


def test_run_requeues_stale_players_at_startup_and_on_the_periodic_interval():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    requeues = []
    calls = []

    def fake_requeue(conn_):
        requeues.append(conn_)
        return 0

    def fake_crawl_player(conn_, client_, uid, season):
        calls.append(uid)
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=-1,  # force the periodic branch immediately
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=fake_requeue,
    )

    assert len(requeues) == 2  # once at startup, once on the periodic tick
    assert all(c is conn for c in requeues)
    assert calls == [1]


def test_run_requeues_even_when_the_reseed_had_to_back_off():
    # Requeueing is local DB work with nothing network-shaped to fail, so a
    # backed-off reseed must not also cost us the frontier refresh.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    requeues = []

    def failing_reseed(conn_, client_):
        raise fetcher.FetchError("boom")

    def fake_crawl_player(conn_, client_, uid, season):
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=failing_reseed,
        requeue_fn=lambda conn_: requeues.append(conn_),
    )

    assert len(requeues) == 1


def test_run_survives_circuit_open_error_from_the_startup_reseed():
    # A reseed makes ~40+ requests; a failure there must not kill the process.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    calls = []
    sleeps = []

    def exploding_reseed(conn_, client_):
        raise fetcher.CircuitOpenError("open")

    def fake_crawl_player(conn_, client_, uid, season):
        calls.append(uid)
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=exploding_reseed,
        circuit_cooldown_seconds=0,
        sleep_fn=sleeps.append,
    )

    assert sleeps == [0]  # cooled down instead of crashing
    assert calls == [1]  # and carried on crawling


def test_run_survives_fetch_error_from_the_periodic_reseed():
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    calls = []
    reseeds = []

    def flaky_reseed(conn_, client_):
        reseeds.append(1)
        if len(reseeds) > 1:  # the periodic (not startup) reseed
            raise fetcher.FetchError("boom")
        return 0

    def fake_crawl_player(conn_, client_, uid, season):
        calls.append(uid)
        flag.requested = True
        return "done"

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=-1,  # force the periodic reseed immediately
        crawl_player_fn=fake_crawl_player,
        reseed_fn=flaky_reseed,
    )

    assert len(reseeds) == 2  # startup + periodic
    assert calls == [1]  # the loop continued past the failed reseed


def test_status_flag_prints_progress_and_exits_without_crawling(monkeypatch, capsys, tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = db.connect(db_path)
    db.init_schema(conn)
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "done", "last_crawled_at": db.now()})
    conn.commit()
    conn.close()

    def fail_if_called(*a, **k):
        raise AssertionError("should not run a crawl loop for --status")

    def fail_if_client_constructed():
        raise AssertionError("should not construct a network client for --status")

    monkeypatch.setattr(main, "run", fail_if_called)
    monkeypatch.setattr(fetcher, "RivalsMetaClient", fail_if_client_constructed)

    main.main(["--db-path", db_path, "--status"])

    captured = capsys.readouterr()
    assert "done=1" in captured.out


# ---------------------------------------------------------------------------
# --workers 1 backward compatibility
# ---------------------------------------------------------------------------


def _record_run_calls(conn, flag, n_players, **run_kwargs):
    """Drive run() over `n_players` pending players with fully instrumented
    fakes, and return the ordered log of everything it did. Used to compare the
    default (workers unset) path against an explicit --workers 1."""
    for uid in range(1, n_players + 1):
        db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    conn.commit()

    events = []

    def fake_crawl_player(conn_, client_, uid, season):
        events.append(("crawl", uid, season))
        db.upsert(
            conn_,
            "players",
            ["uid"],
            {"uid": uid, "crawl_status": "done", "last_crawled_at": db.now(), "claimed_at": None},
        )
        conn_.commit()
        return "done"

    def fake_reseed(conn_, client_):
        events.append(("reseed",))
        return 0

    def fake_requeue(conn_):
        events.append(("requeue",))
        return 0

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=fake_reseed,
        requeue_fn=fake_requeue,
        sleep_fn=lambda s: events.append(("sleep", s)),
        **run_kwargs,
    )
    return events


def test_workers_one_makes_exactly_the_same_call_sequence_as_the_unset_default():
    # The strict backward-compatibility requirement: opting out of concurrency
    # (or simply not opting in) must reproduce the pre-concurrency loop exactly
    # — same startup reseed+requeue, same one-player-at-a-time claim/crawl in
    # frontier-priority order, same exit the moment the queue drains.
    default_conn = make_conn()
    default_events = _record_run_calls(
        default_conn, main.ShutdownFlag(), 5, reseed_interval_seconds=10**9
    )

    explicit_conn = make_conn()
    explicit_events = _record_run_calls(
        explicit_conn, main.ShutdownFlag(), 5, reseed_interval_seconds=10**9, workers=1
    )

    assert default_events == explicit_events
    assert default_events == [
        ("reseed",),
        ("requeue",),
        ("crawl", 1, 19),
        ("crawl", 2, 19),
        ("crawl", 3, 19),
        ("crawl", 4, 19),
        ("crawl", 5, 19),
    ]
    # Every player finished, nothing was left claimed, and the run exited on
    # its own when the frontier emptied rather than idling.
    statuses = dict(default_conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert statuses == {uid: "done" for uid in range(1, 6)}


def test_workers_one_runs_on_the_calling_thread_with_no_pool_and_no_extra_connection():
    # "Identical behaviour" includes not quietly becoming concurrent: a single
    # worker must not spawn a pool, and must not need (or open) a second
    # connection to the database.
    conn = make_conn()
    db.upsert(conn, "players", ["uid"], {"uid": 1, "crawl_status": "pending"})
    flag = main.ShutdownFlag()
    threads = []
    conns_used = []

    def fake_crawl_player(conn_, client_, uid, season):
        threads.append(threading.current_thread())
        conns_used.append(conn_)
        db.upsert(conn_, "players", ["uid"], {"uid": uid, "crawl_status": "done"})
        conn_.commit()
        return "done"

    def exploding_worker_conn():
        raise AssertionError("a single worker must reuse the caller's connection")

    main.run(
        conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        worker_conn_fn=exploding_worker_conn,
    )

    assert threads == [threading.main_thread()]
    assert conns_used == [conn]


def test_main_defaults_to_a_single_worker_so_concurrency_stays_opt_in(monkeypatch, tmp_path):
    # Not passing --workers must leave the crawler exactly as conservative as
    # it has always been. (If the intended shipping default is instead 3, this
    # is the one line to change in main.main.)
    db_path = str(tmp_path / "w.db")
    captured = {}

    monkeypatch.setattr(fetcher, "RivalsMetaClient", lambda: object())
    monkeypatch.setattr(rivalsmeta, "resolve_current_season", lambda client: 19)
    monkeypatch.setattr(main, "run", lambda *a, **k: captured.update(k))

    main.main(["--db-path", db_path])

    assert captured["workers"] == 1


def test_main_passes_the_requested_worker_count_and_a_per_worker_connection_factory(
    monkeypatch, tmp_path
):
    db_path = str(tmp_path / "w.db")
    captured = {}

    monkeypatch.setattr(fetcher, "RivalsMetaClient", lambda: object())
    monkeypatch.setattr(rivalsmeta, "resolve_current_season", lambda client: 19)
    monkeypatch.setattr(main, "run", lambda *a, **k: captured.update(k))

    main.main(["--db-path", db_path, "--workers", "3"])

    assert captured["workers"] == 3
    # Each worker opens its own connection to the same file, because sqlite3
    # connections cannot be shared across threads.
    worker_conn = captured["worker_conn_fn"]()
    assert worker_conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 0
    worker_conn.close()


def test_main_rejects_a_worker_count_below_one(tmp_path):
    with pytest.raises(SystemExit):
        main.main(["--db-path", str(tmp_path / "w.db"), "--workers", "0"])


def test_main_rejects_multiple_workers_against_an_in_memory_database():
    # ":memory:" gives every connection its OWN empty database, so the workers
    # would silently crawl into nothing.
    with pytest.raises(SystemExit):
        main.main(["--db-path", ":memory:", "--workers", "3"])


def test_run_refuses_multiple_workers_without_a_connection_factory():
    conn = make_conn()
    with pytest.raises(ValueError, match="worker_conn_fn"):
        main.run(
            conn,
            object(),
            season=19,
            shutdown_flag=main.ShutdownFlag(),
            reseed_interval_seconds=10**9,
            crawl_player_fn=lambda *a: "done",
            reseed_fn=lambda conn_, client_: 0,
            workers=3,
        )


# ---------------------------------------------------------------------------
# the worker pool
# ---------------------------------------------------------------------------


def _pool_db(tmp_path, n_players):
    path = str(tmp_path / "pool.db")
    setup = db.connect(path)
    db.init_schema(setup)
    for uid in range(1, n_players + 1):
        db.upsert(setup, "players", ["uid"], {"uid": uid, "crawl_status": "pending"})
    setup.commit()
    return path, setup


def test_worker_pool_crawls_every_player_exactly_once_and_drains_the_queue(tmp_path):
    # The end-to-end claim: three real threads against one real on-disk DB, and
    # afterwards every player is done, none was handed out twice, and none was
    # left stranded in 'pending' or 'claimed'.
    path, coordinator_conn = _pool_db(tmp_path, 20)
    crawled = []
    lock = threading.Lock()

    def fake_crawl_player(conn_, client_, uid, season):
        time.sleep(0.002)  # a little real work, so the threads genuinely overlap
        with lock:
            crawled.append(uid)
        db.upsert(
            conn_,
            "players",
            ["uid"],
            {"uid": uid, "crawl_status": "done", "last_crawled_at": db.now(), "claimed_at": None},
        )
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=lambda conn_: 0,
        workers=3,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    assert sorted(crawled) == list(range(1, 21))  # all 20
    assert len(crawled) == len(set(crawled))  # none twice
    statuses = dict(coordinator_conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert statuses == {uid: "done" for uid in range(1, 21)}
    assert coordinator_conn.execute(
        "SELECT COUNT(*) FROM players WHERE claimed_at IS NOT NULL"
    ).fetchone()[0] == 0
    coordinator_conn.close()


def test_worker_pool_actually_overlaps_its_workers(tmp_path):
    # Guards against the pool degenerating into a sequential crawl (e.g. if the
    # shared limiter's lock were ever held across the request): assert more
    # than one worker is inside crawl_player at the same moment.
    path, coordinator_conn = _pool_db(tmp_path, 12)
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def fake_crawl_player(conn_, client_, uid, season):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        db.upsert(conn_, "players", ["uid"], {"uid": uid, "crawl_status": "done"})
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=lambda conn_: 0,
        workers=3,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    assert peak > 1, "workers never overlapped; the pool is effectively sequential"
    coordinator_conn.close()


def test_worker_pool_reseeds_and_requeues_from_one_place_only(tmp_path):
    # A reseed is ~40+ leaderboard requests per hero-refresh cycle. Running it
    # per worker would multiply that by the worker count for no benefit, so it
    # must stay on the coordinator no matter how wide the pool is.
    path, coordinator_conn = _pool_db(tmp_path, 9)
    reseed_conns = []
    requeue_conns = []
    lock = threading.Lock()

    def fake_reseed(conn_, client_):
        with lock:
            reseed_conns.append(conn_)
        return 0

    def fake_requeue(conn_):
        with lock:
            requeue_conns.append(conn_)
        return 0

    def fake_crawl_player(conn_, client_, uid, season):
        db.upsert(conn_, "players", ["uid"], {"uid": uid, "crawl_status": "done"})
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=10**9,  # startup only; no periodic tick
        crawl_player_fn=fake_crawl_player,
        reseed_fn=fake_reseed,
        requeue_fn=fake_requeue,
        workers=3,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    assert len(reseed_conns) == 1  # once, not once per worker
    assert len(requeue_conns) == 1
    # And on the coordinator's own connection, never a worker's.
    assert reseed_conns == [coordinator_conn]
    assert requeue_conns == [coordinator_conn]
    coordinator_conn.close()


def test_worker_pool_lets_every_in_flight_player_finish_on_shutdown(tmp_path):
    # Graceful shutdown has always meant "finish the player you are on". With a
    # pool that must hold for all of them: no player may be abandoned mid-crawl
    # and left stuck in 'claimed'.
    path, coordinator_conn = _pool_db(tmp_path, 30)
    flag = main.ShutdownFlag()
    lock = threading.Lock()
    started = []
    finished = []

    def fake_crawl_player(conn_, client_, uid, season):
        with lock:
            started.append(uid)
            trip = len(started) == 6
        if trip:
            flag.requested = True  # a signal arrives mid-run
        time.sleep(0.02)  # ... while several workers are mid-player
        db.upsert(
            conn_,
            "players",
            ["uid"],
            {"uid": uid, "crawl_status": "done", "last_crawled_at": db.now(), "claimed_at": None},
        )
        conn_.commit()
        with lock:
            finished.append(uid)
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=flag,
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=lambda conn_: 0,
        workers=3,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    # run() returned only after joining the pool: everything started finished.
    assert sorted(started) == sorted(finished)
    assert len(started) < 30  # it really did stop early rather than draining
    left_claimed = coordinator_conn.execute(
        "SELECT COUNT(*) FROM players WHERE crawl_status='claimed'"
    ).fetchone()[0]
    assert left_claimed == 0
    coordinator_conn.close()


def test_worker_pool_survives_a_periodic_reseed_that_writes_before_it_fetches(
    tmp_path, monkeypatch, capsys
):
    """The regression this pool was missing: a REAL periodic reseed tick,
    running concurrently with real workers doing real claims.

    Every other pool test either disables the reseed (interval 10**9) or uses a
    no-op fake, so none of them ever exercised a reseed that writes, then goes
    to the network, then commits. That is exactly crawler.reseed's shape: it
    upserts heroes and seeds players inside a ~42-iteration loop and commits
    only at the very end. pysqlite opens the write transaction at the first
    write, so the coordinator holds SQLite's single write lock across every one
    of those network calls, and every worker's claim/ingest/status write blocks
    for busy_timeout and then raises OperationalError.

    The fake below reproduces that shape at test speed: write, sleep (standing
    in for one hero-leaderboard fetch), commit. Worker connections use a short
    busy_timeout so the contention surfaces in milliseconds rather than the 5s
    the real one would take.
    """
    monkeypatch.setattr(main, "DB_BUSY_SLEEP_SECONDS", 0.02)  # keep the test quick
    path, coordinator_conn = _pool_db(tmp_path, 40)
    reseed_calls = []
    lock = threading.Lock()
    # Tick 1 is the startup reseed, which runs before any worker exists and so
    # cannot contend; ticks 2..4 land while the pool is crawling.
    hold_ticks = 4

    def reseed_that_holds_the_write_lock(conn_, client_):
        with lock:
            reseed_calls.append(1)
            n = len(reseed_calls)
        if n > hold_ticks:
            # Later ticks do nothing. A reseed that held the lock on every pass
            # forever would starve the pool outright — an impossible schedule
            # (the real one runs once a day), not the contention under test.
            return 0
        db.upsert(conn_, "heroes", ["hero_id"], {"hero_id": n, "last_seeded_at": db.now()})
        # Stands in for one hero-leaderboard fetch made inside the open write
        # transaction. Long enough that workers (busy_timeout 50ms) really do
        # hit "database is locked", short enough that a worker's retries
        # outlast it — the point is that contention is survivable, not that a
        # writer may hog the lock for longer than anyone is willing to wait.
        time.sleep(0.08)
        conn_.commit()
        return 0

    def short_timeout_worker_conn():
        conn_ = db.connect(path)
        conn_.execute("PRAGMA busy_timeout=50")
        return conn_

    def fake_crawl_player(conn_, client_, uid, season):
        # ~0.27s of total work across 3 workers, so the pool is still busy when
        # the lock-holding ticks land 0.05s apart — otherwise the queue can
        # drain before any contention happens and the test silently stops
        # exercising the thing it exists for.
        time.sleep(0.02)
        db.upsert(
            conn_,
            "players",
            ["uid"],
            {"uid": uid, "crawl_status": "done", "last_crawled_at": db.now(), "claimed_at": None},
        )
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=0.05,  # real periodic ticks, spaced out
        crawl_player_fn=fake_crawl_player,
        reseed_fn=reseed_that_holds_the_write_lock,
        requeue_fn=lambda conn_: 0,
        workers=3,
        worker_conn_fn=short_timeout_worker_conn,
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    # The contention really happened, rather than the pool quietly draining
    # before the first tick ever took the lock — otherwise this test could stop
    # exercising the bug without anyone noticing.
    assert "database busy" in capsys.readouterr().err
    # No OperationalError escaped run(), every player finished, and none was
    # left stranded in 'claimed' by a worker that died on a locked database.
    statuses = dict(coordinator_conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert statuses == {uid: "done" for uid in range(1, 41)}
    assert coordinator_conn.execute(
        "SELECT COUNT(*) FROM players WHERE crawl_status='claimed' OR claimed_at IS NOT NULL"
    ).fetchone()[0] == 0
    coordinator_conn.close()


def test_worker_pool_surfaces_a_worker_crash_instead_of_swallowing_it(tmp_path):
    # An unexpected exception used to kill the process outright. Buried in a
    # future it would instead leave the crawl silently running short-handed —
    # or hang the coordinator waiting on a drain the dead worker's claim can
    # never deliver.
    path, coordinator_conn = _pool_db(tmp_path, 5)

    def exploding_crawl_player(conn_, client_, uid, season):
        raise RuntimeError("worker bug")

    with pytest.raises(RuntimeError, match="worker bug"):
        main.run(
            coordinator_conn,
            object(),
            season=19,
            shutdown_flag=main.ShutdownFlag(),
            reseed_interval_seconds=10**9,
            crawl_player_fn=exploding_crawl_player,
            reseed_fn=lambda conn_, client_: 0,
            requeue_fn=lambda conn_: 0,
            workers=2,
            worker_conn_fn=lambda: db.connect(path),
            worker_idle_sleep_seconds=0.01,
            coordinator_poll_seconds=0.01,
        )
    coordinator_conn.close()


def test_worker_pool_surfaces_a_worker_crash_even_when_the_coordinator_also_raises(tmp_path):
    # The f.result() loop lives in a finally for this case: if the coordinator
    # body raises, its exception would otherwise sail straight past the loop
    # and a worker's real exception would stay buried in its future, unseen.
    path, coordinator_conn = _pool_db(tmp_path, 5)
    worker_crashed = threading.Event()

    def exploding_crawl_player(conn_, client_, uid, season):
        worker_crashed.set()
        raise RuntimeError("worker bug")

    def exploding_requeue(conn_):
        # Quiet at startup (which runs before the pool exists, so raising there
        # would never reach the futures at all), then fails on the first
        # periodic tick after a worker has actually crashed.
        if not worker_crashed.is_set():
            return 0
        raise RuntimeError("coordinator bug")

    with pytest.raises(RuntimeError) as excinfo:
        main.run(
            coordinator_conn,
            object(),
            season=19,
            shutdown_flag=main.ShutdownFlag(),
            reseed_interval_seconds=-1,  # force the periodic tick, and the raise
            crawl_player_fn=exploding_crawl_player,
            reseed_fn=lambda conn_, client_: 0,
            requeue_fn=exploding_requeue,
            workers=2,
            worker_conn_fn=lambda: db.connect(path),
            worker_idle_sleep_seconds=0.01,
            coordinator_poll_seconds=0.01,
        )

    # Both failures are visible: whichever surfaces, the other is its context.
    chain = []
    exc = excinfo.value
    while exc is not None:
        chain.append(str(exc))
        exc = exc.__context__
    assert any("worker bug" in m for m in chain), chain
    coordinator_conn.close()


def test_worker_pool_stops_even_when_a_claim_cannot_be_released(tmp_path):
    # A 'claimed' row that no live worker holds must not wedge the pool: it can
    # be a leftover from a killed process, or from a release that lost a race
    # with a busy database. Waiting on it is waiting for a drain nothing alive
    # can deliver — observed as 39 of 40 players finished and the pool spinning
    # until killed. Freeing it is the orphaned-claim sweep's job, not shutdown's.
    path, coordinator_conn = _pool_db(tmp_path, 3)
    # A pre-existing orphan from some earlier, long-dead process.
    db.upsert(
        coordinator_conn,
        "players",
        ["uid"],
        {"uid": 99, "crawl_status": "claimed", "claimed_at": db.now()},
    )
    coordinator_conn.commit()

    def fake_crawl_player(conn_, client_, uid, season):
        db.upsert(conn_, "players", ["uid"], {"uid": uid, "crawl_status": "done", "claimed_at": None})
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=lambda conn_: 0,
        workers=2,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    # run() returned rather than spinning forever, the real queue drained, and
    # the orphan is left exactly as found for the sweep to deal with.
    statuses = dict(coordinator_conn.execute("SELECT uid, crawl_status FROM players").fetchall())
    assert statuses == {1: "done", 2: "done", 3: "done", 99: "claimed"}
    coordinator_conn.close()


def test_worker_pool_shares_one_client_across_every_worker(tmp_path):
    # The design's core claim: N workers, ONE rate limiter and circuit breaker,
    # so the site sees a single adaptively-paced traffic pattern rather than N
    # uncoordinated crawlers.
    path, coordinator_conn = _pool_db(tmp_path, 12)
    client = object()
    lock = threading.Lock()
    clients_seen = []
    threads_seen = set()

    def fake_crawl_player(conn_, client_, uid, season):
        with lock:
            clients_seen.append(client_)
            threads_seen.add(threading.current_thread().name)
        # Slow enough that the queue cannot be drained by whichever worker
        # happens to start first, so all three really do take part.
        time.sleep(0.02)
        db.upsert(conn_, "players", ["uid"], {"uid": uid, "crawl_status": "done"})
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        client,
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=lambda conn_: 0,
        workers=3,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    assert len(threads_seen) > 1  # genuinely several workers
    assert all(c is client for c in clients_seen)  # all on the one client
    coordinator_conn.close()


def test_worker_pool_gives_each_worker_its_own_connection(tmp_path):
    # sqlite3 connections are bound to their creating thread, so sharing one
    # would raise "SQLite objects created in a thread can only be used in that
    # same thread" the moment a second worker touched it.
    path, coordinator_conn = _pool_db(tmp_path, 12)
    lock = threading.Lock()
    conn_by_thread = {}

    def fake_crawl_player(conn_, client_, uid, season):
        with lock:
            conn_by_thread.setdefault(threading.current_thread().name, set()).add(id(conn_))
        time.sleep(0.02)  # so all three workers really get some of the queue
        db.upsert(conn_, "players", ["uid"], {"uid": uid, "crawl_status": "done"})
        conn_.commit()
        return "done"

    main.run(
        coordinator_conn,
        object(),
        season=19,
        shutdown_flag=main.ShutdownFlag(),
        reseed_interval_seconds=10**9,
        crawl_player_fn=fake_crawl_player,
        reseed_fn=lambda conn_, client_: 0,
        requeue_fn=lambda conn_: 0,
        workers=3,
        worker_conn_fn=lambda: db.connect(path),
        worker_idle_sleep_seconds=0.01,
        coordinator_poll_seconds=0.01,
    )

    # One connection per worker thread, and never the coordinator's.
    assert len(conn_by_thread) > 1  # several workers really took part
    assert all(len(ids) == 1 for ids in conn_by_thread.values())
    all_ids = {next(iter(ids)) for ids in conn_by_thread.values()}
    assert len(all_ids) == len(conn_by_thread)
    assert id(coordinator_conn) not in all_ids
    coordinator_conn.close()
