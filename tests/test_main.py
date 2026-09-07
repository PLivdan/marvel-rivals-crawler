import db
import fetcher
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

    assert calls == [1, 1]
    assert sleeps == [0]
    # The fake crawl_player_fn never writes to the DB itself (that's real
    # crawler.crawl_player's job, covered by Task 6's tests) — this asserts
    # the narrower thing this test actually owns: run() must NOT mark a
    # player "error" after a CircuitOpenError the way it does for a
    # FetchError, since a tripped circuit isn't this player's fault.
    status = conn.execute("SELECT crawl_status FROM players WHERE uid=1").fetchone()[0]
    assert status == "pending"


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
