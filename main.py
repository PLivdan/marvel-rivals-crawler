import argparse
import concurrent.futures
import random
import signal
import sqlite3
import sys
import threading
import time

import db
import crawler
import fetcher
import rivalsmeta

# How long a worker waits before re-attempting a claim that came back empty.
# Empty means either "the queue is drained" or "another worker won the race for
# the row I saw", and both want the same thing: come back shortly.
WORKER_IDLE_SLEEP_SECONDS = 1.5
# How often the coordinator wakes to check its reseed/progress timers and
# whether the queue has drained. It does no network work between ticks.
COORDINATOR_POLL_SECONDS = 1.0
# How long to wait out a locked database. Reaching this means busy_timeout
# (5s) already expired against another writer, so something is holding the
# write lock unusually long and hammering it again immediately will not help.
DB_BUSY_SLEEP_SECONDS = 5.0

# Outcomes of one claim-and-crawl attempt, distinguished because the caller
# treats them differently: an empty queue ends a single-threaded run, and a
# back-off already slept, so it must not also be counted as progress.
_CRAWLED = "crawled"
_BACKOFF = "backoff"
_QUEUE_EMPTY = "queue_empty"
_BLOCKED = "blocked"
_UNEXPECTED = "unexpected"

# What run() returns. A block ends the run: the site is refusing us, and every
# request sent while it does only risks extending it (2026-09-22 review, C2).
OUTCOME_DONE = None
OUTCOME_BLOCKED = "blocked"
OUTCOME_SEASON_CHANGED = "season_changed"
EXIT_BLOCKED = 75                     # EX_TEMPFAIL: do not restart for at least a day and a half

# A player whose crawl raises something unexpected is marked 'error' and the
# crawl goes on; before, one such player (a 404 on match history, a malformed
# page) killed every worker, and the restart crashed on the same player again
# (review I1). A run of them in a row is a bug in our code, not bad data, so it
# still aborts.
MAX_CONSECUTIVE_UNEXPECTED = 5

# A restart within this long of the previous run's last request holds the
# pacer's slow start for WARM_RESTART_HOLD_SECONDS before ramping. Both blocks
# (13 Sep, 20 Sep) began within ~2 minutes of a restart that came straight back
# at full speed (review C3).
WARM_RESTART_WINDOW_SECONDS = 3600
WARM_RESTART_HOLD_SECONDS = 1800
META_LAST_REQUEST = "last_request_at"


def log(message):
    """One stderr line, with the time and the thread. Every earlier timeline of an
    incident had to be rebuilt from database timestamps because nothing logged
    when it happened (review I4)."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{threading.current_thread().name}] {message}",
          file=sys.stderr, flush=True)


def warm_restart_hold_seconds(last_request_at, now):
    if last_request_at is None or now - last_request_at > WARM_RESTART_WINDOW_SECONDS:
        return 0
    return WARM_RESTART_HOLD_SECONDS


def format_request_stats(client, window_seconds=60):
    """Requests since the last call, by status; the pacer's cap; the 24 h budget used."""
    counts = client.take_status_counts()
    total = sum(counts.values())
    parts = [f"requests={total} ({total / window_seconds:.2f}/s)",
             "status=" + ",".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))) if counts else "status=-"]
    limiter = getattr(client, "limiter", None)
    if limiter is not None and hasattr(limiter, "current_rate"):
        parts.append(f"cap={limiter.current_rate():.2f}/s")
        parts.append(f"24h={limiter.requests_in_window():,}/{limiter.daily_budget:,}")
    if hasattr(client, "take_latency_stats"):
        lat = client.take_latency_stats()
        if lat:
            parts.append("latency p50/p90/max={:.2f}/{:.2f}/{:.2f}s".format(*lat))
    remaining = client.circuit_seconds_remaining()
    if remaining > 0:
        parts.append(f"circuit_open={remaining:.0f}s")
    return " ".join(parts)


class ShutdownFlag:
    def __init__(self):
        self.requested = False

    def request(self, signum, frame):
        self.requested = True


def format_progress_line(conn, window_seconds=600):
    status_counts = dict(
        conn.execute(
            "SELECT crawl_status, COUNT(*) FROM players GROUP BY crawl_status"
        ).fetchall()
    )
    n_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    hero_counts = [row[0] for row in conn.execute("SELECT n FROM hero_coverage").fetchall()]
    if hero_counts:
        hero_counts.sort()
        hero_min = hero_counts[0]
        hero_median = hero_counts[len(hero_counts) // 2]
        hero_max = hero_counts[-1]
    else:
        hero_min = hero_median = hero_max = 0

    pending = status_counts.get("pending", 0)
    cutoff = db.now() - window_seconds
    recently_finished = conn.execute(
        "SELECT COUNT(*) FROM players WHERE last_crawled_at IS NOT NULL AND last_crawled_at >= ?",
        (cutoff,),
    ).fetchone()[0]
    rate_per_min = recently_finished / (window_seconds / 60)
    eta = _format_eta(pending, rate_per_min)

    status_part = " ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
    return (
        f"matches={n_matches} {status_part} "
        f"hero_coverage(min/median/max)={hero_min}/{hero_median}/{hero_max} "
        f"rate={rate_per_min:.1f}/min eta_to_drain_queue={eta}"
    )


def _format_eta(pending, rate_per_min):
    if pending == 0:
        return "queue empty"
    if rate_per_min <= 0:
        return "unknown (no recent throughput)"
    minutes = pending / rate_per_min
    if minutes < 60:
        return f"~{minutes:.0f}m"
    hours = minutes / 60
    if hours < 48:
        return f"~{hours:.1f}h"
    return f"~{hours / 24:.1f}d"


def _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn):
    """Run a reseed, absorbing fetch failures instead of killing the process.

    A reseed makes ~40+ requests (one per hero leaderboard plus the global
    board), so it is the single most likely place to meet a transient
    failure. Unlike a player crawl there is no per-player status to mark on
    failure — a reseed is simply retried on the next cycle.

    Each handler commits first. A reseed that raised part-way through may have
    left a write transaction open, and this connection would then hold
    SQLite's single write lock for the whole back-off — blocking every worker's
    claim and ingest until they time out. Committing keeps whatever the cycle
    already seeded (all of it idempotent) and, more importantly, releases the
    lock before we go to sleep on it."""
    try:
        reseed_fn(conn, client)
    except fetcher.BlockedError as exc:
        conn.commit()
        log(f"{exc} (during reseed); stopping")
        return OUTCOME_BLOCKED
    except (fetcher.CircuitOpenError, fetcher.RateLimitedError) as exc:
        conn.commit()
        log(f"backing off during reseed ({exc}); pausing {circuit_cooldown_seconds}s")
        sleep_fn(circuit_cooldown_seconds)
    except fetcher.FetchError as exc:
        conn.commit()
        log(f"reseed failed (retrying next cycle): {exc}")
    return None


def _rollback_quietly(conn):
    """Clear any transaction left open by a failed statement.

    This is mandatory after a "database is locked" error, not tidiness. When an
    UPDATE fails on a busy database, pysqlite has already issued its implicit
    BEGIN, so the connection is left mid-transaction. The next claim attempt's
    SELECT then runs inside that stale transaction and pins a read snapshot;
    the UPDATE that follows has to upgrade read->write, and if anything else
    committed in the meantime SQLite answers SQLITE_BUSY_SNAPSHOT — which
    busy_timeout does NOT wait out, because only a rollback can resolve it.
    Every subsequent retry fails the same way, so the worker livelocks
    permanently on a database that is no longer even contended. Observed: three
    workers making zero progress for 24s after the lock was released.
    """
    try:
        conn.rollback()
    except Exception:
        pass


def _release_quietly(conn, uid, attempts=3):
    """Best-effort claim release on a path that is already failing. Never
    raises: it must not mask the exception it is cleaning up after.

    Retried, because the usual reason a release fails is the same busy database
    that caused the failure we are cleaning up after — and a swallowed release
    leaves the player claimed until the orphan sweep runs an hour later."""
    for _ in range(attempts):
        try:
            crawler.release_player(conn, uid)
            return True
        except sqlite3.OperationalError:
            _rollback_quietly(conn)
        except Exception:
            _rollback_quietly(conn)
            return False
    log(f"could not release the claim on player {uid}; leaving it for the orphaned-claim sweep")
    return False


def _set_status_quietly(conn, uid, status, attempts=3):
    """set_status for a failure path: retried against a busy database, never raising,
    so a locked DB cannot turn one player's failure into a dead worker (review M1)."""
    for _ in range(attempts):
        try:
            crawler.set_status(conn, uid, status)
            return True
        except sqlite3.OperationalError:
            _rollback_quietly(conn)
        except Exception:
            _rollback_quietly(conn)
            return False
    log(f"could not mark player {uid} '{status}'; leaving it for the orphaned-claim sweep")
    return False


def _claim_and_crawl_one(conn, client, season, crawl_player_fn, circuit_cooldown_seconds, sleep_fn):
    """Claim one player and crawl them, with the per-player failure handling
    the crawl loop has always had. Shared verbatim by the single-threaded loop
    and by every pool worker, so the two can never drift apart."""
    try:
        uid = crawler.claim_next_player(conn)
    except sqlite3.OperationalError as exc:
        # Almost always a busy_timeout expiring against another writer. It is
        # a reason to wait, not to take the whole run down — which is what
        # letting it escape a worker thread would do. The rollback is what
        # makes the retry able to succeed at all; see _rollback_quietly.
        _rollback_quietly(conn)
        log(f"database busy while claiming ({exc}); pausing {DB_BUSY_SLEEP_SECONDS}s")
        sleep_fn(DB_BUSY_SLEEP_SECONDS)
        return _BACKOFF
    if uid is None:
        return _QUEUE_EMPTY

    try:
        crawl_player_fn(conn, client, uid, season)
    except fetcher.BlockedError as exc:
        # The site is refusing us. Hand the player back untouched and tell the
        # caller to stop every worker; no player is marked for it.
        _release_quietly(conn, uid)
        log(f"{exc}; released player {uid}, stopping all workers")
        return _BLOCKED
    except fetcher.CircuitOpenError as exc:
        log(f"{exc}; pausing {circuit_cooldown_seconds}s")
        # Hand the player back before sleeping. Pre-concurrency this player was
        # simply left 'pending' and retried; releasing the claim reproduces
        # that, and also stops a shutdown during the pause from stranding them
        # until the orphaned-claim sweep notices.
        _release_quietly(conn, uid)
        sleep_fn(circuit_cooldown_seconds)
        return _BACKOFF
    except fetcher.RateLimitedError as exc:
        # A sustained 429/403 is site-wide throttling, not this player's
        # fault: back off and leave them 'pending' to be retried. Marking
        # them 'error' would drop them from the frontier forever.
        log(f"rate limited ({exc}); pausing {circuit_cooldown_seconds}s")
        _release_quietly(conn, uid)
        sleep_fn(circuit_cooldown_seconds)
        return _BACKOFF
    except fetcher.FetchError as exc:
        log(f"transient failure crawling player {uid}: {exc}")
        _set_status_quietly(conn, uid, "error")
    except sqlite3.OperationalError as exc:
        log(f"database busy crawling player {uid} ({exc}); pausing {DB_BUSY_SLEEP_SECONDS}s")
        # Roll back BEFORE trying to release, so the release's own UPDATE is
        # not itself issued inside the stale transaction that just failed.
        _rollback_quietly(conn)
        _release_quietly(conn, uid)
        sleep_fn(DB_BUSY_SLEEP_SECONDS)
        return _BACKOFF
    except Exception as exc:
        # Anything else raised while crawling this one player: a payload that
        # changed shape, a bug that only this player's data triggers. It is
        # marked 'error' (retried as a fresh first crawl an hour later) and the
        # crawl goes on; the caller aborts only on a run of these in a row.
        _rollback_quietly(conn)
        log(f"unexpected {type(exc).__name__} crawling player {uid}: {exc}; marking it 'error'")
        _set_status_quietly(conn, uid, "error")
        return (_UNEXPECTED, exc)
    except BaseException:
        # KeyboardInterrupt, SystemExit: release the claim and re-raise unchanged.
        _release_quietly(conn, uid)
        raise

    return _CRAWLED


def run(
    conn,
    client,
    season,
    shutdown_flag,
    reseed_interval_seconds=86400,
    crawl_player_fn=None,
    reseed_fn=None,
    requeue_fn=None,
    circuit_cooldown_seconds=60,
    sleep_fn=time.sleep,
    workers=1,
    worker_conn_fn=None,
    worker_idle_sleep_seconds=WORKER_IDLE_SLEEP_SECONDS,
    coordinator_poll_seconds=COORDINATOR_POLL_SECONDS,
    worker_start_stagger_seconds=0.0,
    season_fn=None,
):
    """Crawl until the frontier drains, shutdown is requested, or the site blocks us.
    Returns OUTCOME_DONE, OUTCOME_BLOCKED, or OUTCOME_SEASON_CHANGED."""
    crawl_player_fn = crawl_player_fn or crawler.crawl_player
    reseed_fn = reseed_fn or crawler.reseed
    requeue_fn = requeue_fn or crawler.requeue_stale_players

    # Requeueing rides along with the reseed cadence, including the startup
    # call: a process that has been down for a while should pick up everything
    # that went stale meanwhile instead of waiting out a full interval. It is
    # pure local DB work, so it runs even when the reseed above had to back
    # off — there is nothing network-shaped to fail.
    if _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn) == OUTCOME_BLOCKED:
        return OUTCOME_BLOCKED
    requeue_fn(conn)

    periodic = lambda: _periodic_maintenance(  # noqa: E731
        conn, client, season, reseed_fn, requeue_fn, circuit_cooldown_seconds, sleep_fn, season_fn)
    if workers <= 1:
        return _run_single_threaded(
            conn, client, season, shutdown_flag, reseed_interval_seconds,
            crawl_player_fn, periodic, circuit_cooldown_seconds, sleep_fn,
        )
    return _run_worker_pool(
        conn, client, season, shutdown_flag, reseed_interval_seconds,
        crawl_player_fn, periodic, circuit_cooldown_seconds, sleep_fn,
        workers, worker_conn_fn, worker_idle_sleep_seconds, coordinator_poll_seconds,
        worker_start_stagger_seconds,
    )


def _periodic_maintenance(conn, client, season, reseed_fn, requeue_fn, circuit_cooldown_seconds, sleep_fn, season_fn):
    """The reseed/requeue tick, plus a check that the season has not rolled over:
    the season is resolved once at startup, so a run spanning the rollover would
    otherwise keep requesting and labelling the old one (review M2)."""
    if _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn) == OUTCOME_BLOCKED:
        return OUTCOME_BLOCKED
    requeue_fn(conn)
    if season_fn is None:
        return OUTCOME_DONE
    try:
        current = season_fn(client)
    except fetcher.BlockedError as exc:
        log(f"{exc} (during the season check); stopping")
        return OUTCOME_BLOCKED
    except Exception as exc:
        log(f"could not re-check the season ({type(exc).__name__}: {exc}); continuing")
        return OUTCOME_DONE
    if current != season:
        log(f"the season changed from {season} to {current}; stopping so a restart picks it up")
        return OUTCOME_SEASON_CHANGED
    return OUTCOME_DONE


def _report(conn, client):
    """The per-minute log lines, and the last-request time a restart reads."""
    log(format_progress_line(conn))
    if hasattr(client, "take_status_counts"):
        log(format_request_stats(client))
    last = getattr(client, "last_request_at", None)
    if last:
        try:
            db.set_meta(conn, META_LAST_REQUEST, f"{last:.0f}")
        except sqlite3.OperationalError:
            _rollback_quietly(conn)


def _run_single_threaded(
    conn, client, season, shutdown_flag, reseed_interval_seconds,
    crawl_player_fn, periodic, circuit_cooldown_seconds, sleep_fn,
):
    """The pre-concurrency loop, unchanged in shape: one connection, one
    request stream, reseed/requeue on its own timer between players, and an
    exit as soon as the frontier is empty."""
    last_reseed = time.time()
    last_progress_log = time.time()
    consecutive_unexpected = 0

    while not shutdown_flag.requested:
        if time.time() - last_reseed > reseed_interval_seconds:
            result = periodic()
            if result:
                return result
            last_reseed = time.time()

        outcome = _claim_and_crawl_one(
            conn, client, season, crawl_player_fn, circuit_cooldown_seconds, sleep_fn
        )
        if outcome is _QUEUE_EMPTY:
            break
        if outcome is _BLOCKED:
            return OUTCOME_BLOCKED
        if isinstance(outcome, tuple):
            consecutive_unexpected += 1
            if consecutive_unexpected >= MAX_CONSECUTIVE_UNEXPECTED:
                log(f"{consecutive_unexpected} players in a row raised unexpected errors; aborting")
                raise outcome[1]
            continue
        if outcome is _BACKOFF:
            continue
        consecutive_unexpected = 0

        if time.time() - last_progress_log > 60:
            _report(conn, client)
            last_progress_log = time.time()
    return OUTCOME_DONE


def _run_worker_pool(
    conn, client, season, shutdown_flag, reseed_interval_seconds,
    crawl_player_fn, periodic, circuit_cooldown_seconds, sleep_fn,
    workers, worker_conn_fn, worker_idle_sleep_seconds, coordinator_poll_seconds,
    worker_start_stagger_seconds=0.0,
):
    """N worker threads sharing ONE client, coordinated by this thread.

    The sharing is the whole point: every worker goes through the same client,
    so the pacer's cap is a cap on the total and a block any worker meets stops
    all of them. With the GlobalPacer the worker count only hides latency; it
    does not raise the request rate (2026-09-22 review, C1).

    Workers start `worker_start_stagger_seconds` apart. Reseed, requeue and
    progress logging stay here on the coordinator rather than in the workers.
    """
    if worker_conn_fn is None:
        raise ValueError(
            "worker_conn_fn is required when workers > 1: sqlite3 connections are "
            "not safe to share across threads, so each worker opens its own"
        )

    stop = threading.Event()
    blocked = threading.Event()
    # How many workers are inside a claim-and-crawl attempt right now. This,
    # not the 'claimed' rows in the database, is what tells the coordinator
    # whether real work is still in flight — see _queue_is_drained.
    active_lock = threading.Lock()
    active = [0]
    unexpected = [0]

    def pool_is_idle():
        with active_lock:
            return active[0] == 0

    def worker(index):
        if index and worker_start_stagger_seconds:
            sleep_fn(index * worker_start_stagger_seconds)
        # Opened inside the thread that will use it: sqlite3 connections are
        # bound to their creating thread. Several connections to one WAL-mode
        # file is the normal, supported way to do this.
        worker_conn = worker_conn_fn()
        try:
            while not (shutdown_flag.requested or stop.is_set()):
                with active_lock:
                    active[0] += 1
                try:
                    outcome = _claim_and_crawl_one(
                        worker_conn, client, season, crawl_player_fn,
                        circuit_cooldown_seconds, sleep_fn,
                    )
                finally:
                    with active_lock:
                        active[0] -= 1
                if outcome is _BLOCKED:
                    blocked.set()
                    stop.set()
                    break
                if isinstance(outcome, tuple):
                    with active_lock:
                        unexpected[0] += 1
                        n = unexpected[0]
                    if n >= MAX_CONSECUTIVE_UNEXPECTED:
                        log(f"{n} players in a row raised unexpected errors; aborting")
                        raise outcome[1]
                    continue
                if outcome is _CRAWLED:
                    with active_lock:
                        unexpected[0] = 0
                if outcome is _QUEUE_EMPTY:
                    # Jittered so an idle pool does not re-poll in lockstep.
                    sleep_fn(worker_idle_sleep_seconds * random.uniform(0.75, 1.25))
        finally:
            worker_conn.close()

    futures = []
    result = OUTCOME_DONE
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="crawl-worker"
        ) as executor:
            futures = [executor.submit(worker, i) for i in range(workers)]
            last_reseed = time.time()
            last_progress_log = time.time()
            try:
                while not shutdown_flag.requested:
                    if blocked.is_set():
                        result = OUTCOME_BLOCKED
                        break
                    if time.time() - last_reseed > reseed_interval_seconds:
                        tick = periodic()
                        if tick:
                            result = tick
                            break
                        last_reseed = time.time()

                    if time.time() - last_progress_log > 60:
                        _report(conn, client)
                        last_progress_log = time.time()

                    if any(f.done() for f in futures):
                        # A worker returned on its own: it was blocked (handled
                        # above on the next pass) or it raised. Stop rather than
                        # crawl on short-handed.
                        if blocked.is_set():
                            result = OUTCOME_BLOCKED
                        break

                    # Both conditions, and idleness first: if no worker is
                    # inside an attempt and the frontier is empty, nothing can
                    # claim anything, so there is genuinely nothing left to do.
                    if pool_is_idle() and _queue_is_drained(conn):
                        break

                    sleep_fn(coordinator_poll_seconds)
            finally:
                # Workers check this at the top of each claim attempt, so
                # shutting down lets each finish the player it has in flight.
                stop.set()
    finally:
        # In the finally so it runs even when the coordinator body itself
        # raised: otherwise a worker's real exception stays buried in its future.
        for f in futures:
            f.result()
    if blocked.is_set():
        result = OUTCOME_BLOCKED
    return result


def _queue_is_drained(conn):
    """True when nothing is waiting: no first crawl and no revisit.

    Deliberately does NOT look at 'claimed'. Whether a player is still being
    worked on is a fact about THIS process's threads, tracked precisely by the
    pool's in-flight counter — whereas a 'claimed' row can also be a leftover
    from a killed process. Waiting on those would wedge the coordinator forever.
    An existence check, not COUNT(*): this runs on every coordinator poll."""
    return conn.execute(
        "SELECT 1 FROM players WHERE crawl_status IN ('pending', 'revisit') LIMIT 1"
    ).fetchone() is None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/rivals.db")
    parser.add_argument("--reseed-interval-hours", type=float, default=24.0)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current progress/ETA and exit — no crawling, no network access.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent crawl worker threads (default 1). All workers share "
            "one GlobalPacer, so --rate caps the TOTAL request rate however many there "
            "are; more workers only hide per-request latency."
        ),
    )
    parser.add_argument("--rate", type=float, default=3.0,
                        help="Cap on total requests per second across all workers (default 3.0).")
    parser.add_argument("--start-rate", type=float, default=0.5,
                        help="Request rate at startup, ramped linearly to --rate (default 0.5).")
    parser.add_argument("--ramp-minutes", type=float, default=15.0,
                        help="Minutes to ramp from --start-rate to --rate (default 15).")
    parser.add_argument("--daily-budget", type=int, default=250_000,
                        help="Most requests in any rolling 24 h (default 250,000).")
    parser.add_argument("--stagger-seconds", type=float, default=10.0,
                        help="Delay between worker start-ups (default 10 s).")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers > 1 and args.db_path == ":memory:":
        # Each worker opens its own connection, and ":memory:" gives every
        # connection a *separate* empty database — the workers would silently
        # crawl into nothing.
        parser.error("--workers > 1 needs a real database file, not ':memory:'")
    if args.rate <= 0 or args.start_rate <= 0:
        parser.error("--rate and --start-rate must be positive")

    if args.status:
        # Read-only, and deliberately no init_schema: --status is meant to be
        # run against a DB a live crawl process is writing, so it must not take
        # a write lock. The tables already exist if a crawl has ever run.
        conn = db.connect_readonly(args.db_path)
        print(format_progress_line(conn))
        return

    conn = db.connect(args.db_path)
    db.init_schema(conn)

    last = db.get_meta(conn, META_LAST_REQUEST)
    hold = warm_restart_hold_seconds(float(last) if last else None, time.time())
    pacer = fetcher.GlobalPacer(
        rate=args.rate, start_rate=args.start_rate, ramp_seconds=args.ramp_minutes * 60,
        hold_seconds=hold, daily_budget=args.daily_budget,
    )
    log(f"starting: {args.workers} workers, cap {args.rate:g} req/s, start {args.start_rate:g} req/s"
        + (f" held {hold / 60:.0f} min (last request {time.time() - float(last):.0f}s ago)" if hold else "")
        + f", ramp {args.ramp_minutes:g} min, 24 h budget {args.daily_budget:,}")
    client = fetcher.RivalsMetaClient(limiter=pacer)

    try:
        season = rivalsmeta.resolve_current_season(client)
    except fetcher.BlockedError as exc:
        log(f"{exc} (resolving the season); not starting")
        sys.exit(EXIT_BLOCKED)

    shutdown_flag = ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown_flag.request)
    signal.signal(signal.SIGTERM, shutdown_flag.request)

    outcome = run(
        conn,
        client,
        season,
        shutdown_flag,
        reseed_interval_seconds=args.reseed_interval_hours * 3600,
        workers=args.workers,
        worker_conn_fn=lambda: db.connect(args.db_path),
        worker_start_stagger_seconds=args.stagger_seconds,
        season_fn=rivalsmeta.resolve_current_season,
    )
    _report(conn, client)
    if outcome == OUTCOME_BLOCKED:
        log(f"STOPPED: the site is blocking this client ({client.blocked}). "
            f"Do not restart for at least 36 hours. Exit code {EXIT_BLOCKED}.")
        sys.exit(EXIT_BLOCKED)
    if outcome == OUTCOME_SEASON_CHANGED:
        log("STOPPED: the season changed; restart the crawler to follow it.")
        return
    log("stopped" + (" on request" if shutdown_flag.requested else ": the queue is empty"))


if __name__ == "__main__":
    main()
