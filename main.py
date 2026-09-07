import argparse
import concurrent.futures
import random
import signal
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

# Outcomes of one claim-and-crawl attempt, distinguished because the caller
# treats them differently: an empty queue ends a single-threaded run, and a
# back-off already slept, so it must not also be counted as progress.
_CRAWLED = "crawled"
_BACKOFF = "backoff"
_QUEUE_EMPTY = "queue_empty"


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
    failure — a reseed is simply retried on the next cycle."""
    try:
        reseed_fn(conn, client)
    except (fetcher.CircuitOpenError, fetcher.RateLimitedError) as exc:
        print(f"backing off during reseed ({exc}); pausing {circuit_cooldown_seconds}s", file=sys.stderr)
        sleep_fn(circuit_cooldown_seconds)
    except fetcher.FetchError as exc:
        print(f"reseed failed (retrying next cycle): {exc}", file=sys.stderr)


def _claim_and_crawl_one(conn, client, season, crawl_player_fn, circuit_cooldown_seconds, sleep_fn):
    """Claim one player and crawl them, with the per-player failure handling
    the crawl loop has always had. Shared verbatim by the single-threaded loop
    and by every pool worker, so the two can never drift apart."""
    uid = crawler.claim_next_player(conn)
    if uid is None:
        return _QUEUE_EMPTY

    try:
        crawl_player_fn(conn, client, uid, season)
    except fetcher.CircuitOpenError:
        print(f"circuit open; pausing {circuit_cooldown_seconds}s", file=sys.stderr)
        # Hand the player back before sleeping. Pre-concurrency this player was
        # simply left 'pending' and retried; releasing the claim reproduces
        # that, and also stops a shutdown during the pause from stranding them
        # until the orphaned-claim sweep notices.
        crawler.release_player(conn, uid)
        sleep_fn(circuit_cooldown_seconds)
        return _BACKOFF
    except fetcher.RateLimitedError as exc:
        # A sustained 429/403 is site-wide throttling, not this player's
        # fault: back off and leave them 'pending' to be retried. Marking
        # them 'error' would drop them from the frontier forever.
        print(f"rate limited ({exc}); pausing {circuit_cooldown_seconds}s", file=sys.stderr)
        crawler.release_player(conn, uid)
        sleep_fn(circuit_cooldown_seconds)
        return _BACKOFF
    except fetcher.FetchError as exc:
        print(f"transient failure crawling player {uid}: {exc}", file=sys.stderr)
        db.upsert(
            conn,
            "players",
            ["uid"],
            {"uid": uid, "crawl_status": "error", "last_crawled_at": db.now(), "claimed_at": None},
        )
        conn.commit()

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
):
    crawl_player_fn = crawl_player_fn or crawler.crawl_player
    reseed_fn = reseed_fn or crawler.reseed
    requeue_fn = requeue_fn or crawler.requeue_stale_players

    # Requeueing rides along with the reseed cadence, including the startup
    # call: a process that has been down for a while should pick up everything
    # that went stale meanwhile instead of waiting out a full interval. It is
    # pure local DB work, so it runs even when the reseed above had to back
    # off — there is nothing network-shaped to fail.
    _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn)
    requeue_fn(conn)

    if workers <= 1:
        _run_single_threaded(
            conn, client, season, shutdown_flag, reseed_interval_seconds,
            crawl_player_fn, reseed_fn, requeue_fn, circuit_cooldown_seconds, sleep_fn,
        )
        return

    _run_worker_pool(
        conn, client, season, shutdown_flag, reseed_interval_seconds,
        crawl_player_fn, reseed_fn, requeue_fn, circuit_cooldown_seconds, sleep_fn,
        workers, worker_conn_fn, worker_idle_sleep_seconds, coordinator_poll_seconds,
    )


def _run_single_threaded(
    conn, client, season, shutdown_flag, reseed_interval_seconds,
    crawl_player_fn, reseed_fn, requeue_fn, circuit_cooldown_seconds, sleep_fn,
):
    """The pre-concurrency loop, unchanged in shape: one connection, one
    request stream, reseed/requeue on its own timer between players, and an
    exit as soon as the frontier is empty. `--workers 1` runs exactly this, so
    not opting in to concurrency costs nothing and changes nothing."""
    last_reseed = time.time()
    last_progress_log = time.time()

    while not shutdown_flag.requested:
        if time.time() - last_reseed > reseed_interval_seconds:
            _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn)
            requeue_fn(conn)
            last_reseed = time.time()

        outcome = _claim_and_crawl_one(
            conn, client, season, crawl_player_fn, circuit_cooldown_seconds, sleep_fn
        )
        if outcome is _QUEUE_EMPTY:
            break
        if outcome is _BACKOFF:
            continue

        if time.time() - last_progress_log > 60:
            print(format_progress_line(conn), file=sys.stderr)
            last_progress_log = time.time()


def _run_worker_pool(
    conn, client, season, shutdown_flag, reseed_interval_seconds,
    crawl_player_fn, reseed_fn, requeue_fn, circuit_cooldown_seconds, sleep_fn,
    workers, worker_conn_fn, worker_idle_sleep_seconds, coordinator_poll_seconds,
):
    """N worker threads sharing ONE client, coordinated by this thread.

    The sharing is the whole point: because every worker goes through the same
    RivalsMetaClient, the site still sees a single adaptively-paced,
    circuit-breaker-protected traffic pattern — just a wider one. A 429 any one
    worker meets immediately widens the delay all of them use next.

    Reseed, requeue and progress logging stay here on the coordinator rather
    than in the workers: a reseed is ~40+ leaderboard requests per hero-refresh
    cycle, and running it per worker would multiply that for no benefit.
    """
    if worker_conn_fn is None:
        raise ValueError(
            "worker_conn_fn is required when workers > 1: sqlite3 connections are "
            "not safe to share across threads, so each worker opens its own"
        )

    stop = threading.Event()

    def worker():
        # Opened inside the thread that will use it: sqlite3 connections are
        # bound to their creating thread. Several connections to one WAL-mode
        # file is the normal, supported way to do this.
        worker_conn = worker_conn_fn()
        try:
            while not (shutdown_flag.requested or stop.is_set()):
                outcome = _claim_and_crawl_one(
                    worker_conn, client, season, crawl_player_fn,
                    circuit_cooldown_seconds, sleep_fn,
                )
                if outcome is _QUEUE_EMPTY:
                    # Jittered so an idle pool does not re-poll in lockstep.
                    sleep_fn(worker_idle_sleep_seconds * random.uniform(0.75, 1.25))
        finally:
            worker_conn.close()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="crawl-worker"
    ) as executor:
        futures = [executor.submit(worker) for _ in range(workers)]
        last_reseed = time.time()
        last_progress_log = time.time()
        try:
            while not shutdown_flag.requested:
                if time.time() - last_reseed > reseed_interval_seconds:
                    _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn)
                    requeue_fn(conn)
                    last_reseed = time.time()

                if time.time() - last_progress_log > 60:
                    print(format_progress_line(conn), file=sys.stderr)
                    last_progress_log = time.time()

                if any(f.done() for f in futures):
                    # A worker returned on its own, which it only does by
                    # raising. Stop rather than silently crawling on with a
                    # shrunken pool — and never wait for a drain that a worker
                    # holding a dead claim can no longer deliver.
                    break

                if _queue_is_drained(conn):
                    break

                sleep_fn(coordinator_poll_seconds)
        finally:
            # Workers check this at the top of each claim attempt, so shutting
            # down lets each finish the player it has in flight, exactly as the
            # single-threaded loop's between-players check always did.
            stop.set()

    # The executor's context manager already did shutdown(wait=True), so every
    # worker has finished. Surface a worker's exception instead of swallowing
    # it in its future, matching how an unexpected error escaped the old loop.
    for f in futures:
        f.result()


def _queue_is_drained(conn):
    """True when there is nothing left to crawl: no 'pending' rows and no
    'claimed' ones still in flight. It is the pool's equivalent of the
    single-threaded loop's break on an empty queue; the extra 'claimed' check
    is what stops the coordinator from calling it quits while a worker is still
    mid-player."""
    return conn.execute(
        "SELECT COUNT(*) FROM players WHERE crawl_status IN ('pending','claimed')"
    ).fetchone()[0] == 0


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
            "Number of concurrent crawl worker threads (default: 1, the single "
            "sequential request stream this crawler has always used). Opting in to "
            "more trades politeness margin for throughput; 3 is the suggested "
            "value. All workers share one rate limiter and circuit breaker, so the "
            "site still sees one adaptively-paced traffic pattern, just wider."
        ),
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers > 1 and args.db_path == ":memory:":
        # Each worker opens its own connection, and ":memory:" gives every
        # connection a *separate* empty database — the workers would silently
        # crawl into nothing.
        parser.error("--workers > 1 needs a real database file, not ':memory:'")

    if args.status:
        # Read-only, and deliberately no init_schema: --status is meant to be
        # run against a DB a live crawl process is writing, so it must not take
        # a write lock. The tables already exist if a crawl has ever run.
        conn = db.connect_readonly(args.db_path)
        print(format_progress_line(conn))
        return

    conn = db.connect(args.db_path)
    db.init_schema(conn)

    client = fetcher.RivalsMetaClient()
    season = rivalsmeta.resolve_current_season(client)

    shutdown_flag = ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown_flag.request)
    signal.signal(signal.SIGTERM, shutdown_flag.request)

    run(
        conn,
        client,
        season,
        shutdown_flag,
        reseed_interval_seconds=args.reseed_interval_hours * 3600,
        workers=args.workers,
        worker_conn_fn=lambda: db.connect(args.db_path),
    )


if __name__ == "__main__":
    main()
