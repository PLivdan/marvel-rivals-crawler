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


def run(
    conn,
    client,
    season,
    shutdown_flag,
    reseed_interval_seconds=86400,
    crawl_player_fn=None,
    reseed_fn=None,
    circuit_cooldown_seconds=60,
    sleep_fn=time.sleep,
):
    crawl_player_fn = crawl_player_fn or crawler.crawl_player
    reseed_fn = reseed_fn or crawler.reseed

    _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn)
    last_reseed = time.time()
    last_progress_log = time.time()

    while not shutdown_flag.requested:
        if time.time() - last_reseed > reseed_interval_seconds:
            _reseed_guarded(reseed_fn, conn, client, circuit_cooldown_seconds, sleep_fn)
            last_reseed = time.time()

        uid = crawler.select_next_player(conn)
        if uid is None:
            break

        try:
            crawl_player_fn(conn, client, uid, season)
        except fetcher.CircuitOpenError:
            print(f"circuit open; pausing {circuit_cooldown_seconds}s", file=sys.stderr)
            sleep_fn(circuit_cooldown_seconds)
            continue
        except fetcher.RateLimitedError as exc:
            # A sustained 429/403 is site-wide throttling, not this player's
            # fault: back off and leave them 'pending' to be retried. Marking
            # them 'error' would drop them from the frontier forever.
            print(f"rate limited ({exc}); pausing {circuit_cooldown_seconds}s", file=sys.stderr)
            sleep_fn(circuit_cooldown_seconds)
            continue
        except fetcher.FetchError as exc:
            print(f"transient failure crawling player {uid}: {exc}", file=sys.stderr)
            db.upsert(conn, "players", ["uid"], {"uid": uid, "crawl_status": "error", "last_crawled_at": db.now()})
            conn.commit()

        if time.time() - last_progress_log > 60:
            print(format_progress_line(conn), file=sys.stderr)
            last_progress_log = time.time()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/rivals.db")
    parser.add_argument("--reseed-interval-hours", type=float, default=24.0)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current progress/ETA and exit — no crawling, no network access.",
    )
    args = parser.parse_args(argv)

    conn = db.connect(args.db_path)
    db.init_schema(conn)

    if args.status:
        print(format_progress_line(conn))
        return

    client = fetcher.RivalsMetaClient()
    season = rivalsmeta.resolve_current_season(client)

    shutdown_flag = ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown_flag.request)
    signal.signal(signal.SIGTERM, shutdown_flag.request)

    run(conn, client, season, shutdown_flag, reseed_interval_seconds=args.reseed_interval_hours * 3600)


if __name__ == "__main__":
    main()
