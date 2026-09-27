"""GlobalPacer: one request schedule shared by every crawl worker.

The 2026-09-22 review measured the old per-worker limiter at N / (delay + latency):
8 workers settled at ~20 req/s within seconds of start, because every worker slept the
shared 0.2 s floor independently and nothing capped the total. These tests pin the
properties that replaced it: an aggregate cap no matter how many threads call wait(),
a slow start, a rolling daily budget, and back-off that only ever cuts the rate."""
import threading
import time

import pytest

from fetcher import GlobalPacer


class FakeClock:
    """`t` is the awake-only monotonic clock; the wall clock also runs while the
    machine is suspended, as time.monotonic (mach_absolute_time) and time.time do
    on macOS."""
    def __init__(self):
        self.t = 1000.0
        self.asleep = 0.0

    def now(self):
        return self.t

    def wall(self):
        return 1_700_000_000.0 + self.t + self.asleep

    def sleep(self, s):
        self.t += max(0.0, s)

    def system_sleep(self, s):
        self.asleep += s


def pacer(clock, **kw):
    kw.setdefault("jitter", 0.0)
    return GlobalPacer(clock=clock.now, sleep=clock.sleep, wall_clock=clock.wall, **kw)


def test_the_cap_holds_for_the_aggregate_not_per_caller():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    start = clock.t
    for _ in range(300):
        p.wait()
    # 300 requests at 3 req/s take ~100 s regardless of which worker asked.
    assert clock.t - start == pytest.approx(299 / 3.0, rel=0.01)


def test_startup_ramps_linearly_from_the_start_rate():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=0.5, ramp_seconds=900)
    p.wait(); t1 = clock.t
    p.wait(); t2 = clock.t
    assert t2 - t1 == pytest.approx(2.0, rel=0.02)            # 0.5 req/s at the start
    clock.t += 900                                            # ramp complete
    p.wait(); a = clock.t
    p.wait(); b = clock.t
    assert b - a == pytest.approx(1 / 3.0, rel=0.02)


def test_a_warm_restart_holds_the_start_rate_before_ramping():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=0.5, ramp_seconds=900, hold_seconds=1800)
    clock.t += 1700
    assert p.current_rate() == pytest.approx(0.5)
    clock.t += 100 + 450                                      # halfway up the ramp
    assert p.current_rate() == pytest.approx(0.5 + 2.5 / 2, rel=0.01)


def test_the_rolling_daily_budget_defers_requests_instead_of_exceeding_it():
    clock = FakeClock()
    p = pacer(clock, rate=10.0, start_rate=10.0, ramp_seconds=0, daily_budget=5)
    first = None
    for i in range(5):
        p.wait()
        first = first if first is not None else clock.t
    p.wait()
    assert clock.t >= first + 86400                           # the 6th waits for the 1st to age out


def test_slow_responses_cut_the_rate_and_it_recovers_only_up_to_the_cap():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for _ in range(GlobalPacer.LATENCY_WINDOW):
        p.record_success(2.5)                                 # a struggling origin
    assert p.current_rate() == pytest.approx(3.0 * 0.7)
    for _ in range(20):                                       # 20 clean recovery periods
        clock.t += GlobalPacer.RECOVERY_SECONDS
        p.wait()
    assert p.current_rate() == pytest.approx(3.0)             # never above the cap


def test_fast_responses_never_raise_the_rate_above_the_cap():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for _ in range(500):
        p.record_success(0.01)
    assert p.current_rate() == pytest.approx(3.0)


def test_penalize_cuts_the_rate_multiplicatively():
    clock = FakeClock()
    p = pacer(clock, rate=4.0, start_rate=4.0, ramp_seconds=0)
    p.penalize(0.5)
    assert p.current_rate() == pytest.approx(2.0)


def test_eight_real_threads_share_one_aggregate_rate():
    # Real time: 8 threads x 15 requests at a 100 req/s cap must take at least ~1.2 s.
    # The old per-worker design would have finished in a fraction of that.
    p = GlobalPacer(rate=100.0, start_rate=100.0, ramp_seconds=0, jitter=0.0)
    def work():
        for _ in range(15):
            p.wait()
    threads = [threading.Thread(target=work) for _ in range(8)]
    t0 = time.monotonic()
    for t in threads: t.start()
    for t in threads: t.join()
    assert time.monotonic() - t0 >= (8 * 15 - 1) / 100.0 * 0.95


def test_the_last_request_time_is_reported_for_warm_restart_detection():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    assert p.requests_in_window() == 0
    p.wait(); p.wait()
    assert p.requests_in_window() == 2


# ---- 2026-09-23: back off on the failure RATE, not on single errors -----------------------------
# Overnight, 20 connection errors in 61,000 requests (0.03%, every one retried successfully)
# each cut the rate 30%; recovery was 10% per 10 min, so from 05:30 the cuts compounded
# and the cap sat at 0.33 req/s against the 3.0 requested.

def test_an_isolated_error_among_healthy_requests_does_not_cut_the_rate():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for i in range(1000):
        if i % 500 == 250:
            p.record_failure()                                 # 0.2%, well under the threshold
        else:
            p.record_success(0.2)
    assert p.current_rate() == pytest.approx(3.0)


def test_a_sustained_failure_rate_cuts_the_rate():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for i in range(GlobalPacer.FAILURE_WINDOW):
        (p.record_failure if i % 20 == 0 else lambda: p.record_success(0.2))()   # 5% failing
    assert p.current_rate() == pytest.approx(3.0 * 0.7)


def test_cuts_from_the_failure_rate_are_spaced_out():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for _ in range(GlobalPacer.FAILURE_WINDOW):
        p.record_failure()
    assert p.current_rate() == pytest.approx(3.0 * 0.7)        # one cut, not two hundred
    clock.t += GlobalPacer.CUT_SPACING_SECONDS
    p.record_failure()
    assert p.current_rate() == pytest.approx(3.0 * 0.7 * 0.7)


def test_failures_just_after_the_machine_wakes_do_not_cut_the_rate():
    """2026-09-26 run: a request in flight when the Mac slept timed out on the next
    maintenance wake, 1-4 per 45 s wake, always over the 1% threshold. Each wake
    cut the cap 30% and recovery only counts awake time: 3.0 -> 0.55 req/s overnight
    with the site answering every request that actually reached it."""
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    p.wait()                                                   # last request before the sleep
    clock.system_sleep(1581)                                   # one maintenance-sleep interval
    clock.t += 15                                              # the in-flight request times out
    for _ in range(GlobalPacer.FAILURE_WINDOW):
        p.record_failure()
    assert p.current_rate() == pytest.approx(3.0)


def test_failures_once_the_wake_grace_has_passed_cut_the_rate_as_usual():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    p.wait()
    clock.system_sleep(1581)
    p.wait()                                                   # first request after the wake
    clock.t += GlobalPacer.WAKE_GRACE_SECONDS
    for _ in range(GlobalPacer.FAILURE_WINDOW):
        p.record_failure()
    assert p.current_rate() == pytest.approx(3.0 * 0.7)


def test_failures_ignored_during_a_wake_are_not_judged_later():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for _ in range(GlobalPacer.FAILURE_MIN_SAMPLE):
        p.record_success(0.2)
    p.wait()
    clock.system_sleep(1581)
    for _ in range(4):
        p.record_failure()                                     # the wake's timeouts
    clock.t += GlobalPacer.WAKE_GRACE_SECONDS
    # One real error is 1 in 101, under the 1% threshold; had the wake's four
    # timeouts been kept in the window it would be 5 in 105 and cut.
    p.record_failure()
    assert p.current_rate() == pytest.approx(3.0)


def test_too_few_requests_are_not_judged():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    for _ in range(GlobalPacer.FAILURE_MIN_SAMPLE - 1):
        p.record_failure()
    assert p.current_rate() == pytest.approx(3.0)


def test_recovery_is_ten_percent_per_five_minutes():
    clock = FakeClock()
    p = pacer(clock, rate=3.0, start_rate=3.0, ramp_seconds=0)
    p.penalize(0.5)
    clock.t += 300
    p.wait()
    assert p.current_rate() == pytest.approx(1.5 * 1.1)
