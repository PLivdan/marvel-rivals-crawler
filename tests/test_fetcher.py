import sys
import threading

import pytest
import requests

import fetcher
from fetcher import (
    AdaptiveRateLimiter,
    RivalsMetaClient,
    PlayerNotFoundError,
    CircuitOpenError,
    FetchError,
    RateLimitedError,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        # responses: list of FakeResponse or Exception instances, consumed in order
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class NoSleepLimiter(AdaptiveRateLimiter):
    def wait(self):
        pass  # skip real sleeping in tests


class RecordingLock:
    """A real lock that also reports whether it is currently held, so a test
    can prove where the lock IS taken and — just as importantly — where it is
    not (never across a sleep or a network call)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquisitions = 0
        self.held = False

    def __enter__(self):
        self._lock.acquire()
        self.acquisitions += 1
        self.held = True
        return self

    def __exit__(self, *exc_info):
        self.held = False
        self._lock.release()
        return False


def test_get_json_success_returns_payload():
    session = FakeSession([FakeResponse(200, payload={"ok": True})])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    assert client.get_json("/api/player/1") == {"ok": True}


def test_get_json_404_raises_player_not_found_without_retry():
    session = FakeSession([FakeResponse(404)])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(PlayerNotFoundError):
        client.get_json("/api/player/999")
    assert session.calls == 1


def test_get_json_retries_5xx_then_succeeds():
    session = FakeSession([FakeResponse(500), FakeResponse(200, payload={"ok": True})])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    assert client.get_json("/api/player/1") == {"ok": True}
    assert session.calls == 2


def test_get_json_exhausts_retries_and_raises_fetch_error():
    session = FakeSession([FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(FetchError):
        client.get_json("/api/player/1")
    assert session.calls == 3


def test_sustained_429_raises_rate_limited_error_not_fetch_error():
    # Per the spec, 429/403 is a site-wide throttling signal, not a
    # per-player failure — so it must NOT reach main.py as a FetchError,
    # which would permanently mark that player 'error'.
    session = FakeSession([FakeResponse(429)] * 3)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(RateLimitedError):
        client.get_json("/api/player/1")
    assert session.calls == RivalsMetaClient.MAX_RETRIES


def test_sustained_403_raises_rate_limited_error():
    session = FakeSession([FakeResponse(403)] * 3)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(RateLimitedError):
        client.get_json("/api/player/1")


def test_rate_limited_error_is_not_a_fetch_error():
    session = FakeSession([FakeResponse(429)] * 3)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(RateLimitedError) as excinfo:
        client.get_json("/api/player/1")
    assert not isinstance(excinfo.value, FetchError)


def test_sustained_5xx_still_raises_fetch_error_not_rate_limited_error():
    session = FakeSession([FakeResponse(503)] * 3)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(FetchError) as excinfo:
        client.get_json("/api/player/1")
    assert not isinstance(excinfo.value, RateLimitedError)


def test_connection_errors_exhaust_retries_as_fetch_error_not_raw_requests_error():
    # A raw requests.RequestException escaping _request would sail straight
    # past main.py's `except fetcher.FetchError` and kill the whole run.
    session = FakeSession([requests.ConnectionError("no route to host")] * 3)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    with pytest.raises(FetchError) as excinfo:
        client.get_json("/api/player/1")
    assert not isinstance(excinfo.value, requests.RequestException)
    assert isinstance(excinfo.value.__cause__, requests.ConnectionError)
    assert session.calls == RivalsMetaClient.MAX_RETRIES


def test_circuit_opens_after_repeated_failures_and_blocks_further_calls():
    responses = [FakeResponse(500)] * (RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD * RivalsMetaClient.MAX_RETRIES)
    session = FakeSession(responses)
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    for _ in range(RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD):
        with pytest.raises(FetchError):
            client.get_json("/api/player/1")
    with pytest.raises(CircuitOpenError):
        client.get_json("/api/player/1")


def test_default_session_carries_the_declared_user_agent():
    client = RivalsMetaClient(limiter=NoSleepLimiter())
    assert client.session.headers["User-Agent"] == RivalsMetaClient.USER_AGENT


def test_injected_session_is_not_mutated_and_needs_no_headers_attribute():
    # Test fakes are plain objects with only .get(); constructing a client
    # around one must not touch (or require) a .headers mapping.
    session = FakeSession([FakeResponse(200, payload={"ok": True})])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    assert client.session is session
    assert not hasattr(session, "headers")


def test_rate_limiter_speeds_up_on_success_and_slows_down_on_failure():
    # UPDATED for the cooldown semantics: a single fast success no longer eases
    # the delay down — it takes `cooldown_successes` consecutive fast responses
    # to earn that, so the site must prove it's comfortable before we speed up.
    # Hence the loop where this test previously made one record_success call.
    limiter = AdaptiveRateLimiter(initial_delay=1.0, min_delay=0.2, max_delay=8.0)
    for _ in range(limiter.cooldown_successes):
        limiter.record_success(latency=0.1)
    assert limiter.current_delay < 1.0
    before = limiter.current_delay
    limiter.record_failure()
    assert limiter.current_delay > before


def test_rate_limiter_does_not_speed_up_before_the_cooldown_is_earned():
    limiter = AdaptiveRateLimiter(initial_delay=1.0, cooldown_successes=3)
    limiter.record_success(latency=0.1)
    assert limiter.current_delay == 1.0  # one fast response is not enough
    limiter.record_success(latency=0.1)
    assert limiter.current_delay == 1.0  # nor two
    limiter.record_success(latency=0.1)
    assert limiter.current_delay < 1.0  # the third earns it


def test_rate_limiter_keeps_easing_down_on_a_clean_run_once_past_the_cooldown():
    limiter = AdaptiveRateLimiter(initial_delay=1.0, cooldown_successes=3)
    for _ in range(3):
        limiter.record_success(latency=0.1)
    after_first_ease = limiter.current_delay
    limiter.record_success(latency=0.1)
    assert limiter.current_delay < after_first_ease


def test_rate_limiter_backs_off_on_a_slow_success_instead_of_easing_down():
    # A 200 that took ages still means the origin is straining — speeding up
    # into it would be exactly the wrong move.
    limiter = AdaptiveRateLimiter(initial_delay=1.0, latency_threshold=2.0, cooldown_successes=3)
    for _ in range(3):
        limiter.record_success(latency=0.1)
    eased = limiter.current_delay
    assert eased < 1.0

    limiter.record_success(latency=5.0)
    assert limiter.current_delay > eased
    assert limiter.consecutive_fast_successes == 0  # cooldown clock restarted

    # And the restarted cooldown really is enforced: one fast response after a
    # slow one must not immediately resume easing.
    after_slow = limiter.current_delay
    limiter.record_success(latency=0.1)
    assert limiter.current_delay == after_slow


def test_rate_limiter_never_exceeds_max_delay_when_backing_off_on_slow_successes():
    limiter = AdaptiveRateLimiter(initial_delay=1.0, max_delay=2.0, latency_threshold=2.0)
    for _ in range(10):
        limiter.record_success(latency=9.0)
    assert limiter.current_delay == 2.0


def test_rate_limiter_failure_resets_the_cooldown_clock():
    limiter = AdaptiveRateLimiter(initial_delay=1.0, cooldown_successes=3)
    limiter.record_success(latency=0.1)
    limiter.record_success(latency=0.1)
    limiter.record_failure()
    assert limiter.consecutive_fast_successes == 0

    after_failure = limiter.current_delay
    limiter.record_success(latency=0.1)
    assert limiter.current_delay == after_failure  # must re-earn the cooldown


# --- thread safety: one limiter and one circuit breaker shared by N workers ---


def test_rate_limiter_jitters_each_sleep_around_the_current_delay(monkeypatch):
    # Concurrent workers that all sleep exactly `current_delay` would fire in
    # lockstep bursts; jitter desynchronizes them. It is also plain good
    # manners single-threaded — a metronome-exact cadence is a machine
    # signature no human traffic produces.
    limiter = AdaptiveRateLimiter(initial_delay=2.0, jitter=0.12)
    slept = []
    monkeypatch.setattr(fetcher.time, "sleep", slept.append)

    for _ in range(200):
        limiter.wait()

    assert all(2.0 * 0.88 <= s <= 2.0 * 1.12 for s in slept)
    assert len(set(slept)) > 1  # actually varying, not a fixed offset
    # Centred on the configured delay, so throughput is unchanged on average.
    assert 1.9 < sum(slept) / len(slept) < 2.1


def test_rate_limiter_does_not_hold_its_lock_while_sleeping(monkeypatch):
    # The one rule that keeps a shared limiter from serializing the whole pool:
    # the delay is read under the lock, then slept OUTSIDE it. Holding it here
    # would make every worker's wait strictly sequential and collapse the pool
    # back to a single request at a time.
    limiter = AdaptiveRateLimiter(initial_delay=0.01)
    lock = RecordingLock()
    limiter._lock = lock
    held_during_sleep = []
    monkeypatch.setattr(fetcher.time, "sleep", lambda _d: held_during_sleep.append(lock.held))

    limiter.wait()

    assert lock.acquisitions == 1  # the delay was read under the lock
    assert held_during_sleep == [False]  # and released before sleeping


def test_rate_limiter_mutations_happen_under_the_lock():
    limiter = AdaptiveRateLimiter()
    lock = RecordingLock()
    limiter._lock = lock

    limiter.record_success(latency=0.1)
    limiter.record_success(latency=99.0)  # the slow-response branch
    limiter.record_failure()

    assert lock.acquisitions == 3
    assert lock.held is False  # every acquisition was released


def test_rate_limiter_survives_concurrent_access_from_many_real_threads():
    limiter = AdaptiveRateLimiter(initial_delay=1.0, min_delay=0.2, max_delay=8.0)
    out_of_bounds = []
    errors = []
    barrier = threading.Barrier(8)

    def hammer(seed):
        try:
            barrier.wait()  # maximize real overlap
            for i in range(500):
                if (i + seed) % 3 == 0:
                    limiter.record_failure()
                elif (i + seed) % 3 == 1:
                    limiter.record_success(latency=0.01)
                else:
                    limiter.record_success(latency=99.0)
                # Sampled from inside the race, not just at the end: the delay
                # must never be observable outside its configured bounds.
                delay = limiter.current_delay
                if not (limiter.min_delay <= delay <= limiter.max_delay):
                    out_of_bounds.append(delay)
        except Exception as exc:  # pragma: no cover - only on a real defect
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert out_of_bounds == []
    assert limiter.min_delay <= limiter.current_delay <= limiter.max_delay
    assert limiter.consecutive_fast_successes >= 0


def test_circuit_breaker_state_is_mutated_under_the_lock():
    # The deterministic half of the circuit-breaker thread-safety proof, and
    # the one that actually fails if the locking is dropped.
    #
    # It is white-box on purpose. The black-box version — many threads
    # registering failures, then asserting an exact count — cannot fail on
    # CPython 3.12 even with the lock removed (verified): the eval breaker is
    # only polled at jumps and calls, so the straight-line read-modify-write in
    # `+= 1` is never preempted. That makes the exact-count assertion a test of
    # this interpreter's scheduling, not of our locking. Asserting the lock is
    # taken tests the thing we actually control.
    client = RivalsMetaClient(session=FakeSession([]), limiter=NoSleepLimiter())
    lock = RecordingLock()
    client._state_lock = lock

    client._register_failure()
    client._reset_failures()

    assert lock.acquisitions == 2
    assert lock.held is False  # both acquisitions were released


def test_circuit_breaker_survives_concurrent_registration_from_many_threads():
    # The black-box companion: whatever the interpreter's scheduling does, many
    # threads registering failures at once must not corrupt the counter, raise,
    # or leave the circuit deadline in a nonsensical state.
    client = RivalsMetaClient(session=FakeSession([]), limiter=NoSleepLimiter())
    client.CIRCUIT_FAILURE_THRESHOLD = 10**9  # keep the circuit shut; count only
    barrier = threading.Barrier(8)
    errors = []

    def hammer():
        try:
            barrier.wait()
            for _ in range(2000):
                client._register_failure()
        except Exception as exc:  # pragma: no cover - only on a real defect
            errors.append(exc)

    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # maximize preemption between the threads
    try:
        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(original_interval)

    assert errors == []
    assert client._consecutive_failures == 8 * 2000
    assert client._circuit_open_until == 0.0  # threshold never reached


def test_circuit_opens_for_every_worker_at_once_once_the_threshold_is_hit():
    # The shared-breaker payoff: the deadline one worker's failures set is the
    # same deadline every other worker checks, so the whole pool stops together
    # rather than each worker having to discover the outage for itself.
    session = FakeSession(
        [FakeResponse(500)] * (RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD * RivalsMetaClient.MAX_RETRIES)
    )
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    for _ in range(RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD):
        with pytest.raises(FetchError):
            client.get_json("/api/player/1")

    seen = []

    def other_worker():
        try:
            client.get_json("/api/player/2")
        except Exception as exc:
            seen.append(type(exc))

    threads = [threading.Thread(target=other_worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == [CircuitOpenError] * 3
    assert session.calls == RivalsMetaClient.CIRCUIT_FAILURE_THRESHOLD * RivalsMetaClient.MAX_RETRIES


def test_request_does_not_hold_the_circuit_lock_across_the_network_call():
    # The mirror of the limiter rule: workers must serialize only on the brief
    # failure-counter bookkeeping, never on each other's HTTP round trips.
    lock = RecordingLock()
    held_during_get = []

    class ProbingSession:
        def get(self, url, params=None, timeout=None):
            held_during_get.append(lock.held)
            return FakeResponse(200, payload={"ok": True})

    client = RivalsMetaClient(session=ProbingSession(), limiter=NoSleepLimiter())
    client._state_lock = lock

    assert client.get_json("/api/player/1") == {"ok": True}
    assert held_during_get == [False]
    assert lock.acquisitions >= 2  # the circuit check, then the success reset
    assert lock.held is False


def test_a_failure_seen_by_one_thread_immediately_widens_every_thread_delay():
    # The point of sharing one limiter across the pool: a throttling signal any
    # single worker meets is applied to what all of them do next, instead of
    # each worker learning it separately by getting throttled itself.
    limiter = AdaptiveRateLimiter(initial_delay=1.0, max_delay=8.0)
    observed = []

    def other_worker():
        observed.append(limiter.current_delay)

    limiter.record_failure()  # worker A meets a 429
    t = threading.Thread(target=other_worker)  # worker B's very next request
    t.start()
    t.join()

    assert observed == [2.0]
