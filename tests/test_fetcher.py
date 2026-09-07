import pytest
import requests

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
