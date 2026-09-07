import pytest
import requests

from fetcher import AdaptiveRateLimiter, RivalsMetaClient, PlayerNotFoundError, CircuitOpenError, FetchError


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
    limiter = AdaptiveRateLimiter(initial_delay=1.0, min_delay=0.2, max_delay=8.0)
    limiter.record_success(latency=0.1)
    assert limiter.current_delay < 1.0
    before = limiter.current_delay
    limiter.record_failure()
    assert limiter.current_delay > before
