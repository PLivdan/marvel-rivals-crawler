"""How the client reacts to the site refusing us.

Evidence behind these rules (2026-09-13 and 2026-09-20): when rivalsmeta blocks the
IP, every /api/* route answers HTTP 400 with a plain-text body "error". The old client
treated that 400 as a success, failed only at JSON parsing, retried each call three
times, and after its circuit cooldown kept probing every few minutes for as long as
the process lived. The site never sent a 429 or 403 first, so there was no gentler
signal to act on. The rule now: a block signal is answered with zero retries and a
process-wide halt."""
import pytest

import fetcher
from fetcher import BlockedError, FetchError, RateLimitedError, RivalsMetaClient

from test_fetcher import FakeResponse, FakeSession, NoSleepLimiter


class Resp(FakeResponse):
    def __init__(self, status_code, payload=None, text="", headers=None):
        super().__init__(status_code, payload=payload, text=text)
        self.headers = headers or {}


def client_with(responses):
    session = FakeSession(responses)
    return RivalsMetaClient(session=session, limiter=NoSleepLimiter()), session


@pytest.mark.parametrize("status", [400, 401, 403, 451])
def test_a_block_status_halts_with_no_retry(status):
    client, session = client_with([Resp(status, text="error")])
    with pytest.raises(BlockedError) as excinfo:
        client.get_json("/api/player/1")
    assert session.calls == 1
    assert client.blocked
    assert f"status {status}" in str(excinfo.value) and "error" in str(excinfo.value)


def test_once_blocked_no_further_request_is_ever_sent():
    client, session = client_with([Resp(400, text="error")])
    with pytest.raises(BlockedError):
        client.get_json("/api/player/1")
    with pytest.raises(BlockedError):
        client.get_json("/api/player/2")
    assert session.calls == 1


def test_a_block_is_not_a_fetch_error_so_no_player_is_marked_error_for_it():
    assert not issubclass(BlockedError, FetchError)
    assert not issubclass(BlockedError, RateLimitedError)


def test_429_honours_retry_after_and_is_not_retried():
    client, session = client_with([Resp(429, headers={"Retry-After": "120"})])
    with pytest.raises(RateLimitedError):
        client.get_json("/api/player/1")
    assert session.calls == 1
    assert client.circuit_seconds_remaining() == pytest.approx(120, abs=2)


def test_429_without_retry_after_defaults_to_five_minutes():
    client, session = client_with([Resp(429)])
    with pytest.raises(RateLimitedError):
        client.get_json("/api/player/1")
    assert client.circuit_seconds_remaining() == pytest.approx(300, abs=2)


def test_429_cuts_the_shared_rate_when_the_limiter_supports_it():
    class Recording(NoSleepLimiter):
        cuts = []
        def penalize(self, factor):
            self.cuts.append(factor)
    session = FakeSession([Resp(429, headers={"Retry-After": "0"})])
    client = RivalsMetaClient(session=session, limiter=Recording())
    with pytest.raises(RateLimitedError):
        client.get_json("/api/player/1")
    assert Recording.cuts == [0.5]


def test_repeated_429s_within_the_window_escalate_to_a_block():
    client, session = client_with([Resp(429, headers={"Retry-After": "0"}) for _ in range(3)])
    for _ in range(2):
        with pytest.raises(RateLimitedError):
            client.get_json("/api/player/1")
    with pytest.raises(BlockedError):
        client.get_json("/api/player/1")


def test_a_non_json_2xx_is_a_single_failure_not_a_retry_loop():
    client, session = client_with([Resp(200, payload=ValueError("bad"), text="<html>")])
    with pytest.raises(FetchError):
        client.get_json("/api/player/1")
    assert session.calls == 1
    assert not client.blocked


def test_consecutive_non_json_2xx_responses_are_treated_as_a_block():
    bad = [Resp(200, payload=ValueError("bad"), text="<html>challenge") for _ in range(3)]
    client, session = client_with(bad)
    for _ in range(2):
        with pytest.raises(FetchError):
            client.get_json("/api/player/1")
    with pytest.raises(BlockedError):
        client.get_json("/api/player/1")


def test_a_good_parse_resets_the_non_json_streak():
    bad = Resp(200, payload=ValueError("bad"), text="")
    good = Resp(200, payload={"ok": True})
    client, session = client_with([bad, bad, good, bad, bad, good])
    for r in ("f", "f", "ok", "f", "f", "ok"):
        if r == "f":
            with pytest.raises(FetchError):
                client.get_json("/api/x")
        else:
            assert client.get_json("/api/x") == {"ok": True}
    assert not client.blocked


def test_the_circuit_cooldown_doubles_on_consecutive_trips(monkeypatch):
    monkeypatch.setattr(RivalsMetaClient, "CIRCUIT_FAILURE_THRESHOLD", 2)
    client, session = client_with([Resp(500) for _ in range(20)])
    for _ in range(2):
        with pytest.raises(FetchError):
            client.get_json("/api/x")
    first = client.circuit_seconds_remaining()
    client._circuit_open_until = 0.0                          # cooldown elapses
    for _ in range(2):
        with pytest.raises(FetchError):
            client.get_json("/api/x")
    assert client.circuit_seconds_remaining() == pytest.approx(2 * first, abs=2)


def test_request_counts_are_tallied_by_status_for_the_log():
    client, session = client_with([Resp(200, payload={}), Resp(404), Resp(200, payload={})])
    client.get_json("/api/a")
    with pytest.raises(fetcher.PlayerNotFoundError):
        client.get_json("/api/b")
    client.get_json("/api/c")
    assert client.take_status_counts() == {200: 2, 404: 1}
    assert client.take_status_counts() == {}                 # taking resets the window


def test_a_request_waiting_for_its_slot_is_not_sent_once_another_worker_saw_a_block():
    session = FakeSession([])
    class BlockDuringWait(NoSleepLimiter):
        def wait(self):
            client.blocked = "BLOCKED: seen by another worker"
    client = RivalsMetaClient(session=session, limiter=BlockDuringWait())
    with pytest.raises(BlockedError):
        client.get_json("/api/player/1")
    assert session.calls == 0


def test_connection_errors_are_counted_by_type():
    import requests
    session = FakeSession([requests.ReadTimeout("slow"), Resp(200, payload={})])
    client = RivalsMetaClient(session=session, limiter=NoSleepLimiter())
    client.get_json("/api/a")
    assert client.take_status_counts() == {"ReadTimeout": 1, 200: 1}


def test_latency_percentiles_are_reported_and_reset(monkeypatch):
    client, session = client_with([Resp(200, payload={}) for _ in range(10)])
    ticks = iter([0, 0.1, 1, 1.2, 2, 2.3, 3, 3.4, 4, 4.5, 5, 5.6, 6, 6.7, 7, 7.8, 8, 8.9, 9, 11.0])
    monkeypatch.setattr(fetcher.time, "monotonic", lambda: next(ticks))
    for _ in range(10):
        client.get_json("/api/a")
    p50, p90, worst = client.take_latency_stats()
    assert (round(p50, 2), round(p90, 2), round(worst, 2)) == (0.5, 0.9, 2.0)
    assert client.take_latency_stats() is None
