import time

import requests


class PlayerNotFoundError(Exception):
    pass


class CircuitOpenError(Exception):
    pass


class FetchError(Exception):
    pass


class RateLimitedError(Exception):
    """Retries exhausted against a sustained 429/403. Deliberately NOT a
    FetchError: per the spec, 429/403 is a site-wide throttling signal, not a
    per-player failure, so the caller must back off and leave the player
    'pending' rather than marking them 'error' (which nothing ever resets)."""


class AdaptiveRateLimiter:
    """Token-bucket-ish delay that only speeds up once the site has proved it
    is comfortable. Two guards the spec (§8) asks for that a naive
    ease-down-on-every-200 loop misses:

    - **Latency awareness.** A 200 that took a long time still means the origin
      is straining, so it is treated as a soft-failure and backs the delay off
      rather than speeding up into a struggling server.
    - **A cooldown before speeding up.** After a failure (or a slow response),
      a single fast 200 must not immediately resume easing the delay down; it
      takes `cooldown_successes` consecutive fast responses to earn that. Once
      earned, every further fast response keeps easing down.
    """

    def __init__(
        self,
        initial_delay=1.0,
        min_delay=0.2,
        max_delay=8.0,
        latency_threshold=2.0,
        cooldown_successes=3,
    ):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.latency_threshold = latency_threshold
        self.cooldown_successes = cooldown_successes
        self.consecutive_fast_successes = 0

    def wait(self):
        time.sleep(self.current_delay)

    def record_success(self, latency):
        if latency > self.latency_threshold:
            # A slow 200 is a strain signal, not a green light. Back off
            # (gentler than a hard failure) and restart the cooldown clock.
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
            self.consecutive_fast_successes = 0
            return

        self.consecutive_fast_successes += 1
        # The counter is deliberately NOT reset after easing down: the cooldown
        # gates when easing may *start*, then a clean run keeps easing.
        if self.consecutive_fast_successes >= self.cooldown_successes:
            self.current_delay = max(self.min_delay, self.current_delay * 0.95)

    def record_failure(self):
        self.current_delay = min(self.max_delay, self.current_delay * 2)
        self.consecutive_fast_successes = 0


class RivalsMetaClient:
    BASE_URL = "https://rivalsmeta.com"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    MAX_RETRIES = 3
    CIRCUIT_FAILURE_THRESHOLD = 10
    CIRCUIT_COOLDOWN_SECONDS = 300

    def __init__(self, session=None, limiter=None):
        if session is None:
            # Only stamp the User-Agent on a session we own. An injected
            # session belongs to the caller (tests pass plain fakes with no
            # .headers at all), so it is never mutated here.
            session = requests.Session()
            session.headers["User-Agent"] = self.USER_AGENT
        self.session = session
        self.limiter = limiter or AdaptiveRateLimiter()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def get_json(self, path, params=None):
        return self._request(path, params).json()

    def get_text(self, path, params=None):
        return self._request(path, params).text

    def _request(self, path, params=None):
        if time.time() < self._circuit_open_until:
            raise CircuitOpenError(f"circuit open until {self._circuit_open_until}")

        url = f"{self.BASE_URL}{path}"
        last_exc = None
        for _ in range(self.MAX_RETRIES):
            self.limiter.wait()
            start = time.time()
            try:
                resp = self.session.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                last_exc = exc
                self.limiter.record_failure()
                continue

            latency = time.time() - start

            if resp.status_code == 404:
                self.limiter.record_success(latency)
                self._consecutive_failures = 0
                raise PlayerNotFoundError(path)

            if resp.status_code in (429, 403):
                self.limiter.record_failure()
                last_exc = RateLimitedError(f"status {resp.status_code} for {path}")
                continue

            if resp.status_code >= 500:
                self.limiter.record_failure()
                last_exc = FetchError(f"status {resp.status_code} for {path}")
                continue

            self.limiter.record_success(latency)
            self._consecutive_failures = 0
            return resp

        self._register_failure()
        # Every exhausted-retry path must surface as a FetchError. When all
        # attempts died on requests.RequestException, last_exc is that raw
        # exception, which main.py's `except fetcher.FetchError` would not
        # catch — crashing the whole run on a transient connection blip.
        if last_exc is not None and not isinstance(last_exc, (FetchError, RateLimitedError)):
            raise FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}") from last_exc
        raise last_exc or FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}")

    def _register_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open_until = time.time() + self.CIRCUIT_COOLDOWN_SECONDS
