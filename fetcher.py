import random
import threading
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

    One limiter instance is shared by every crawl worker thread, which is what
    makes a concurrent crawl a single coordinated traffic pattern rather than N
    independent crawlers: a failure any one worker sees immediately widens the
    delay every other worker will use next. All read-modify-writes of the
    shared delay/counter therefore happen under `_lock` — but the lock is NEVER
    held across the sleep in wait(), so workers serialize only on the brief
    bookkeeping, never on each other's waiting.
    """

    def __init__(
        self,
        initial_delay=1.0,
        min_delay=0.2,
        max_delay=8.0,
        latency_threshold=2.0,
        cooldown_successes=3,
        jitter=0.12,
    ):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.latency_threshold = latency_threshold
        self.cooldown_successes = cooldown_successes
        self.consecutive_fast_successes = 0
        # Randomizing each sleep by +/- `jitter` keeps concurrent workers from
        # firing in lockstep bursts (they desynchronize on their own), and is
        # good manners even single-threaded: a metronome-exact request cadence
        # is a machine signature no human traffic produces.
        self.jitter = jitter
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            delay = self.current_delay
        # Sleeping outside the lock is the point: holding it here would make
        # every worker's wait strictly sequential, collapsing the pool back to
        # one request at a time.
        #
        # Clamped at min_delay so jitter cannot undercut the floor: without it,
        # a delay already eased down to min_delay=0.2 would sleep as little as
        # 0.176s, and min_delay would no longer be the hard floor it is
        # documented to be. Jitter may only ever slow a request down.
        time.sleep(max(self.min_delay, delay * random.uniform(1.0 - self.jitter, 1.0 + self.jitter)))

    def record_success(self, latency):
        with self._lock:
            if latency > self.latency_threshold:
                # A slow 200 is a strain signal, not a green light. Back off
                # (gentler than a hard failure) and restart the cooldown clock.
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)
                self.consecutive_fast_successes = 0
                return

            self.consecutive_fast_successes += 1
            # The counter is deliberately NOT reset after easing down: the
            # cooldown gates when easing may *start*, then a clean run keeps
            # easing.
            if self.consecutive_fast_successes >= self.cooldown_successes:
                self.current_delay = max(self.min_delay, self.current_delay * 0.95)

    def record_failure(self):
        with self._lock:
            self.current_delay = min(self.max_delay, self.current_delay * 2)
            self.consecutive_fast_successes = 0


class RivalsMetaClient:
    """One client instance is shared by every crawl worker thread, so the
    adaptive delay and the circuit breaker are genuinely shared signals rather
    than per-worker guesses. `_state_lock` guards the failure counter and the
    circuit deadline; like the limiter's lock it is never held across
    `session.get` or `limiter.wait()`, only across the counter bookkeeping."""

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
        self._state_lock = threading.Lock()

    def get_json(self, path, params=None):
        # A response _request() already treated as a success (2xx) can still
        # fail to parse: an empty body, or an HTML challenge/error page
        # served with a 200 status instead of one of the statuses _request()
        # itself already retries on. Without this, a raw
        # requests.exceptions.JSONDecodeError (a ValueError subclass) would
        # escape as an "unexpected exception" and crash the whole run —
        # observed live, not hypothetical.
        #
        # This does NOT also feed the circuit breaker: _request()'s success
        # path already calls _reset_failures() on every 2xx before handing
        # the response back here, so a _register_failure() call made after
        # the fact would just be undone by the next attempt's own 2xx —
        # accumulating nothing. Fixing that needs _request() itself to know
        # in advance that its caller may still reject a "successful"
        # response, which is a bigger change than this fix calls for. The
        # rate limiter backoff below is unaffected by that reset and still
        # applies correctly.
        last_exc = None
        for _ in range(self.MAX_RETRIES):
            resp = self._request(path, params)
            try:
                return resp.json()
            except ValueError as exc:
                last_exc = exc
                self.limiter.record_failure()
        raise FetchError(f"non-JSON response after {self.MAX_RETRIES} attempts: {path}") from last_exc

    def get_text(self, path, params=None):
        return self._request(path, params).text

    def _request(self, path, params=None):
        with self._state_lock:
            open_until = self._circuit_open_until
        if time.time() < open_until:
            raise CircuitOpenError(f"circuit open until {open_until}")

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
                self._reset_failures()
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
            self._reset_failures()
            return resp

        self._register_failure()
        # Every exhausted-retry path must surface as a FetchError. When all
        # attempts died on requests.RequestException, last_exc is that raw
        # exception, which main.py's `except fetcher.FetchError` would not
        # catch — crashing the whole run on a transient connection blip.
        if last_exc is not None and not isinstance(last_exc, (FetchError, RateLimitedError)):
            raise FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}") from last_exc
        raise last_exc or FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}")

    def _reset_failures(self):
        with self._state_lock:
            self._consecutive_failures = 0

    def _register_failure(self):
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
                self._circuit_open_until = time.time() + self.CIRCUIT_COOLDOWN_SECONDS
