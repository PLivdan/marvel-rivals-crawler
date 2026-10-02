import collections
import ctypes
import random
import statistics
import sys
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
    """A 429: a site-wide throttling signal, not a per-player failure, so the
    caller backs off and leaves the player 'pending' rather than marking them
    'error'. (403 used to land here too; it is now a BlockedError.)"""


class BlockedError(Exception):
    """The site is refusing this client: HTTP 400/401/403/451, repeated non-JSON
    2xx bodies, or repeated 429s. Deliberately neither a FetchError nor a
    RateLimitedError: nothing about it is the current player's fault, and
    nothing is gained by retrying. Once raised, the client refuses to send
    any further request for the life of the process, and main.py stops every
    worker and exits with EXIT_BLOCKED.

    Observed on 2026-09-13 and 2026-09-20: every /api/* route answered 400 with
    the plain-text body "error" for roughly two days. The old client retried
    each call three times and kept probing every few minutes afterwards."""


def _unbiased_interrupt_time():
    """Seconds of awake time on Windows. time.monotonic there is GetTickCount64, which
    keeps counting while the machine sleeps, so the pacer's wake detection (the wall
    clock jumping ahead of its clock) never fired; QueryUnbiasedInterruptTime stops
    during sleep, like time.monotonic on macOS and Linux."""
    t = ctypes.c_ulonglong()
    ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(t))
    return t.value / 1e7


# The pacer's clock: one that stops while the machine sleeps, on every platform.
awake_clock = _unbiased_interrupt_time if sys.platform == "win32" else time.monotonic


class GlobalPacer:
    """One request schedule shared by every worker thread and the reseed.

    Replaces AdaptiveRateLimiter as the client's default. That limiter had each
    worker sleep a shared delay independently, so the aggregate rate was
    N / (delay + latency), and it eased the delay to its 0.2 s floor within a
    few seconds because this site never sends the 429s it backs off on:
    ~20 req/s at 8 workers, measured in the 2026-09-22 review. Here the rate is
    a cap on the total, whatever the worker count.

    - wait() reserves the next free slot under the lock and sleeps outside it,
      so workers never serialize on each other's sleeping.
    - The rate starts at start_rate, optionally holds there (hold_seconds, for
      a restart soon after a previous run), then ramps linearly to `rate`.
    - A rolling 24 h budget defers requests rather than exceeding it.
    - Back-off only ever cuts the rate, by x0.7, at most once per
      CUT_SPACING_SECONDS: when the median of the last LATENCY_WINDOW latencies
      exceeds LATENCY_THRESHOLD, or when at least FAILURE_RATE of the last
      FAILURE_WINDOW requests failed (judged once FAILURE_MIN_SAMPLE are in).
      It recovers 10% per RECOVERY_SECONDS and never exceeds `rate`.

      Judging the failure RATE rather than each failure is the 2026-09-23 fix:
      20 connection errors in 61,000 requests (0.03%, all retried successfully)
      each cut 30% under the first design, and by 08:00 the cap sat at
      0.33 req/s against 3.0 requested.
    - Failures within WAKE_GRACE_SECONDS of the machine waking from sleep are
      not judged at all (2026-09-27 fix). A request in flight when the Mac
      slept times out on the next wake, 1-4 per 45 s maintenance wake, and
      requests sent before Wi-Fi is back fail too: a failure rate over 1% on
      every wake, each cutting 30%, with recovery counting only awake time.
      The 2026-09-26 run went 3.0 -> 0.55 req/s overnight while the site
      answered every request that reached it. A sleep shows up as the wall
      clock jumping ahead of `clock`, which stops while the machine is asleep
      (time.monotonic is mach_absolute_time on macOS, CLOCK_MONOTONIC on Linux,
      and on Windows awake_clock uses QueryUnbiasedInterruptTime instead).
      Blocks and 429s are handled by the client, not here, so a real refusal
      is still caught immediately.
    """

    LATENCY_WINDOW = 20
    LATENCY_THRESHOLD = 1.5
    FAILURE_WINDOW = 200
    FAILURE_MIN_SAMPLE = 100
    FAILURE_RATE = 0.01
    CUT_SPACING_SECONDS = 300
    RECOVERY_SECONDS = 300
    MIN_MULTIPLIER = 0.1
    SLEEP_DETECT_SECONDS = 5.0
    WAKE_GRACE_SECONDS = 60.0

    def __init__(self, rate=3.0, start_rate=0.5, ramp_seconds=900, hold_seconds=0,
                 daily_budget=250_000, jitter=0.2, clock=awake_clock, sleep=time.sleep,
                 wall_clock=time.time):
        self.rate = float(rate)
        self.start_rate = float(min(start_rate, rate))
        self.ramp_seconds = float(ramp_seconds)
        self.hold_seconds = float(hold_seconds)
        self.daily_budget = int(daily_budget)
        self.jitter = float(jitter)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._t0 = clock()
        self._next = self._t0
        self._mult = 1.0
        self._last_cut = self._t0
        self._latencies = collections.deque(maxlen=self.LATENCY_WINDOW)
        self._outcomes = collections.deque(maxlen=self.FAILURE_WINDOW)   # True = success
        self._last_backoff = float("-inf")
        self._window = collections.deque()
        self._wall = wall_clock
        self._wall_offset = wall_clock() - self._t0
        self._woke_at = float("-inf")

    def _just_woke(self, now):
        """Note a system sleep since the last call, and say whether `now` is within
        WAKE_GRACE_SECONDS of the latest wake. Called under the lock on every wait()
        and record_*(), so the wake is dated by the first event after it."""
        offset = self._wall() - now
        if offset - self._wall_offset > self.SLEEP_DETECT_SECONDS:
            self._woke_at = now
        self._wall_offset = offset
        return now - self._woke_at < self.WAKE_GRACE_SECONDS

    def _base_rate(self, now):
        elapsed = now - self._t0 - self.hold_seconds
        if elapsed <= 0:
            return self.start_rate
        if self.ramp_seconds <= 0 or elapsed >= self.ramp_seconds:
            return self.rate
        return self.start_rate + (self.rate - self.start_rate) * elapsed / self.ramp_seconds

    def current_rate(self):
        with self._lock:
            return self._base_rate(self._clock()) * self._mult

    def wait(self):
        with self._lock:
            now = self._clock()
            self._just_woke(now)
            if self._mult < 1.0 and now - self._last_cut >= self.RECOVERY_SECONDS:
                self._mult = min(1.0, self._mult * 1.1)
                self._last_cut = now
            while self._window and self._window[0] <= now - 86400:
                self._window.popleft()
            slot = max(now, self._next)
            if len(self._window) >= self.daily_budget:
                slot = max(slot, self._window[0] + 86400)
            r = max(1e-6, self._base_rate(slot) * self._mult)
            spread = random.uniform(1.0 - self.jitter, 1.0 + self.jitter) if self.jitter else 1.0
            self._next = slot + spread / r
            self._window.append(slot)
        if slot > now:
            self._sleep(slot - now)

    def record_success(self, latency):
        with self._lock:
            self._just_woke(self._clock())
            self._outcomes.append(True)
            self._latencies.append(latency)
            if (len(self._latencies) == self.LATENCY_WINDOW
                    and statistics.median(self._latencies) > self.LATENCY_THRESHOLD):
                self._latencies.clear()
                self._backoff()

    def record_failure(self):
        with self._lock:
            if self._just_woke(self._clock()):
                return                                       # the machine's sleep, not the site
            self._outcomes.append(False)
            n = len(self._outcomes)
            if n >= self.FAILURE_MIN_SAMPLE and self._outcomes.count(False) / n >= self.FAILURE_RATE:
                self._backoff()

    def _backoff(self):
        now = self._clock()
        if now - self._last_backoff >= self.CUT_SPACING_SECONDS:
            self._last_backoff = now
            self._cut(0.7)

    def penalize(self, factor):
        with self._lock:
            self._cut(factor)

    def _cut(self, factor):
        self._mult = max(self.MIN_MULTIPLIER, self._mult * factor)
        self._last_cut = self._clock()

    def requests_in_window(self):
        with self._lock:
            return len(self._window)


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


def _body_snippet(text, limit=120):
    """First `limit` characters of a response body, whitespace collapsed to a
    single line so it fits in one log entry. An HTML challenge page is a few
    KB; the opening tag and <title> are enough to tell what it was."""
    if not text:
        return "<empty>"
    flat = " ".join(text.split())
    if len(flat) > limit:
        return flat[:limit] + "..."
    return flat


class RivalsMetaClient:
    """One client instance is shared by every crawl worker thread, so pacing, the
    circuit breaker and the block state are single, shared signals. `_state_lock`
    guards the counters and deadlines; it is never held across `session.get` or
    the pacer's sleep.

    Response handling:
    - 2xx: parsed; a non-JSON body is one FetchError (no retry), and
      BLOCK_NON_JSON_STREAK of them in a row is a block.
    - 404: PlayerNotFoundError.
    - 400/401/403/451: BlockedError at once, no retry, and the client never
      sends again.
    - 429: RateLimitedError at once, the circuit opens for Retry-After (default
      300 s), the pacer's rate is halved; BLOCK_429_COUNT of them within
      BLOCK_429_WINDOW is a block.
    - 5xx / connection errors: retried with RETRY_BACKOFF_SECONDS; a call that
      exhausts them counts toward the circuit, whose cooldown doubles on each
      consecutive trip up to CIRCUIT_MAX_COOLDOWN_SECONDS."""

    BASE_URL = "https://rivalsmeta.com"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = (2.0, 8.0)
    CIRCUIT_FAILURE_THRESHOLD = 10
    CIRCUIT_COOLDOWN_SECONDS = 300
    CIRCUIT_MAX_COOLDOWN_SECONDS = 4 * 3600
    BLOCK_STATUSES = (400, 401, 403, 451)
    BLOCK_NON_JSON_STREAK = 3
    BLOCK_429_COUNT = 3
    BLOCK_429_WINDOW = 1800
    DEFAULT_RETRY_AFTER = 300

    def __init__(self, session=None, limiter=None):
        if session is None:
            # Only stamp the User-Agent on a session we own. An injected
            # session belongs to the caller (tests pass plain fakes with no
            # .headers at all), so it is never mutated here.
            session = requests.Session()
            session.headers["User-Agent"] = self.USER_AGENT
        self.session = session
        self.limiter = limiter or GlobalPacer()
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0
        self._consecutive_bad_json = 0
        self._circuit_open_until = 0.0
        self._circuit_trips = 0
        self._recent_429 = collections.deque()
        self.blocked = None                                  # the BlockedError message, once blocked
        self._status_counts = collections.Counter()
        self._latency_log = []                               # seconds, since the last take_latency_stats()
        self.last_request_at = None                          # wall-clock time of the last request sent

    # ---- public ------------------------------------------------------------------------------
    def get_json(self, path, params=None):
        resp = self._request(path, params)
        try:
            result = resp.json()
        except ValueError as exc:
            self.limiter.record_failure()
            streak = self._register_bad_json()
            detail = f"{path} (status {resp.status_code}, body={_body_snippet(resp.text)})"
            if streak >= self.BLOCK_NON_JSON_STREAK:
                self._block(f"{streak} consecutive non-JSON responses, last: {detail}")
            raise FetchError(f"non-JSON response: {detail}") from exc
        self._reset_bad_json()
        return result

    def get_text(self, path, params=None):
        return self._request(path, params).text

    def circuit_seconds_remaining(self):
        with self._state_lock:
            return max(0.0, self._circuit_open_until - time.time())

    def take_latency_stats(self):
        """(p50, p90, max) response latency in seconds since the last call, then reset;
        None if no response arrived in between."""
        with self._state_lock:
            lat, self._latency_log = sorted(self._latency_log), []
        if not lat:
            return None
        def pct(p):
            return lat[max(0, -(-int(p * len(lat) * 1000) // 1000) - 1)]
        return pct(0.5), pct(0.9), lat[-1]

    def take_status_counts(self):
        """The per-status request counts since the last call, then reset (for the stats log)."""
        with self._state_lock:
            counts, self._status_counts = dict(self._status_counts), collections.Counter()
        return counts

    # ---- internals ---------------------------------------------------------------------------
    def _request(self, path, params=None):
        if self.blocked:
            raise BlockedError(self.blocked)
        with self._state_lock:
            open_until = self._circuit_open_until
        if time.time() < open_until:
            raise CircuitOpenError(f"circuit open for another {open_until - time.time():.0f}s")

        url = f"{self.BASE_URL}{path}"
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            if attempt:
                backoff = self.RETRY_BACKOFF_SECONDS[min(attempt - 1, len(self.RETRY_BACKOFF_SECONDS) - 1)]
                if backoff:
                    time.sleep(backoff)
            if self.blocked:                                 # another worker saw the block meanwhile
                raise BlockedError(self.blocked)
            self.limiter.wait()
            if self.blocked:                                 # ...or while this one waited for its slot
                raise BlockedError(self.blocked)
            self.last_request_at = time.time()
            start = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                self._count(type(exc).__name__)             # ReadTimeout, ConnectionError, ...
                last_exc = exc
                self.limiter.record_failure()
                continue
            latency = time.monotonic() - start
            with self._state_lock:
                self._latency_log.append(latency)
            status = resp.status_code
            self._count(status)

            if status in self.BLOCK_STATUSES:
                self._block(f"status {status} for {path} (body={_body_snippet(getattr(resp, 'text', ''))})")
            if status == 429:
                self._rate_limited(path, getattr(resp, "headers", {}) or {})
            if status == 404:
                self.limiter.record_success(latency)
                self._reset_failures()
                raise PlayerNotFoundError(path)
            if status >= 500:
                self.limiter.record_failure()
                last_exc = FetchError(f"status {status} for {path}")
                continue
            self.limiter.record_success(latency)
            self._reset_failures()
            return resp

        self._register_failure()
        # Every exhausted-retry path surfaces as a FetchError, never a raw
        # requests exception, which the crawl loop's handlers would not catch.
        if isinstance(last_exc, FetchError):
            raise last_exc
        raise FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}") from last_exc

    def _block(self, reason):
        with self._state_lock:
            if not self.blocked:
                self.blocked = f"BLOCKED: {reason}"
            message = self.blocked
        raise BlockedError(message)

    def _rate_limited(self, path, headers):
        retry_after = _parse_retry_after(headers.get("Retry-After"), self.DEFAULT_RETRY_AFTER)
        now = time.time()
        with self._state_lock:
            self._recent_429.append(now)
            while self._recent_429 and self._recent_429[0] < now - self.BLOCK_429_WINDOW:
                self._recent_429.popleft()
            repeated = len(self._recent_429)
            self._circuit_open_until = max(self._circuit_open_until, now + retry_after)
        penalize = getattr(self.limiter, "penalize", None)
        if penalize:
            penalize(0.5)
        if repeated >= self.BLOCK_429_COUNT:
            self._block(f"{repeated} responses with status 429 within {self.BLOCK_429_WINDOW // 60} min, last for {path}")
        raise RateLimitedError(f"status 429 for {path}; pausing {retry_after:.0f}s")

    def _count(self, key):
        with self._state_lock:
            self._status_counts[key] += 1

    def _reset_failures(self):
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_trips = 0

    def _register_failure(self):
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
                cooldown = min(self.CIRCUIT_MAX_COOLDOWN_SECONDS, self.CIRCUIT_COOLDOWN_SECONDS * 2 ** self._circuit_trips)
                self._circuit_open_until = time.time() + cooldown
                self._circuit_trips += 1
                self._consecutive_failures = 0

    def _reset_bad_json(self):
        with self._state_lock:
            self._consecutive_bad_json = 0

    def _register_bad_json(self):
        with self._state_lock:
            self._consecutive_bad_json += 1
            return self._consecutive_bad_json


def _parse_retry_after(value, default):
    """Seconds from a Retry-After header: either delta-seconds or an HTTP date."""
    if value is None:
        return float(default)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, IndexError):
        return float(default)
