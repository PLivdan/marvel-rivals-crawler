import time

import requests


class PlayerNotFoundError(Exception):
    pass


class CircuitOpenError(Exception):
    pass


class FetchError(Exception):
    pass


class AdaptiveRateLimiter:
    def __init__(self, initial_delay=1.0, min_delay=0.2, max_delay=8.0):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay

    def wait(self):
        time.sleep(self.current_delay)

    def record_success(self, latency):
        self.current_delay = max(self.min_delay, self.current_delay * 0.95)

    def record_failure(self):
        self.current_delay = min(self.max_delay, self.current_delay * 2)


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
        self.session = session or requests.Session()
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

            if resp.status_code in (429, 403) or resp.status_code >= 500:
                self.limiter.record_failure()
                last_exc = FetchError(f"status {resp.status_code} for {path}")
                continue

            self.limiter.record_success(latency)
            self._consecutive_failures = 0
            return resp

        self._register_failure()
        raise last_exc or FetchError(f"failed after {self.MAX_RETRIES} attempts: {path}")

    def _register_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open_until = time.time() + self.CIRCUIT_COOLDOWN_SECONDS
