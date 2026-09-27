import pytest

import fetcher


@pytest.fixture(autouse=True)
def _no_real_retry_backoff(monkeypatch):
    # The client sleeps between 5xx retries (2 s, then 8 s). Tests must not.
    monkeypatch.setattr(fetcher.RivalsMetaClient, "RETRY_BACKOFF_SECONDS", (0.0, 0.0), raising=False)
