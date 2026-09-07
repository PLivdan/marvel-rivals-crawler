# Concurrent worker pool — implementation report

Branch: `worktree-concurrent-crawl`
Commits: `ce5bd43`, `41d6417`, `9adacda`
Tests: **117 passed, 3 deselected** (was 75 passed, 3 deselected)

---

## 1. What I implemented, file by file

### `db.py`

| Change | Why |
| --- | --- |
| `PRAGMA busy_timeout=5000` in `connect` | A worker whose commit meets another worker's write lock blocks and retries for up to 5s instead of failing immediately with `sqlite3.OperationalError: database is locked`. |
| `PRAGMA busy_timeout=5000` in `connect_readonly` | Same, for `--status` run against a live crawl. Safe on a read-only handle: unlike `journal_mode`, `busy_timeout` is a pure connection-level setting requiring no write. |
| New `players.claimed_at INTEGER` column | Records when a worker took a player. **Deliberately not `last_crawled_at`** — see §5.1, this is the one significant deviation from the brief. |
| `ADDED_COLUMNS` + `_add_missing_columns()`, called from `init_schema` | `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new column does *not* self-apply to a production DB file the way the existing `CREATE INDEX IF NOT EXISTS` statements do. This is a guarded `ALTER TABLE` that preserves `init_schema`'s documented "safe to re-run on every startup" property. Idempotent, and tested against a DB file built with the pre-`claimed_at` schema. |

### `fetcher.py`

- `AdaptiveRateLimiter` gains `self._lock = threading.Lock()`.
  - `wait()` — acquires the lock, reads `current_delay`, **releases**, *then* sleeps. The lock is never held across the sleep.
  - `record_success()` / `record_failure()` — the entire read-modify-write of `current_delay` and `consecutive_fast_successes` runs under the lock.
- **Jitter**: `wait()` now sleeps `delay * random.uniform(0.88, 1.12)` (configurable via a new `jitter=0.12` constructor arg). Desynchronizes concurrent workers, and closes the no-jitter politeness gap the final review flagged. Mean delay is unchanged, so throughput is unaffected.
- `RivalsMetaClient` gains `self._state_lock = threading.Lock()` guarding `_consecutive_failures` and `_circuit_open_until`.
  - `_request`'s circuit check reads `_circuit_open_until` under the lock into a local, then compares/raises outside it.
  - The two `self._consecutive_failures = 0` resets became `self._reset_failures()` (locked).
  - `_register_failure()` runs entirely under the lock.
  - The lock is **never** held across `session.get()` or `limiter.wait()`.
- The 404 / 429·403 / 5xx **decision** logic is untouched, as required — only the state those decisions mutate is now guarded.

Attribute naming note: the brief referred to `_consecutive_fast_successes`, but the existing code and four existing tests use the public `consecutive_fast_successes`. I kept the existing public name rather than break those tests.

### `crawler.py`

- **`claim_next_player(conn)`** — new. Calls the existing `select_next_player`, then
  `UPDATE players SET crawl_status='claimed', claimed_at=? WHERE uid=? AND crawl_status='pending'`,
  commits, and returns `None` if `rowcount == 0`. Relies only on SQLite's ordinary write serialization — no `RETURNING`, no newer pragma.
- **`select_next_player` kept unchanged** as the read-only helper `claim_next_player` builds on internally. Answering the brief's open question: keeping it is the right call — four existing tests exercise it directly (frontier priority, untagged de-prioritization, empty queue, and the 2,000-player performance guard), and those test the *ordering*, which is orthogonal to claiming. Building the claim on top means the ordering has exactly one definition.
- **`release_player(conn, uid)`** — new. Returns a claimed player to `pending` and clears `claimed_at`, without touching `last_crawled_at`. Needed to preserve existing behaviour: see §5.2.
- **`requeue_stale_players`** — third category added, `claimed_stale_seconds=600`. Any `claimed` row with `claimed_at < now - 600` resets to `pending` with `last_crawled_at=NULL` **and** `claimed_at=NULL`, following the exact asymmetry the existing docstring already documents for `error` (a crawl that never finished must retry with fresh-crawl semantics). Docstring extended in the same voice; return value now sums all three categories.
- **`_set_status`** — also writes `claimed_at: None`, so the invariant "`claimed_at` is set iff a worker is holding this player right now" holds, which is what the orphan sweep keys on.
- **`crawl_player` internals: unchanged.** I walked through it and confirm the brief's claim. It reads `latest_known_level` and `last_crawled_at`; it never reads `crawl_status`. Its own profile upsert omits `crawl_status`, so `db.upsert`'s partial-update semantics leave the claim intact. It only ever *writes* a status via `_set_status`. It is genuinely indifferent to whether the row was `pending` or `claimed` on entry.

### `main.py`

Restructured into three functions. `format_progress_line`, `ShutdownFlag`, `_format_eta` and `_reseed_guarded` are untouched, as instructed.

- **`_claim_and_crawl_one(...)`** — claims one player and crawls them, carrying the exact per-player `CircuitOpenError` / `RateLimitedError` / `FetchError` handling `run()` always had. Shared *verbatim* by both the single-threaded loop and every pool worker, so the two paths cannot drift apart. Returns one of `_CRAWLED` / `_BACKOFF` / `_QUEUE_EMPTY` — the tri-state exists to preserve today's control flow exactly, where a back-off `continue`s and skips the progress-log check.
- **`run(...)`** — does the startup reseed + requeue (unchanged), then dispatches to one of the two loops. New kwargs: `workers=1`, `worker_conn_fn=None`, `worker_idle_sleep_seconds`, `coordinator_poll_seconds`.
- **`_run_single_threaded(...)`** — the pre-concurrency loop, unchanged in shape.
- **`_run_worker_pool(...)`** — `ThreadPoolExecutor(max_workers=N)`, N submitted worker loops, coordinator on the calling thread:
  - Each worker opens its **own** connection via `worker_conn_fn()`, *inside* the thread (sqlite3 connections are bound to their creating thread), and closes it in a `finally`.
  - All workers share the **one** `client` passed to `run()`.
  - Workers loop: check `shutdown_flag.requested or stop.is_set()` → `_claim_and_crawl_one` → on `_QUEUE_EMPTY`, sleep `1.5s ± 25%` jitter and retry.
  - Coordinator owns reseed, requeue and progress logging exclusively.
  - Coordinator breaks on shutdown, on a drained queue (`_queue_is_drained`: no `pending` **and** no `claimed` rows — the `claimed` half is what stops it quitting while a worker is mid-player), or if any worker future is `done` (see §5.3).
  - `stop.set()` in a `finally`, then the executor's `__exit__` does `shutdown(wait=True)`, so every in-flight player finishes.
  - Afterwards `f.result()` on each future re-raises a worker's exception rather than swallowing it.
- **CLI**: `--workers N` (default 1). Rejects `< 1`, and rejects `> 1` against `:memory:` (which would give each worker its own empty database and silently crawl into nothing). `main()` passes `worker_conn_fn=lambda: db.connect(args.db_path)`.

---

## 2. The race-condition test

`tests/test_crawler.py::test_two_connections_racing_for_the_same_player_produce_exactly_one_claim`

Two real `db.connect()` connections to the same on-disk WAL file (`tmp_path`, not `:memory:`), one pending player. The interleaving is constructed by monkeypatching `crawler.select_next_player` so that connection A is frozen at precisely the dangerous instant — after it has chosen its candidate, before it has written its claim:

```
A: select_next_player()            -> 1        # A has its candidate
        |
        |  <-- the race window: exactly where the old
        |      select-then-crawl pair let both workers start
        v
B:   claim_next_player(conn_b)
       select_next_player()        -> 1        # B sees the same row
       UPDATE ... WHERE uid=1 AND crawl_status='pending'
       COMMIT                                   # rowcount 1 -> B wins
        |
        v
A:   UPDATE ... WHERE uid=1 AND crawl_status='pending'
     COMMIT                                     # rowcount 0 -> A returns None
```

Assertions: `b_result == [1]`, `a_result is None`, the sorted pair is exactly `[1, None]` (**one** claim — not both, not neither), and the row ends `crawl_status='claimed'` with a non-NULL `claimed_at`.

**Why this proves atomicity.** The claim's correctness rests entirely on the `AND crawl_status='pending'` guard plus SQLite's write serialization: whichever `UPDATE` commits first flips the row out of `pending`, so the loser's predicate matches zero rows. This test forces the loser's `UPDATE` to run strictly *after* the winner's `COMMIT` — the worst case — and shows `rowcount` is 0 and the function reports `None`. Note this is deterministic, not probabilistic: A's `SELECT` runs in autocommit (pysqlite only opens a transaction before DML), so it holds no read lock, and B's claim completes fully inside A's window.

**Verified load-bearing by mutation.** With `AND crawl_status='pending'` removed from the UPDATE:

```
>       assert a_result is None  # A lost, and says so rather than double-claiming
E       assert 1 is None
FAILED tests/test_crawler.py::test_two_connections_racing_for_the_same_player_produce_exactly_one_claim
```

Both connections claim the same player, which is exactly the bug. Guard restored, test passes.

Complementing it, `test_worker_pool_crawls_every_player_exactly_once_and_drains_the_queue` runs 3 real threads over a 20-player on-disk DB and asserts all 20 end `done`, `len(crawled) == len(set(crawled))` (none twice), and none left `pending`/`claimed`.

---

## 3. `--workers 1` backward-compatibility verification

Four independent checks:

**a) Identical call sequence (`test_workers_one_makes_exactly_the_same_call_sequence_as_the_unset_default`).**
The same 5-player scenario is driven twice — once with `workers` left unset, once with `workers=1` — through fully instrumented fakes that log every `reseed`, `requeue` and `crawl` call in order. Asserts the two logs are equal *and* equal to the expected literal:

```
[("reseed",), ("requeue",), ("crawl",1,19), ("crawl",2,19),
 ("crawl",3,19), ("crawl",4,19), ("crawl",5,19)]
```

That pins startup reseed+requeue ordering, one-player-at-a-time claim/crawl in frontier-priority order, and the exit-on-drain.

**b) No pool, no extra connection (`test_workers_one_runs_on_the_calling_thread_...`).**
Asserts `crawl_player_fn` runs on `threading.main_thread()` and receives the caller's own `conn`; the injected `worker_conn_fn` raises `AssertionError` if called. So a single worker cannot quietly become concurrent or open a second connection.

**c) Every pre-existing `run()` test still passes unmodified.** Nine of the eleven were untouched — including the two that use `reseed_interval_seconds=-1` to force the periodic branch and assert exact reseed/requeue counts, which is the tightest existing pin on reseed timing. Only two needed adaptation (§5.2).

**d) The CLI default is 1** (`test_main_defaults_to_a_single_worker_so_concurrency_stays_opt_in`).

**e) Beyond the test suite — real end-to-end data equivalence.** Every pool test above uses a fake `crawl_player`, so I separately ran the *real* `crawl_player` + `ingest_match` under a fake network (fixture payloads, deliberately overlapping match pool so workers race to ingest the same `match_uid`), at 1 worker and at 3 workers:

```
{'workers': 1, 'statuses': {'done': 36}, 'matches': 10, 'match_players': 120,
 'match_player_heroes': 190, 'requests': 82, 'still_claimed': 0,
 'integrity': 'ok', 'fk_violations': []}
{'workers': 3, 'statuses': {'done': 36}, 'matches': 10, 'match_players': 120,
 'match_player_heroes': 190, 'requests': 85, 'still_claimed': 0,
 'integrity': 'ok', 'fk_violations': []}
```

Identical collected data, `PRAGMA integrity_check` ok, no FK violations, nothing stranded. Reproduced 4/4 runs. (The 82 vs 85 request delta is discussed in §5.4.)

**f) Real SIGINT against a real 3-worker process.** Ran `main.main([... "--workers", "3"])` in a subprocess with a slow fake network, sent `SIGINT` mid-crawl through the real signal handlers. The process printed `RETURNED_CLEANLY` (i.e. `run()` joined the pool and returned normally rather than dying on a traceback), and the DB afterwards showed:

```
status counts: {'done': 89, 'pending': 822}
rows still holding a claim: 0
matches collected: 267
integrity_check: ok
foreign_key_check: []
```

Zero players abandoned mid-crawl.

---

## 4. Full test suite

Command (`pytest.ini` excludes `-m live` by default; the live suite was **not** run, as instructed):

```
$ python3 -m pytest -q
117 passed, 3 deselected in 2.07s
```

Baseline before this work was `75 passed, 3 deselected`. Run 8× consecutively with no flakes. New tests:

**`tests/test_db.py`** (3): busy_timeout on `connect` and `connect_readonly`; `claimed_at` migration onto a legacy DB file.

**`tests/test_crawler.py`** (16): `claim_next_player` empty-queue / stamps `claimed_at` / priority order / two sequential claims give two different uids both `claimed` / does not stamp `last_crawled_at` / preserves an existing one; a claimed first-time player is still crawled as a fresh first crawl; `_set_status` clears `claimed_at`; the two-connection race; `release_player` round-trip and its no-op on unclaimed rows; four `requeue_stale_players` orphan-claim cases (resets a stale claim with `last_crawled_at=NULL`, leaves a fresh claim alone, counts alongside the other categories, ignores a NULL `claimed_at`).

**`tests/test_fetcher.py`** (9): jitter bounds/variation/mean; `wait()` does not hold the lock while sleeping; limiter mutations happen under the lock; 8 threads × 500 mixed ops with bounds sampled *from inside* the race; circuit state mutated under the lock; 8 threads × 2000 `_register_failure`; circuit opens for all workers at once; `_request` does not hold `_state_lock` across the network call; a failure in one thread immediately widens another thread's delay.

**`tests/test_main.py`** (14): the `--workers 1` equivalence pair; 4 CLI-wiring/validation tests; `run()` refuses `workers>1` without `worker_conn_fn`; 7 pool tests (drain-exactly-once, real overlap, reseed/requeue from one place only, graceful shutdown finishes in-flight players, worker crash surfaces, one shared client across threads, one connection per worker); plus the new claim-release test.

---

## 5. Concerns, deviations, and findings

### 5.1 DEVIATION (important): the claim stamps `claimed_at`, not `last_crawled_at`

The brief's `claim_next_player` writes `last_crawled_at=?` at claim time. **That silently regresses commit `a01862d` ("don't lose a player's history when a first crawl is interrupted").**

`crawl_player` reads `last_crawled_at IS NULL` as "this player has never *completed* a crawl" and uses it to choose skip-past-known-matches (first crawl) vs break-at-first-known (revisit). Stamping it at claim time makes *every* claimed player look like a revisit. I reproduced this before writing any code — a player with one already-ingested match and one unseen match, claimed per the brief's function, then crawled:

```
status: done
match-detail fetches: []
EXPECTED (pre-claim behaviour): ['5517519_1788641229_1421093_11001_11']
```

The unseen match is never fetched, the player is marked `done`, and the rest of their history is stranded permanently. This hits both freshly-seeded players and `error`-requeued players (whose `last_crawled_at` was deliberately cleared to NULL for exactly this reason).

The brief's stated assumption — "`crawl_player` never reads the row's current `crawl_status`" — is **correct**; the problem is a different column, so I judged this a fix-and-flag rather than a stop-and-report. A dedicated `claimed_at` column resolves it cleanly with `crawl_player` completely untouched, and has two bonus effects: `format_progress_line`'s `recently_finished` throughput window is not inflated by in-flight claims, and the orphan sweep gets an unambiguous key. Cost is the `ALTER TABLE` migration in `db.py`. Guarded by `test_claim_next_player_does_not_stamp_last_crawled_at` and `test_a_claimed_first_time_player_is_still_crawled_as_a_fresh_first_crawl`.

I kept the brief's requeue instruction as written (orphaned claims reset with `last_crawled_at=NULL`), and I agree with it: a crashed *revisit* strands matches for the same reason a crashed first crawl does, so forcing fresh-crawl semantics on recovery is right in both cases.

### 5.2 DEVIATION: two existing tests adapted, and a new `release_player`

With claiming, a `CircuitOpenError`/`RateLimitedError` would leave the player `claimed` — whereas the pre-concurrency loop left them `pending` for an immediate retry, which `test_run_pauses_on_circuit_open_error_then_retries` and `test_run_backs_off_on_rate_limited_error_and_leaves_player_pending` both assert. Rather than accept a 10-minute stall, I added `crawler.release_player`, restoring the old behaviour exactly.

Those two tests then still fail on their *final* assertion only, because their fake `crawl_player_fn` never writes a terminal status, so the row ends `claimed` after the second claim. I preserved their coverage rather than deleting it:
- `assert status == "pending"` → `assert status != "error"`, which is the invariant they actually own (a tripped circuit isn't the player's fault, and nothing ever resets `error`).
- Their existing `assert calls == [1, 1]` now carries *more* weight than before: a still-claimed row is invisible to `claim_next_player`, so the same player being handed out twice is itself proof the release happened.
- Added `test_run_releases_the_claim_when_the_circuit_opens`, which asserts the row is back to `pending` with `claimed_at IS NULL` directly.

### 5.3 Coordinator must detect a dead worker, or it hangs

Not in the brief, found while building. If a worker thread dies from an unexpected exception, its player stays `claimed`, so `_queue_is_drained` can never become true — the coordinator would spin forever (the orphan sweep only runs on the 24h reseed tick). The pre-concurrency behaviour was that an unexpected exception killed the process. I restored parity: the coordinator breaks if `any(f.done() for f in futures)`, and `f.result()` after the join re-raises. Covered by `test_worker_pool_surfaces_a_worker_crash_instead_of_swallowing_it`.

### 5.4 Concurrency costs a small number of duplicate match fetches

Two workers can both check "is this `match_uid` known?" before either commits its ingest, so the same match detail is occasionally fetched twice. In the deliberately-overlapping e2e above that was 85 vs 82 requests (+3.7%); in production, overlap between two random diamond players' histories is far lower. It is *safe* — every write is an idempotent upsert keyed on natural PKs, and the collected data was byte-identical. Eliminating it would need cross-worker match-level locking, which would serialize the workers and defeat the purpose. Flagging as a known, accepted cost, not a defect.

### 5.5 CLI default is 1, not 3 — please confirm

The brief says both "Default worker count: 3" and (under out-of-scope) "`--workers 1` must remain the safe, conservative, fully-tested default behavior if a user doesn't opt in", with the feature framed throughout as **opt-in**. These conflict. I chose `default=1` because the later constraint is more explicit, "opt-in" implies off-by-default, and the risk is asymmetric (accidentally tripling live traffic to a real site vs. requiring one flag). `--help` documents 3 as the suggested value.

**If you meant the shipping default to be 3, it is a one-line change:** `default=1` → `default=3` in `main.main`'s `--workers` argument (`main.py:323`). `test_main_defaults_to_a_single_worker_so_concurrency_stays_opt_in` pins the current choice and would need updating with it.

### 5.6 A hard kill now strands one player for up to 10 minutes

Previously, `SIGKILL` left the in-flight player `pending`, so a restart picked them up immediately. Now they are `claimed`, and the startup `requeue_stale_players` won't free them until `claimed_at` is 10 minutes old. One player, self-healing, and inherent to any claim-based queue. Graceful `SIGINT`/`SIGTERM` is unaffected (verified: zero rows left claimed). I deliberately did **not** shorten the window at startup, because a zero window would let a starting process steal a live worker's player if two crawler processes ever share a DB.

### 5.7 Minor observability change

`--status` and the periodic progress line can now show a `claimed=N` bucket (N ≤ worker count) for players currently in flight. Previously those showed as `pending`. Informative rather than wrong, but worth knowing before someone reads a status line.

### 5.8 `requests.Session` sharing

All workers share one `requests.Session`. `Session` is not formally documented as thread-safe, but the parts that matter here are: urllib3's connection pool is thread-safe, and `http.cookiejar.CookieJar` guards itself with an `RLock`. We only issue GETs with a fixed header dict and never mutate the session after construction. I did **not** put a lock around `session.get` — that would serialize network I/O, which the brief explicitly forbids. Noting it as a known, standard-practice risk.

### 5.9 Transaction-shape check (no `SQLITE_BUSY_SNAPSHOT` risk)

I checked for the classic WAL deadlock where a deferred transaction starts with a read and later upgrades to a write — `busy_timeout` does *not* retry that. It cannot occur here: pysqlite's legacy isolation mode only emits `BEGIN` immediately before DML, so every transaction in this codebase (`claim_next_player`, `release_player`, `requeue_stale_players`, `ingest_match`, `_set_status`) begins with a write and takes the write lock up front. Confirmed empirically — no lock errors across the e2e and SIGINT runs, `integrity_check` ok.

### 5.10 Note on one test's honesty

The obvious black-box test for the circuit breaker lock — N threads incrementing, assert an exact count — **cannot fail on CPython 3.12 even with the lock removed** (I verified this by mutation, including at `sys.setswitchinterval(1e-6)`): the eval breaker is only polled at jumps and calls, so the straight-line read-modify-write in `+= 1` is never preempted. That test would have been a test of the interpreter's scheduling, not of our locking. I kept it as a no-corruption smoke test but renamed it accordingly, and added `test_circuit_breaker_state_is_mutated_under_the_lock`, which asserts the lock is actually taken and **does** fail when it is removed. Same treatment for `test_rate_limiter_does_not_hold_its_lock_while_sleeping` (verified: fails when `wait()` sleeps inside the lock).

---

## 6. Out-of-scope items — confirmed untouched

- No multiprocessing, no asyncio. Threads only.
- `ingest.py` unchanged (zero diff).
- `rivalsmeta.py` unchanged (zero diff).
- `fetcher.py`'s retry/circuit-breaker *decision* logic unchanged — only locking added around the state it mutates.
- Reseed remains single-threaded and sequential, on the coordinator only (`test_worker_pool_reseeds_and_requeues_from_one_place_only`).
- `format_progress_line`, `ShutdownFlag`, `_format_eta`, `_reseed_guarded` unchanged.
- The `-m live` suite was not run.

---
---

# Addendum: review-round fixes

Commit: `ce5a71e`
Tests: **125 passed, 3 deselected** (was 117); 8 consecutive clean full-suite runs.

Every fix below was reproduced empirically before being written, and the two
most important ones were then verified by mutation (break the fix, watch the
test fail; restore it, watch it pass).

## A. CRITICAL — the reseed held the write lock across ~40 network calls

**Reproduced first.** A probe drives the real `crawler.reseed` with a client
that, from a *second* connection, tries to write at the exact moment reseed is
inside a network call. Against the old trailing-commit-only code:

```
assert ['writable', 'LOCKED', 'LOCKED', 'LOCKED'] == ['writable'] * 4
```

Probe 1 is writable because no write has been issued yet; from the first
`_seed_player` onward the coordinator holds SQLite's single write lock for the
rest of the reseed. After the fix all four probes report `writable`.

**Fixed:**
- `crawler.reseed` commits after each hero, so no network call is ever made
  while this connection holds the write lock. The global-leaderboard fetch now
  also happens with no transaction open, and its seeding loop (which makes no
  network calls) commits once after it.
- `_reseed_guarded` commits at the top of both `except` blocks, so an
  interrupted reseed cannot hold the lock across the back-off sleep.
- Workers catch `sqlite3.OperationalError` and treat it as a back-off signal
  instead of letting it kill the process.

**Tests:** `test_reseed_never_holds_the_write_lock_across_a_network_call`
(root cause, in `tests/test_crawler.py`) and
`test_worker_pool_survives_a_periodic_reseed_that_writes_before_it_fetches`
(defense in depth, a real periodic tick against 3 real workers). Both verified
load-bearing by mutation.

## A2. Two further bugs found while verifying the above

Neither was in the review. Both were found by instrumenting a real pool rather
than by reading the code, and both would have shipped.

**A2.1 — the `OperationalError` guard livelocked the pool permanently.**
Catching the error without rolling back is worse than not catching it. A failed
`UPDATE` leaves pysqlite's implicit `BEGIN` open; the next attempt's `SELECT`
then pins a read snapshot *inside* that stale transaction, and the following
read→write upgrade fails with `SQLITE_BUSY_SNAPSHOT` — which `busy_timeout`
does **not** wait out, because only a rollback can resolve it. Every retry then
fails identically, forever.

Measured, with the lock released after 0.94s:

```
!!! WATCHDOG at 25.0s: crawled=0/40 reseeds=1899
```

Zero progress in 24 seconds on a database nothing was contending for. With
`_rollback_quietly` in the handler: `40/40 crawled, run() returned at 1.03s`.

This is precisely the hazard §5.9 of the original report called impossible —
correctly, at the time: it only became reachable because my own new handler
introduced the "transaction left open" precondition that §5.9 depended on not
existing.

**A2.2 — the coordinator waited for a drain nothing alive could deliver.**
A release that lost its own race with the busy database left one player
`claimed` forever; the drain check counted `claimed` rows, so the pool spun
until killed:

```
!!! HUNG at 20s: crawled=39/40
!!! last 8 polls (in_transaction, coordinator_sees, fresh_conn_sees, drained):
    (False, 1, 1, False)   x8
```

The fresh-connection column proves this was a genuinely stuck row, not a stale
snapshot. Fixed by separating the two questions:
- `_queue_is_drained` now means "the frontier is empty" (`pending` only).
- Whether work is in flight is tracked by an in-flight counter over *this
  process's* threads — the only thing that actually knows, and unlike a
  `claimed` row it cannot be a leftover from a killed process.
- `_release_quietly` also retries now, so stuck claims are far rarer.

**Test:** `test_worker_pool_stops_even_when_a_claim_cannot_be_released`.

## B. IMPORTANT — an unexpected exception stranded a claimed player

`except BaseException: release; raise` added around the crawl. The review's two
concrete paths both check out: `crawl_player` wraps only the *match-detail*
fetch in `except PlayerNotFoundError`, so a 404 from
`get_player_match_history_page` escapes unwrapped; and a `JSONDecodeError` is a
`RequestException`, not a `FetchError`.
**Test:** `test_run_releases_the_claim_when_the_crawl_raises_something_unexpected`.

## C. IMPORTANT — orphan window raised to 1 hour

`claimed_stale_seconds` 600 → 3600, docstring rewritten to cite the real
numbers (~112 matches, `max_delay=8s`, so ~15 minutes for one legitimate
crawl) rather than the old "generously longer" claim, which was false.
**Tests:** `test_requeue_stale_players_default_orphan_window_outlasts_a_real_crawl`;
an existing count test was updated, since a 20-minute-old claim is now
correctly *not* reaped.

## D. Jitter could sleep below `min_delay`

Clamped: `time.sleep(max(self.min_delay, delay * random.uniform(...)))`.
Jitter may now only ever slow a request down.
**Test:** `test_rate_limiter_jitter_never_sleeps_below_the_min_delay_floor`.

Per the review, delay is **not** scaled by worker count. The aggregate request
rate scaling roughly linearly with `--workers` is the intended trade, and is
now stated plainly in both `--help` and `_run_worker_pool`'s docstring so it
reads as a deliberate choice rather than a surprise.

## E. Minors

- `f.result()` moved into a `finally`, so a worker's exception is surfaced even
  when the coordinator body itself raises.
  **Test:** `test_worker_pool_surfaces_a_worker_crash_even_when_the_coordinator_also_raises`.
- `_queue_is_drained` is an existence check (`SELECT 1 ... LIMIT 1`), not
  `COUNT(*)` over ~32k pending rows on every ~1s poll — the same unindexed-scan
  mistake this project already fixed once in `select_next_player`.
- The `error` write routes through `crawler.set_status` (promoted from
  `_set_status`; nothing outside `crawler.py` referenced it) instead of
  duplicating the `claimed_at` invariant inline.
  **Test:** `test_run_marks_a_fetch_error_through_the_one_terminal_status_writer`.

## F. Re-verification after all fixes

- Full suite: **125 passed, 3 deselected**, 8/8 consecutive clean runs, ~3.1s.
  The live suite was not run.
- 1-worker vs 3-worker end-to-end data equivalence (real `crawl_player` +
  `ingest_match`, fake network) — identical on every table, `integrity_check`
  ok, no FK violations, and now identical request counts too (82 vs 82).
- Real `SIGINT` against a real 3-worker process: `RETURNED_CLEANLY`,
  94 done / 817 pending, **0 rows still holding a claim**, integrity ok.

## G. Deliberately not addressed (per instruction)

- Scaling delay by worker count — would defeat the feature.
- Retrying across the top-5 claim candidates — negligible at 3 workers.
- **Known documentation gap:** `_upsert_discovered_player`'s write-lock
  invariant is still uncommented. Real but not urgent; noted here rather than
  fixed.
- The shared circuit breaker's symmetric-reset nuance, and double-Ctrl-C
  shutdown latency — accepted behaviours.

## H. One process-level note

Two of the new tests were flaky when first written, and both flakes were mine,
not the code's:
1. The contention test originally used `reseed_interval_seconds=-1`, which
   fires a tick on *every* ~10ms coordinator pass — a reseed holding the write
   lock essentially continuously. That is an impossible schedule (the real one
   runs once a day) and it starved the pool outright. It now uses spaced ticks
   with a hold short enough that a worker's retries outlast it.
2. Asserting "contention happened" via reseed tick counts was racy; it now
   asserts on the `database busy` log line, and the fake crawl is slowed so the
   pool is still working when the lock-holding ticks land.

Both were caught by running the suite 8-10x rather than once. Worth keeping
that habit for this file specifically — a concurrency test that passes once
has not told you very much.
