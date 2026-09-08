# Hero Adjusted Plus-Minus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate each Marvel Rivals hero's contribution to win probability, net of player skill, lineup, composition, side and team-ups, with econometrics-journal-grade inference.

**Architecture:** A new `apm/` package reading the crawler's SQLite database read-only. Design matrices are built with a pluggable hero-attribution rule; rank-deficient blocks (hero, composition) are reduced through an explicit sum-to-zero contrast basis so estimation is unpenalized and standard inference applies. Two co-headline estimators — a match-level logit and a player-match LPM absorbing player fixed effects — are compared against each other. Results and full run metadata are written back into the same database.

**Tech Stack:** Python 3.12, numpy, scipy, pandas, statsmodels, matplotlib, sqlite3, pytest. All already installed.

**Spec:** `docs/superpowers/specs/2026-09-07-hero-apm-model-design.md`

## Global Constraints

- **No new dependencies.** numpy, scipy, pandas, statsmodels, matplotlib only. `pyfixest` and `linearmodels` are NOT installed and must NOT be added; one-way fixed-effect absorption is done by within-transformation, which is exact.
- **No regularization in headline specifications.** Ridge appears only in the Task 9 appendix diagnostic. Never in Spec A or Spec B.
- **Rank-deficient blocks use the sum-to-zero contrast basis** from Task 1, never a dropped reference category. Hero effects are always reported relative to the average hero.
- **Forfeit floor default: 240 seconds.** Sensitivity reported at 0, 180, 240, 300.
- **Rank-band robustness cut: mean pre-match score >= 4750.** Never >= 5000 as a headline split.
- **Bootstrap replications: 1000.** BH FDR at **q = 0.05** across the 55-hero family.
- **Attribution rules:** `W1` play-time-weighted (primary), `W2` dominant-hero, `W3` W1 within duration strata. Team-up and composition features are ALWAYS built under W2 even when hero contrasts use W1 (spec 5.5).
- **Pre-match score is `new_score - add_score`.** Never `new_score` alone.
- **Always filter `is_win IN (0,1)`.** The value 2 is neither win nor loss.
- **Database access is read-only for analysis** (`db.connect_readonly`); only `apm/report.py` writes, and only to `apm_*` tables.
- **No network in the default test suite.** Tests build fixture databases with `db.connect(":memory:")` + `db.init_schema(conn)`, matching `tests/test_db.py`.
- **Estimator correctness is verified against synthetic data with known coefficients.** An estimator test that does not recover planted parameters is not a test.

---

### Task 1: Sum-to-zero contrast basis

**Files:**
- Create: `apm/__init__.py`
- Create: `apm/contrasts.py`
- Test: `tests/test_apm_contrasts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `sum_to_zero_basis(k: int) -> np.ndarray` returning shape `(k, k-1)`, orthonormal columns each summing to zero.
  - `reduce_design(X: np.ndarray, basis: np.ndarray) -> np.ndarray` returning `X @ basis`, shape `(n, k-1)`.
  - `effects_from_free(gamma: np.ndarray, basis: np.ndarray) -> np.ndarray` returning shape `(k,)` summing to zero.
  - `cov_from_free(cov_gamma: np.ndarray, basis: np.ndarray) -> np.ndarray` returning shape `(k, k)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_contrasts.py`:

```python
import numpy as np
import pytest

from apm import contrasts


def test_basis_has_orthonormal_columns_that_sum_to_zero():
    basis = contrasts.sum_to_zero_basis(5)
    assert basis.shape == (5, 4)
    np.testing.assert_allclose(basis.sum(axis=0), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)


def test_effects_recovered_from_free_parameters_sum_to_zero():
    basis = contrasts.sum_to_zero_basis(6)
    gamma = np.array([0.4, -1.1, 0.25, 0.9, -0.3])
    beta = contrasts.effects_from_free(gamma, basis)
    assert beta.shape == (6,)
    assert abs(beta.sum()) < 1e-12


def test_reduced_design_preserves_the_linear_predictor():
    # X rows sum to zero across columns, exactly as hero contrasts do because
    # both teams field six heroes. The reduced design must give an identical
    # linear predictor, which is what makes the reparameterisation lossless.
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 6))
    X = raw - raw.mean(axis=1, keepdims=True)
    basis = contrasts.sum_to_zero_basis(6)
    gamma = rng.normal(size=5)
    beta = contrasts.effects_from_free(gamma, basis)
    np.testing.assert_allclose(
        contrasts.reduce_design(X, basis) @ gamma, X @ beta, atol=1e-10
    )


def test_covariance_transform_is_symmetric_and_positive_semidefinite():
    basis = contrasts.sum_to_zero_basis(4)
    a = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 0.9]])
    cov = contrasts.cov_from_free(a, basis)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_basis_rejects_degenerate_size():
    with pytest.raises(ValueError):
        contrasts.sum_to_zero_basis(1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_contrasts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/__init__.py` as an empty file.

Create `apm/contrasts.py`:

```python
"""Sum-to-zero reparameterisation for rank-deficient design blocks.

Every team fields exactly six heroes, so the hero contrast columns sum to zero
in every row and the all-ones vector lies in the null space of that block. The
same holds for composition shape, since each team has exactly one shape. Rather
than drop a reference category (which leaves the dropped level without a
standard error) we reduce the block onto an orthonormal basis of the sum-to-zero
subspace, fit the free parameters, then map back. The recovered effects satisfy
sum(beta) = 0 by construction, which is exactly the "value relative to the
average hero" normalisation the design calls for.
"""

import numpy as np


def sum_to_zero_basis(k):
    """Orthonormal (k, k-1) basis for {b in R^k : sum(b) = 0}.

    Helmert contrasts: column j puts 1 on the first j+1 levels, -(j+1) on level
    j+1, and 0 elsewhere. Each column sums to zero, and the columns are mutually
    orthogonal, so normalising gives an orthonormal basis.
    """
    if k < 2:
        raise ValueError(f"need at least 2 levels to form contrasts, got {k}")
    basis = np.zeros((k, k - 1))
    for j in range(k - 1):
        basis[: j + 1, j] = 1.0
        basis[j + 1, j] = -(j + 1)
        basis[:, j] /= np.linalg.norm(basis[:, j])
    return basis


def reduce_design(X, basis):
    """Project a rank-deficient block onto its sum-to-zero basis."""
    return np.asarray(X) @ basis


def effects_from_free(gamma, basis):
    """Map free parameters back to per-level effects summing to zero."""
    return basis @ np.asarray(gamma)


def cov_from_free(cov_gamma, basis):
    """Delta-method covariance of the recovered effects."""
    return basis @ np.asarray(cov_gamma) @ basis.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_contrasts.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/__init__.py apm/contrasts.py tests/test_apm_contrasts.py
git commit -m "feat(apm): sum-to-zero contrast basis for rank-deficient blocks"
```

---

### Task 2: Sample construction with exclusion accounting

**Files:**
- Create: `apm/sample.py`
- Test: `tests/test_apm_sample.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ExclusionReport` dataclass with int fields `starting`, `dropped_roster`, `dropped_draw`, `dropped_missing_score`, `dropped_missing_playtime`, `dropped_short`, `remaining`.
  - `build_sample(conn, forfeit_floor_seconds: int = 240) -> tuple[pandas.DataFrame, ExclusionReport]`. The DataFrame has one row per eligible match with columns `match_uid` (str), `camp0_win` (int 0/1), `duration` (float), `timestamp` (int), sorted by `timestamp`.
  - `temporal_holdout(sample: pandas.DataFrame, holdout_days: int = 7) -> tuple[pandas.DataFrame, pandas.DataFrame]` returning `(train, test)` split on `timestamp`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_sample.py`:

```python
import db
from apm import sample


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def add_match(conn, uid, *, duration=700.0, timestamp=1_787_000_000,
              camp0_wins=True, n_players=12, draw=False,
              missing_score=False, missing_playtime=False):
    """Insert one synthetic match with a full 6v6 roster."""
    conn.execute(
        "INSERT INTO matches (match_uid, match_time_stamp, match_play_duration) "
        "VALUES (?,?,?)", (uid, timestamp, duration))
    for i in range(n_players):
        camp = 0 if i < 6 else 1
        won = (camp == 0) == camp0_wins
        is_win = 2 if (draw and i == 0) else int(won)
        conn.execute(
            "INSERT INTO match_players (match_uid, player_uid, camp, cur_hero_id,"
            " is_win, add_score, new_score) VALUES (?,?,?,?,?,?,?)",
            (uid, 1000 + i, camp, 1011 + i, is_win,
             None if missing_score else 10.0, None if missing_score else 4500.0))
        if not missing_playtime:
            conn.execute(
                "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id,"
                " play_time) VALUES (?,?,?,?)", (uid, 1000 + i, 1011 + i, 600.0))
    conn.commit()


def test_clean_match_is_retained():
    conn = make_conn()
    add_match(conn, "m1")
    frame, report = sample.build_sample(conn)
    assert list(frame["match_uid"]) == ["m1"]
    assert frame.loc[0, "camp0_win"] == 1
    assert report.remaining == 1


def test_outcome_flips_with_the_winning_camp():
    conn = make_conn()
    add_match(conn, "m1", camp0_wins=False)
    frame, _ = sample.build_sample(conn)
    assert frame.loc[0, "camp0_win"] == 0


def test_each_exclusion_rule_drops_its_match_and_is_counted():
    conn = make_conn()
    add_match(conn, "keep")
    add_match(conn, "short", duration=100.0)
    add_match(conn, "draw", draw=True)
    add_match(conn, "noscore", missing_score=True)
    add_match(conn, "notime", missing_playtime=True)
    add_match(conn, "roster", n_players=10)
    frame, report = sample.build_sample(conn, forfeit_floor_seconds=240)
    assert list(frame["match_uid"]) == ["keep"]
    assert report.starting == 6
    assert report.dropped_roster == 1
    assert report.dropped_draw == 1
    assert report.dropped_missing_score == 1
    assert report.dropped_missing_playtime == 1
    assert report.dropped_short == 1
    assert report.remaining == 1


def test_forfeit_floor_is_configurable():
    conn = make_conn()
    add_match(conn, "m1", duration=200.0)
    assert sample.build_sample(conn, forfeit_floor_seconds=0)[1].remaining == 1
    assert sample.build_sample(conn, forfeit_floor_seconds=240)[1].remaining == 0


def test_temporal_holdout_splits_on_time_not_at_random():
    conn = make_conn()
    day = 86_400
    for i in range(10):
        add_match(conn, f"m{i}", timestamp=1_787_000_000 + i * day)
    frame, _ = sample.build_sample(conn)
    train, test = sample.temporal_holdout(frame, holdout_days=3)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(frame)
    assert len(test) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_sample.py -v`
Expected: FAIL with `ImportError: cannot import name 'sample' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/sample.py`:

```python
"""Analysis-sample construction with per-rule exclusion accounting.

Filters are applied in a fixed order and each one's cost is counted, because a
published table has to state what was thrown away and why. Rules follow the
design spec section 4.
"""

from dataclasses import dataclass

import pandas as pd

DEFAULT_FORFEIT_FLOOR_SECONDS = 240


@dataclass
class ExclusionReport:
    starting: int
    dropped_roster: int
    dropped_draw: int
    dropped_missing_score: int
    dropped_missing_playtime: int
    dropped_short: int
    remaining: int


def build_sample(conn, forfeit_floor_seconds=DEFAULT_FORFEIT_FLOOR_SECONDS):
    """Return (match-level frame, exclusion report)."""
    all_uids = {r[0] for r in conn.execute("SELECT match_uid FROM matches")}

    complete = {
        r[0]
        for r in conn.execute(
            "SELECT match_uid FROM match_players GROUP BY match_uid "
            "HAVING COUNT(*)=12 AND SUM(camp=0)=6 AND SUM(camp=1)=6"
        )
    }
    drew = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT match_uid FROM match_players WHERE is_win=2")
    }
    no_score = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT match_uid FROM match_players "
            "WHERE add_score IS NULL OR new_score IS NULL")
    }
    # A match qualifies only if every one of its 12 players has positive
    # play time; counting distinct players with play time is cheaper than
    # a correlated per-player existence check.
    timed = {
        r[0] for r in conn.execute(
            "SELECT match_uid FROM match_player_heroes WHERE play_time>0 "
            "GROUP BY match_uid HAVING COUNT(DISTINCT player_uid)=12")
    }
    long_enough = {
        r[0] for r in conn.execute(
            "SELECT match_uid FROM matches WHERE match_play_duration IS NOT NULL "
            "AND match_play_duration>=?", (forfeit_floor_seconds,))
    }

    after_roster = all_uids & complete
    after_draw = after_roster - drew
    after_score = after_draw - no_score
    after_time = after_score & timed
    kept = after_time & long_enough

    report = ExclusionReport(
        starting=len(all_uids),
        dropped_roster=len(all_uids) - len(after_roster),
        dropped_draw=len(after_roster) - len(after_draw),
        dropped_missing_score=len(after_draw) - len(after_score),
        dropped_missing_playtime=len(after_score) - len(after_time),
        dropped_short=len(after_time) - len(kept),
        remaining=len(kept),
    )

    rows = conn.execute(
        "SELECT m.match_uid, m.match_time_stamp, m.match_play_duration, "
        "       MAX(CASE WHEN p.camp=0 THEN p.is_win END) "
        "FROM matches m JOIN match_players p ON p.match_uid=m.match_uid "
        "GROUP BY m.match_uid"
    ).fetchall()
    frame = pd.DataFrame(
        [r for r in rows if r[0] in kept],
        columns=["match_uid", "timestamp", "duration", "camp0_win"],
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["match_uid", "camp0_win", "duration", "timestamp"]], report


def temporal_holdout(sample, holdout_days=7):
    """Split on time, never at random: a random split leaks meta drift."""
    if sample.empty:
        return sample.copy(), sample.copy()
    cutoff = sample["timestamp"].max() - holdout_days * 86_400
    train = sample[sample["timestamp"] <= cutoff].reset_index(drop=True)
    test = sample[sample["timestamp"] > cutoff].reset_index(drop=True)
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_sample.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/sample.py tests/test_apm_sample.py
git commit -m "feat(apm): analysis sample construction with exclusion accounting"
```

---

### Task 3: Hero attribution weights

**Files:**
- Create: `apm/attribution.py`
- Test: `tests/test_apm_attribution.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `attribution_weights(conn, match_uids: set[str], rule: str) -> pandas.DataFrame` with columns `match_uid`, `camp`, `hero_id`, `weight`. `rule` is `"W1"` (play-time share) or `"W2"` (dominant hero). Weights sum to 6.0 per (match, camp).
  - `RULES = ("W1", "W2")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_attribution.py`:

```python
import pytest

import db
from apm import attribution


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def add_player(conn, match_uid, player_uid, camp, hero_playtimes):
    conn.execute(
        "INSERT INTO match_players (match_uid, player_uid, camp, is_win) "
        "VALUES (?,?,?,1)", (match_uid, player_uid, camp))
    for hero_id, seconds in hero_playtimes.items():
        conn.execute(
            "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id,"
            " play_time) VALUES (?,?,?,?)", (match_uid, player_uid, hero_id, seconds))
    conn.commit()


def test_w1_splits_a_swapping_player_across_heroes_by_time():
    conn = make_conn()
    add_player(conn, "m1", 1, 0, {1011: 300.0, 1022: 100.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W1")
    weights = dict(zip(frame["hero_id"], frame["weight"]))
    assert weights[1011] == pytest.approx(0.75)
    assert weights[1022] == pytest.approx(0.25)


def test_w2_gives_the_whole_slot_to_the_dominant_hero():
    conn = make_conn()
    add_player(conn, "m1", 1, 0, {1011: 300.0, 1022: 100.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W2")
    assert list(frame["hero_id"]) == [1011]
    assert frame.loc[0, "weight"] == pytest.approx(1.0)


def test_weights_sum_to_six_per_team_under_both_rules():
    conn = make_conn()
    for i in range(6):
        add_player(conn, "m1", i, 0, {1011 + i: 400.0, 1040: 200.0})
    for i in range(6):
        add_player(conn, "m1", 100 + i, 1, {1030 + i: 600.0})
    for rule in attribution.RULES:
        frame = attribution.attribution_weights(conn, {"m1"}, rule)
        totals = frame.groupby("camp")["weight"].sum()
        assert totals[0] == pytest.approx(6.0)
        assert totals[1] == pytest.approx(6.0)


def test_zero_playtime_heroes_are_ignored():
    conn = make_conn()
    add_player(conn, "m1", 1, 0, {1011: 400.0, 1022: 0.0})
    frame = attribution.attribution_weights(conn, {"m1"}, "W1")
    assert set(frame["hero_id"]) == {1011}


def test_unknown_rule_is_rejected():
    conn = make_conn()
    with pytest.raises(ValueError):
        attribution.attribution_weights(conn, set(), "W9")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_attribution.py -v`
Expected: FAIL with `ImportError: cannot import name 'attribution' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/attribution.py`:

```python
"""Hero attribution: turning a player-match into hero weights.

No pick data exists (is_pick is always 0), and roughly 55% of player-matches
involve a mid-match swap whose timing is endogenous to the match state. So
"which hero did this player play" is a modelling choice, not an observation,
and it is exposed here as a parameter rather than baked into the feature code.

W1 (primary) splits a player's slot across heroes in proportion to play time.
It is the only rule that does not condition on the outcome: taking the final
hero blames losses on whoever was swapped to, and keeping only non-swappers
selects on winning (65.6% vs 37.3% in the full data).

W2 assigns the whole slot to the longest-played hero. It discards swap
information but yields integer lineups, which team-up and composition features
require because those need a definite set of six.
"""

import pandas as pd

RULES = ("W1", "W2")


def attribution_weights(conn, match_uids, rule):
    if rule not in RULES:
        raise ValueError(f"unknown attribution rule {rule!r}; expected one of {RULES}")
    if not match_uids:
        return pd.DataFrame(columns=["match_uid", "camp", "hero_id", "weight"])

    rows = conn.execute(
        "SELECT h.match_uid, p.camp, h.player_uid, h.hero_id, h.play_time "
        "FROM match_player_heroes h "
        "JOIN match_players p ON p.match_uid=h.match_uid "
        "                    AND p.player_uid=h.player_uid "
        "WHERE h.play_time>0"
    ).fetchall()
    frame = pd.DataFrame(
        rows, columns=["match_uid", "camp", "player_uid", "hero_id", "play_time"]
    )
    frame = frame[frame["match_uid"].isin(match_uids)]
    if frame.empty:
        return pd.DataFrame(columns=["match_uid", "camp", "hero_id", "weight"])

    keys = ["match_uid", "player_uid"]
    if rule == "W1":
        total = frame.groupby(keys)["play_time"].transform("sum")
        frame = frame.assign(weight=frame["play_time"] / total)
    else:
        best = frame.groupby(keys)["play_time"].transform("max")
        # A tie would otherwise give a player two full slots and break the
        # sums-to-six invariant, so keep exactly one row per player.
        frame = frame[frame["play_time"] == best]
        frame = frame.drop_duplicates(subset=keys, keep="first").assign(weight=1.0)

    return (
        frame.groupby(["match_uid", "camp", "hero_id"], as_index=False)["weight"]
        .sum()
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_attribution.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/attribution.py tests/test_apm_attribution.py
git commit -m "feat(apm): pluggable hero attribution rules W1 and W2"
```

---

### Task 4: Design matrix assembly

**Files:**
- Create: `apm/features.py`
- Test: `tests/test_apm_features.py`

**Interfaces:**
- Consumes: `apm.attribution.attribution_weights`, `apm.contrasts.sum_to_zero_basis`, `apm.contrasts.reduce_design`.
- Produces:
  - `DesignMatrix` dataclass with fields `X` (np.ndarray, `(n, p)`), `y` (np.ndarray `(n,)`), `column_names` (list[str]), `hero_ids` (list[int]), `hero_basis` (np.ndarray), `hero_slice` (slice), `teamup_ids` (list[int]), `match_uids` (list[str]).
  - `build_design(conn, sample, attribution_rule="W1", min_shape_count=500) -> DesignMatrix`.
  - `hero_contrast_frame(weights) -> pandas.DataFrame` — pivot of weights into `match_uid` x `hero_id`, camp 0 minus camp 1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_features.py`:

```python
import numpy as np
import pytest

import db
from apm import features


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1011,'A','Tank')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1022,'B','Tank')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1033,'C','Damage')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1044,'D','Damage')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1055,'E','Support')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1066,'F','Support')")
    conn.commit()
    return conn


def add_match(conn, uid, camp0_heroes, camp1_heroes, camp0_win=1,
              camp0_score=4500.0, camp1_score=4500.0, timestamp=1_787_000_000):
    conn.execute(
        "INSERT INTO matches (match_uid, match_time_stamp, match_play_duration) "
        "VALUES (?,?,700.0)", (uid, timestamp))
    for i, hero in enumerate(camp0_heroes + camp1_heroes):
        camp = 0 if i < 6 else 1
        score = camp0_score if camp == 0 else camp1_score
        conn.execute(
            "INSERT INTO match_players (match_uid, player_uid, camp, cur_hero_id,"
            " is_win, add_score, new_score) VALUES (?,?,?,?,?,?,?)",
            (uid, hash((uid, i)) % 10**6, camp, hero,
             camp0_win if camp == 0 else 1 - camp0_win, 10.0, score + 10.0))
        conn.execute(
            "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id,"
            " play_time) VALUES (?,?,?,600.0)",
            (uid, hash((uid, i)) % 10**6, hero))
    conn.commit()


SIX_A = [1011, 1022, 1033, 1044, 1055, 1066]
SIX_B = [1022, 1011, 1044, 1033, 1066, 1055]


def test_hero_block_is_reduced_by_one_column_and_recovers_zero_sum_effects():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_B)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    # Six heroes present -> five free hero parameters.
    assert len(design.hero_ids) == 6
    assert design.hero_basis.shape == (6, 5)
    assert design.X.shape[0] == 1


def test_mirror_lineups_give_a_zero_hero_contrast():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_A)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    hero_block = design.X[:, design.hero_slice]
    np.testing.assert_allclose(hero_block, np.zeros_like(hero_block), atol=1e-12)


def test_skill_control_is_the_pre_match_score_differential():
    conn = make_conn()
    # new_score - add_score is the pre-match score, so camp0 enters 200 higher.
    add_match(conn, "m1", SIX_A, SIX_B, camp0_score=4700.0, camp1_score=4500.0)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    idx = design.column_names.index("skill_diff")
    assert design.X[0, idx] == pytest.approx(200.0)


def test_intercept_column_is_present_for_the_side_advantage():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_B)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    idx = design.column_names.index("intercept")
    assert design.X[0, idx] == pytest.approx(1.0)


def test_outcome_vector_matches_the_sample():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_B, camp0_win=0)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    assert design.y.tolist() == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'features' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/features.py`:

```python
"""Design-matrix assembly for the adjusted plus-minus model.

Column layout, in order: intercept, hero free parameters, team-up contrasts,
composition free parameters, skill differential. The hero and composition
blocks are rank-deficient (each sums to zero across levels within a row) and so
are reduced onto sum-to-zero bases; the recovered per-level effects come back
via apm.contrasts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apm import attribution, contrasts


@dataclass
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    column_names: list
    hero_ids: list
    hero_basis: np.ndarray
    hero_slice: slice
    teamup_ids: list
    match_uids: list


def hero_contrast_frame(weights):
    """Pivot attribution weights to match x hero, camp 0 minus camp 1."""
    signed = weights.assign(
        signed=np.where(weights["camp"] == 0, weights["weight"], -weights["weight"])
    )
    return signed.pivot_table(
        index="match_uid", columns="hero_id", values="signed",
        aggfunc="sum", fill_value=0.0,
    )


def _lineup_sets(conn, match_uids):
    """Six dominant heroes per (match, camp). Team-ups and composition need a
    definite set, so these always use W2 regardless of the hero rule."""
    w2 = attribution.attribution_weights(conn, match_uids, "W2")
    return {
        key: set(group["hero_id"])
        for key, group in w2.groupby(["match_uid", "camp"])
    }


def build_design(conn, sample, attribution_rule="W1", min_shape_count=500):
    # `order` may legitimately contain the same match twice: a cluster
    # bootstrap resamples players with replacement, so their matches repeat.
    # Every row-wise loop below must therefore index by position, never by a
    # {uid: row} dict, which would keep only the last occurrence.
    match_uids = set(sample["match_uid"])
    order = list(sample["match_uid"])
    n = len(order)

    weights = attribution.attribution_weights(conn, match_uids, attribution_rule)
    hero_frame = hero_contrast_frame(weights).reindex(order, fill_value=0.0)
    hero_ids = [int(h) for h in hero_frame.columns]
    hero_raw = hero_frame.to_numpy(dtype=float)
    hero_basis = contrasts.sum_to_zero_basis(len(hero_ids))
    hero_block = contrasts.reduce_design(hero_raw, hero_basis)

    lineups = _lineup_sets(conn, match_uids)
    roles = dict(conn.execute("SELECT hero_id, role FROM hero_info"))

    teamups = {}
    for tid, hid in conn.execute("SELECT teamup_id, hero_id FROM teamup_heroes"):
        teamups.setdefault(tid, set()).add(hid)
    teamup_ids = sorted(teamups)
    teamup_block = np.zeros((n, len(teamup_ids)))
    for j, tid in enumerate(teamup_ids):
        members = teamups[tid]
        for row, uid in enumerate(order):
            in0 = members <= lineups.get((uid, 0), set())
            in1 = members <= lineups.get((uid, 1), set())
            teamup_block[row, j] = float(in0) - float(in1)

    def shape_of(uid, camp):
        heroes = lineups.get((uid, camp), set())
        counts = [0, 0, 0]
        for hero in heroes:
            role = roles.get(hero)
            if role == "Tank":
                counts[0] += 1
            elif role == "Damage":
                counts[1] += 1
            elif role == "Support":
                counts[2] += 1
        return "{}-{}-{}".format(*counts)

    shapes0 = [shape_of(uid, 0) for uid in order]
    shapes1 = [shape_of(uid, 1) for uid in order]
    counts = pd.Series(shapes0 + shapes1).value_counts()
    common = sorted(counts[counts >= min_shape_count].index)
    labels = common + ["other"]
    canon = lambda s: s if s in common else "other"

    shape_raw = np.zeros((n, len(labels)))
    for i, (s0, s1) in enumerate(zip(shapes0, shapes1)):
        shape_raw[i, labels.index(canon(s0))] += 1.0
        shape_raw[i, labels.index(canon(s1))] -= 1.0
    if len(labels) >= 2:
        shape_basis = contrasts.sum_to_zero_basis(len(labels))
        shape_block = contrasts.reduce_design(shape_raw, shape_basis)
        shape_names = [f"shape_free_{i}" for i in range(shape_block.shape[1])]
    else:
        shape_block = np.zeros((n, 0))
        shape_names = []

    score_rows = conn.execute(
        "SELECT match_uid, camp, AVG(new_score-add_score) FROM match_players "
        "WHERE add_score IS NOT NULL GROUP BY match_uid, camp"
    ).fetchall()
    pre = {(r[0], r[1]): r[2] for r in score_rows}
    skill = np.array(
        [[pre.get((uid, 0), 0.0) - pre.get((uid, 1), 0.0)] for uid in order]
    )

    intercept = np.ones((n, 1))
    X = np.hstack([intercept, hero_block, teamup_block, shape_block, skill])
    names = (
        ["intercept"]
        + [f"hero_free_{i}" for i in range(hero_block.shape[1])]
        + [f"teamup_{t}" for t in teamup_ids]
        + shape_names
        + ["skill_diff"]
    )
    hero_slice = slice(1, 1 + hero_block.shape[1])
    y = sample["camp0_win"].to_numpy(dtype=int)
    return DesignMatrix(X, y, names, hero_ids, hero_basis, hero_slice,
                        teamup_ids, order)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_features.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/features.py tests/test_apm_features.py
git commit -m "feat(apm): design matrix with hero, team-up, composition and skill blocks"
```

---

### Task 5: Specification A — match-level differential logit

**Files:**
- Create: `apm/estimate.py`
- Test: `tests/test_apm_estimate_logit.py`

**Interfaces:**
- Consumes: `apm.features.DesignMatrix`, `apm.contrasts.effects_from_free`, `apm.contrasts.cov_from_free`.
- Produces:
  - `FitResult` dataclass with fields `params` (np.ndarray), `cov` (np.ndarray), `column_names` (list[str]), `hero_effects` (dict[int, float]), `hero_cov` (np.ndarray), `hero_ids` (list[int]), `loglike` (float), `n` (int).
  - `fit_logit(design: DesignMatrix) -> FitResult`.
  - `predict_proba(design: DesignMatrix, fit: FitResult) -> np.ndarray`.
  - `log_loss(y, p) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_estimate_logit.py`:

```python
import numpy as np
import pytest

from apm import contrasts, estimate
from apm.features import DesignMatrix


def synthetic_design(n=40_000, seed=0):
    """Matches with six heroes a side drawn at random and a KNOWN hero effect
    vector. Recovering planted coefficients is the only honest test of an
    estimator."""
    rng = np.random.default_rng(seed)
    k = 8
    true_beta = np.array([0.5, 0.3, 0.1, 0.0, -0.1, -0.2, -0.3, -0.3])
    true_beta -= true_beta.mean()
    raw = np.zeros((n, k))
    for i in range(n):
        pick0 = rng.choice(k, size=3, replace=False)
        pick1 = rng.choice(k, size=3, replace=False)
        for h in pick0:
            raw[i, h] += 1.0
        for h in pick1:
            raw[i, h] -= 1.0
    eta = 0.05 + raw @ true_beta
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    basis = contrasts.sum_to_zero_basis(k)
    block = contrasts.reduce_design(raw, basis)
    X = np.hstack([np.ones((n, 1)), block])
    names = ["intercept"] + [f"hero_free_{i}" for i in range(block.shape[1])]
    design = DesignMatrix(X, y, names, list(range(k)), basis,
                          slice(1, 1 + block.shape[1]), [],
                          [f"m{i}" for i in range(n)])
    return design, true_beta


def test_logit_recovers_planted_hero_effects():
    design, true_beta = synthetic_design()
    fit = estimate.fit_logit(design)
    estimated = np.array([fit.hero_effects[h] for h in design.hero_ids])
    assert np.max(np.abs(estimated - true_beta)) < 0.05


def test_recovered_hero_effects_sum_to_zero():
    design, _ = synthetic_design(n=5_000)
    fit = estimate.fit_logit(design)
    assert abs(sum(fit.hero_effects.values())) < 1e-9


def test_every_hero_including_the_last_gets_a_standard_error():
    design, _ = synthetic_design(n=5_000)
    fit = estimate.fit_logit(design)
    assert fit.hero_cov.shape == (len(design.hero_ids), len(design.hero_ids))
    assert np.all(np.diag(fit.hero_cov) > 0)


def test_intercept_recovers_the_side_advantage():
    design, _ = synthetic_design(n=40_000)
    fit = estimate.fit_logit(design)
    assert fit.params[0] == pytest.approx(0.05, abs=0.05)


def test_log_loss_beats_a_coin_flip_on_the_training_data():
    design, _ = synthetic_design(n=20_000)
    fit = estimate.fit_logit(design)
    p = estimate.predict_proba(design, fit)
    assert estimate.log_loss(design.y, p) < np.log(2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_estimate_logit.py -v`
Expected: FAIL with `ImportError: cannot import name 'estimate' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/estimate.py`:

```python
"""Estimators for the two co-headline specifications.

Deliberately unpenalised. Ridge is the conventional choice for a collinear hero
design, but its coefficients are biased and carry no valid standard errors,
which is fatal when the entire deliverable is a ranked table with intervals.
The collinearity here is structural, not incidental, and is removed exactly by
the sum-to-zero reparameterisation in apm.contrasts.
"""

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from apm import contrasts


@dataclass
class FitResult:
    params: np.ndarray
    cov: np.ndarray
    column_names: list
    hero_effects: dict
    hero_cov: np.ndarray
    hero_ids: list
    loglike: float
    n: int


def _hero_pieces(design, params, cov):
    gamma = params[design.hero_slice]
    cov_gamma = cov[design.hero_slice, design.hero_slice]
    beta = contrasts.effects_from_free(gamma, design.hero_basis)
    hero_cov = contrasts.cov_from_free(cov_gamma, design.hero_basis)
    return dict(zip(design.hero_ids, beta)), hero_cov


def fit_logit(design):
    model = sm.Logit(design.y, design.X)
    res = model.fit(disp=0, method="newton", maxiter=200)
    effects, hero_cov = _hero_pieces(design, res.params, res.cov_params())
    return FitResult(
        params=res.params,
        cov=res.cov_params(),
        column_names=design.column_names,
        hero_effects=effects,
        hero_cov=hero_cov,
        hero_ids=design.hero_ids,
        loglike=res.llf,
        n=len(design.y),
    )


def predict_proba(design, fit):
    eta = design.X @ fit.params
    return 1.0 / (1.0 + np.exp(-eta))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_estimate_logit.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/estimate.py tests/test_apm_estimate_logit.py
git commit -m "feat(apm): specification A, match-level differential logit"
```

---

### Task 6: Specification B — player-match LPM with absorbed player fixed effects

**Files:**
- Modify: `apm/estimate.py` (append; do not alter Task 5 functions)
- Test: `tests/test_apm_estimate_lpm.py`

**Interfaces:**
- Consumes: `apm.estimate.FitResult`.
- Produces:
  - `within_transform(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning group-demeaned `(X, y)`.
  - `fit_absorbed_lpm(X, y, groups, hero_ids, hero_basis, hero_slice, column_names) -> FitResult`, using degrees of freedom `n - k - n_groups`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_estimate_lpm.py`:

```python
import numpy as np

from apm import contrasts, estimate


def test_within_transform_removes_group_means():
    X = np.array([[1.0], [3.0], [10.0], [20.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    groups = np.array([0, 0, 1, 1])
    Xw, yw = estimate.within_transform(X, y, groups)
    np.testing.assert_allclose(Xw.ravel(), [-1.0, 1.0, -5.0, 5.0])
    np.testing.assert_allclose(yw, [-1.0, 1.0, -1.0, 1.0])


def test_absorbed_lpm_recovers_slopes_despite_large_group_effects():
    # Group intercepts are enormous relative to the slope. An estimator that
    # failed to absorb them would be swamped; a correct one is unaffected.
    rng = np.random.default_rng(0)
    n_groups, per = 500, 8
    groups = np.repeat(np.arange(n_groups), per)
    n = n_groups * per
    X = rng.normal(size=(n, 2))
    alpha = rng.normal(scale=50.0, size=n_groups)[groups]
    true = np.array([0.30, -0.20])
    y = alpha + X @ true + rng.normal(scale=0.1, size=n)
    fit = estimate.fit_absorbed_lpm(
        X, y, groups, hero_ids=[], hero_basis=np.zeros((0, 0)),
        hero_slice=slice(0, 0), column_names=["a", "b"])
    np.testing.assert_allclose(fit.params, true, atol=0.02)


def test_absorbed_lpm_recovers_zero_sum_hero_effects():
    rng = np.random.default_rng(1)
    k = 6
    true_beta = np.array([0.20, 0.10, 0.05, -0.05, -0.10, -0.20])
    true_beta -= true_beta.mean()
    n_groups, per = 800, 6
    groups = np.repeat(np.arange(n_groups), per)
    n = n_groups * per
    raw = np.zeros((n, k))
    for i in range(n):
        raw[i, rng.integers(k)] = 1.0
    raw -= raw.mean(axis=1, keepdims=True)
    basis = contrasts.sum_to_zero_basis(k)
    block = contrasts.reduce_design(raw, basis)
    alpha = rng.normal(scale=5.0, size=n_groups)[groups]
    y = alpha + raw @ true_beta + rng.normal(scale=0.05, size=n)
    fit = estimate.fit_absorbed_lpm(
        block, y, groups, hero_ids=list(range(k)), hero_basis=basis,
        hero_slice=slice(0, block.shape[1]),
        column_names=[f"h{i}" for i in range(block.shape[1])])
    estimated = np.array([fit.hero_effects[h] for h in range(k)])
    assert np.max(np.abs(estimated - true_beta)) < 0.02
    assert abs(sum(fit.hero_effects.values())) < 1e-9


def test_degrees_of_freedom_account_for_absorbed_groups():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(100), 5)
    X = rng.normal(size=(500, 2))
    y = rng.normal(size=500)
    fit = estimate.fit_absorbed_lpm(
        X, y, groups, hero_ids=[], hero_basis=np.zeros((0, 0)),
        hero_slice=slice(0, 0), column_names=["a", "b"])
    # 500 observations - 2 slopes - 100 absorbed intercepts
    assert fit.n == 500
    assert np.all(np.diag(fit.cov) > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_estimate_lpm.py -v`
Expected: FAIL with `AttributeError: module 'apm.estimate' has no attribute 'within_transform'`

- [ ] **Step 3: Write minimal implementation**

Append to `apm/estimate.py`:

```python
def within_transform(X, y, groups):
    """Demean X and y by group.

    With a single fixed-effect dimension this recovers the OLS slope estimates
    exactly (Frisch-Waugh-Lovell), so no iterative absorption library is needed
    and none is installed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    codes, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    x_sums = np.zeros((len(codes), X.shape[1]))
    np.add.at(x_sums, inverse, X)
    y_sums = np.zeros(len(codes))
    np.add.at(y_sums, inverse, y)
    return X - (x_sums / counts[:, None])[inverse], y - (y_sums / counts)[inverse]


def fit_absorbed_lpm(X, y, groups, hero_ids, hero_basis, hero_slice,
                     column_names):
    """Linear probability model with one absorbed fixed-effect dimension.

    An LPM rather than a fixed-effects logit: with roughly 5.6 matches per
    player, a nonlinear model with a parameter per player is inconsistent
    (incidental parameters). Predicted probabilities can fall outside [0,1];
    that cost is reported as a diagnostic rather than hidden.
    """
    Xw, yw = within_transform(X, y, groups)
    n, k = Xw.shape
    n_groups = len(np.unique(groups))
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    params = xtx_inv @ (Xw.T @ yw)
    resid = yw - Xw @ params
    dof = max(n - k - n_groups, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * xtx_inv
    if len(hero_ids):
        gamma = params[hero_slice]
        beta = contrasts.effects_from_free(gamma, hero_basis)
        hero_cov = contrasts.cov_from_free(cov[hero_slice, hero_slice], hero_basis)
        effects = dict(zip(hero_ids, beta))
    else:
        effects, hero_cov = {}, np.zeros((0, 0))
    return FitResult(params=params, cov=cov, column_names=column_names,
                     hero_effects=effects, hero_cov=hero_cov,
                     hero_ids=list(hero_ids), loglike=float("nan"), n=n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_estimate_lpm.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add apm/estimate.py tests/test_apm_estimate_lpm.py
git commit -m "feat(apm): specification B, LPM absorbing player fixed effects"
```

---

### Task 7: Inference — cluster bootstrap and FDR control

**Files:**
- Create: `apm/inference.py`
- Test: `tests/test_apm_inference.py`

**Interfaces:**
- Consumes: `apm.features.DesignMatrix`, `apm.estimate.fit_logit`.
- Produces:
  - `benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]` returning `(adjusted_pvalues, rejected_mask)`.
  - `player_clusters(conn, match_uids: list[str]) -> dict[str, list[int]]`.
  - `cluster_bootstrap(design, fit_fn, clusters, n_reps=1000, seed=0) -> np.ndarray` of shape `(n_reps, n_heroes)`.
  - `percentile_interval(draws: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_inference.py`:

```python
import numpy as np

from apm import inference


def test_bh_rejects_nothing_when_all_pvalues_are_uniform_noise():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    _, rejected = inference.benjamini_hochberg(p, q=0.05)
    # Under the null, expected rejections are far below 5% of tests.
    assert rejected.sum() <= 5


def test_bh_rejects_obvious_signal_and_is_less_strict_than_bonferroni():
    p = np.array([1e-8, 1e-7, 1e-6, 0.4, 0.6, 0.9])
    adjusted, rejected = inference.benjamini_hochberg(p, q=0.05)
    assert rejected[:3].all()
    assert not rejected[3:].any()
    assert np.all(adjusted >= p)
    assert adjusted[2] < p[2] * len(p)


def test_bh_adjusted_values_are_monotone():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    adjusted, _ = inference.benjamini_hochberg(p)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_percentile_interval_brackets_the_truth():
    rng = np.random.default_rng(1)
    draws = rng.normal(loc=0.3, scale=0.1, size=(2000, 1))
    lo, hi = inference.percentile_interval(draws)
    assert lo[0] < 0.3 < hi[0]


def test_cluster_bootstrap_returns_one_row_per_replication():
    rng = np.random.default_rng(2)

    class FakeDesign:
        match_uids = [f"m{i}" for i in range(50)]
        hero_ids = [1, 2, 3]

    clusters = {f"m{i}": [i % 10] for i in range(50)}

    def fake_fit(match_uids):
        return {h: rng.normal() for h in FakeDesign.hero_ids}

    draws = inference.cluster_bootstrap(
        FakeDesign(), fake_fit, clusters, n_reps=25, seed=0)
    assert draws.shape == (25, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_inference.py -v`
Expected: FAIL with `ImportError: cannot import name 'inference' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/inference.py`:

```python
"""Inference respecting how the sample was actually produced.

Matches are not independent draws: the crawl enumerates players and harvests
their matches, and a single match belongs to twelve overlapping player clusters.
That is cross-classified rather than nested, so conventional one-way clustering
does not apply. The reported intervals come from resampling *players* and
refitting, which mirrors the sampling process directly.
"""

import numpy as np


def benjamini_hochberg(pvalues, q=0.05):
    """Step-up FDR control. Returns (adjusted p-values, rejected mask).

    Fifty-five hero coefficients are tested at once; unadjusted stars on that
    family would be indefensible.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= q


def player_clusters(conn, match_uids):
    """Map each match to the players in it, for the cluster bootstrap."""
    wanted = set(match_uids)
    out = {}
    for match_uid, player_uid in conn.execute(
        "SELECT match_uid, player_uid FROM match_players"
    ):
        if match_uid in wanted:
            out.setdefault(match_uid, []).append(player_uid)
    return out


def cluster_bootstrap(design, fit_fn, clusters, n_reps=1000, seed=0):
    """Resample players with replacement, refit on their matches.

    `fit_fn` takes a list of match_uids and returns {hero_id: effect}.
    """
    rng = np.random.default_rng(seed)
    by_player = {}
    for match_uid, players in clusters.items():
        for player in players:
            by_player.setdefault(player, []).append(match_uid)
    players = list(by_player)
    draws = np.full((n_reps, len(design.hero_ids)), np.nan)
    for rep in range(n_reps):
        picked = rng.choice(len(players), size=len(players), replace=True)
        uids = []
        for idx in picked:
            uids.extend(by_player[players[idx]])
        effects = fit_fn(uids)
        draws[rep] = [effects.get(h, np.nan) for h in design.hero_ids]
    return draws


def percentile_interval(draws, alpha=0.05):
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return lo, hi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_inference.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/inference.py tests/test_apm_inference.py
git commit -m "feat(apm): player cluster bootstrap and Benjamini-Hochberg FDR"
```

---

### Task 8: Results schema and persistence

**Files:**
- Modify: `db.py` (append three tables to `SCHEMA`, immediately before the closing `"""`)
- Create: `apm/report.py`
- Test: `tests/test_apm_report.py`

**Interfaces:**
- Consumes: `apm.estimate.FitResult`.
- Produces:
  - `record_run(conn, spec, attribution_rule, forfeit_floor, n_matches, n_players, exclusions, git_commit) -> int` returning the new `run_id`.
  - `write_hero_effects(conn, run_id, fit, intervals, adjusted_p) -> None`.
  - `hero_table(conn, run_id) -> pandas.DataFrame` joined to `hero_info` for names.
  - `to_probability_points(beta: float) -> float` returning `beta / 4 * 100`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_report.py`:

```python
import numpy as np
import pytest

import db
from apm import report
from apm.estimate import FitResult


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1011,'Hulk','Tank')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1022,'Cap','Tank')")
    conn.commit()
    return conn


def test_schema_creates_the_result_tables():
    conn = make_conn()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"apm_runs", "apm_hero_effects", "apm_teamup_effects"} <= tables


def test_run_metadata_round_trips():
    conn = make_conn()
    run_id = report.record_run(
        conn, spec="A", attribution_rule="W1", forfeit_floor=240,
        n_matches=1000, n_players=12000, exclusions="{}", git_commit="abc123")
    row = conn.execute(
        "SELECT spec, attribution_rule, forfeit_floor, n_matches, git_commit "
        "FROM apm_runs WHERE run_id=?", (run_id,)).fetchone()
    assert row == ("A", "W1", 240, 1000, "abc123")


def test_hero_effects_are_written_and_join_to_names():
    conn = make_conn()
    run_id = report.record_run(
        conn, spec="A", attribution_rule="W1", forfeit_floor=240,
        n_matches=10, n_players=120, exclusions="{}", git_commit="abc")
    fit = FitResult(
        params=np.zeros(2), cov=np.eye(2), column_names=["a", "b"],
        hero_effects={1011: 0.20, 1022: -0.20},
        hero_cov=np.diag([0.0004, 0.0004]), hero_ids=[1011, 1022],
        loglike=-1.0, n=10)
    report.write_hero_effects(
        conn, run_id, fit,
        intervals={1011: (0.15, 0.25), 1022: (-0.25, -0.15)},
        adjusted_p={1011: 0.001, 1022: 0.001})
    table = report.hero_table(conn, run_id)
    assert set(table["name"]) == {"Hulk", "Cap"}
    hulk = table[table["name"] == "Hulk"].iloc[0]
    assert hulk["effect_logodds"] == pytest.approx(0.20)
    assert hulk["effect_pp"] == pytest.approx(5.0)


def test_probability_point_conversion_uses_the_balanced_match_margin():
    # dP/dx = beta * p * (1-p); at p = 0.5 that is beta/4.
    assert report.to_probability_points(0.4) == pytest.approx(10.0)


def test_runs_are_independent():
    conn = make_conn()
    a = report.record_run(conn, spec="A", attribution_rule="W1",
                          forfeit_floor=240, n_matches=1, n_players=1,
                          exclusions="{}", git_commit="x")
    b = report.record_run(conn, spec="B", attribution_rule="W2",
                          forfeit_floor=0, n_matches=2, n_players=2,
                          exclusions="{}", git_commit="y")
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'report' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

In `db.py`, append to the `SCHEMA` string immediately before the closing `"""`:

```sql
-- Model results. Coefficients are meaningless without the run metadata that
-- produced them, so apm_runs records the specification, attribution rule,
-- sample filters and git commit alongside every fit.
CREATE TABLE IF NOT EXISTS apm_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    spec TEXT NOT NULL,
    attribution_rule TEXT NOT NULL,
    forfeit_floor INTEGER,
    n_matches INTEGER,
    n_players INTEGER,
    exclusions TEXT,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS apm_hero_effects (
    run_id INTEGER NOT NULL REFERENCES apm_runs(run_id),
    hero_id INTEGER NOT NULL,
    effect_logodds REAL,
    std_error REAL,
    ci_low REAL,
    ci_high REAL,
    p_adjusted REAL,
    PRIMARY KEY (run_id, hero_id)
);

CREATE TABLE IF NOT EXISTS apm_teamup_effects (
    run_id INTEGER NOT NULL REFERENCES apm_runs(run_id),
    teamup_id INTEGER NOT NULL,
    effect_logodds REAL,
    std_error REAL,
    PRIMARY KEY (run_id, teamup_id)
);
```

Create `apm/report.py`:

```python
"""Result persistence and formatted tables.

Results are written back into the crawler's database so a coefficient can be
traced to the exact sample and specification that produced it months later.
"""

import numpy as np
import pandas as pd


def to_probability_points(beta):
    """Convert a log-odds coefficient to win-probability points.

    The logit marginal effect is beta * p * (1-p), which at a balanced match
    (p = 0.5) is beta / 4. Reported in points, so multiply by 100.
    """
    return beta / 4.0 * 100.0


def record_run(conn, spec, attribution_rule, forfeit_floor, n_matches,
               n_players, exclusions, git_commit):
    cur = conn.execute(
        "INSERT INTO apm_runs (spec, attribution_rule, forfeit_floor,"
        " n_matches, n_players, exclusions, git_commit) VALUES (?,?,?,?,?,?,?)",
        (spec, attribution_rule, forfeit_floor, n_matches, n_players,
         exclusions, git_commit),
    )
    conn.commit()
    return cur.lastrowid


def write_hero_effects(conn, run_id, fit, intervals, adjusted_p):
    errors = np.sqrt(np.diag(fit.hero_cov))
    for i, hero_id in enumerate(fit.hero_ids):
        low, high = intervals.get(hero_id, (None, None))
        conn.execute(
            "INSERT OR REPLACE INTO apm_hero_effects (run_id, hero_id,"
            " effect_logodds, std_error, ci_low, ci_high, p_adjusted)"
            " VALUES (?,?,?,?,?,?,?)",
            (run_id, hero_id, float(fit.hero_effects[hero_id]),
             float(errors[i]), low, high, adjusted_p.get(hero_id)),
        )
    conn.commit()


def hero_table(conn, run_id):
    rows = conn.execute(
        "SELECT e.hero_id, h.name, h.role, e.effect_logodds, e.std_error,"
        "       e.ci_low, e.ci_high, e.p_adjusted "
        "FROM apm_hero_effects e LEFT JOIN hero_info h ON h.hero_id=e.hero_id "
        "WHERE e.run_id=? ORDER BY e.effect_logodds DESC", (run_id,)
    ).fetchall()
    frame = pd.DataFrame(rows, columns=[
        "hero_id", "name", "role", "effect_logodds", "std_error",
        "ci_low", "ci_high", "p_adjusted"])
    frame["effect_pp"] = frame["effect_logodds"].apply(to_probability_points)
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_report.py tests/test_db.py -v`
Expected: PASS, 5 new tests plus the existing `test_db.py` suite still green

- [ ] **Step 5: Commit**

```bash
git add db.py apm/report.py tests/test_apm_report.py
git commit -m "feat(apm): result schema and persistence with run metadata"
```

---

### Task 9: Diagnostics — placebo, Oster bounds, calibration

**Files:**
- Create: `apm/diagnostics.py`
- Test: `tests/test_apm_diagnostics.py`

**Interfaces:**
- Consumes: `apm.features.DesignMatrix`, `apm.estimate.fit_logit`, `apm.estimate.log_loss`.
- Produces:
  - `permute_hero_block(design, seed=0) -> DesignMatrix` — placebo design with hero rows shuffled.
  - `oster_delta(beta_short, r_short, beta_full, r_full, r_max) -> float`.
  - `calibration_table(y, p, bins=10) -> pandas.DataFrame` with columns `bin`, `mean_predicted`, `mean_actual`, `n`.
  - `nested_log_losses(design, fit_fn, blocks: dict[str, slice]) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_diagnostics.py`:

```python
import numpy as np
import pytest

from apm import contrasts, diagnostics, estimate
from apm.features import DesignMatrix


def make_design(n=6000, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    k = 6
    beta = np.array([0.6, 0.3, 0.0, 0.0, -0.4, -0.5]) if signal else np.zeros(k)
    beta = beta - beta.mean()
    raw = np.zeros((n, k))
    for i in range(n):
        for h in rng.choice(k, size=2, replace=False):
            raw[i, h] += 1.0
        for h in rng.choice(k, size=2, replace=False):
            raw[i, h] -= 1.0
    eta = raw @ beta
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    basis = contrasts.sum_to_zero_basis(k)
    block = contrasts.reduce_design(raw, basis)
    X = np.hstack([np.ones((n, 1)), block])
    return DesignMatrix(X, y, ["intercept"] + [f"h{i}" for i in range(k - 1)],
                        list(range(k)), basis, slice(1, k), [],
                        [f"m{i}" for i in range(n)])


def test_placebo_destroys_the_hero_signal():
    design = make_design()
    real = estimate.fit_logit(design)
    placebo = estimate.fit_logit(diagnostics.permute_hero_block(design, seed=1))
    real_max = max(abs(v) for v in real.hero_effects.values())
    placebo_max = max(abs(v) for v in placebo.hero_effects.values())
    assert placebo_max < real_max / 3


def test_placebo_preserves_shape_and_outcome():
    design = make_design(n=500)
    permuted = diagnostics.permute_hero_block(design, seed=2)
    assert permuted.X.shape == design.X.shape
    np.testing.assert_array_equal(permuted.y, design.y)


def test_oster_delta_is_zero_when_the_coefficient_is_already_zero():
    assert diagnostics.oster_delta(0.5, 0.01, 0.0, 0.05, 0.10) == pytest.approx(0.0)


def test_oster_delta_grows_when_the_coefficient_is_stable_under_controls():
    # A coefficient that barely moves when controls are added requires a much
    # larger unobserved confounder to overturn.
    stable = diagnostics.oster_delta(0.50, 0.01, 0.49, 0.05, 0.10)
    fragile = diagnostics.oster_delta(0.50, 0.01, 0.20, 0.05, 0.10)
    assert stable > fragile


def test_calibration_table_is_well_calibrated_for_honest_probabilities():
    rng = np.random.default_rng(3)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(int)
    table = diagnostics.calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert np.max(np.abs(table["mean_predicted"] - table["mean_actual"])) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_diagnostics.py -v`
Expected: FAIL with `ImportError: cannot import name 'diagnostics' from 'apm'`

- [ ] **Step 3: Write minimal implementation**

Create `apm/diagnostics.py`:

```python
"""Falsification and sensitivity checks.

A placebo that fails invalidates the pipeline, so these are not optional
extras; they run alongside every reported fit.
"""

import dataclasses

import numpy as np
import pandas as pd


def permute_hero_block(design, seed=0):
    """Shuffle hero rows against outcomes. Coefficients must collapse."""
    rng = np.random.default_rng(seed)
    X = design.X.copy()
    block = X[:, design.hero_slice]
    X[:, design.hero_slice] = block[rng.permutation(len(block))]
    return dataclasses.replace(design, X=X)


def oster_delta(beta_short, r_short, beta_full, r_full, r_max):
    """Oster (2019) proportional-selection ratio.

    Returns the delta at which the coefficient would be driven to zero: how
    strong selection on unobservables would have to be, relative to selection
    on the observed controls. Larger means more robust. Returns inf when the
    controls do not move the coefficient at all.
    """
    numerator = beta_full * (r_full - r_short)
    denominator = (beta_short - beta_full) * (r_max - r_full)
    if denominator == 0:
        return float("inf")
    return abs(numerator / denominator)


def calibration_table(y, p, bins=10):
    """Predicted versus realised frequency, in equal-width probability bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append({
            "bin": b,
            "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
            "mean_actual": float(y[mask].mean()) if mask.any() else np.nan,
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def nested_log_losses(design, fit_fn, blocks):
    """Out-of-sample log loss as each block of columns is added.

    `blocks` maps a label to the slice of columns cumulatively included.
    """
    out = {}
    for label, columns in blocks.items():
        reduced = dataclasses.replace(design, X=design.X[:, columns])
        out[label] = fit_fn(reduced)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_diagnostics.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add apm/diagnostics.py tests/test_apm_diagnostics.py
git commit -m "feat(apm): placebo, Oster bounds and calibration diagnostics"
```

---

### Task 10: CLI wiring

**Files:**
- Create: `apm_main.py`
- Test: `tests/test_apm_main.py`

**Interfaces:**
- Consumes: every `apm` module.
- Produces: `main(argv=None) -> int`, plus `run_specification_a(conn, args) -> int` returning the `run_id`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apm_main.py`:

```python
import numpy as np

import apm_main
import db


def build_fixture(path, n_matches=400):
    """A database large enough for the logit to converge."""
    conn = db.connect(path)
    db.init_schema(conn)
    heroes = [1011, 1022, 1033, 1044, 1055, 1066, 1077, 1088]
    roles = ["Tank", "Tank", "Damage", "Damage", "Support", "Support",
             "Damage", "Support"]
    for hero, role in zip(heroes, roles):
        conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (?,?,?)",
                     (hero, f"H{hero}", role))
    rng = np.random.default_rng(0)
    for m in range(n_matches):
        uid = f"m{m}"
        conn.execute(
            "INSERT INTO matches (match_uid, match_time_stamp,"
            " match_play_duration) VALUES (?,?,700.0)",
            (uid, 1_787_000_000 + m * 600))
        pick0 = rng.choice(heroes, size=6, replace=False)
        pick1 = rng.choice(heroes, size=6, replace=False)
        win0 = int(rng.random() < 0.5)
        for i, hero in enumerate(list(pick0) + list(pick1)):
            camp = 0 if i < 6 else 1
            player = m * 12 + i
            conn.execute(
                "INSERT INTO match_players (match_uid, player_uid, camp,"
                " cur_hero_id, is_win, add_score, new_score)"
                " VALUES (?,?,?,?,?,?,?)",
                (uid, player, camp, int(hero),
                 win0 if camp == 0 else 1 - win0, 10.0, 4510.0))
            conn.execute(
                "INSERT INTO match_player_heroes (match_uid, player_uid,"
                " hero_id, play_time) VALUES (?,?,?,600.0)",
                (uid, player, int(hero)))
    conn.commit()
    conn.close()


def test_cli_runs_end_to_end_and_persists_results(tmp_path):
    path = str(tmp_path / "t.db")
    build_fixture(path)
    assert apm_main.main(["--db-path", path, "--bootstrap-reps", "0"]) == 0
    conn = db.connect(path)
    runs = conn.execute("SELECT COUNT(*) FROM apm_runs").fetchone()[0]
    effects = conn.execute("SELECT COUNT(*) FROM apm_hero_effects").fetchone()[0]
    assert runs == 1
    assert effects == 8


def test_cli_records_the_attribution_rule_it_was_given(tmp_path):
    path = str(tmp_path / "t.db")
    build_fixture(path)
    apm_main.main(["--db-path", path, "--attribution", "W2",
                   "--bootstrap-reps", "0"])
    conn = db.connect(path)
    assert conn.execute(
        "SELECT attribution_rule FROM apm_runs").fetchone()[0] == "W2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apm_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apm_main'`

- [ ] **Step 3: Write minimal implementation**

Create `apm_main.py`:

```python
"""CLI for the hero adjusted plus-minus model.

Usage: python3 apm_main.py --db-path data/rivals.db
"""

import argparse
import dataclasses
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

import db
from apm import attribution, estimate, features, inference, report, sample


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_specification_a(conn, args):
    frame, exclusions = sample.build_sample(conn, args.forfeit_floor)
    if frame.empty:
        raise SystemExit("no matches survived the sample filters")
    design = features.build_design(conn, frame, args.attribution)
    fit = estimate.fit_logit(design)

    errors = np.sqrt(np.diag(fit.hero_cov))
    z = np.array([fit.hero_effects[h] for h in fit.hero_ids]) / np.where(
        errors > 0, errors, np.nan)
    raw_p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    adjusted, _ = inference.benjamini_hochberg(raw_p)
    adjusted_p = dict(zip(fit.hero_ids, adjusted))

    intervals = {}
    if args.bootstrap_reps > 0:
        clusters = inference.player_clusters(conn, design.match_uids)

        def refit(match_uids):
            # Multiplicity matters: a match drawn three times must appear three
            # times, so filtering with isin() alone would silently turn the
            # cluster bootstrap into a subsample bootstrap.
            counts = pd.Series(match_uids).value_counts()
            subset = frame[frame["match_uid"].isin(counts.index)]
            subset = subset.loc[
                subset.index.repeat(subset["match_uid"].map(counts))
            ].reset_index(drop=True)
            sub_design = features.build_design(conn, subset, args.attribution)
            return estimate.fit_logit(sub_design).hero_effects

        draws = inference.cluster_bootstrap(
            design, refit, clusters, n_reps=args.bootstrap_reps)
        lo, hi = inference.percentile_interval(draws)
        intervals = {h: (float(lo[i]), float(hi[i]))
                     for i, h in enumerate(fit.hero_ids)}
    else:
        for i, hero in enumerate(fit.hero_ids):
            half = 1.96 * errors[i]
            intervals[hero] = (fit.hero_effects[hero] - half,
                               fit.hero_effects[hero] + half)

    n_players = conn.execute(
        "SELECT COUNT(*) FROM match_players").fetchone()[0]
    run_id = report.record_run(
        conn, spec="A", attribution_rule=args.attribution,
        forfeit_floor=args.forfeit_floor, n_matches=len(frame),
        n_players=n_players,
        exclusions=json.dumps(dataclasses.asdict(exclusions)),
        git_commit=_git_commit())
    report.write_hero_effects(conn, run_id, fit, intervals, adjusted_p)
    return run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/rivals.db")
    parser.add_argument("--attribution", default="W1",
                        choices=list(attribution.RULES))
    parser.add_argument("--forfeit-floor", type=int,
                        default=sample.DEFAULT_FORFEIT_FLOOR_SECONDS)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args(argv)

    conn = db.connect(args.db_path)
    db.init_schema(conn)
    run_id = run_specification_a(conn, args)
    table = report.hero_table(conn, run_id)
    print(table.to_string(index=False), file=sys.stderr)
    print(f"run_id={run_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apm_main.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Run the full suite and commit**

Run: `python3 -m pytest -q`
Expected: all previous tests still pass

```bash
git add apm_main.py tests/test_apm_main.py
git commit -m "feat(apm): CLI wiring specification A end to end"
```

---

## Deferred to a follow-up plan

These spec items are real requirements but are not blocking a first working
model. Each becomes a task once Task 10 runs against the real database:

- Spec B wired through the CLI at player-match level (Task 6 provides the
  estimator; the player-match feature builder is the missing piece).
- Two-way (match x player) clustered standard errors for Spec B.
- Robustness battery items 3-8: attribution sensitivity, duration strata,
  temporal stability and structural break, rank-band refits at >= 4750,
  forfeit-floor sensitivity, ridge appendix.
- `total_value` = solo effect plus expected team-up contribution.
- Ban rate correlation as external validation.
- Nested out-of-sample evaluation table on the temporal holdout.
