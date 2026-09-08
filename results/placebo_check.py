"""Placebo and sanity checks for a fitted APM run.

Nothing in apm_main.py runs these — the full robustness battery is deferred to
the follow-up plan — but a placebo that does not collapse invalidates the whole
pipeline, so no result should be reported before this passes.

Three checks:

1. PLACEBO. Permute the hero block against outcomes and refit. Real hero
   effects must be substantially larger than permuted ones. If shuffled data
   still produces large hero effects, the estimator is fitting noise structure
   rather than hero strength and every number in the table is meaningless.

2. SUM-TO-ZERO. Recovered effects must sum to zero by construction. A non-zero
   sum means the contrast basis was applied wrongly somewhere in the chain.

3. AGREEMENT WITH RAW WIN RATES. The play-time-weighted raw win rate is a crude
   but assumption-light measure of the same thing. The APM estimates should
   correlate positively with it. They should NOT match it exactly — the whole
   point of adjustment is to remove skill and lineup confounding — but a
   near-zero or negative rank correlation means the adjustment is doing
   something other than adjusting.

Usage: python3 results/placebo_check.py
"""

import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, ".")

import db
from apm import diagnostics, estimate, features, sample
from apm_main import _drop_zero_variance_extra_columns


def main():
    conn = db.connect_readonly("data/rivals.db")

    t0 = time.time()
    frame, report = sample.build_sample(conn, 240)
    print(f"sample: {len(frame):,} matches ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    design = features.build_design(conn, frame, "W1")
    before = design.X.shape[1]
    # Returns one design; the names it drops go to stderr, not a return value.
    design = _drop_zero_variance_extra_columns(design)
    print(
        f"design: {design.X.shape} ({time.time() - t0:.0f}s), "
        f"dropped {before - design.X.shape[1]} zero-variance columns",
        flush=True,
    )

    t0 = time.time()
    real = estimate.fit_logit(design)
    print(f"real fit: {time.time() - t0:.0f}s", flush=True)

    real_effects = np.array([real.hero_effects[h] for h in real.hero_ids])
    print("\n=== CHECK 2: sum-to-zero ===")
    print(f"  sum of hero effects = {real_effects.sum():.3e} (must be ~0)")

    print("\n=== CHECK 1: placebo ===")
    t0 = time.time()
    placebo = estimate.fit_logit(diagnostics.permute_hero_block(design, seed=1))
    placebo_effects = np.array([placebo.hero_effects[h] for h in placebo.hero_ids])
    print(f"  placebo fit: {time.time() - t0:.0f}s")
    real_max = np.abs(real_effects).max()
    placebo_max = np.abs(placebo_effects).max()
    print(f"  real max |effect|    = {real_max:.4f}")
    print(f"  placebo max |effect| = {placebo_max:.4f}")
    print(f"  ratio placebo/real   = {placebo_max / real_max:.3f}")
    print(f"  VERDICT: {'PASS' if placebo_max < real_max / 3 else 'FAIL — DO NOT REPORT'}")

    print("\n=== CHECK 3: agreement with play-time-weighted raw win rates ===")
    raw = dict(
        conn.execute(
            """SELECT h.hero_id,
                      100.0*SUM(h.play_time*mp.is_win)/NULLIF(SUM(h.play_time),0)
               FROM match_player_heroes h
               JOIN match_players mp ON mp.match_uid=h.match_uid
                                    AND mp.player_uid=h.player_uid
               WHERE mp.is_win IN (0,1) AND h.play_time>0
               GROUP BY h.hero_id"""
        ).fetchall()
    )
    ids = [h for h in real.hero_ids if h in raw]
    apm_v = [real.hero_effects[h] for h in ids]
    raw_v = [raw[h] for h in ids]
    rho, p = stats.spearmanr(apm_v, raw_v)
    print(f"  n = {len(ids)} heroes")
    print(f"  Spearman rho = {rho:.3f} (p = {p:.2e})")
    print(f"  VERDICT: {'PASS' if rho > 0.3 else 'INVESTIGATE — adjustment may be misbehaving'}")


if __name__ == "__main__":
    main()
