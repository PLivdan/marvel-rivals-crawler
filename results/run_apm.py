"""Run Specification A against the live database and persist the results.

Why this exists rather than just calling apm_main.py: the crawler is writing to
data/rivals.db continuously from 8 workers, and db.connect's 5s busy_timeout is
not enough to win the write lock at the end of a long run. The first attempt
completed the entire fit and then died with "database is locked" while writing
the run row, destroying the computation.

So this runner does two things differently:

1. It writes the results to CSV/JSON on disk BEFORE attempting any database
   write. Compute that took minutes must never be lost to a lock that takes
   seconds to resolve.
2. It uses a long busy_timeout for the write connection, so it waits the
   crawler out instead of failing.

Usage: python3 results/run_apm.py [--attribution W1] [--tag W1]
"""

import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")

import db
from apm import estimate, features, inference, report, sample
from apm_main import (
    _count_sample_players,
    _drop_zero_variance_extra_columns,
    _git_commit,
    _hero_adjusted_p_values,
)

WRITE_BUSY_TIMEOUT_MS = 600_000  # 10 minutes; the crawler's commits are short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/rivals.db")
    ap.add_argument("--attribution", default="W1")
    ap.add_argument("--forfeit-floor", type=int, default=240)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.attribution

    conn = db.connect_readonly(args.db_path)

    t0 = time.time()
    frame, exclusions = sample.build_sample(conn, args.forfeit_floor)
    print(f"[{time.time()-t0:.0f}s] sample: {len(frame):,} matches", flush=True)

    t0 = time.time()
    design = features.build_design(conn, frame, args.attribution)
    before = design.X.shape[1]
    design = _drop_zero_variance_extra_columns(design)
    dropped = before - design.X.shape[1]
    print(
        f"[{time.time()-t0:.0f}s] design: {design.X.shape}, "
        f"{len(design.hero_ids)} heroes, {len(design.teamup_ids)} team-ups, "
        f"{dropped} zero-variance columns dropped",
        flush=True,
    )

    t0 = time.time()
    fit = estimate.fit_logit(design)
    print(f"[{time.time()-t0:.0f}s] fit converged, loglike={fit.loglike:,.1f}", flush=True)

    errors = np.sqrt(np.diag(fit.hero_cov))
    adjusted_p = _hero_adjusted_p_values(fit.hero_ids, fit.hero_effects, fit.hero_cov)
    intervals = {
        h: (fit.hero_effects[h] - 1.96 * errors[i], fit.hero_effects[h] + 1.96 * errors[i])
        for i, h in enumerate(fit.hero_ids)
    }

    names = dict(conn.execute("SELECT hero_id, name FROM hero_info"))
    roles = dict(conn.execute("SELECT hero_id, role FROM hero_info"))

    # Within-role normalisation.
    #
    # The raw coefficients are dominated by a ROLE-COMPOSITION effect: teams
    # fielding more Tanks win more, and because a role count is a linear
    # function of the hero indicators, that effect lands entirely on the
    # individual hero coefficients. The composition-shape controls cannot
    # absorb it — they are a coarse non-linear summary, not the linear
    # combination that actually carries it. Measured on the first run: role
    # alone explains 64.6% of the variance in the raw effects.
    #
    # So the raw number answers a counterfactual nobody wants ("replace an
    # average hero drawn across ALL roles with this one"), since a real team
    # must keep a role balance. Subtracting each hero's own-role mean gives
    # the interpretable quantity: value relative to an average hero OF THE
    # SAME ROLE. That is a linear contrast, so its covariance follows exactly
    # by the delta method rather than being approximated.
    hero_ids = list(fit.hero_ids)
    k = len(hero_ids)
    role_of = [roles.get(h) for h in hero_ids]
    L = np.eye(k)
    for i in range(k):
        peers = [j for j in range(k) if role_of[j] == role_of[i]]
        for j in peers:
            L[i, j] -= 1.0 / len(peers)
    beta = np.array([fit.hero_effects[h] for h in hero_ids])
    beta_wr = L @ beta
    cov_wr = L @ fit.hero_cov @ L.T
    err_wr = np.sqrt(np.clip(np.diag(cov_wr), 0, None))

    rows = []
    for i, h in enumerate(hero_ids):
        b = float(fit.hero_effects[h])
        lo, hi = intervals[h]
        rows.append({
            "hero_id": h,
            "name": names.get(h),
            "role": roles.get(h),
            "effect_logodds": b,
            "effect_pp": report.to_probability_points(b),
            "std_error": float(errors[i]),
            "ci_low_pp": report.to_probability_points(lo),
            "ci_high_pp": report.to_probability_points(hi),
            "p_adjusted": float(adjusted_p[h]),
            "within_role_logodds": float(beta_wr[i]),
            "within_role_pp": report.to_probability_points(float(beta_wr[i])),
            "within_role_se": float(err_wr[i]),
            "within_role_ci_low_pp": report.to_probability_points(
                float(beta_wr[i] - 1.96 * err_wr[i])),
            "within_role_ci_high_pp": report.to_probability_points(
                float(beta_wr[i] + 1.96 * err_wr[i])),
        })
    table = pd.DataFrame(rows).sort_values("within_role_pp", ascending=False)
    np.savez(
        f"results/apm_cov_{tag}.npz",
        hero_ids=np.array(hero_ids), beta=beta, hero_cov=fit.hero_cov,
        beta_within_role=beta_wr, cov_within_role=cov_wr,
    )

    # Team-up coefficients. These are estimated but were never extracted by
    # apm_main, which reports heroes only. A team-up column is +1 when both
    # members are on camp 0's dominant lineup and -1 when both are on camp 1,
    # so the coefficient is the pair-synergy premium ON TOP OF whatever the two
    # heroes contribute individually.
    tnames = dict(conn.execute("SELECT teamup_id, name FROM teamups"))
    tse = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    trows = []
    for j, name in enumerate(fit.column_names):
        if not name.startswith("teamup_"):
            continue
        tid = int(name[len("teamup_"):])
        trows.append({
            "teamup_id": tid,
            "name": tnames.get(tid),
            "effect_logodds": float(fit.params[j]),
            "effect_pp": report.to_probability_points(float(fit.params[j])),
            "std_error": float(tse[j]),
        })
    if trows:
        pd.DataFrame(trows).sort_values("effect_pp", ascending=False).to_csv(
            f"results/apm_teamup_table_{tag}.csv", index=False)
        print(f"saved results/apm_teamup_table_{tag}.csv ({len(trows)} team-ups)", flush=True)

    # Save BEFORE touching the write lock, so a lock failure cannot destroy this.
    table.to_csv(f"results/apm_hero_table_{tag}.csv", index=False)
    meta = {
        "attribution_rule": args.attribution,
        "forfeit_floor": args.forfeit_floor,
        "n_matches": len(frame),
        "exclusions": dataclasses.asdict(exclusions),
        "zero_variance_columns_dropped": dropped,
        "n_heroes": len(fit.hero_ids),
        "n_teamups": len(design.teamup_ids),
        "loglike": fit.loglike,
        "intercept_logodds": float(fit.params[0]),
        "git_commit": _git_commit(),
        "interval_method": "analytic 1.96*HC1 SE (bootstrap infeasible at this scale)",
    }
    with open(f"results/apm_run_meta_{tag}.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved results/apm_hero_table_{tag}.csv and meta", flush=True)

    print("\n" + table.to_string(index=False), flush=True)

    # Count on the READ-ONLY connection, before opening any writer.
    #
    # The first attempt did this on the write connection and died with
    # "database is locked" despite a 10-minute busy_timeout. The reason is
    # subtle: running a SELECT first leaves the connection in a deferred READ
    # transaction, and when the later INSERT tries to UPGRADE that to a write
    # transaction while another writer (the crawler) has intervened, SQLite
    # returns SQLITE_BUSY *immediately* — busy_timeout is not honoured for
    # lock upgrades, only for acquiring a lock from a clean start. Keeping the
    # reads off the writer means its first statement is already a write.
    t0 = time.time()
    n_players = _count_sample_players(conn, frame["match_uid"])
    print(f"[{time.time()-t0:.0f}s] n_players (participations) = {n_players:,}", flush=True)

    # Persist, waiting the crawler out. Autocommit so each statement takes and
    # releases the write lock cleanly rather than holding one open transaction.
    t0 = time.time()
    try:
        wconn = db.connect(args.db_path)
        wconn.isolation_level = None
        wconn.execute(f"PRAGMA busy_timeout={WRITE_BUSY_TIMEOUT_MS}")
        db.init_schema(wconn)
        run_id = report.record_run(
            wconn,
            spec="A",
            attribution_rule=args.attribution,
            forfeit_floor=args.forfeit_floor,
            n_matches=len(frame),
            n_players=n_players,
            exclusions=json.dumps(dataclasses.asdict(exclusions)),
            git_commit=_git_commit(),
        )
        report.write_hero_effects(wconn, run_id, fit, intervals, adjusted_p)
        print(f"\n[{time.time()-t0:.0f}s] persisted run_id={run_id}", flush=True)
    except Exception as exc:
        # The CSV/JSON/npz above are already on disk, so a lock failure costs
        # nothing but the database copy. Report it, do not fail the run.
        print(f"\nPERSISTENCE FAILED ({exc}); results are on disk regardless",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
