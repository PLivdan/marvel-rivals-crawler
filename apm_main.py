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
