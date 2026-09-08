"""CLI for the hero adjusted plus-minus model.

Usage: python3 apm_main.py --db-path data/rivals.db
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import warnings

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


def _hero_adjusted_p_values(hero_ids, hero_effects, hero_cov):
    """BH-adjusted p-values per hero, quarantining degenerate heroes first.

    benjamini_hochberg's step-up procedure runs a cumulative minimum over the
    sorted p-values (apm/inference.py), so a single NaN — from a hero whose
    standard error is exactly zero, giving z = NaN via the errors>0 guard
    below — propagates through np.minimum.accumulate and silently blanks
    every OTHER hero's adjusted p-value too, not just its own. Excluding such
    heroes from the input entirely (and reporting NaN only for them) keeps
    one degenerate hero from poisoning the whole family-wise result.
    """
    errors = np.sqrt(np.diag(hero_cov))
    z = np.array([hero_effects[h] for h in hero_ids]) / np.where(
        errors > 0, errors, np.nan)
    raw_p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    finite = np.isfinite(raw_p)

    adjusted_p = {h: float("nan") for h in hero_ids}
    if not finite.all():
        excluded = [h for h, ok in zip(hero_ids, finite) if not ok]
        warnings.warn(
            f"_hero_adjusted_p_values: excluding {len(excluded)} hero(es) "
            f"with a non-finite p-value from the BH adjustment (hero_id(s) "
            f"{excluded}); their standard error was zero or otherwise "
            "degenerate. Reporting p_adjusted=NaN for those heroes rather "
            "than letting them blank every other hero's result.",
            RuntimeWarning,
        )
    if finite.any():
        adjusted_finite, _ = inference.benjamini_hochberg(raw_p[finite])
        for h, p in zip(np.asarray(hero_ids)[finite], adjusted_finite):
            adjusted_p[h] = float(p)
    return adjusted_p


def _count_sample_players(conn, match_uids):
    """Count match_players rows scoped to exactly the sample's matches.

    apm_runs exists so a coefficient can be traced back to the exact sample
    that produced it, and n_matches already varies per run with the sample
    filters (--forfeit-floor and the roster/draw/score/playtime exclusions in
    apm.sample). n_players must vary the same way or it is false precision:
    a static whole-database count sitting next to a real per-run n_matches.

    At the ~200,000-match scale this runs at, a single parameterized
    `WHERE match_uid IN (...)` would either blow past SQLite's bound
    parameter limit or need manual chunking. Loading the sample's match ids
    into a temp table and joining does it as one query with no such limit;
    the join is indexed on both sides (match_uid is the temp table's primary
    key and leads match_players' composite primary key).
    """
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS _apm_sample_matches "
        "(match_uid TEXT PRIMARY KEY)"
    )
    conn.execute("DELETE FROM _apm_sample_matches")
    conn.executemany(
        "INSERT INTO _apm_sample_matches (match_uid) VALUES (?)",
        [(uid,) for uid in match_uids],
    )
    count = conn.execute(
        "SELECT COUNT(*) FROM match_players p "
        "JOIN _apm_sample_matches s ON s.match_uid = p.match_uid"
    ).fetchone()[0]
    conn.execute("DROP TABLE _apm_sample_matches")
    return count


def run_specification_a(conn, args):
    frame, exclusions = sample.build_sample(conn, args.forfeit_floor)
    if frame.empty:
        raise SystemExit("no matches survived the sample filters")
    design = features.build_design(conn, frame, args.attribution)
    fit = estimate.fit_logit(design)

    errors = np.sqrt(np.diag(fit.hero_cov))
    adjusted_p = _hero_adjusted_p_values(
        fit.hero_ids, fit.hero_effects, fit.hero_cov)

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

    n_players = _count_sample_players(conn, frame["match_uid"])
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
