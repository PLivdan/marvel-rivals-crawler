"""Inference respecting how the sample was actually produced.

Matches are not independent draws: the crawl enumerates players and harvests
their matches, and a single match belongs to twelve overlapping player clusters.
That is cross-classified rather than nested, so conventional one-way clustering
does not apply. The reported intervals come from resampling *players* and
refitting, which mirrors the sampling process directly.
"""

import warnings

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
    """Percentile bootstrap interval, column-wise.

    A hero absent from every resampled fit leaves an all-NaN column. That is
    a real, reportable fact (the interval is genuinely undefined there), but
    letting numpy's bare "All-NaN slice encountered" warning stand for it
    would be indistinguishable from background noise at 3am. Detect it
    explicitly, name the affected hero columns, and return NaN for exactly
    those without ever handing an all-NaN column to nanpercentile.
    """
    draws = np.asarray(draws, dtype=float)
    n_cols = draws.shape[1]
    all_nan = np.all(np.isnan(draws), axis=0)
    lo = np.full(n_cols, np.nan)
    hi = np.full(n_cols, np.nan)
    if all_nan.any():
        bad = np.flatnonzero(all_nan)
        warnings.warn(
            f"percentile_interval: {bad.size} of {n_cols} column(s) have no "
            f"non-NaN bootstrap draws (indices {bad.tolist()}); returning "
            "NaN for those intervals.",
            RuntimeWarning,
        )
    good = ~all_nan
    if good.any():
        lo[good] = np.nanpercentile(draws[:, good], 100 * alpha / 2, axis=0)
        hi[good] = np.nanpercentile(draws[:, good], 100 * (1 - alpha / 2), axis=0)
    return lo, hi
