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
