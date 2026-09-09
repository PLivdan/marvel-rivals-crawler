"""Synthetic-recovery tests for fit_logit_weighted.

Flat file, plain pytest, no conftest -- matches the style of
tests/test_apm_estimate_logit.py. This estimator underlies the player-cluster
bootstrap: each replication reuses the fixed design and supplies a frequency-
weight vector over the fixed rows (weight 3 = "this match was drawn three
times"), warm-started from the full-sample solution so a replication costs
seconds instead of re-querying and rebuilding the design.
"""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.special import expit

from apm import estimate


def make_design(n, k=6, seed=0, beta_scale=0.6):
    """Plain (non-collinear) logistic design with a known planted beta.

    Unlike the hero-contrast fixture in test_apm_estimate_logit.py, this
    estimator takes a raw (X, y, weights) design directly -- no sum-to-zero
    reparameterisation involved -- so a simple intercept-plus-normal-
    covariates design is the honest fixture.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    beta = rng.normal(scale=beta_scale, size=k)
    eta = X @ beta
    p = expit(eta)
    y = (rng.random(n) < p).astype(float)
    return X, y, beta


def test_unit_weights_reproduce_the_unweighted_fit():
    X, y, _ = make_design(n=20_000, seed=1)
    ones = np.ones(len(y))
    beta, info = estimate.fit_logit_weighted(X, y, ones)
    ref = sm.Logit(y, X).fit(disp=0)
    assert info["converged"]
    np.testing.assert_allclose(beta, ref.params, atol=1e-6)


def test_integer_weights_equal_row_duplication():
    # This is the property the bootstrap depends on: a frequency weight of 3
    # must behave exactly like the row appearing three times.
    X, y, _ = make_design(n=5_000, seed=2)
    rng = np.random.default_rng(3)
    w = rng.integers(0, 4, size=len(y)).astype(float)
    beta_w, info = estimate.fit_logit_weighted(X, y, w)
    Xr = np.repeat(X, w.astype(int), axis=0)
    yr = np.repeat(y, w.astype(int))
    ref = sm.Logit(yr, Xr).fit(disp=0)
    assert info["converged"]
    np.testing.assert_allclose(beta_w, ref.params, atol=1e-6)


def test_zero_weight_rows_are_ignored():
    X, y, _ = make_design(n=5_000, seed=4)
    rng = np.random.default_rng(5)
    keep = rng.random(len(y)) > 0.3
    w = keep.astype(float)
    beta_w, info = estimate.fit_logit_weighted(X, y, w)
    ref = sm.Logit(y[keep], X[keep]).fit(disp=0)
    assert info["converged"]
    assert np.all(np.isfinite(beta_w))
    np.testing.assert_allclose(beta_w, ref.params, atol=1e-6)


def test_planted_coefficients_are_recovered():
    X, y, true_beta = make_design(n=40_000, seed=6)
    ones = np.ones(len(y))
    beta, info = estimate.fit_logit_weighted(X, y, ones)
    assert info["converged"]
    assert np.max(np.abs(beta - true_beta)) < 0.05


def test_warm_start_converges_in_fewer_iterations_than_cold_start():
    X, y, _ = make_design(n=20_000, seed=7)
    ones = np.ones(len(y))
    beta_hat, ref_info = estimate.fit_logit_weighted(X, y, ones)
    assert ref_info["converged"]

    # Poisson(1) weights mimic a player-cluster bootstrap draw: most rows get
    # weight 0-3, mean weight 1, so the full-sample solution is a good but
    # inexact warm start for the resampled fit.
    rng = np.random.default_rng(8)
    w = rng.poisson(1.0, size=len(y)).astype(float)

    beta_cold, info_cold = estimate.fit_logit_weighted(X, y, w)
    beta_warm, info_warm = estimate.fit_logit_weighted(X, y, w, beta0=beta_hat)

    assert info_cold["converged"]
    assert info_warm["converged"]
    assert info_warm["n_iter"] < info_cold["n_iter"]


def test_loglike_matches_statsmodels_at_unit_weights():
    X, y, _ = make_design(n=10_000, seed=9)
    ones = np.ones(len(y))
    beta, info = estimate.fit_logit_weighted(X, y, ones)
    ref = sm.Logit(y, X).fit(disp=0)
    assert info["converged"]
    assert info["loglike"] == pytest.approx(ref.llf, rel=1e-8)
