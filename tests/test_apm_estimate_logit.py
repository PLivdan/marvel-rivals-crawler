import numpy as np
import pytest

from apm import contrasts, estimate
from apm.features import DesignMatrix


def synthetic_design(n=40_000, seed=0):
    """Matches with three heroes a side drawn at random and a KNOWN hero
    effect vector. Three rather than six only to keep the fixture small; the
    sum-to-zero structure is identical. Recovering planted coefficients is the
    only honest test of an estimator."""
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
