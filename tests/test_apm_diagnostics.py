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
