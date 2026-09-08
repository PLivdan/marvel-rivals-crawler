import numpy as np
import pytest

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

    # Pin the exact dof arithmetic: cov = sigma2 * pinv(Xw'Xw), where sigma2
    # uses dof = n - k - n_groups = 500 - 2 - 100 = 398. Recompute sigma2 and
    # cov independently (by hand, from the demeaned residuals) and compare
    # against fit.cov with a tight tolerance, so a regression to the wrong dof
    # (e.g. n - k = 498, omitting n_groups) changes the expected value and the
    # assertion fails, unlike the mere positivity check above.
    Xw, yw = estimate.within_transform(X, y, groups)
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    params = xtx_inv @ (Xw.T @ yw)
    resid = yw - Xw @ params
    n, k, n_groups = 500, 2, 100
    expected_dof = n - k - n_groups
    assert expected_dof == 398
    expected_sigma2 = float(resid @ resid) / expected_dof
    expected_cov = expected_sigma2 * xtx_inv
    np.testing.assert_allclose(fit.cov, expected_cov, rtol=1e-10)


def test_fit_absorbed_lpm_warns_on_rank_deficient_design():
    # A column that is an exact duplicate of another stays rank-deficient
    # after group-demeaning (demeaning is linear, so identical columns remain
    # identical). This mirrors the real risk in Spec B: with ~5.6 matches per
    # player against 55 heroes, a player who only ever plays one hero demeans
    # to zero on that hero's column, and pinv would otherwise silently hand
    # back a minimum-norm "fit" for an unidentified direction.
    rng = np.random.default_rng(3)
    n_groups, per = 50, 4
    groups = np.repeat(np.arange(n_groups), per)
    n = n_groups * per
    x1 = rng.normal(size=n)
    X = np.column_stack([x1, x1])
    y = rng.normal(size=n)
    with pytest.warns(RuntimeWarning, match="rank-deficient"):
        estimate.fit_absorbed_lpm(
            X, y, groups, hero_ids=[], hero_basis=np.zeros((0, 0)),
            hero_slice=slice(0, 0), column_names=["a", "b"])
