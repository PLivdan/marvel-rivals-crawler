import numpy as np
import pytest

from apm import contrasts


def test_basis_has_orthonormal_columns_that_sum_to_zero():
    basis = contrasts.sum_to_zero_basis(5)
    assert basis.shape == (5, 4)
    np.testing.assert_allclose(basis.sum(axis=0), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)


def test_effects_recovered_from_free_parameters_sum_to_zero():
    basis = contrasts.sum_to_zero_basis(6)
    gamma = np.array([0.4, -1.1, 0.25, 0.9, -0.3])
    beta = contrasts.effects_from_free(gamma, basis)
    assert beta.shape == (6,)
    assert abs(beta.sum()) < 1e-12


def test_reduced_design_preserves_the_linear_predictor():
    # X rows sum to zero across columns, exactly as hero contrasts do because
    # both teams field six heroes. The reduced design must give an identical
    # linear predictor, which is what makes the reparameterisation lossless.
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 6))
    X = raw - raw.mean(axis=1, keepdims=True)
    basis = contrasts.sum_to_zero_basis(6)
    gamma = rng.normal(size=5)
    beta = contrasts.effects_from_free(gamma, basis)
    np.testing.assert_allclose(
        contrasts.reduce_design(X, basis) @ gamma, X @ beta, atol=1e-10
    )


def test_covariance_transform_is_symmetric_and_positive_semidefinite():
    basis = contrasts.sum_to_zero_basis(4)
    a = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 0.9]])
    cov = contrasts.cov_from_free(a, basis)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_basis_rejects_degenerate_size():
    with pytest.raises(ValueError):
        contrasts.sum_to_zero_basis(1)


def test_within_role_basis_sums_to_zero_inside_each_role():
    """Three constraints, not one.

    The shape block can reproduce the role-count differential EXACTLY (set
    delta_s = tankcount(s) and sum_s delta_s c_s equals t'x), so with a single
    global sum-to-zero constraint the hero and shape blocks are collinear in the
    role-count directions and differ only on teams in the pooled "other" bucket
    -- 1,502 of 961,697 team-instances. Constraining beta to sum to zero WITHIN
    each role removes those directions from the hero block entirely, forcing all
    composition into delta where it is estimable from the full sample.
    """
    roles = ["Tank", "Tank", "Tank", "Damage", "Damage", "Support", "Support"]
    basis = contrasts.within_role_basis(roles)
    assert basis.shape == (7, 7 - 3)          # k - number of roles
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    gamma = np.array([0.7, -0.2, 1.1, 0.4])
    beta = contrasts.effects_from_free(gamma, basis)
    for role in ("Tank", "Damage", "Support"):
        idx = [i for i, r in enumerate(roles) if r == role]
        assert abs(beta[idx].sum()) < 1e-12, f"{role} does not sum to zero"


def test_within_role_basis_is_orthogonal_to_every_role_indicator():
    """The point of the constraint: beta cannot express "all Tanks get +c",
    so a role-count effect has nowhere to go but the composition block."""
    roles = ["Tank"] * 4 + ["Damage"] * 3
    basis = contrasts.within_role_basis(roles)
    for role in ("Tank", "Damage"):
        indicator = np.array([1.0 if r == role else 0.0 for r in roles])
        np.testing.assert_allclose(indicator @ basis, np.zeros(basis.shape[1]), atol=1e-12)


def test_within_role_basis_pins_a_singleton_role_to_zero():
    """A role with one hero has no within-role deviation to estimate: its
    effect is definitionally zero, so it contributes no free parameter."""
    roles = ["Tank", "Tank", "Damage"]
    basis = contrasts.within_role_basis(roles)
    assert basis.shape == (3, 1)
    beta = contrasts.effects_from_free(np.array([0.9]), basis)
    assert beta[2] == pytest.approx(0.0)
