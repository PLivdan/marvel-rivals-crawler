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
