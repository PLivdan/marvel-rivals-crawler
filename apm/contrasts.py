"""Sum-to-zero reparameterisation for rank-deficient design blocks.

Every team fields exactly six heroes, so the hero contrast columns sum to zero
in every row and the all-ones vector lies in the null space of that block. The
same holds for composition shape, since each team has exactly one shape. Rather
than drop a reference category (which leaves the dropped level without a
standard error) we reduce the block onto an orthonormal basis of the sum-to-zero
subspace, fit the free parameters, then map back. The recovered effects satisfy
sum(beta) = 0 by construction, which is exactly the "value relative to the
average hero" normalisation the design calls for.
"""

import numpy as np


def sum_to_zero_basis(k):
    """Orthonormal (k, k-1) basis for {b in R^k : sum(b) = 0}.

    Helmert contrasts: column j puts 1 on the first j+1 levels, -(j+1) on level
    j+1, and 0 elsewhere. Each column sums to zero, and the columns are mutually
    orthogonal, so normalising gives an orthonormal basis.
    """
    if k < 2:
        raise ValueError(f"need at least 2 levels to form contrasts, got {k}")
    basis = np.zeros((k, k - 1))
    for j in range(k - 1):
        basis[: j + 1, j] = 1.0
        basis[j + 1, j] = -(j + 1)
        basis[:, j] /= np.linalg.norm(basis[:, j])
    return basis


def reduce_design(X, basis):
    """Project a rank-deficient block onto its sum-to-zero basis."""
    return np.asarray(X) @ basis


def effects_from_free(gamma, basis):
    """Map free parameters back to per-level effects summing to zero."""
    return basis @ np.asarray(gamma)


def cov_from_free(cov_gamma, basis):
    """Delta-method covariance of the recovered effects."""
    return basis @ np.asarray(cov_gamma) @ basis.T
