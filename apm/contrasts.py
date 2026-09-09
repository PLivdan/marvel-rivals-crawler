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


def within_role_basis(roles):
    """Block-diagonal basis for {b : sum of b within each role is zero}.

    `roles` is the role label of each hero, in hero order. Returns a
    (k, k - n_roles) matrix with orthonormal columns.

    Why three constraints rather than one. The composition-shape block can
    reproduce the role-count differential *exactly*: choosing delta_s equal to
    the tank count of shape s makes sum_s delta_s c_s identical to t'x, where t
    is the tank indicator. Under a single global sum-to-zero constraint the
    hero block can express that same direction, so the two blocks are collinear
    in it and are separated only by teams whose shape was pooled into "other"
    -- 1,502 of 961,697 team-instances, or 0.16%. The role component of beta is
    then identified off almost nothing, which is why a raw fit put twelve Tanks
    at the top of the table.

    Constraining beta to sum to zero within each role makes it orthogonal to
    every role indicator, so the hero block cannot represent a role-count
    effect at all and composition has nowhere to go but delta, where the full
    sample identifies it. Within-role reporting then becomes structural rather
    than a post-hoc projection of an ill-conditioned fit.

    A role holding a single hero contributes no column: with nothing to compare
    that hero against, its within-role deviation is definitionally zero.
    """
    roles = list(roles)
    k = len(roles)
    if k == 0:
        raise ValueError("need at least one hero to build a within-role basis")

    order = []
    for role in dict.fromkeys(roles):          # first-appearance order, stable
        order.append([i for i, r in enumerate(roles) if r == role])

    blocks = [(idx, sum_to_zero_basis(len(idx))) for idx in order if len(idx) >= 2]
    width = sum(b.shape[1] for _, b in blocks)
    basis = np.zeros((k, width))
    col = 0
    for idx, block in blocks:
        basis[np.ix_(idx, range(col, col + block.shape[1]))] = block
        col += block.shape[1]
    return basis
