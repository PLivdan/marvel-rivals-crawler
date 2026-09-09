"""Estimators for the two co-headline specifications.

Deliberately unpenalised. Ridge is the conventional choice for a collinear hero
design, but its coefficients are biased and carry no valid standard errors,
which is fatal when the entire deliverable is a ranked table with intervals.
The collinearity here is structural, not incidental, and is removed exactly by
the sum-to-zero reparameterisation in apm.contrasts.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from apm import contrasts


@dataclass
class FitResult:
    params: np.ndarray
    cov: np.ndarray
    column_names: list
    hero_effects: dict
    hero_cov: np.ndarray
    hero_ids: list
    loglike: float
    n: int


def _hero_pieces(design, params, cov):
    gamma = params[design.hero_slice]
    cov_gamma = cov[design.hero_slice, design.hero_slice]
    beta = contrasts.effects_from_free(gamma, design.hero_basis)
    hero_cov = contrasts.cov_from_free(cov_gamma, design.hero_basis)
    return dict(zip(design.hero_ids, beta)), hero_cov


def fit_logit(design):
    model = sm.Logit(design.y, design.X)
    # HC1: spec section 8 calls for heteroskedasticity-robust match-level
    # standard errors. These analytic errors are the reported intervals
    # whenever the bootstrap is off (its default -- see apm_main), so they
    # must not be the classical non-robust MLE covariance.
    res = model.fit(disp=0, method="newton", maxiter=200, cov_type="HC1")
    effects, hero_cov = _hero_pieces(design, res.params, res.cov_params())
    return FitResult(
        params=res.params,
        cov=res.cov_params(),
        column_names=design.column_names,
        hero_effects=effects,
        hero_cov=hero_cov,
        hero_ids=design.hero_ids,
        loglike=res.llf,
        n=len(design.y),
    )


def predict_proba(design, fit):
    eta = design.X @ fit.params
    return expit(eta)


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def within_transform(X, y, groups):
    """Demean X and y by group.

    With a single fixed-effect dimension this recovers the OLS slope estimates
    exactly (Frisch-Waugh-Lovell), so no iterative absorption library is needed
    and none is installed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    codes, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    x_sums = np.zeros((len(codes), X.shape[1]))
    np.add.at(x_sums, inverse, X)
    y_sums = np.zeros(len(codes))
    np.add.at(y_sums, inverse, y)
    return X - (x_sums / counts[:, None])[inverse], y - (y_sums / counts)[inverse]


def fit_absorbed_lpm(X, y, groups, hero_ids, hero_basis, hero_slice,
                     column_names):
    """Linear probability model with one absorbed fixed-effect dimension.

    An LPM rather than a fixed-effects logit: with roughly 5.6 matches per
    player, a nonlinear model with a parameter per player is inconsistent
    (incidental parameters). Predicted probabilities can fall outside [0,1];
    that cost is reported as a diagnostic rather than hidden.
    """
    Xw, yw = within_transform(X, y, groups)
    n, k = Xw.shape
    n_groups = len(np.unique(groups))
    xtx = Xw.T @ Xw
    rank = np.linalg.matrix_rank(xtx)
    if rank < k:
        warnings.warn(
            f"within-transformed design is rank-deficient: rank {rank} of {k} "
            f"parameters ({k - rank} unidentified dimension(s)). pinv returns a "
            "minimum-norm solution; coefficients and standard errors along the "
            "unidentified directions are not meaningful.",
            RuntimeWarning,
        )
    xtx_inv = np.linalg.pinv(xtx)
    params = xtx_inv @ (Xw.T @ yw)
    resid = yw - Xw @ params
    dof = max(n - k - n_groups, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * xtx_inv
    if len(hero_ids):
        gamma = params[hero_slice]
        beta = contrasts.effects_from_free(gamma, hero_basis)
        hero_cov = contrasts.cov_from_free(cov[hero_slice, hero_slice], hero_basis)
        effects = dict(zip(hero_ids, beta))
    else:
        effects, hero_cov = {}, np.zeros((0, 0))
    return FitResult(params=params, cov=cov, column_names=column_names,
                     hero_effects=effects, hero_cov=hero_cov,
                     hero_ids=list(hero_ids), loglike=float("nan"), n=n)


def _weighted_loglike(X, y, w, beta):
    """Weighted Bernoulli log-likelihood via logaddexp (never exp(-eta)).

    Zero-weight rows are zeroed out before multiplying by w, not after, so a
    row with an extreme eta (0/1 saturated probability) can never turn into a
    0 * (+/-inf) NaN just because it happens to carry zero weight.
    """
    eta = X @ beta
    ll_i = y * eta - np.logaddexp(0.0, eta)
    contrib = np.where(w > 0, w * ll_i, 0.0)
    return float(np.sum(contrib))


def fit_logit_weighted(X, y, weights, beta0=None, max_iter=25, tol=1e-8):
    """Frequency-weighted logistic regression by IRLS.

    X: (n, k) float array. y: (n,) 0/1. weights: (n,) non-negative; a zero
    weight drops the row; weights need not be integers. beta0: warm start,
    (k,); zeros if None. Returns (beta, info) where info is a dict with keys
    'converged' (bool), 'n_iter' (int), 'loglike' (weighted log-likelihood at
    the solution).

    Built for the player-cluster bootstrap: the design (X, y) is fixed and
    built once, and each replication supplies its own frequency-weight
    vector over the same rows (weight 3 = "this match was drawn three
    times"), warm-started from the full-sample solution so IRLS converges in
    a handful of iterations instead of the ~10-20 a cold start needs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    n, k = X.shape

    beta = np.zeros(k) if beta0 is None else np.array(beta0, dtype=float, copy=True)

    loglike = _weighted_loglike(X, y, w, beta)
    converged = False
    singular = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        eta = X @ beta
        p = expit(eta)
        W = w * p * (1.0 - p)
        r = w * (y - p)

        # X' W X via a single weighted matrix product -- scale X by W once
        # (n, k) and matrix-multiply, rather than materialising an (n, n)
        # diagonal weight matrix. Essential at n ~= 500,000.
        XtWX = X.T @ (X * W[:, None])
        Xtr = X.T @ r

        try:
            delta = np.linalg.solve(XtWX, Xtr)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(XtWX, Xtr, rcond=None)
            singular = True

        beta = beta + delta
        new_loglike = _weighted_loglike(X, y, w, beta)
        max_change = float(np.max(np.abs(delta))) if delta.size else 0.0
        ll_improved = abs(new_loglike - loglike)
        loglike = new_loglike

        if max_change < tol or ll_improved < tol:
            converged = True
            break

    info = {
        "converged": converged,
        "n_iter": n_iter,
        "loglike": loglike,
        "singular": singular,
    }
    return beta, info
