"""Estimators for the two co-headline specifications.

Deliberately unpenalised. Ridge is the conventional choice for a collinear hero
design, but its coefficients are biased and carry no valid standard errors,
which is fatal when the entire deliverable is a ranked table with intervals.
The collinearity here is structural, not incidental, and is removed exactly by
the sum-to-zero reparameterisation in apm.contrasts.
"""

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
    res = model.fit(disp=0, method="newton", maxiter=200)
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
