"""Pilot: fit the penalized model on the 40k test design, time it, and verify (a) eta_lineups reproduces the
cached design's eta, (b) swapping camps complements the prediction."""
import sys, time, numpy as np
sys.path.insert(0, ".")
from lineup.fit import LineupDesign
from scipy.special import expit
d = LineupDesign(sys.argv[1] if len(sys.argv) > 1 else "results/lineup_design_test")
print("P (free parameters):", d.P, {k: s.stop - s.start for k, s in d.slices.items()})
w = np.ones(d.n); w[int(0.8 * d.n):] = 0.0                     # last 20% by play time held out
lams = [1.0, 30.0, 30.0, 30.0, 300.0]
t = time.time(); fit = d.fit(lams, w); print(f"fit: {fit['seconds']:.0f}s, {fit['n_iter']} newton steps, converged {fit['converged']}")
val = np.nonzero(w == 0)[0]; ev = d.evaluate(fit, val); ev.pop("loss_vector")
print("validation:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()})
# constant-prediction and headline-like references on the same validation rows
p0 = d.y[w > 0].mean(); print("constant log loss:", round(float(-np.mean(d.y[val] * np.log(p0) + (1 - d.y[val]) * np.log(1 - p0))), 4))
# consistency and symmetry
rows = np.random.default_rng(1).choice(d.n, 3000, replace=False)
e1 = d.eta(fit["phi"], fit["scalers"], rows)
e2 = d.eta_lineups(fit["phi"], fit["scalers"], d.s0[rows], d.s1[rows], d.map_idx[rows], d.lobby[rows], d.drank[rows])
print("eta rebuilt from lineups matches cached design:", np.allclose(e1, e2, atol=1e-8), f"max|diff| {np.max(np.abs(e1-e2)):.2e}")
e3 = d.eta_lineups(fit["phi"], fit["scalers"], d.s1[rows], d.s0[rows], d.map_idx[rows], d.lobby[rows], -d.drank[rows])
alpha = d.native(fit["phi"])["maps"][d.map_idx[rows]]
print("camp swap (same map side) flips everything except the map intercept:", np.allclose(e1 - alpha, -(e3 - alpha), atol=1e-8))
np.savez("results/lineup_pilot_fit.npz", phi=fit["phi"])
