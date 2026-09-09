"""SPIKE -- throwaway. Is Specification C tractable, and what does it do to beta?

Fits the full model with twelve player effects per match: dense block D (map
intercepts, hero contrasts, team-ups, shapes, skill) unpenalised, plus a sparse
player block P (+1 own side, -1 opposing) under a ridge penalty. Objective is
minimised by L-BFGS with two matvecs per evaluation. Reports timing, the fitted
hero effects, and how far they move from Spec A+.
"""
import sys, time, json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit
sys.path.insert(0, ".")
import db
from apm import sample, features, contrasts
from apm_main import _drop_zero_variance_extra_columns

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
t_all = time.time()
conn = db.connect_readonly("data/rivals.db")
frame, _ = sample.build_sample(conn, 240)
design = features.build_design(conn, frame, "W0", min_shape_count=100,
                               constraint="within_role", map_intercepts=True)
design = _drop_zero_variance_extra_columns(design)
D = design.X.astype(np.float64); y = design.y.astype(np.float64)
n, kd = D.shape
print(f"[{time.time()-t_all:.0f}s] dense block {D.shape}", flush=True)

# Sparse player block: one column per distinct player in the sample.
rows = conn.execute("SELECT match_uid, player_uid, camp FROM match_players").fetchall()
order = {uid: i for i, uid in enumerate(design.match_uids)}
players = {}
ri, ci, vv = [], [], []
for muid, puid, camp in rows:
    r = order.get(muid)
    if r is None: continue
    c = players.setdefault(puid, len(players))
    ri.append(r); ci.append(c); vv.append(1.0 if camp == 0 else -1.0)
P = sp.csr_matrix((vv, (ri, ci)), shape=(n, len(players)))
kp = P.shape[1]
print(f"[{time.time()-t_all:.0f}s] player block {P.shape}, nnz {P.nnz:,}", flush=True)

def f_and_g(theta):
    b, a = theta[:kd], theta[kd:]
    eta = D @ b + P @ a
    p = expit(eta)
    # log-loss in a numerically safe form
    nll = np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * LAM * a @ a
    r = p - y
    g = np.concatenate([D.T @ r, P.T @ r + LAM * a])
    return nll, g

theta0 = np.zeros(kd + kp)
t0 = time.time()
res = minimize(f_and_g, theta0, jac=True, method="L-BFGS-B",
               options={"maxiter": 2000, "maxfun": 4000, "ftol": 1e-10, "gtol": 1e-6})
fit_s = time.time() - t0
print(f"[{fit_s:.0f}s] L-BFGS: {res.message}, nit={res.nit}, nfev={res.nfev}, "
      f"penalised nll={res.fun:,.1f}", flush=True)

b, a = res.x[:kd], res.x[kd:]
beta_C = contrasts.effects_from_free(b[design.hero_slice], design.hero_basis)
loglik_C = -np.sum(np.logaddexp(0.0, D @ b + P @ a) - y * (D @ b + P @ a))
print(f"unpenalised loglik at optimum: {loglik_C:,.1f}  (Spec A+ was -320,067.7)")
print(f"player effects: sd {a.std():.4f}, |a|>0.5 for {(np.abs(a)>0.5).sum():,} players, "
      f"max |a| {np.abs(a).max():.3f}")

import pandas as pd
A = pd.read_csv("results/apm_hero_table_W0_specAplus.csv").set_index("hero_id")
out = pd.DataFrame({"hero_id": design.hero_ids, "beta_C_pp": beta_C * 25.0})
out["beta_A_pp"] = out["hero_id"].map(A["within_role_pp"]); out["name"] = out["hero_id"].map(A["name"])
out["role"] = out["hero_id"].map(A["role"]); out["shift_pp"] = out["beta_C_pp"] - out["beta_A_pp"]
from scipy import stats
print(f"\nA -> C: Spearman {stats.spearmanr(out.beta_A_pp, out.beta_C_pp)[0]:.3f}, "
      f"sd A {out.beta_A_pp.std():.2f} -> C {out.beta_C_pp.std():.2f}, "
      f"mean |shift| {out.shift_pp.abs().mean():.2f}pp")
print("\nlargest movers (A -> C):")
for _, r in out.reindex(out.shift_pp.abs().sort_values(ascending=False).index).head(8).iterrows():
    print(f"  {r['name']:<22}{r['role']:<9}{r.beta_A_pp:+6.2f} -> {r.beta_C_pp:+6.2f}  ({r.shift_pp:+.2f})")
out.to_csv(f"results/spike_specC_lam{LAM}.csv", index=False)
json.dump({"lambda": LAM, "n": int(n), "k_dense": int(kd), "k_players": int(kp),
           "fit_seconds": fit_s, "nit": int(res.nit), "loglik": float(loglik_C),
           "converged": bool(res.success)}, open(f"results/spike_specC_lam{LAM}.json", "w"), indent=1)
print(f"\n[{time.time()-t_all:.0f}s total]")
