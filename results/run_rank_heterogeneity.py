"""Refit the headline specification within rank terciles (DB step, run once).

Matches are split into terciles of the mean pre-match rank score of their twelve players; the
Spec A+ design is built once and refitted on each tercile by weighted IRLS with HC1 covariance.
Output: results/heterogeneity_rank_W0_specAplus.csv with each hero's within-role APM and SE per
tercile, and the top-minus-bottom difference with its SE."""
import sys, time, json
import numpy as np, pandas as pd
from scipy.special import expit
sys.path.insert(0, ".")
import db
from apm import contrasts, sample
from apm.estimate import fit_logit_weighted
from apm.features import build_design
from apm_main import _drop_zero_variance_extra_columns
t0 = time.time()
conn = db.connect_readonly("data/rivals.db")
frame, _ = sample.build_sample(conn, 240)
design = _drop_zero_variance_extra_columns(build_design(conn, frame, "W0", min_shape_count=100, constraint="within_role", map_intercepts=True))
X, y = design.X.astype(float), design.y.astype(float); uids = list(design.match_uids); n = len(y)
score = dict(conn.execute("SELECT match_uid, AVG(new_score-add_score) FROM match_players GROUP BY match_uid"))
ms = np.array([score[u] for u in uids]); edges = np.quantile(ms, [1/3, 2/3])
terc = np.digitize(ms, edges)
labels = {0: f"bottom third (<{edges[0]:.0f})", 1: f"middle third ({edges[0]:.0f}-{edges[1]:.0f})", 2: f"top third (>{edges[1]:.0f})"}
print(f"[{time.time()-t0:.0f}s] design {X.shape}; tercile edges {edges.round(0)}", flush=True)
info = {h: (nm, r) for h, nm, r in conn.execute("SELECT hero_id, name, role FROM hero_info")}
hs, B = design.hero_slice, design.hero_basis
beta_full, _ = fit_logit_weighted(X, y, np.ones(n))
out = pd.DataFrame({"hero_id": design.hero_ids, "name": [info[h][0] for h in design.hero_ids], "role": [info[h][1] for h in design.hero_ids]})
for t in range(3):
    w = (terc == t).astype(float)
    beta, inf = fit_logit_weighted(X, y, w, beta0=beta_full)
    p = expit(X @ beta); W = w * p * (1 - p); e2 = w * (y - p) ** 2
    H = X.T @ (X * W[:, None]); M = X.T @ (X * e2[:, None]); Hinv = np.linalg.inv(H)
    cov = Hinv @ M @ Hinv * w.sum() / (w.sum() - X.shape[1])
    eff = contrasts.effects_from_free(beta[hs], B) * 25.0
    se = np.sqrt(np.diag(contrasts.cov_from_free(cov[hs, hs], B))) * 25.0
    out[f"apm_t{t}"] = eff; out[f"se_t{t}"] = se
    print(f"[{time.time()-t0:.0f}s] tercile {t} {labels[t]}: n={int(w.sum()):,}, converged {inf['converged']} in {inf['n_iter']} iters", flush=True)
out["top_minus_bottom"] = out.apm_t2 - out.apm_t0
out["se_diff"] = np.sqrt(out.se_t2 ** 2 + out.se_t0 ** 2)
out["z_diff"] = out.top_minus_bottom / out.se_diff
out.to_csv("results/heterogeneity_rank_W0_specAplus.csv", index=False)
json.dump({"edges": edges.tolist(), "labels": labels, "n": {str(t): int((terc == t).sum()) for t in range(3)}}, open("results/heterogeneity_rank_meta.json", "w"), indent=1)
for nm in ("Elsa Bloodstone", "Black Cat", "Magik", "The Thing", "Mantis"):
    r = out[out.name == nm].iloc[0]
    print(f"{nm:16s} bottom {r.apm_t0:+.2f} ({r.se_t0:.2f})  middle {r.apm_t1:+.2f} ({r.se_t1:.2f})  top {r.apm_t2:+.2f} ({r.se_t2:.2f})  top-bottom {r.top_minus_bottom:+.2f} (z {r.z_diff:+.1f})")
print("largest top-minus-bottom:"); print(out.sort_values("top_minus_bottom", ascending=False).head(8)[["name", "role", "apm_t0", "apm_t2", "top_minus_bottom", "z_diff"]].round(2).to_string(index=False))
print("most negative:"); print(out.sort_values("top_minus_bottom").head(8)[["name", "role", "apm_t0", "apm_t2", "top_minus_bottom", "z_diff"]].round(2).to_string(index=False))
print(f"[{time.time()-t0:.0f}s] done", flush=True)
