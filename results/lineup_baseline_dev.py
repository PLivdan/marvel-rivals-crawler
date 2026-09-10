"""Refit the published headline specification (Spec A+) on the DEVELOPMENT rows of the unified sample
(same matches, legal lineups, play time <= T_CONF), for the locked confirmation comparison and the
development ladder. Predictions for the confirmation slice are STORED, not evaluated (stage 8 opens them)."""
import json, sys, time, numpy as np, pandas as pd
sys.path.insert(0, ".")
import db
from apm import contrasts
from apm.features import build_design
from apm.estimate import fit_logit_weighted
from apm_main import _drop_zero_variance_extra_columns
from scipy.special import expit
T_CONF = 1788519632
t0 = time.time()
meta = json.load(open("results/lineup_design/meta.json")); uids = meta["match_uids"]
snap = pd.read_csv("results/dev_snapshot/matches.csv").set_index("match_uid").loc[uids].reset_index()
conn = db.connect_readonly("data/rivals.db")
design = _drop_zero_variance_extra_columns(build_design(conn, snap, "W0", min_shape_count=100, constraint="within_role", map_intercepts=True))
X, y = design.X.astype(float), design.y.astype(float); ts = snap.set_index("match_uid").loc[list(design.match_uids), "timestamp"].to_numpy()
dev = ts <= T_CONF; conf = ~dev
print(f"[{time.time()-t0:.0f}s] baseline design {X.shape}; dev {dev.sum():,}, confirmation {conf.sum():,}", flush=True)
# development folds identical in definition to lineup_tune.py (by play-time order of dev rows)
order = np.nonzero(dev)[0][np.argsort(ts[dev], kind="stable")]
cuts = [int(len(order) * f) for f in (0.60, 0.73, 0.87, 1.0)]
folds = []
for k in range(3):
    tr, va = order[:cuts[k]], order[cuts[k]:cuts[k + 1]]
    w = np.zeros(len(y)); w[tr] = 1
    b, inf = fit_logit_weighted(X, y, w)
    p = np.clip(expit(X[va] @ b), 1e-12, 1 - 1e-12)
    folds.append({"fold": k, "n_train": int(len(tr)), "n_val": int(len(va)), "logloss": float(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))), "converged": bool(inf["converged"])})
    print(f"[{time.time()-t0:.0f}s] fold {k}: val log loss {folds[-1]['logloss']:.5f}", flush=True)
w = dev.astype(float); b, inf = fit_logit_weighted(X, y, w)
p_conf = expit(X[conf] @ b)
np.savez("results/lineup_baseline_dev.npz", beta=b, conf_uids=np.array(design.match_uids, dtype=object)[conf], p_conf=p_conf, column_names=np.array(design.column_names))
json.dump({"folds": folds, "mean_val_logloss": float(np.mean([f["logloss"] for f in folds])), "n_dev": int(dev.sum()), "n_conf": int(conf.sum()),
           "loglike_dev": inf["loglike"], "converged": bool(inf["converged"]), "k": int(X.shape[1])}, open("results/lineup_baseline_dev.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] baseline dev fit done; confirmation predictions stored (not evaluated)", flush=True)
