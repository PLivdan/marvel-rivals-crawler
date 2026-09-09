"""Out-of-sample fit quality for the headline specification (DB step, run once).

Temporal split: the last 7 days of matches are held out (a random split would
leak meta drift). The design is built once on the full sample so columns are
identical, then the model is fitted on the training rows only and evaluated
on the holdout. Produces:
  results/calibration_<tag>.csv    predicted vs realised win frequency, 25 equal-count bins,
                                   for the holdout (out-of-sample) and the full-sample fit
  results/fit_quality_<tag>.json   holdout log loss / Brier / AUC / calibration slope, and a
                                   nested ladder of holdout log losses as blocks are added
The report generator reads these files; it never touches the database.
"""
import json, sys, time
import numpy as np, pandas as pd
from scipy.special import expit, logit
sys.path.insert(0, ".")
import db
from apm import features, sample
from apm.estimate import fit_logit_weighted, log_loss
from apm_main import _drop_zero_variance_extra_columns

TAG = sys.argv[1] if len(sys.argv) > 1 else "W0_specAplus"
HOLDOUT_DAYS, BINS = 7, 25
t0 = time.time()
conn = db.connect_readonly("data/rivals.db")
frame, _ = sample.build_sample(conn, 240)
train, test = sample.temporal_holdout(frame, HOLDOUT_DAYS)
print(f"[{time.time()-t0:.0f}s] sample {len(frame):,}: train {len(train):,}, holdout {len(test):,} "
      f"(last {HOLDOUT_DAYS} days)", flush=True)
design = _drop_zero_variance_extra_columns(
    features.build_design(conn, frame, "W0", min_shape_count=100, constraint="within_role", map_intercepts=True))
X, y, names = design.X.astype(float), design.y.astype(float), list(design.column_names)
test_set = set(test.match_uid)
is_test = np.fromiter((u in test_set for u in design.match_uids), bool, len(design.match_uids))
w_train = (~is_test).astype(float)
print(f"[{time.time()-t0:.0f}s] design {X.shape}; holdout rows {is_test.sum():,}", flush=True)

def block(prefixes):
    return [i for i, c in enumerate(names) if any(c.startswith(p) for p in prefixes)]
LADDER = [("Map intercepts only", ["map_", "intercept"]),
          ("+ rank-score differential", ["skill_"]),
          ("+ team-composition shapes", ["shape_"]),
          ("+ hero contrasts", ["hero_"]),
          ("+ team-up contrasts", ["teamup_"])]
ladder, cols = [], []
for label, pref in LADDER:
    cols = sorted(set(cols) | set(block(pref)))
    b, info = fit_logit_weighted(X[:, cols], y, w_train)
    p = expit(X[:, cols] @ b)
    ladder.append({"model": label, "k": len(cols), "converged": bool(info["converged"]),
                   "loglik_train": info["loglike"],
                   "logloss_train": log_loss(y[~is_test], p[~is_test]),
                   "logloss_holdout": log_loss(y[is_test], p[is_test])})
    print(f"[{time.time()-t0:.0f}s] {label:28s} k={len(cols):4d} holdout log loss {ladder[-1]['logloss_holdout']:.5f}", flush=True)
assert cols == list(range(X.shape[1])), "ladder did not exhaust the design columns"
p_hold, y_hold = p[is_test], y[is_test]

def quantile_bins(p, y, bins):
    order = np.argsort(p); groups = np.array_split(order, bins); rows = []
    for g in groups:
        a = y[g].mean(); n = len(g)
        rows.append({"mean_predicted": p[g].mean(), "mean_actual": a, "n": n,
                     "se": np.sqrt(max(a * (1 - a), 1e-12) / n)})
    return pd.DataFrame(rows)
cal = quantile_bins(p_hold, y_hold, BINS); cal["sample"] = "holdout"
b_full, _ = fit_logit_weighted(X, y, np.ones(len(y)), beta0=b)
p_full = expit(X @ b_full)
cal_in = quantile_bins(p_full, y, BINS); cal_in["sample"] = "in_sample"
pd.concat([cal, cal_in]).to_csv(f"results/calibration_{TAG}.csv", index=False)

# Cox calibration: y ~ a + c*logit(p) on the holdout; ideal (a, c) = (0, 1)
Z = np.column_stack([np.ones(len(p_hold)), logit(np.clip(p_hold, 1e-9, 1 - 1e-9))])
bc, _ = fit_logit_weighted(Z, y_hold, np.ones(len(y_hold)))
pc = expit(Z @ bc); V = np.linalg.inv(Z.T @ (Z * (pc * (1 - pc))[:, None]))
# AUC by rank statistic
r = pd.Series(p_hold).rank().to_numpy(); n1 = y_hold.sum(); n0 = len(y_hold) - n1
auc = (r[y_hold == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
out = {"tag": TAG, "holdout_days": HOLDOUT_DAYS, "n_train": int((~is_test).sum()), "n_holdout": int(is_test.sum()),
       "holdout_start": int(test.timestamp.min()), "holdout_end": int(test.timestamp.max()),
       "logloss_holdout": log_loss(y_hold, p_hold), "logloss_holdout_constant": log_loss(y_hold, np.full_like(y_hold, y[~is_test].mean())),
       "brier_holdout": float(np.mean((p_hold - y_hold) ** 2)), "auc_holdout": float(auc),
       "calibration_intercept": float(bc[0]), "calibration_intercept_se": float(np.sqrt(V[0, 0])),
       "calibration_slope": float(bc[1]), "calibration_slope_se": float(np.sqrt(V[1, 1])),
       "holdout_base_rate": float(y_hold.mean()), "p_holdout_min": float(p_hold.min()), "p_holdout_max": float(p_hold.max()),
       "share_holdout_between_35_65": float(np.mean((p_hold > .35) & (p_hold < .65))),
       "bins": BINS, "ladder": ladder}
json.dump(out, open(f"results/fit_quality_{TAG}.json", "w"), indent=2)
print(f"[{time.time()-t0:.0f}s] done. holdout log loss {out['logloss_holdout']:.5f} vs constant "
      f"{out['logloss_holdout_constant']:.5f}; AUC {auc:.4f}; calibration slope {bc[1]:.3f} ({np.sqrt(V[1,1]):.3f})", flush=True)
