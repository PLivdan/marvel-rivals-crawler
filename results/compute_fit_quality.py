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
from apm.features import _lineup_rosters
HERO = {h: (n, r) for h, n, r in conn.execute("SELECT hero_id, name, role FROM hero_info")}
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
    if label.startswith("+ team-composition"):
        p_nohero = p.copy()
    ladder.append({"model": label, "k": len(cols), "converged": bool(info["converged"]),
                   "loglik_train": info["loglike"],
                   "logloss_train": log_loss(y[~is_test], p[~is_test]),
                   "logloss_holdout": log_loss(y[is_test], p[is_test])})
    print(f"[{time.time()-t0:.0f}s] {label:28s} k={len(cols):4d} holdout log loss {ladder[-1]['logloss_holdout']:.5f}", flush=True)
assert cols == list(range(X.shape[1])), "ladder did not exhaust the design columns"
p_hold, y_hold = p[is_test], y[is_test]

# --- out-of-sample aggregates by starting hero and by team composition -------
# Each holdout match contributes two team-sides. A side's predicted win
# probability is p (camp 0) or 1-p (camp 1); realised is y or 1-y. Every
# starting hero on that side (duplicates count twice) and the side's shape
# receive that pair. The no-hero model (maps + skill + shapes) is the benchmark.
hold_uids = [u for u, t in zip(design.match_uids, is_test) if t]
rosters = _lineup_rosters(conn, hold_uids, "W0")
p_by, pn_by, y_by = ({u: v for u, v in zip(design.match_uids, arr)} for arr in (p, p_nohero, y))
acc = {}
def add(level, key, model, pred, real):
    a = acc.setdefault((level, key, model), [0, 0.0, 0.0]); a[0] += 1; a[1] += pred; a[2] += real
for (uid, camp), heroes in rosters.items():
    sgn = 1 if camp == 0 else -1
    pairs = {"full": (0.5 + sgn * (p_by[uid] - 0.5), 0.5 + sgn * (y_by[uid] - 0.5)),
             "no_hero": (0.5 + sgn * (pn_by[uid] - 0.5), 0.5 + sgn * (y_by[uid] - 0.5))}
    counts = {"Tank": 0, "Damage": 0, "Support": 0}
    for h in heroes:
        role = HERO.get(h, (None, None))[1]
        if role in counts:
            counts[role] += 1
        for model, (pr, re_) in pairs.items():
            add("hero", h, model, pr, re_)
    shape = f"{counts['Tank']}-{counts['Damage']}-{counts['Support']}"
    for model, (pr, re_) in pairs.items():
        add("shape", shape, model, pr, re_)
rows = []
for (level, key, model), (n, sp, sr) in acc.items():
    a = sr / n
    rows.append({"level": level, "key": HERO[key][0] if level == "hero" else key,
                 "role": HERO[key][1] if level == "hero" else None, "model": model, "n": n,
                 "mean_predicted": sp / n, "mean_realised": a, "se_realised": np.sqrt(max(a * (1 - a), 1e-12) / n)})
grp = pd.DataFrame(rows).sort_values(["level", "model", "n"], ascending=[True, True, False])
grp.to_csv(f"results/holdout_groups_{TAG}.csv", index=False)
def summary(level, model, min_n=1):
    d = grp[(grp.level == level) & (grp.model == model) & (grp.n >= min_n)]
    return {"n_groups": int(len(d)), "corr": float(np.corrcoef(d.mean_predicted, d.mean_realised)[0, 1]),
            "mean_abs_gap_pp": float(100 * (d.mean_predicted - d.mean_realised).abs().mean()),
            "sd_predicted_pp": float(100 * d.mean_predicted.std()), "sd_realised_pp": float(100 * d.mean_realised.std())}
group_summary = {"hero_full": summary("hero", "full"), "hero_no_hero": summary("hero", "no_hero"),
                 "shape_full": summary("shape", "full", 100), "shape_no_hero": summary("shape", "no_hero", 100)}
print(f"[{time.time()-t0:.0f}s] hero-level holdout: full corr {group_summary['hero_full']['corr']:.3f} "
      f"(no-hero {group_summary['hero_no_hero']['corr']:.3f}); mean |gap| {group_summary['hero_full']['mean_abs_gap_pp']:.2f}pp", flush=True)

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
       "bins": BINS, "ladder": ladder, "groups": group_summary}
json.dump(out, open(f"results/fit_quality_{TAG}.json", "w"), indent=2)
print(f"[{time.time()-t0:.0f}s] done. holdout log loss {out['logloss_holdout']:.5f} vs constant "
      f"{out['logloss_holdout_constant']:.5f}; AUC {auc:.4f}; calibration slope {bc[1]:.3f} ({np.sqrt(V[1,1]):.3f})", flush=True)
