"""Stage 8: ONE locked confirmation evaluation. Usage: python3 results/lineup_confirm.py <candidate_fit.npz>
Order of operations is the point: (1) write results/lineup_lock.json with the sha256 of the candidate's
specification, penalties and coefficients; (2) only then read the confirmation outcomes; (3) apply the
predeclared criterion from results/confirmation_protocol.json; (4) write results/lineup_confirmation.json.
Refuses to run twice."""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy.special import expit
sys.path.insert(0, ".")
from lineup.fit import LineupDesign
T_CONF = 1788519632
if os.path.exists("results/lineup_confirmation.json"):
    raise SystemExit("confirmation already evaluated; a revised candidate needs a new confirmation window")
fit_path = sys.argv[1]; z = np.load(fit_path, allow_pickle=True); phi, lams, sc = z["phi"], [float(x) for x in z["lams"]], json.loads(str(z["scalers"]))
d = LineupDesign("results/lineup_design")
# ---- 1. lock -------------------------------------------------------------------------------------------
spec = {"design_summary": json.load(open("results/lineup_design/summary.json")), "lams": lams, "scalers": sc,
        "phi_sha256": hashlib.sha256(np.ascontiguousarray(phi).tobytes()).hexdigest(), "fit_file": fit_path,
        "code": {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in ("lineup/design.py", "lineup/fit.py", "lineup/summaries.py", "results/lineup_tune.py")},
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(spec, open("results/lineup_lock.json", "w"), indent=1)
print("locked:", spec["phi_sha256"][:12], spec["locked_at_utc"], flush=True)
# ---- 2. read confirmation outcomes (first and only time) ---------------------------------------------------
conf = np.nonzero(d.ts > T_CONF)[0]; y = d.y[conf]
p_cand = np.clip(expit(d.eta(phi, sc, conf)), 1e-12, 1 - 1e-12)
base = np.load("results/lineup_baseline_dev.npz", allow_pickle=True)
bmap = dict(zip(base["conf_uids"], base["p_conf"]))
uids = [d.meta["match_uids"][i] for i in conf]
p_base = np.clip(np.array([bmap[u] for u in uids]), 1e-12, 1 - 1e-12)
def loss(p): return -(y * np.log(p) + (1 - y) * np.log(1 - p))
lc, lb = loss(p_cand), loss(p_base); diff = lb - lc; D = float(diff.mean()); n = len(y)
se_iid = float(diff.std(ddof=1) / np.sqrt(n)); ci_iid = [D - 1.96 * se_iid, D + 1.96 * se_iid]
# player-multiplicity bootstrap over the confirmation matches' players
P = np.load("results/lineup_design/players.npz")["players"][conf]
uniq, inv = np.unique(P.ravel(), return_inverse=True); inv = inv.reshape(n, 12)
rng = np.random.default_rng(2026); boots = []
for _ in range(1000):
    c = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq))); w = c[inv].sum(1).astype(float)
    boots.append(float(np.average(diff, weights=w)))
ci_boot = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
def cox(p, yy):
    e = np.log(p / (1 - p)); Z = np.column_stack([np.ones(len(e)), e]); b = np.zeros(2)
    for _ in range(30):
        q = expit(Z @ b); b = b + np.linalg.solve(Z.T @ (Z * (q * (1 - q))[:, None]) + 1e-10 * np.eye(2), Z.T @ (yy - q))
    return float(b[0]), float(b[1])
dev = np.nonzero(d.ts <= T_CONF)[0]; edges = np.quantile(d.lobby[dev], [1/3, 2/3]); thirds = np.digitize(d.lobby[conf], edges)
a_all, s_all = cox(p_cand, y)
thirds_res = []
for t in range(3):
    m = thirds == t; a, s_ = cox(p_cand[m], y[m]); thirds_res.append({"third": t, "n": int(m.sum()), "D": float(diff[m].mean()), "cal_intercept": a, "cal_slope": s_})
crit = {"D_positive_and_boot_ci_excludes_zero": D > 0 and ci_boot[0] > 0,
        "calibration_overall": 0.85 <= s_all <= 1.15 and abs(a_all) <= 0.05,
        "calibration_by_third": all(0.85 <= t["cal_slope"] <= 1.15 and abs(t["cal_intercept"]) <= 0.05 for t in thirds_res),
        "no_third_worse_than_minus_0.001": all(t["D"] >= -0.001 for t in thirds_res), "sample_at_least_10000": n >= 10000}
if all(crit.values()): verdict = "PROMOTE"
elif (ci_boot[1] < 0) or not crit["calibration_overall"] or not crit["calibration_by_third"]: verdict = "RETAIN HEADLINE"
else: verdict = "INCONCLUSIVE"
out = {"n": n, "D": D, "se_iid": se_iid, "ci_iid": ci_iid, "ci_boot": ci_boot, "logloss_candidate": float(lc.mean()), "logloss_baseline": float(lb.mean()),
       "cal_intercept": a_all, "cal_slope": s_all, "cal_slope_thirds": [t["cal_slope"] for t in thirds_res], "by_third": thirds_res,
       "criteria": crit, "verdict": verdict, "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(out, open("results/lineup_confirmation.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "by_third"}, indent=1)); print(json.dumps(thirds_res, indent=1))
