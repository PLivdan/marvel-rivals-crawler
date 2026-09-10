"""Descriptive fit check on the internal temporal evaluation slice: predicted against realised win rates of
team-sides grouped by starting hero and by composition shape, for the selected unified fit and for the
refitted headline specification. No fitting: the selected coefficients (results/lineup_selected.npz) are
applied to the cached design rows after the confirmation cut, and the headline predictions are the ones saved
by results/lineup_baseline_dev.py. Writes results/lineup_fit_scatter.json (group-level aggregates only).
Usage: PYTHONPATH=. python3 results/lineup_fit_scatter.py"""
import json, sys
import numpy as np, pandas as pd
from scipy.special import expit
sys.path.insert(0, ".")
from lineup.fit import LineupDesign
T_CONF = 1788519632; MIN_SHAPE_SIDES = 100
z = np.load("results/lineup_selected.npz", allow_pickle=True); phi, sc = z["phi"], json.loads(str(z["scalers"]))
d = LineupDesign("results/lineup_design")
conf = np.nonzero(d.ts > T_CONF)[0]; y = d.y[conf].astype(float)
p = expit(d.eta(phi, sc, conf))
base = np.load("results/lineup_baseline_dev.npz", allow_pickle=True); bmap = dict(zip(base["conf_uids"], base["p_conf"]))
pb = np.array([bmap[d.meta["match_uids"][i]] for i in conf])
names = d.meta["names"]; roles = np.array(d.meta["roles"])
# team-sides: side 0 wins with probability p, side 1 with 1 - p
s0, s1 = d.s0[conf], d.s1[conf]
side_p = np.concatenate([p, 1 - p]); side_pb = np.concatenate([pb, 1 - pb]); side_y = np.concatenate([y, 1 - y])
side_heroes = np.vstack([s0, s1])
def shape_of(S):
    cnt = np.zeros((len(S), 3), int)
    for i in range(6):
        rr = roles[S[:, i]]; cnt[:, 0] += rr == "Tank"; cnt[:, 1] += rr == "Damage"; cnt[:, 2] += rr == "Support"
    return ["{}-{}-{}".format(*c) for c in cnt]
side_shape = np.array(shape_of(side_heroes))
rows = []
for h in range(d.K):
    m = (side_heroes == h).any(axis=1)
    rows.append({"name": names[h], "role": roles[h], "n_sides": int(m.sum()), "pred": float(side_p[m].mean()), "pred_base": float(side_pb[m].mean()), "real": float(side_y[m].mean())})
H = pd.DataFrame(rows)
srows = []
for s in sorted(set(side_shape)):
    m = side_shape == s
    if m.sum() >= MIN_SHAPE_SIDES:
        srows.append({"shape": s, "n_sides": int(m.sum()), "pred": float(side_p[m].mean()), "pred_base": float(side_pb[m].mean()), "real": float(side_y[m].mean())})
S = pd.DataFrame(srows)
def stats(df, col):
    return {"corr": float(np.corrcoef(df[col], df.real)[0, 1]), "mean_abs_gap_pp": float(100 * (df[col] - df.real).abs().mean()), "n_groups": int(len(df))}
out = {"n_matches": int(len(conf)), "n_sides": int(len(side_p)), "min_shape_sides": MIN_SHAPE_SIDES,
       "hero": {"unified": stats(H, "pred"), "headline": stats(H, "pred_base"), "points": H.to_dict("records")},
       "shape": {"unified": stats(S, "pred"), "headline": stats(S, "pred_base"), "points": S.to_dict("records")}}
json.dump(out, open("results/lineup_fit_scatter.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out["hero"].items() if k != "points"}, indent=1)); print(json.dumps({k: v for k, v in out["shape"].items() if k != "points"}, indent=1))
print(H.sort_values("pred").to_string(index=False)); print(S.to_string(index=False))
