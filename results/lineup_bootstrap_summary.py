"""Aggregate the fixed-penalty bootstrap draws into results/lineup_bootstrap_summary.json.
Usage: python3 results/lineup_bootstrap_summary.py [summary_prefix]
Stability statements only: sign agreement across replications and across-replication standard deviations
of the reported quantities (replacement scores, four-lineup contrasts, hero coefficients at the median rank)."""
import json, sys
import numpy as np, pandas as pd
prefix = sys.argv[1] if len(sys.argv) > 1 else "results/lineup_selected"
z = np.load("results/lineup_bootstrap_draws.npz", allow_pickle=True)
repl = z["replacement_obs_meta"]; beta = z["hero_beta_p50"]; ad = z["allied_did"]; od = z["opposing_did"]
reps = repl.shape[0]
R = pd.read_csv(f"{prefix}_replacement.csv"); P = pd.read_csv(f"{prefix}_pairs_p50.csv")
names = list(z["replacement_names"]); assert names == list(R.name), "replacement row order differs from the summary table"
A = P[P.kind == "allied"].reset_index(drop=True); O = P[P.kind == "opposing"].reset_index(drop=True)
assert len(A) == ad.shape[1] and len(O) == od.shape[1]
point = R.obs_meta_pp.to_numpy()
sd = np.nanstd(repl, axis=0, ddof=1)
sign_agree = np.nanmean(np.sign(repl) == np.sign(point)[None, :], axis=0)
top10 = np.argsort(-np.nan_to_num(R.common_ref_pp.to_numpy(), nan=-99))[:10]
# listed pairs: same rules as the report (support >= 500; team-ups; top 25 +/- non-team-up; top 50 opposing by |contrast|)
MIN = 500
A_ok = A[A.support_dev >= MIN]; O_ok = O[O.support_dev >= MIN]
tu_idx = A_ok[A_ok.is_teamup].index.to_numpy()
nt = A_ok[~A_ok.is_teamup]; nt_idx = np.concatenate([nt.sort_values("did", ascending=False).head(25).index.to_numpy(), nt.sort_values("did").head(25).index.to_numpy()])
o_idx = O_ok.assign(a=O_ok.did.abs()).sort_values("a", ascending=False).head(50).index.to_numpy()
def stable(draws, idx, point_vals, thr=0.95):
    agree = np.mean(np.sign(draws[:, idx]) == np.sign(point_vals[idx])[None, :], axis=0)
    return int((agree >= thr).sum()), int(len(idx)), agree
a_list_idx = np.concatenate([tu_idx, nt_idx])
n_a_stable, n_a, a_agree = stable(ad, a_list_idx, A.did.to_numpy())
n_o_stable, n_o, o_agree = stable(od, o_idx, O.did.to_numpy())
out = {"reps": int(reps), "pool_size": int(z["pool_size"]),
       "replacement_sd_median": float(np.nanmedian(sd)), "replacement_sd_max": float(np.nanmax(sd)),
       "top10_sign_stable": int((sign_agree[top10] >= 0.999).sum()),
       "hero_beta_sd_median": float(np.median(beta.std(axis=0, ddof=1))),
       "allied_listed_sign_stable": n_a_stable, "allied_listed_n": n_a, "opposing_listed_sign_stable": n_o_stable, "opposing_listed_n": n_o,
       "allied_did_sd_median": float(np.median(ad.std(axis=0, ddof=1))), "opposing_did_sd_median": float(np.median(od.std(axis=0, ddof=1))),
       "per_hero": [{"name": n, "point": float(p), "sd": float(s), "sign_agreement": float(g)} for n, p, s, g in zip(names, point, sd, sign_agree)]}
json.dump(out, open("results/lineup_bootstrap_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_hero"}, indent=1))
