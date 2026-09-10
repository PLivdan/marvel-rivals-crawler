"""Stage 6 driver: all reported quantities from ONE fit. Usage:
  python3 results/lineup_summaries.py <fit.npz> <out_prefix> [pool_size]
The fit file holds phi, lams, and the scalers (r_mu, r_sd, dr_sd) of the training rows."""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from lineup.fit import LineupDesign
from lineup import summaries as S
T_CONF = 1788519632
t0 = time.time()
fit_path, prefix = sys.argv[1], sys.argv[2]; pool_size = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
z = np.load(fit_path, allow_pickle=True); phi = z["phi"]; sc = json.loads(str(z["scalers"])) if "scalers" in z.files else None
d = LineupDesign("results/lineup_design")
dev = np.nonzero(d.ts <= T_CONF)[0]
if sc is None:
    w = np.zeros(d.n); w[dev] = 1; sc = d.scalers(w)
lob = d.lobby[dev]
q = {"p5": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}
r_values = {k: float((np.quantile(lob, v) - sc["r_mu"]) / sc["r_sd"]) for k, v in q.items()}
lobby_at = {k: float(np.quantile(lob, v)) for k, v in q.items()}
thirds = np.quantile(lob, [1/3, 2/3])
print(f"[{time.time()-t0:.0f}s] rank grid (lobby score): {lobby_at}; thirds {thirds.round(0)}", flush=True)
# 1. hero coefficients by rank
H = S.hero_rank_table(d, phi, sc, {k: r_values[k] for k in ("p25", "p50", "p75")})
H.to_csv(f"{prefix}_hero_by_rank.csv", index=False)
# 2. replacement scores on a fixed development pool
rng = np.random.default_rng(2026); pool = np.sort(rng.choice(dev, size=min(pool_size, len(dev)), replace=False))
R = S.replacement_scores(d, phi, sc, pool, r_thirds=thirds)
R.to_csv(f"{prefix}_replacement.csv", index=False)
print(f"[{time.time()-t0:.0f}s] replacement scores over {len(pool):,} pool matches", flush=True)
# 3. pair tables at the median rank, with support by third (development rows only)
dev_mask = np.zeros(d.n, bool); dev_mask[dev] = True
sup = S.support_by_rank(d, thirds, rows_mask=dev_mask)
P = S.pair_tables(d, phi, sc, r_values["p50"], sup)
P.to_csv(f"{prefix}_pairs_p50.csv", index=False)
for lab in ("p25", "p75"):
    S.pair_tables(d, phi, sc, r_values[lab], sup)[["kind", "hero_a", "hero_b", "coef", "did"]].to_csv(f"{prefix}_pairs_{lab}.csv", index=False)
json.dump({"fit": fit_path, "r_values": r_values, "lobby_at": lobby_at, "thirds": thirds.tolist(), "pool_size": int(len(pool)),
           "scalers": sc, "lams": z["lams"].tolist() if "lams" in z.files else None}, open(f"{prefix}_summary_meta.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] wrote {prefix}_*.csv", flush=True)
print(H.sort_values("beta_p50", ascending=False).head(5)[["name", "role", "beta_p25", "beta_p50", "beta_p75"]].round(3).to_string(index=False))
print(R.sort_values("obs_meta_pp", ascending=False).head(5)[["name", "role", "obs_meta_pp", "coverage", "common_ref_pp"]].round(2).to_string(index=False))
print(P[P.kind == "allied"].sort_values("did", ascending=False).head(5)[["hero_a", "hero_b", "coef", "did", "is_teamup", "support_dev"]].round(3).to_string(index=False))
print(P[P.kind == "opposing"].assign(a=lambda x: x.did.abs()).sort_values("a", ascending=False).head(5)[["hero_a", "hero_b", "coef", "did", "support_dev"]].round(3).to_string(index=False))
