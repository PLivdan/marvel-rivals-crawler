"""Stage 7: player-multiplicity bootstrap at FIXED penalties (stability, not frequentist coverage).

Weights. Draw player counts n_i ~ Multinomial(N_players, uniform) over the distinct players of the
development rows. A match's raw weight is the sum of its twelve players' counts, w_m = sum_{i in m} n_i,
so a match whose players were drawn several times counts several times and matches sharing players move
together. Raw weights are rescaled to mean 1 over development rows, w_m <- w_m * n_dev / sum w, so the
data term keeps its original scale and the fixed penalties keep their meaning. Each replication refits
from the selected fit (warm start, 3 Newton steps) and recomputes the reported quantities.
Usage: python3 results/lineup_bootstrap.py <selected_fit.npz> <reps> [pool_size]"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from lineup.fit import LineupDesign
from lineup import summaries as S
T_CONF = 1788519632
fit_path, reps = sys.argv[1], int(sys.argv[2]); pool_size = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
t0 = time.time()
d = LineupDesign("results/lineup_design"); z = np.load(fit_path, allow_pickle=True)
phi_sel, lams, sc = z["phi"], list(z["lams"]), json.loads(str(z["scalers"]))
dev = np.nonzero(d.ts <= T_CONF)[0]; n_dev = len(dev)
pl = np.load("results/lineup_design/players.npz"); P = pl["players"]            # n x 12 player ids
uniq, inv = np.unique(P[dev].ravel(), return_inverse=True); inv = inv.reshape(len(dev), 12)
lob = d.lobby[dev]; thirds = np.quantile(lob, [1/3, 2/3]); r50 = float((np.median(lob) - sc["r_mu"]) / sc["r_sd"])
rng = np.random.default_rng(7); pool = np.sort(rng.choice(dev, size=min(pool_size, n_dev), replace=False))
dev_mask = np.zeros(d.n, bool); dev_mask[dev] = True; sup = S.support_by_rank(d, thirds, rows_mask=dev_mask)
draws = {"hero_beta_p50": [], "replacement_obs_meta": [], "allied_did": [], "opposing_did": []}
for rep in range(reps):
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
    w_raw = counts[inv].sum(1).astype(float)
    w = np.zeros(d.n); w[dev] = w_raw * n_dev / w_raw.sum()
    fit = d.fit(lams, w, phi0=phi_sel, max_iter=3, verbose=False, sc=sc)          # scalers fixed at the selected fit's
    nat = d.native(fit["phi"])
    draws["hero_beta_p50"].append(nat["hero"] + r50 * nat["hero_r"])
    R = S.replacement_scores(d, fit["phi"], sc, pool, r_thirds=thirds)
    draws["replacement_obs_meta"].append(R.obs_meta_pp.to_numpy())
    Pt = S.pair_tables(d, fit["phi"], sc, r50, sup)
    draws["allied_did"].append(Pt[Pt.kind == "allied"].did.to_numpy()); draws["opposing_did"].append(Pt[Pt.kind == "opposing"].did.to_numpy())
    print(f"[{time.time()-t0:.0f}s] rep {rep+1}/{reps}: fit {fit['seconds']:.0f}s, {fit['n_iter']} steps, loglik {fit['loglik']:,.1f}", flush=True)
    np.savez("results/lineup_bootstrap_draws.npz", **{k: np.array(v) for k, v in draws.items()}, hero_names=np.array(d.meta["names"]),
             replacement_names=R.name.to_numpy(), allied_pairs=np.array(d.meta["allied_pairs"]), opposing_pairs=np.array(d.meta["opposing_pairs"]), pool_size=len(pool))
print(f"[{time.time()-t0:.0f}s] done: {reps} replications")
