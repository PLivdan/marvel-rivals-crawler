"""Player-cluster bootstrap for the headline specification, made feasible.

The design is built ONCE (and cached to disk), then each replication resamples
players with replacement, converts their matches to frequency weights on the
fixed rows, and refits by warm-started IRLS. That replaces a ~40-minute
re-query-and-rebuild per replication (656 hours for 1,000) with ~15 seconds.

Why players, not matches: the crawl enumerates players and harvests their
histories, and a match belongs to twelve overlapping player clusters. Resampling
players mirrors the sampling process; resampling matches would not.

Usage: python3 results/run_bootstrap.py [--reps 1000] [--tag W0_specAplus]
Outputs: results/bootstrap_draws_<tag>.npz, and bootstrap_* columns added to
results/apm_hero_table_<tag>.csv.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import db
from apm import contrasts, features, sample
from apm.estimate import fit_logit, fit_logit_weighted
from apm_main import _drop_zero_variance_extra_columns

ap = argparse.ArgumentParser()
ap.add_argument("--db-path", default="data/rivals.db")
ap.add_argument("--tag", default="W0_specAplus")
ap.add_argument("--reps", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--checkpoint-every", type=int, default=50)
ap.add_argument("--attribution", default="W0")
ap.add_argument("--min-shape-count", type=int, default=100)
args = ap.parse_args()

cache = f"results/design_cache_{args.tag}.npz"
t_all = time.time()
conn = db.connect_readonly(args.db_path)

# ---- design, built once and cached ----------------------------------------
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    X, y = z["X"], z["y"]
    hero_ids, hero_basis = list(z["hero_ids"]), z["hero_basis"]
    hero_slice = slice(int(z["hero_slice"][0]), int(z["hero_slice"][1]))
    match_uids = list(z["match_uids"])
    print(f"[{time.time()-t_all:.0f}s] design loaded from cache {X.shape}", flush=True)
else:
    frame, _ = sample.build_sample(conn, 240)
    design = features.build_design(conn, frame, args.attribution,
                                   min_shape_count=args.min_shape_count,
                                   constraint="within_role", map_intercepts=True)
    design = _drop_zero_variance_extra_columns(design)
    X, y = design.X.astype(np.float64), design.y.astype(np.float64)
    hero_ids, hero_basis, hero_slice = design.hero_ids, design.hero_basis, design.hero_slice
    match_uids = design.match_uids
    np.savez(cache, X=X, y=y, hero_ids=np.array(hero_ids), hero_basis=hero_basis,
             hero_slice=np.array([hero_slice.start, hero_slice.stop]),
             match_uids=np.array(match_uids, dtype=object))
    print(f"[{time.time()-t_all:.0f}s] design built and cached {X.shape}", flush=True)
n, k = X.shape

# ---- full-sample fit: the warm start ----------------------------------------
t0 = time.time()
beta0, info0 = fit_logit_weighted(X, y, np.ones(n))
assert info0["converged"], info0
print(f"[{time.time()-t0:.0f}s] full-sample fit converged in {info0['n_iter']} iters, "
      f"loglike {info0['loglike']:,.1f}", flush=True)

# ---- player -> rows --------------------------------------------------------
row_of = {uid: i for i, uid in enumerate(match_uids)}
by_player = {}
for muid, puid in conn.execute("SELECT match_uid, player_uid FROM match_players"):
    r = row_of.get(muid)
    if r is not None:
        by_player.setdefault(puid, []).append(r)
players = list(by_player)
# flatten to arrays for fast weight accumulation
offsets = np.zeros(len(players) + 1, dtype=np.int64)
rows_flat = []
for j, p in enumerate(players):
    rows_flat.extend(by_player[p]); offsets[j + 1] = len(rows_flat)
rows_flat = np.asarray(rows_flat, dtype=np.int64)
print(f"[{time.time()-t_all:.0f}s] {len(players):,} players -> {len(rows_flat):,} player-match rows", flush=True)

# ---- replications --------------------------------------------------------
draws_path = f"results/bootstrap_draws_{args.tag}.npz"
draws = np.full((args.reps, len(hero_ids)), np.nan)
start = 0
if os.path.exists(draws_path):
    prev = np.load(draws_path)["draws"]
    m = min(len(prev), args.reps)
    draws[:m] = prev[:m]; start = int(np.isfinite(prev[:m, 0]).sum())
    print(f"resuming at replication {start}", flush=True)
rng = np.random.default_rng(args.seed + start)
t_reps = time.time(); n_iters = []; n_fail = 0
for rep in range(start, args.reps):
    picked = rng.integers(0, len(players), size=len(players))
    counts = np.bincount(picked, minlength=len(players))
    # weight of row r = number of times any player in that match was drawn
    w = np.zeros(n)
    np.add.at(w, rows_flat, np.repeat(counts, np.diff(offsets)))
    beta, info = fit_logit_weighted(X, y, w, beta0=beta0)
    if not info["converged"]:
        n_fail += 1
    n_iters.append(info["n_iter"])
    draws[rep] = contrasts.effects_from_free(beta[hero_slice], hero_basis) * 25.0  # pp
    if (rep + 1) % args.checkpoint_every == 0 or rep + 1 == args.reps:
        np.savez(draws_path, draws=draws, hero_ids=np.array(hero_ids))
        done = rep + 1 - start; el = time.time() - t_reps
        eta = (args.reps - rep - 1) * el / max(done, 1)
        print(f"  rep {rep+1}/{args.reps}  {el/done:.1f}s/rep  mean iters {np.mean(n_iters):.1f}  "
              f"unconverged {n_fail}  ETA {eta/60:.0f} min", flush=True)

# ---- intervals into the hero table -----------------------------------------
lo, hi = np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)
se = np.nanstd(draws, axis=0, ddof=1)
t = pd.read_csv(f"results/apm_hero_table_{args.tag}.csv").set_index("hero_id")
for i, h in enumerate(hero_ids):
    t.loc[h, "bootstrap_ci_low_pp"] = lo[i]; t.loc[h, "bootstrap_ci_high_pp"] = hi[i]
    t.loc[h, "bootstrap_se_pp"] = se[i]
t["bootstrap_over_hc1"] = t["bootstrap_se_pp"] / (t["within_role_se"] * 25.0)
t.reset_index().sort_values("within_role_pp", ascending=False).to_csv(
    f"results/apm_hero_table_{args.tag}.csv", index=False)
meta_p = f"results/apm_run_meta_{args.tag}.json"
meta = json.load(open(meta_p))
meta["bootstrap"] = {"reps": int(np.isfinite(draws[:, 0]).sum()), "unit": "player", "seed": args.seed,
                     "unconverged": int(n_fail), "mean_iters": float(np.mean(n_iters)) if n_iters else None,
                     "seconds_per_rep": float((time.time() - t_reps) / max(len(n_iters), 1)),
                     "median_se_ratio_to_hc1": float(np.nanmedian(t["bootstrap_over_hc1"]))}
json.dump(meta, open(meta_p, "w"), indent=2)
print(f"\n[{time.time()-t_all:.0f}s total] bootstrap SE / HC1 SE: median "
      f"{meta['bootstrap']['median_se_ratio_to_hc1']:.2f}x, range "
      f"{t['bootstrap_over_hc1'].min():.2f}-{t['bootstrap_over_hc1'].max():.2f}x", flush=True)
