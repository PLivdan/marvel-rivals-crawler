"""Stage 5: tune the five ridge penalties on three chronological development folds (resumable).

Development rows = design rows with play time <= T_CONF (the internal confirmation slice, play time
> T_CONF, is never trained on, never evaluated here). Folds by play-time order of development rows:
  fold k: train = rows before cut_k, validate = rows in [cut_k, cut_{k+1}), cuts at 60 / 73 / 87 / 100 %.
Search: bounded coordinate descent on log10(lambda) with warm starts; every evaluation is logged to
results/lineup_tuning.json so the run can resume. Usage:
  python3 results/lineup_tune.py pilot          # one full-size fit, timing only
  python3 results/lineup_tune.py coarse [sub]   # coarse pass (optional row subsample fraction for training)
  python3 results/lineup_tune.py fine
  python3 results/lineup_tune.py reduced        # genuine reduced models at the selected penalties
"""
import json, os, sys, time
import numpy as np
from multiprocessing import get_context
sys.path.insert(0, ".")
from lineup.fit import LineupDesign, GROUPS
WORKERS = int(os.environ.get("LINEUP_WORKERS", "1"))      # parallel processes (fork) over (candidate, fold) jobs; ~4 GB RAM each
TOL = float(os.environ.get("LINEUP_TOL", "1e-5"))        # Newton stopping tolerance for the search (final fit uses 1e-6)

T_CONF = 1788519632                                  # 2026-09-04T11:00:32Z, see results/confirmation_protocol.json
LOG = "results/lineup_tuning.json"
d = LineupDesign("results/lineup_design")
dev = np.nonzero(d.ts <= T_CONF)[0]; conf = np.nonzero(d.ts > T_CONF)[0]
order = dev[np.argsort(d.ts[dev], kind="stable")]
cuts = [int(len(order) * f) for f in (0.60, 0.73, 0.87, 1.0)]
FOLDS = [(order[:cuts[k]], order[cuts[k]:cuts[k + 1]]) for k in range(3)]
print(f"design rows {d.n:,}: development {len(dev):,}, confirmation slice {len(conf):,} (untouched); folds "
      + ", ".join(f"train {len(a):,}/val {len(b):,}" for a, b in FOLDS), flush=True)
state = json.load(open(LOG)) if os.path.exists(LOG) else {"evaluations": [], "warm": {}}
def save(): json.dump(state, open(LOG, "w"), indent=1)
def key(lams): return ",".join(f"{l:g}" for l in lams)
WARM = {}                                            # in-memory warm starts per fold: key -> phi
def _fit_fold(args):
    """One (candidate, fold) job: fit at lams with optional training subsample and warm start; returns (phi, fold record)."""
    f, lams, sub, max_iter, phi0 = args
    tr, va = FOLDS[f]
    w = np.zeros(d.n); w[tr] = 1.0
    if sub < 1.0:
        rng = np.random.default_rng(f); drop = rng.random(len(tr)) > sub; w[tr[drop]] = 0.0
    fit = d.fit(lams, w, phi0=phi0, max_iter=max_iter, tol=TOL, verbose=False)
    ev = d.evaluate(fit, va); ev.pop("loss_vector")
    thirds = np.digitize(d.lobby[va], np.quantile(d.lobby[dev], [1/3, 2/3]))
    by_third = []
    for t in range(3):
        rows = va[thirds == t]; e = d.evaluate(fit, rows); e.pop("loss_vector"); by_third.append({k2: round(v, 4) if isinstance(v, float) else v for k2, v in e.items()})
    rec = {"fold": f, "n_train": fit["n_train"], "n_iter": fit["n_iter"], "converged": fit["converged"], "seconds": round(fit["seconds"]),
           **{k2: round(v, 5) if isinstance(v, float) else v for k2, v in ev.items()}, "by_third": by_third}
    return fit["phi"], rec

def evaluate_batch(cands, sub=1.0, max_iter=6, tags=None):
    """Evaluate several candidate penalty vectors at once: every (candidate, fold) job runs in the process pool."""
    tags = tags or [""] * len(cands)
    todo = [(i, c) for i, c in enumerate(cands) if not any(e["key"] == key(c) and e["sub"] == sub for e in state["evaluations"])]
    t0 = time.time()
    jobs = [(f, list(c), sub, max_iter, WARM.get(f)) for i, c in todo for f in range(len(FOLDS))]
    if jobs:
        if WORKERS > 1:
            with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
                outs = pool.map(_fit_fold, jobs)
        else:
            outs = [_fit_fold(j) for j in jobs]
        for (i, c), chunk in zip(todo, [outs[j:j + len(FOLDS)] for j in range(0, len(outs), len(FOLDS))]):
            res = {"key": key(c), "lams": list(c), "sub": sub, "tag": tags[i], "folds": [rec for _, rec in chunk]}
            res["mean_val_logloss"] = float(np.mean([f["logloss"] for f in res["folds"]]))
            res["mean_cal_slope"] = float(np.mean([f["cal_slope"] for f in res["folds"]]))
            res["seconds"] = round(time.time() - t0)
            state["evaluations"].append(res)
            print(f"  {tags[i]:10s} lams={res['key']:32s} val logloss {res['mean_val_logloss']:.5f}  slope {res['mean_cal_slope']:.3f}  [{res['seconds']}s]", flush=True)
        # warm starts: keep the best candidate's coefficients per fold
        best_i = min(range(len(todo)), key=lambda j: next(e for e in state["evaluations"] if e["key"] == key(todo[j][1]) and e["sub"] == sub)["mean_val_logloss"])
        for f, (phi, _) in enumerate(outs[best_i * len(FOLDS):(best_i + 1) * len(FOLDS)]):
            WARM[f] = phi
        save()
    return [next(e for e in state["evaluations"] if e["key"] == key(c) and e["sub"] == sub)["mean_val_logloss"] for c in cands]

def evaluate(lams, sub=1.0, max_iter=6, tag=""):
    return evaluate_batch([lams], sub, max_iter, [tag])[0]

mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
if mode == "pilot":
    tr, va = FOLDS[0]; w = np.zeros(d.n); w[tr] = 1.0
    fit = d.fit([10, 300, 300, 300, 3000], w, max_iter=4)
    ev = d.evaluate(fit, va); ev.pop("loss_vector"); print("pilot fold-0:", fit["seconds"], "s", ev)
    np.savez("results/lineup_pilot_full.npz", phi=fit["phi"])
elif mode in ("coarse", "fine"):
    sub = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    best = state.get("best_lams") or [10.0, 300.0, 300.0, 300.0, 3000.0]
    steps = [0.1, 10.0] if mode == "coarse" else [1 / 3, 3.0]
    order_groups = [1, 2, 4, 3, 0]                   # most parameters first: allied, opposing, pair slopes, slopes/heromap, hero+shape
    for pas in range(2):
        improved = False
        base = evaluate(best, sub, tag=f"{mode}{pas}")
        for gi in order_groups:
            cands = []
            for mult in steps:
                cand = list(best); cand[gi] = float(np.clip(cand[gi] * mult, 0.01, 1e6)); cands.append(cand)
            vals = evaluate_batch(cands, sub, tags=[f"{mode}{pas}:{GROUPS[gi][:2]}"] * len(cands))
            j = int(np.argmin(vals))
            if vals[j] < base - 1e-6:
                best, base, improved = cands[j], vals[j], True
        state["best_lams"] = best; state["best_val_logloss"] = base; save()
        print(f"pass {pas} best {key(best)} -> {base:.5f}", flush=True)
        if not improved: break
elif mode == "final":
    # the selected penalties refitted on ALL development rows, converged, for the summaries and the lock
    best = state["best_lams"]; w = np.zeros(d.n); w[dev] = 1.0
    fit = d.fit(best, w, max_iter=10)
    np.savez("results/lineup_selected.npz", phi=fit["phi"], lams=np.array(best), scalers=json.dumps(fit["scalers"]))
    state["final"] = {"lams": best, "loglik": fit["loglik"], "n_iter": fit["n_iter"], "converged": fit["converged"], "n_train": fit["n_train"]}; save()
    print("final fit saved: results/lineup_selected.npz", state["final"], flush=True)
elif mode == "reduced":
    best = state["best_lams"]; BIG = 1e9
    variants = {"selected": best,
                "no_pair_slopes": [best[0], best[1], best[2], best[3], BIG],
                "no_pairs": [best[0], BIG, BIG, best[3], BIG],
                "no_pairs_no_slopes_no_heromap": [best[0], BIG, BIG, BIG, BIG]}
    evaluate_batch(list(variants.values()), 1.0, tags=list(variants.keys()))
    state["reduced"] = {name: key(l) for name, l in variants.items()}; save()
