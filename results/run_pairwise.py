"""Spec E: Spec A+ plus pairwise hero interactions on STARTING lineups (DB step).

Two interaction blocks, each one coefficient per unordered hero pair:
  synergy  s_ab = 1[a,b both start on side 0] - 1[both on side 1]
  counter  c_ab = 1[a on side 0, b on side 1] - 1[b on side 0, a on side 1]   (a < b; kappa>0: a beats b)
Pairs with fewer than MIN_PAIR co-occurrences get no coefficient (deviation fixed at zero), except
designated team-ups, which are always synergy columns; the separate team-up block is dropped
because it is a subset of the synergy block.

Identification. Summing a hero's synergy contrasts over its five teammates equals 5x its main-effect
contrast, and its counter contrasts over six opponents equals 6x, so unconstrained blocks contain the
hero main effects exactly. Constraints imposed (null-space parametrisation), over NON-team-up pairs:
  per hero:      sum_b sigma_ab = 0            sum_b sgn(a,b) kappa_ab = 0   (counters: all pairs)
  per role pair: sum_{ab in RR'} sigma_ab = 0   (role-pair sums are functions of composition shape)
Designated team-up columns are left free, exactly as in Spec A+, so the main effects keep the
headline definition (net of team-up premiums) and a non-team-up synergy reads as the deviation from
the hero's typical non-team-up partner. (An earlier variant constrained team-ups too; that redefined
the main effects as averages over all partners and made heroes with strong team-ups show negative
synergies with everyone else. See the ledger.) Fitted by constrained Newton/IRLS with the base design dense and the
interaction blocks sparse; HC1 covariance mapped through the null-space basis.

Outputs (results/): apm_pairwise_synergy_<tag>.csv, apm_pairwise_counter_<tag>.csv,
apm_hero_table_pairwise_<tag>.csv, pairwise_meta_<tag>.json (holdout ladder, comparisons).
"""
import json, sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.linalg import null_space
from scipy.special import expit
from scipy import stats
sys.path.insert(0, ".")
import db
from apm import contrasts, sample
from apm.features import build_design, _starting_hero_per_player
from apm.inference import benjamini_hochberg
from apm_main import _drop_zero_variance_extra_columns

TAG = "W0_specE"; MIN_PAIR = 2500; HOLDOUT_DAYS = 7
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

conn = db.connect_readonly("data/rivals.db")
frame, _ = sample.build_sample(conn, 240)
train, test = sample.temporal_holdout(frame, HOLDOUT_DAYS)
design = _drop_zero_variance_extra_columns(
    build_design(conn, frame, "W0", min_shape_count=100, constraint="within_role", map_intercepts=True))
names = list(design.column_names); uids = list(design.match_uids)
keep = [i for i, c in enumerate(names) if not c.startswith("teamup_")]
Xb = np.ascontiguousarray(design.X[:, keep].astype(np.float64)); base_names = [names[i] for i in keep]
Xfull_A = design.X.astype(np.float64)                     # Spec A+ (with team-up block) for the ladder
y = design.y.astype(np.float64); n = len(y)
test_set = set(test.match_uid); is_test = np.fromiter((u in test_set for u in uids), bool, n)
w_train = (~is_test).astype(float); w_all = np.ones(n)
log(f"design {design.X.shape}; base without team-ups {Xb.shape}; holdout rows {is_test.sum():,}")

# ---- starting lineups aligned to design rows -------------------------------------------------
hero_ids = list(design.hero_ids); K = len(hero_ids); hidx = {h: i for i, h in enumerate(hero_ids)}
info = {h: (nm, r) for h, nm, r in conn.execute("SELECT hero_id, name, role FROM hero_info")}
hname = [info[h][0] for h in hero_ids]; hrole = [info[h][1] for h in hero_ids]
per = _starting_hero_per_player(conn, uids)
g = per.groupby(["match_uid", "camp"]).hero_id.apply(list)
side0 = np.array([[hidx[h] for h in g[(u, 0)]] for u in uids]); side1 = np.array([[hidx[h] for h in g[(u, 1)]] for u in uids])
assert side0.shape == (n, 6) and side1.shape == (n, 6)
tu_pairs = set(); tu = {}
for tid, hid in conn.execute("SELECT teamup_id, hero_id FROM teamup_heroes"):
    tu.setdefault(tid, []).append(hid)
for tid, hs in tu.items():
    if len(hs) == 2 and all(h in hidx for h in hs):
        tu_pairs.add((min(hidx[hs[0]], hidx[hs[1]]), max(hidx[hs[0]], hidx[hs[1]])))
log(f"starting lineups built; {len(tu_pairs)} team-up pairs")

# ---- pair counts on this exact sample -----------------------------------------------------------
same = np.zeros((K, K), int); cross = np.zeros((K, K), int)
for side in (side0, side1):
    for i in range(6):
        for j in range(i + 1, 6):
            np.add.at(same, (side[:, i], side[:, j]), 1); np.add.at(same, (side[:, j], side[:, i]), 1)
for i in range(6):
    for j in range(6):
        np.add.at(cross, (side0[:, i], side1[:, j]), 1); np.add.at(cross, (side1[:, j], side0[:, i]), 1)
syn_pairs = [(a, b) for a in range(K) for b in range(a + 1, K) if same[a, b] >= MIN_PAIR or (a, b) in tu_pairs]
ctr_pairs = [(a, b) for a in range(K) for b in range(a + 1, K) if cross[a, b] >= MIN_PAIR]
Ps, Pc = len(syn_pairs), len(ctr_pairs)
syn_col = {p: j for j, p in enumerate(syn_pairs)}; ctr_col = {p: j for j, p in enumerate(ctr_pairs)}
log(f"synergy columns {Ps} (of 1485), counter columns {Pc} (of 1485), threshold {MIN_PAIR}")

# ---- sparse interaction blocks ------------------------------------------------------------------
rows, cols, vals = [], [], []
for side, sign in ((side0, 1.0), (side1, -1.0)):
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = np.minimum(side[:, i], side[:, j]), np.maximum(side[:, i], side[:, j])
            c = np.array([syn_col.get((x, z), -1) for x, z in zip(a, b)])
            m = c >= 0; rows.append(np.nonzero(m)[0]); cols.append(c[m]); vals.append(np.full(m.sum(), sign))
S = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, Ps)).tocsr()
rows, cols, vals = [], [], []
for i in range(6):
    for j in range(6):
        a, b = side0[:, i], side1[:, j]                     # a on side 0, b on side 1
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        c = np.array([ctr_col.get((x, z), -1) for x, z in zip(lo, hi)])
        m = (c >= 0) & (a != b)
        rows.append(np.nonzero(m)[0]); cols.append(c[m]); vals.append(np.where(a[m] < b[m], 1.0, -1.0))
C = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, Pc)).tocsr()
S.sum_duplicates(); C.sum_duplicates()
log(f"S nnz {S.nnz:,}, C nnz {C.nnz:,}")

# ---- constraints and null-space bases -----------------------------------------------------------
def role_pair(a, b): return "-".join(sorted([hrole[a], hrole[b]]))
A_s = np.zeros((K + 6, Ps)); rp_types = sorted({role_pair(a, b) for a, b in syn_pairs}); rp_idx = {t: i for i, t in enumerate(rp_types)}
for j, (a, b) in enumerate(syn_pairs):
    if (a, b) in tu_pairs:
        continue                                   # designated team-ups stay unconstrained
    A_s[a, j] = 1; A_s[b, j] = 1; A_s[K + rp_idx[role_pair(a, b)], j] = 1
A_c = np.zeros((K, Pc))
for j, (a, b) in enumerate(ctr_pairs):
    A_c[a, j] = 1; A_c[b, j] = -1
N_s, N_c = null_space(A_s), null_space(A_c)
log(f"free parameters: synergy {N_s.shape[1]} (constraints rank {Ps - N_s.shape[1]}), counter {N_c.shape[1]} (rank {Pc - N_c.shape[1]})")

# ---- constrained IRLS ----------------------------------------------------------------------------
def fit(Xb, blocks, y, w, theta0=None, max_iter=40, tol=1e-7):
    """blocks: list of (sparse matrix, null-space basis N). Coefficients theta = [beta_base; theta_1; ...]
    with theta_k = N_k phi_k. Newton in the reduced space; returns theta, reduced HC1 cov, info."""
    kb = Xb.shape[1]; sizes = [B.shape[1] for B, _ in blocks]; frees = [N.shape[1] for _, N in blocks]
    def eta_of(theta):
        e = Xb @ theta[:kb]; o = kb
        for (B, _), sz in zip(blocks, sizes):
            e += B @ theta[o:o + sz]; o += sz
        return e
    def reduced_hessian(Wv):
        """B' X' diag(Wv) X B in the reduced parametrisation."""
        XbW = Xb * Wv[:, None]
        Hbb = Xb.T @ XbW
        parts = [[Hbb]]; cross_bl = []
        for (B, N) in blocks:
            cross_bl.append(((B.T @ XbW).T) @ N)           # kb x free
            parts[0].append(cross_bl[-1])
        for i, (Bi, Ni) in enumerate(blocks):
            row = [cross_bl[i].T]
            for j, (Bj, Nj) in enumerate(blocks):
                row.append(Ni.T @ ((Bi.T @ Bj.multiply(Wv[:, None]).tocsr()).toarray() @ Nj))
            parts.append(row)
        return np.block(parts)
    def reduced_grad(r):
        gs = [Xb.T @ r]
        for (B, N) in blocks:
            gs.append(N.T @ (B.T @ r))
        return np.concatenate(gs)
    def expand(d_r):
        out = [d_r[:kb]]; o = kb
        for (B, N), f in zip(blocks, frees):
            out.append(N @ d_r[o:o + f]); o += f
        return np.concatenate(out)
    theta = np.zeros(kb + sum(sizes)) if theta0 is None else theta0.copy()
    def ll(theta):
        e = eta_of(theta); return float(np.sum(np.where(w > 0, w * (y * e - np.logaddexp(0, e)), 0)))
    cur = ll(theta); converged = False
    for it in range(1, max_iter + 1):
        e = eta_of(theta); p = expit(e); Wv = w * p * (1 - p); r = w * (y - p)
        H = reduced_hessian(Wv); gr = reduced_grad(r)
        d_r = np.linalg.solve(H + 1e-10 * np.eye(len(H)), gr)
        step = 1.0
        while True:
            cand = theta + step * expand(d_r); new = ll(cand)
            if new >= cur - 1e-9 or step < 1e-4: break
            step *= 0.5
        change = float(np.max(np.abs(cand - theta))); theta, prev, cur = cand, cur, new
        log(f"   iter {it}: loglik {cur:,.2f}  max|delta| {change:.2e}  step {step}")
        if change < tol or abs(cur - prev) < 1e-6:
            converged = True; break
    # HC1 in the reduced space: (B'HB)^-1 (B'MB) (B'HB)^-1, M = sum e_i^2 x_i x_i'
    e = eta_of(theta); p = expit(e); Wv = w * p * (1 - p); res2 = w * (y - p) ** 2
    H = reduced_hessian(Wv); M = reduced_hessian(res2)
    kfree = H.shape[0]; n_eff = float(w.sum()); Hinv = np.linalg.inv(H)
    cov_r = Hinv @ M @ Hinv * n_eff / (n_eff - kfree)
    return theta, cov_r, {"converged": converged, "n_iter": it, "loglike": cur, "k_free": kfree}

def logloss(theta, Xb, blocks, mask):
    e = Xb[mask] @ theta[:Xb.shape[1]]; o = Xb.shape[1]
    for (B, _) in blocks:
        e += B[mask] @ theta[o:o + B.shape[1]]; o += B.shape[1]
    p = np.clip(expit(e), 1e-12, 1 - 1e-12); yy = y[mask]
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

# ---- holdout ladder ------------------------------------------------------------------------------
ladder = []
log("ladder 1/3: Spec A+ (base with team-up block), train rows")
thA, _, infoA = fit(Xfull_A, [], y, w_train)
ladder.append({"model": "Spec A+ (heroes, maps, skill, shapes, team-ups)", "k_free": infoA["k_free"], "logloss_holdout": logloss(thA, Xfull_A, [], is_test), "converged": infoA["converged"]})
log(f"   holdout log loss {ladder[-1]['logloss_holdout']:.5f}")
base0 = thA[[i for i in keep]]
log("ladder 2/3: + synergy block (replaces team-ups)")
thS, _, infoS = fit(Xb, [(S, N_s)], y, w_train, theta0=np.concatenate([base0, np.zeros(Ps)]))
ladder.append({"model": "+ all pairwise synergies (team-ups subsumed)", "k_free": infoS["k_free"], "logloss_holdout": logloss(thS, Xb, [(S, N_s)], is_test), "converged": infoS["converged"]})
log(f"   holdout log loss {ladder[-1]['logloss_holdout']:.5f}")
log("ladder 3/3: + counter block")
thSC, _, infoSC = fit(Xb, [(S, N_s), (C, N_c)], y, w_train, theta0=np.concatenate([thS, np.zeros(Pc)]))
ladder.append({"model": "+ all pairwise counters", "k_free": infoSC["k_free"], "logloss_holdout": logloss(thSC, Xb, [(S, N_s), (C, N_c)], is_test), "converged": infoSC["converged"]})
log(f"   holdout log loss {ladder[-1]['logloss_holdout']:.5f}")

# ---- full-sample fit of Spec E -----------------------------------------------------------------
log("full-sample fit of Spec E")
theta, cov_r, info = fit(Xb, [(S, N_s), (C, N_c)], y, w_all, theta0=thSC)
kb = Xb.shape[1]; fs, fc = N_s.shape[1], N_c.shape[1]
cov_bb = cov_r[:kb, :kb]; cov_ss = cov_r[kb:kb + fs, kb:kb + fs]; cov_cc = cov_r[kb + fs:, kb + fs:]
sig = theta[kb:kb + Ps]; kap = theta[kb + Ps:]
se_sig = np.sqrt(np.einsum("ij,jk,ik->i", N_s, cov_ss, N_s)); se_kap = np.sqrt(np.einsum("ij,jk,ik->i", N_c, cov_cc, N_c))

# main effects (within-role, from the base hero block) vs Spec A+
hs = design.hero_slice
main = contrasts.effects_from_free(theta[hs], design.hero_basis)
main_cov = contrasts.cov_from_free(cov_bb[hs, hs], design.hero_basis)
A = pd.read_csv("results/apm_hero_table_W0_specAplus.csv").set_index("hero_id")
main_pp = main * 25.0; a_pp = A.within_role_pp.reindex(hero_ids).to_numpy()
cmp_main = {"spearman_vs_specAplus": float(stats.spearmanr(main_pp, a_pp)[0]), "mean_abs_diff_pp": float(np.mean(np.abs(main_pp - a_pp))),
            "max_abs_diff_pp": float(np.max(np.abs(main_pp - a_pp))), "sd_pp_specE": float(np.std(main_pp, ddof=1)), "sd_pp_specAplus": float(np.std(a_pp, ddof=1))}
pd.DataFrame({"hero_id": hero_ids, "name": hname, "role": hrole, "within_role_pp": main_pp, "within_role_se_pp": np.sqrt(np.diag(main_cov)) * 25.0,
              "specAplus_pp": a_pp, "diff_pp": main_pp - a_pp}).sort_values("within_role_pp", ascending=False).to_csv(f"results/apm_hero_table_pairwise_{TAG}.csv", index=False)

def table(pairs, est, se, counts, kind):
    z = est / se; pr = 2 * stats.norm.sf(np.abs(z))
    return pd.DataFrame({"kind": kind, "hero_a": [hname[a] for a, b in pairs], "hero_b": [hname[b] for a, b in pairs],
                         "role_a": [hrole[a] for a, b in pairs], "role_b": [hrole[b] for a, b in pairs],
                         "n_cooccur": [int(counts[a, b]) for a, b in pairs], "effect_pp": est * 25.0, "se_pp": se * 25.0, "z": z, "p_raw": pr,
                         "is_teamup": [(a, b) in tu_pairs for a, b in pairs]})
syn = table(syn_pairs, sig, se_sig, same, "synergy"); ctr = table(ctr_pairs, kap, se_kap, cross, "counter")
both = pd.concat([syn, ctr], ignore_index=True)
both["q"], _ = benjamini_hochberg(both.p_raw.to_numpy())          # one family: every interaction tested
syn, ctr = both[both.kind == "synergy"].copy(), both[both.kind == "counter"].copy()
syn.sort_values("effect_pp", ascending=False).to_csv(f"results/apm_pairwise_synergy_{TAG}.csv", index=False)
ctr.assign(abs_pp=lambda d: d.effect_pp.abs()).sort_values("abs_pp", ascending=False).drop(columns="abs_pp").to_csv(f"results/apm_pairwise_counter_{TAG}.csv", index=False)

# team-up premiums: old block vs synergy coefficient for the same pair
old_tu = pd.read_csv("results/apm_teamup_table_W0_specAplus.csv")
tu_rows = syn[syn.is_teamup].copy()
def find_pair(r):
    m = old_tu[((old_tu.anchor == r.hero_a) & (old_tu.partner == r.hero_b)) | ((old_tu.anchor == r.hero_b) & (old_tu.partner == r.hero_a))]
    return float(m.effect_pp.iloc[0]) if len(m) else np.nan
tu_rows["old_premium_pp"] = tu_rows.apply(find_pair, axis=1)
ok = tu_rows.old_premium_pp.notna()
cmp_tu = {"n_matched": int(ok.sum()), "spearman": float(stats.spearmanr(tu_rows.effect_pp[ok], tu_rows.old_premium_pp[ok])[0]),
          "mean_abs_diff_pp": float(np.mean(np.abs(tu_rows.effect_pp[ok] - tu_rows.old_premium_pp[ok])))}
def lookup(df, x, z):
    m = df[((df.hero_a == x) & (df.hero_b == z)) | ((df.hero_a == z) & (df.hero_b == x))]
    if not len(m): return None
    r = m.iloc[0]; sign = 1 if r.hero_a == x else -1
    return {"effect_pp_for_first_named": float(sign * r.effect_pp) if df is ctr else float(r.effect_pp), "se_pp": float(r.se_pp), "q": float(r.q), "n": int(r.n_cooccur)}
examples = {"black_panther_vs_thing": lookup(ctr, "Black Panther", "The Thing"), "jeff_with_dino": lookup(syn, "Jeff The Land Shark", "Devil Dinosaur")}
meta = {"tag": TAG, "min_pair": MIN_PAIR, "n_matches": n, "holdout_days": HOLDOUT_DAYS, "n_train": int(w_train.sum()), "n_holdout": int(is_test.sum()),
        "n_synergy_cols": Ps, "n_counter_cols": Pc, "free_synergy": int(N_s.shape[1]), "free_counter": int(N_c.shape[1]),
        "loglike_full": info["loglike"], "converged": info["converged"], "n_iter": info["n_iter"], "ladder": ladder,
        "main_effects_vs_specAplus": cmp_main, "teamups_vs_old_block": cmp_tu, "examples": examples,
        "n_significant_q05": {"synergy": int((syn.q < .05).sum()), "counter": int((ctr.q < .05).sum())},
        "sd_pp": {"synergy": float(syn.effect_pp.std()), "counter": float(ctr.effect_pp.std())}}
json.dump(meta, open(f"results/pairwise_meta_{TAG}.json", "w"), indent=1)
np.savez(f"results/pairwise_fit_{TAG}.npz", theta=theta, syn_pairs=np.array(syn_pairs), ctr_pairs=np.array(ctr_pairs))
log(json.dumps({k: v for k, v in meta.items() if k in ("ladder", "main_effects_vs_specAplus", "teamups_vs_old_block", "examples", "n_significant_q05", "sd_pp")}, indent=1))
log("top counters:\n" + ctr.assign(abs_pp=lambda d: d.effect_pp.abs()).sort_values("abs_pp", ascending=False).head(12)[["hero_a", "hero_b", "effect_pp", "se_pp", "q", "n_cooccur"]].to_string(index=False))
log("top synergies:\n" + syn.sort_values("effect_pp", ascending=False).head(12)[["hero_a", "hero_b", "effect_pp", "se_pp", "q", "n_cooccur", "is_teamup"]].to_string(index=False))
log("done")
