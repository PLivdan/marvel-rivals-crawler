"""Build and cache the complete starting-lineup design (stage 3).

Blocks (all from starting lineups, side 0 minus side 1 conventions):
  maps      alpha_v          one intercept per map (unpenalized)
  drank     Delta R          team mean pre-match score difference (raw; standardized per fold)
  hero      x_mh = A_mh - B_mh, within-role sum-to-zero basis (52 free of 55)
  shape     composition contrasts c(A) - c(B), sum-to-zero basis over shapes with >= MIN_SHAPE
            team-instances plus a pooled 'other'
  heromap   x_mh * 1[v_m = v], sparse; exact aliases removed (per-map hero sums, per-hero map sums)
  allied    z_mhk = A_mh A_mk - B_mh B_mk, h < k, every pair with nonzero identifying support
  opposing  w_mhk = A_mh B_mk - A_mk B_mh, h < k, every pair with nonzero identifying support
Rank slopes are not stored: the fitter forms them from the base blocks and the standardized lobby
rank r_m. The lobby rank r_m (mean pre-match score of the 12 starters) and Delta R are stored raw.

Constraints: candidate centering directions are verified numerically. A direction is imposed only
if its feature-space image lies in the span of the lower-order blocks on the data (relative
residual < TOL), i.e. it is an exact alias; near-collinear directions are left to the ridge.
"""
import json, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.linalg import null_space
from apm import contrasts
from apm.features import _starting_hero_per_player

MIN_SHAPE = 100
TOL = 1e-6


def log(t0, msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)


def load_snapshot(conn, snap_csv, max_play_time=None):
    snap = pd.read_csv(snap_csv)
    if max_play_time is not None:
        snap = snap[snap.timestamp <= max_play_time]
    return snap.sort_values("timestamp").reset_index(drop=True)


def lineups(conn, uids, hidx):
    per = _starting_hero_per_player(conn, uids)
    g = per.groupby(["match_uid", "camp"]).hero_id.apply(list)
    s0 = np.array([[hidx[h] for h in g[(u, 0)]] for u in uids])
    s1 = np.array([[hidx[h] for h in g[(u, 1)]] for u in uids])
    assert s0.shape == (len(uids), 6) and s1.shape == (len(uids), 6)
    for s in (s0, s1):                                   # legal lineups: six distinct heroes per side
        assert all(len(set(row)) == 6 for row in s), "duplicate starter within a side"
    return s0, s1


def pair_features(s0, s1, K, allied_col=None, opp_col=None):
    """Signed allied and opposing features for lineup arrays (n x 6 each).
    Returns COO triplets keyed by pair index (a<b) unless column maps are given."""
    n = len(s0)
    rows, cols, vals = [], [], []
    for side, sign in ((s0, 1.0), (s1, -1.0)):
        for i in range(6):
            for j in range(i + 1, 6):
                a, b = np.minimum(side[:, i], side[:, j]), np.maximum(side[:, i], side[:, j])
                key = a * K + b
                c = key if allied_col is None else allied_col[key]
                m = c >= 0
                rows.append(np.nonzero(m)[0]); cols.append(c[m]); vals.append(np.full(int(m.sum()), sign))
    allied = (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))
    rows, cols, vals = [], [], []
    for i in range(6):
        for j in range(6):
            a, b = s0[:, i], s1[:, j]                           # a on side 0, b on side 1
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            key = lo * K + hi
            c = key if opp_col is None else opp_col[key]
            m = (c >= 0) & (a != b)
            rows.append(np.nonzero(m)[0]); cols.append(c[m]); vals.append(np.where(a[m] < b[m], 1.0, -1.0))
    opposing = (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))
    return allied, opposing


def to_csr(trip, n, P):
    r, c, v = trip
    M = sp.coo_matrix((v, (r, c)), shape=(n, P)).tocsr(); M.sum_duplicates(); M.eliminate_zeros()
    return M


def build(conn, snap_csv="results/dev_snapshot/matches.csv", out="results/lineup_design", max_play_time=None, limit=None):
    t0 = time.time()
    snap = load_snapshot(conn, snap_csv, max_play_time)
    if limit:
        snap = snap.iloc[:limit].reset_index(drop=True)
    uids = list(snap.match_uid); n = len(uids); uset = set(uids)
    y = snap.camp0_win.to_numpy().astype(float); ts = snap.timestamp.to_numpy()
    info = {h: (nm, r) for h, nm, r in conn.execute("SELECT hero_id, name, role FROM hero_info")}
    # ---- lineups ------------------------------------------------------------------------------
    per = _starting_hero_per_player(conn, uids)
    hero_ids = sorted(per.hero_id.unique()); K = len(hero_ids); hidx = {h: i for i, h in enumerate(hero_ids)}
    names = [info[h][0] for h in hero_ids]; roles = [info[h][1] for h in hero_ids]
    g = per.groupby(["match_uid", "camp"]).hero_id.apply(list)
    s0 = np.array([[hidx[h] for h in g[(u, 0)]] for u in uids]); s1 = np.array([[hidx[h] for h in g[(u, 1)]] for u in uids])
    assert s0.shape == (n, 6) and s1.shape == (n, 6)
    # Legal-lineup invariant: six distinct starters per side. A duplicated first-appearance hero on one
    # side cannot be a simultaneous start (hero uniqueness within a team), so the starting record of that
    # match is unverifiable. Such matches are EXCLUDED, never collapsed; the count is recorded.
    legal = np.array([len(set(a)) == 6 and len(set(b)) == 6 for a, b in zip(s0, s1)])
    n_illegal = int((~legal).sum())
    snap = snap[legal].reset_index(drop=True); uids = list(snap.match_uid); uset = set(uids); n = len(uids)
    y, ts, s0, s1 = y[legal], ts[legal], s0[legal], s1[legal]
    log(t0, f"{n:,} matches after excluding {n_illegal:,} with a duplicated starter on one side; {K} heroes")
    # ---- match-level covariates --------------------------------------------------------------
    mp = pd.DataFrame(conn.execute("SELECT match_uid, camp, new_score-add_score FROM match_players").fetchall(), columns=["match_uid", "camp", "pre"])
    mp = mp[mp.match_uid.isin(uset)]
    lob = mp.groupby("match_uid").pre.mean().loc[uids].to_numpy()
    side_mean = mp.groupby(["match_uid", "camp"]).pre.mean().unstack()
    drank = (side_mean[0] - side_mean[1]).loc[uids].to_numpy()
    map_of = dict(conn.execute("SELECT match_uid, map_id FROM matches"))
    maps = sorted({map_of[u] for u in uids}); midx = {m: i for i, m in enumerate(maps)}
    map_idx = np.array([midx[map_of[u]] for u in uids])
    bans = pd.DataFrame(conn.execute("SELECT match_uid, hero_id FROM match_bans WHERE hero_id>0").fetchall(), columns=["match_uid", "hero_id"])
    bans = bans[bans.match_uid.isin(uset) & bans.hero_id.isin(hidx)]
    uidx = {u: i for i, u in enumerate(uids)}
    ban_mat = sp.coo_matrix((np.ones(len(bans)), (bans.match_uid.map(uidx).to_numpy(), bans.hero_id.map(hidx).to_numpy())), shape=(n, K)).tocsr()
    ban_mat.data[:] = 1.0
    log(t0, f"covariates: {len(maps)} maps, bans for {ban_mat.getnnz(axis=1).astype(bool).sum():,} matches")
    # ---- dense blocks -------------------------------------------------------------------------
    X_map = np.zeros((n, len(maps))); X_map[np.arange(n), map_idx] = 1.0
    A = np.zeros((n, K)); B = np.zeros((n, K))
    for i in range(6):
        A[np.arange(n), s0[:, i]] = 1.0; B[np.arange(n), s1[:, i]] = 1.0
    X_hero_raw = A - B
    hero_basis = contrasts.within_role_basis(roles)                  # K x (K-3)
    X_hero = X_hero_raw @ hero_basis
    def shape_of(side):
        cnt = np.zeros((n, 3))
        for i in range(6):
            r = np.array(roles)[side[:, i]]
            cnt[:, 0] += r == "Tank"; cnt[:, 1] += r == "Damage"; cnt[:, 2] += r == "Support"
        return ["{}-{}-{}".format(*map(int, c)) for c in cnt]
    sh0, sh1 = shape_of(s0), shape_of(s1)
    counts = pd.Series(sh0 + sh1).value_counts()
    common = sorted(counts[counts >= MIN_SHAPE].index); labels = common + ["other"]
    canon = lambda s: s if s in common else "other"
    shape_raw = np.zeros((n, len(labels)))
    for i, (a, b) in enumerate(zip(sh0, sh1)):
        shape_raw[i, labels.index(canon(a))] += 1.0; shape_raw[i, labels.index(canon(b))] -= 1.0
    shape_basis = contrasts.sum_to_zero_basis(len(labels))
    X_shape = shape_raw @ shape_basis
    log(t0, f"dense blocks: hero {X_hero.shape[1]}, shape {X_shape.shape[1]} ({len(common)} shapes + other)")
    # ---- hero-by-map (sparse) ------------------------------------------------------------------
    hm_rows, hm_cols, hm_vals = [], [], []
    for i in range(6):
        for side, sign in ((s0, 1.0), (s1, -1.0)):
            hm_rows.append(np.arange(n)); hm_cols.append(side[:, i] * len(maps) + map_idx); hm_vals.append(np.full(n, sign))
    HM = to_csr((np.concatenate(hm_rows), np.concatenate(hm_cols), np.concatenate(hm_vals)), n, K * len(maps))
    # ---- pair blocks (sparse), all pairs with nonzero identifying support ---------------------
    allied_t, opp_t = pair_features(s0, s1, K)
    def keep_pairs(trip):
        r, c, v = trip
        M = to_csr((r, c, v), n, K * K)
        support = M.getnnz(axis=0)                                 # distinct matches with nonzero signed feature
        keys = np.nonzero(support)[0]
        colmap = np.full(K * K, -1, dtype=np.int64); colmap[keys] = np.arange(len(keys))
        return M[:, keys].tocsr(), keys, colmap, support[keys]
    S, allied_keys, allied_col, allied_support = keep_pairs(allied_t)
    C, opp_keys, opp_col, opp_support = keep_pairs(opp_t)
    allied_pairs = [(int(k // K), int(k % K)) for k in allied_keys]; opp_pairs = [(int(k // K), int(k % K)) for k in opp_keys]
    log(t0, f"pairs: allied {S.shape[1]} (support median {np.median(allied_support):.0f}), opposing {C.shape[1]} (median {np.median(opp_support):.0f}); hero-map nnz {HM.nnz:,}")
    # ---- exact-alias verification and constraint bases -----------------------------------------
    lower = np.hstack([X_map, drank[:, None], X_hero, X_shape])       # lower-order blocks (base, not slopes)
    rng = np.random.default_rng(0); sub = np.sort(rng.choice(n, size=min(n, 60000), replace=False))
    L = lower[sub]; Q, _ = np.linalg.qr(L)
    def alias_residual(vec):                                        # relative residual of vec after projection on lower blocks
        v = vec[sub]; nv = np.linalg.norm(v)
        return float(np.linalg.norm(v - Q @ (Q.T @ v)) / nv) if nv > 0 else 0.0
    def verify_and_basis(M, candidates, label):
        """candidates: list of (name, coefficient-direction vector over M's columns). Impose only exact aliases."""
        imposed, report = [], []
        for name, d in candidates:
            res = alias_residual(np.asarray(M @ d).ravel())
            exact = res < TOL
            report.append({"constraint": name, "relative_residual": res, "imposed": bool(exact)})
            if exact:
                imposed.append(d)
        Acon = np.array(imposed) if imposed else np.zeros((0, M.shape[1]))
        N = null_space(Acon) if len(imposed) else np.eye(M.shape[1])
        log(t0, f"{label}: {len(imposed)} of {len(candidates)} candidate constraints are exact aliases -> {N.shape[1]} free of {M.shape[1]}")
        return N, report
    # hero-map candidates: per-map sum over heroes; per-hero sum over maps
    cand = []
    for v in range(len(maps)):
        d = np.zeros(K * len(maps)); d[np.arange(K) * len(maps) + v] = 1; cand.append((f"heromap: map {maps[v]} sum over heroes", d))
    for h in range(K):
        d = np.zeros(K * len(maps)); d[h * len(maps) + np.arange(len(maps))] = 1; cand.append((f"heromap: hero {names[h]} sum over maps", d))
    N_hm, rep_hm = verify_and_basis(HM, cand, "hero-map")
    # allied candidates: per-hero sums; per-role-pair sums
    cand = []
    for h in range(K):
        d = np.array([1.0 if h in p else 0.0 for p in allied_pairs]); cand.append((f"allied: hero {names[h]} sum over partners", d))
    rp = lambda p: "-".join(sorted([roles[p[0]], roles[p[1]]]))
    for t in sorted({rp(p) for p in allied_pairs}):
        d = np.array([1.0 if rp(p) == t else 0.0 for p in allied_pairs]); cand.append((f"allied: role pair {t} sum", d))
    N_S, rep_S = verify_and_basis(S, cand, "allied")
    cand = []
    for h in range(K):
        d = np.array([1.0 if p[0] == h else (-1.0 if p[1] == h else 0.0) for p in opp_pairs]); cand.append((f"opposing: hero {names[h]} sum over opponents", d))
    N_C, rep_C = verify_and_basis(C, cand, "opposing")
    # ---- sign-symmetry check on random matches ---------------------------------------------------
    chk = rng.choice(n, 2000, replace=False)
    al2, op2 = pair_features(s1[chk], s0[chk], K)                    # camps swapped
    S2 = to_csr(al2, len(chk), K * K)[:, allied_keys]; C2 = to_csr(op2, len(chk), K * K)[:, opp_keys]
    sym_ok = (abs(S[chk] + S2).max() == 0) and (abs(C[chk] + C2).max() == 0) and np.allclose(X_hero_raw[chk], -(B - A)[chk])
    log(t0, f"sign symmetry under camp swap: {'OK' if sym_ok else 'FAILED'}")
    assert sym_ok
    # ---- team-up metadata (labels only) ----------------------------------------------------------
    tu = {}
    for tid, hid in conn.execute("SELECT teamup_id, hero_id FROM teamup_heroes"):
        tu.setdefault(tid, []).append(hid)
    tu_pairs = sorted({(min(hidx[a], hidx[b]), max(hidx[a], hidx[b])) for hs in tu.values() if len(hs) == 2 for a, b in [hs] if a in hidx and b in hidx})
    # ---- cache -----------------------------------------------------------------------------------
    np.savez(f"{out}/dense.npz", X_map=X_map, drank=drank, lobby=lob, X_hero=X_hero, X_shape=X_shape, y=y, ts=ts,
             map_idx=map_idx, s0=s0, s1=s1, X_hero_raw_cols=np.arange(K))
    sp.save_npz(f"{out}/HM.npz", HM); sp.save_npz(f"{out}/S.npz", S); sp.save_npz(f"{out}/C.npz", C); sp.save_npz(f"{out}/bans.npz", ban_mat)
    np.savez(f"{out}/bases.npz", hero_basis=hero_basis, shape_basis=shape_basis, N_hm=N_hm, N_S=N_S, N_C=N_C)
    meta = {"n": n, "hero_ids": [int(h) for h in hero_ids], "names": names, "roles": roles, "maps": [int(m) for m in maps],
            "shape_labels": labels, "min_shape": MIN_SHAPE, "allied_pairs": allied_pairs, "opposing_pairs": opp_pairs,
            "allied_support": [int(x) for x in allied_support], "opposing_support": [int(x) for x in opp_support],
            "teamup_pairs": [list(p) for p in tu_pairs], "play_time_min": int(ts.min()), "play_time_max": int(ts.max()),
            "match_uids": uids, "n_excluded_duplicate_starters": n_illegal,
            "constraints": {"heromap": rep_hm, "allied": rep_S, "opposing": rep_C},
            "free": {"maps": len(maps), "drank": 1, "hero": X_hero.shape[1], "shape": X_shape.shape[1], "heromap": int(N_hm.shape[1]),
                     "allied": int(N_S.shape[1]), "opposing": int(N_C.shape[1])}, "tol": TOL, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(f"{out}/meta.json", "w"))
    log(t0, f"cached to {out}: free base parameters {sum(meta['free'].values())} (+ slopes for hero/shape/allied/opposing, + drank x r)")
    return meta


if __name__ == "__main__":
    import sys, db
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "results/lineup_design"
    import os; os.makedirs(out, exist_ok=True)
    conn = db.connect_readonly("data/rivals.db")
    build(conn, out=out, limit=limit)
