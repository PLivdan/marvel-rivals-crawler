"""Penalized (ridge-group) starting-lineup logit on the cached design (stage 4).

Reduced parameter vector phi = [maps | drank, drank*r | hero | hero*r | shape | shape*r | heromap | allied |
allied*r | opposing | opposing*r], with native coefficients recovered through the bases:
  hero  = hero_basis @ phi_hero,  shape = shape_basis @ phi_shape,  heromap = N_hm @ phi, pairs = N @ phi.
Penalty groups (native scale, lambda * ||theta||^2 / 2):
  G1 hero averages + shape averages      G2 allied averages       G3 opposing averages
  G4 hero slopes + shape slopes + hero-map   G5 allied + opposing slopes
Map intercepts and the two rank-imbalance terms are unpenalized (a 1e-8 ridge for numerical safety).
The lobby rank r and the imbalance Delta R are standardized with TRAINING rows only (weights > 0)."""
import json, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from lineup.design import pair_features, to_csr

GROUPS = ["G1_hero_shape", "G2_allied", "G3_opposing", "G4_slopes_heromap", "G5_pair_slopes"]


class LineupDesign:
    def __init__(self, d="results/lineup_design"):
        z = np.load(f"{d}/dense.npz", allow_pickle=True)
        self.X_map, self.drank, self.lobby = z["X_map"], z["drank"], z["lobby"]
        self.X_hero, self.X_shape, self.y, self.ts = z["X_hero"], z["X_shape"], z["y"], z["ts"]
        self.map_idx, self.s0, self.s1 = z["map_idx"], z["s0"], z["s1"]
        self.HM, self.S, self.C = sp.load_npz(f"{d}/HM.npz").tocsr(), sp.load_npz(f"{d}/S.npz").tocsr(), sp.load_npz(f"{d}/C.npz").tocsr()
        self.bans = sp.load_npz(f"{d}/bans.npz").tocsr()
        b = np.load(f"{d}/bases.npz")
        self.hero_basis, self.shape_basis, self.N_hm, self.N_S, self.N_C = b["hero_basis"], b["shape_basis"], b["N_hm"], b["N_S"], b["N_C"]
        self.meta = json.load(open(f"{d}/meta.json"))
        self.n = len(self.y); self.K = len(self.meta["hero_ids"]); self.M = len(self.meta["maps"])
        K = self.K
        self.allied_col = np.full(K * K, -1, dtype=np.int64); self.opp_col = np.full(K * K, -1, dtype=np.int64)
        for j, (a, b_) in enumerate(self.meta["allied_pairs"]): self.allied_col[a * K + b_] = j
        for j, (a, b_) in enumerate(self.meta["opposing_pairs"]): self.opp_col[a * K + b_] = j
        self.shape_index = {s: i for i, s in enumerate(self.meta["shape_labels"])}
        # reduced layout
        sizes = [("maps", self.M), ("drank", 2), ("hero", self.X_hero.shape[1]), ("hero_r", self.X_hero.shape[1]),
                 ("shape", self.X_shape.shape[1]), ("shape_r", self.X_shape.shape[1]), ("heromap", self.N_hm.shape[1]),
                 ("allied", self.N_S.shape[1]), ("allied_r", self.N_S.shape[1]), ("opp", self.N_C.shape[1]), ("opp_r", self.N_C.shape[1])]
        self.slices, o = {}, 0
        for name, sz in sizes:
            self.slices[name] = slice(o, o + sz); o += sz
        self.P = o
        self.group_of = {"hero": 0, "shape": 0, "allied": 1, "opp": 2, "hero_r": 3, "shape_r": 3, "heromap": 3, "allied_r": 4, "opp_r": 4}

    # ---- standardization -------------------------------------------------------------------------
    def scalers(self, w):
        m = w > 0
        return {"r_mu": float(np.average(self.lobby[m], weights=w[m])), "r_sd": float(np.sqrt(np.average((self.lobby[m] - np.average(self.lobby[m], weights=w[m])) ** 2, weights=w[m]))),
                "dr_sd": float(np.sqrt(np.average(self.drank[m] ** 2, weights=w[m])))}

    def r_std(self, sc, lobby=None):
        return ((self.lobby if lobby is None else lobby) - sc["r_mu"]) / sc["r_sd"]

    def dense_block(self, sc, rows=None):
        idx = slice(None) if rows is None else rows
        r = self.r_std(sc)[idx]; dr = (self.drank / sc["dr_sd"])[idx]
        Xh, Xs = self.X_hero[idx], self.X_shape[idx]
        return np.hstack([self.X_map[idx], dr[:, None], (dr * r)[:, None], Xh, Xh * r[:, None], Xs, Xs * r[:, None]])

    # ---- linear predictor ------------------------------------------------------------------------
    def split(self, phi):
        return {k: phi[s] for k, s in self.slices.items()}

    def eta(self, phi, sc, rows=None):
        idx = slice(None) if rows is None else rows
        p = self.split(phi); r = self.r_std(sc)[idx]
        D = self.dense_block(sc, rows)
        dvec = np.concatenate([p["maps"], p["drank"], p["hero"], p["hero_r"], p["shape"], p["shape_r"]])
        e = D @ dvec
        e += self.HM[idx] @ (self.N_hm @ p["heromap"])
        e += self.S[idx] @ (self.N_S @ p["allied"]) + r * (self.S[idx] @ (self.N_S @ p["allied_r"]))
        e += self.C[idx] @ (self.N_C @ p["opp"]) + r * (self.C[idx] @ (self.N_C @ p["opp_r"]))
        return e

    def native(self, phi):
        p = self.split(phi)
        return {"maps": p["maps"], "drank": p["drank"], "hero": self.hero_basis @ p["hero"], "hero_r": self.hero_basis @ p["hero_r"],
                "shape": self.shape_basis @ p["shape"], "shape_r": self.shape_basis @ p["shape_r"],
                "heromap": (self.N_hm @ p["heromap"]).reshape(self.K, self.M),
                "allied": self.N_S @ p["allied"], "allied_r": self.N_S @ p["allied_r"], "opp": self.N_C @ p["opp"], "opp_r": self.N_C @ p["opp_r"]}

    # ---- penalty matrix (reduced space, native-scale ridge) ---------------------------------------
    def penalty(self, lams):
        L = np.zeros((self.P, self.P))
        BB_h = self.hero_basis.T @ self.hero_basis; BB_s = self.shape_basis.T @ self.shape_basis
        for name, s in self.slices.items():
            if name in ("maps", "drank"):
                L[s, s] += 1e-8 * np.eye(s.stop - s.start); continue
            lam = lams[self.group_of[name]]
            if name in ("hero", "hero_r"): L[s, s] += lam * BB_h
            elif name in ("shape", "shape_r"): L[s, s] += lam * BB_s
            else: L[s, s] += lam * np.eye(s.stop - s.start)          # N bases are orthonormal
        return L

    # ---- reduced Hessian and gradient ---------------------------------------------------------
    def hessian(self, wv, r, sc):
        """B' X' diag(wv) X B assembled by blocks. wv: row weights (w * p(1-p)); r: standardized rank."""
        n = self.n
        D = self.dense_block(sc); DW = D * wv[:, None]
        HM, S, C = self.HM, self.S, self.C
        N_hm, N_S, N_C = self.N_hm, self.N_S, self.N_C
        def gram(A, B, k):                                   # A' diag(wv r^k) B as dense
            wk = wv * (r ** k) if k else wv
            return (A.T @ (B.multiply(wk[:, None]).tocsr())).toarray()
        def dsp(B, k):                                       # D' diag(wv r^k) B
            wk = (wv * r ** k) if k else wv
            return (B.T @ (D * wk[:, None])).T
        H = np.zeros((self.P, self.P)); sl = self.slices
        dslice = slice(0, sl["shape_r"].stop)
        H[dslice, dslice] = D.T @ DW
        blocks = {"heromap": (HM, N_hm, None), "allied": (S, N_S, "allied_r"), "opp": (C, N_C, "opp_r")}
        # dense x sparse
        for name, (B, N, rname) in blocks.items():
            G0 = dsp(B, 0) @ N; H[dslice, sl[name]] = G0; H[sl[name], dslice] = G0.T
            if rname:
                G1 = dsp(B, 1) @ N; H[dslice, sl[rname]] = G1; H[sl[rname], dslice] = G1.T
        # sparse x sparse
        names = list(blocks)
        for i, a in enumerate(names):
            Ba, Na, ra = blocks[a]
            for b in names[i:]:
                Bb, Nb, rb = blocks[b]
                G = [Na.T @ gram(Ba, Bb, k) @ Nb for k in range(3 if (ra and rb) else (2 if (ra or rb) else 1))]
                pairs = [(a, b, G[0])]
                if ra: pairs.append((ra, b, G[1]))
                if rb: pairs.append((a, rb, G[1]))
                if ra and rb: pairs.append((ra, rb, G[2]))
                for x, y_, Gk in pairs:
                    H[sl[x], sl[y_]] = Gk; H[sl[y_], sl[x]] = Gk.T
        return H

    def gradient(self, resid, r, sc):
        D = self.dense_block(sc)
        g = np.zeros(self.P); sl = self.slices
        g[0:sl["shape_r"].stop] = D.T @ resid
        g[sl["heromap"]] = self.N_hm.T @ (self.HM.T @ resid)
        g[sl["allied"]] = self.N_S.T @ (self.S.T @ resid); g[sl["allied_r"]] = self.N_S.T @ (self.S.T @ (resid * r))
        g[sl["opp"]] = self.N_C.T @ (self.C.T @ resid); g[sl["opp_r"]] = self.N_C.T @ (self.C.T @ (resid * r))
        return g

    # ---- fit -------------------------------------------------------------------------------------
    def fit(self, lams, w, phi0=None, max_iter=12, tol=1e-6, verbose=True, sc=None):
        t0 = time.time()
        sc = sc or self.scalers(w); r = self.r_std(sc); L = self.penalty(lams); y = self.y
        phi = np.zeros(self.P) if phi0 is None else phi0.copy()
        def objective(phi):
            e = self.eta(phi, sc); ll = float(np.sum(np.where(w > 0, w * (y * e - np.logaddexp(0, e)), 0.0)))
            return -ll + 0.5 * phi @ (L @ phi), ll
        obj, ll = objective(phi); converged = False; it = 0
        for it in range(1, max_iter + 1):
            e = self.eta(phi, sc); p = expit(e)
            wv = w * p * (1 - p); resid = w * (y - p)
            H = self.hessian(wv, r, sc) + L
            g = self.gradient(resid, r, sc) - L @ phi
            cf = cho_factor(H, lower=True, check_finite=False); d = cho_solve(cf, g, check_finite=False)
            step = 1.0
            while True:
                cand = phi + step * d; new, ll_new = objective(cand)
                if new <= obj + 1e-9 or step < 1e-4: break
                step *= 0.5
            change = float(np.max(np.abs(cand - phi))); rel = abs(obj - new) / max(abs(obj), 1.0)
            phi, obj, ll = cand, new, ll_new
            if verbose: print(f"    newton {it}: obj {obj:,.2f} loglik {ll:,.2f} max|d| {change:.2e} step {step} [{time.time()-t0:.0f}s]", flush=True)
            if change < tol or rel < 1e-9:
                converged = True; break
        return {"phi": phi, "lams": list(lams), "objective": obj, "loglik": ll, "n_iter": it, "converged": converged,
                "scalers": sc, "n_train": int((w > 0).sum()), "seconds": time.time() - t0}

    # ---- evaluation ---------------------------------------------------------------------------------
    def evaluate(self, fit, rows):
        e = self.eta(fit["phi"], fit["scalers"], rows); p = np.clip(expit(e), 1e-12, 1 - 1e-12); y = self.y[rows]
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        # Cox calibration on the log-odds scale
        Z = np.column_stack([np.ones(len(e)), e]); b = np.zeros(2)
        for _ in range(25):
            q = expit(Z @ b); W = q * (1 - q)
            b = b + np.linalg.solve(Z.T @ (Z * W[:, None]) + 1e-10 * np.eye(2), Z.T @ (y - q))
        return {"logloss": float(loss.mean()), "brier": float(np.mean((p - y) ** 2)), "n": int(len(y)),
                "cal_intercept": float(b[0]), "cal_slope": float(b[1]), "loss_vector": loss}

    # ---- features for arbitrary lineups (replacement predictions, symmetry checks) ------------------
    def eta_lineups(self, phi, sc, s0, s1, map_idx, lobby, drank):
        """Linear predictor for lineups given as index arrays (n x 6), rebuilt from scratch."""
        nat = self.native(phi); n = len(s0); K, M = self.K, self.M
        r = (lobby - sc["r_mu"]) / sc["r_sd"]; dr = drank / sc["dr_sd"]
        A = np.zeros((n, K)); B = np.zeros((n, K))
        for i in range(6):
            A[np.arange(n), s0[:, i]] = 1; B[np.arange(n), s1[:, i]] = 1
        x = A - B
        roles = np.array(self.meta["roles"])
        def shape_of(side):
            cnt = np.zeros((n, 3), int)
            for i in range(6):
                rr = roles[side[:, i]]; cnt[:, 0] += rr == "Tank"; cnt[:, 1] += rr == "Damage"; cnt[:, 2] += rr == "Support"
            return ["{}-{}-{}".format(*c) for c in cnt]
        shp = np.zeros((n, len(self.meta["shape_labels"])))
        for i, (a, b) in enumerate(zip(shape_of(s0), shape_of(s1))):
            if a in self.shape_index: shp[i, self.shape_index[a]] += 1
            if b in self.shape_index: shp[i, self.shape_index[b]] -= 1
        e = nat["maps"][map_idx] + dr * (nat["drank"][0] + nat["drank"][1] * r)
        e += x @ nat["hero"] + r * (x @ nat["hero_r"])
        e += shp @ nat["shape"] + r * (shp @ nat["shape_r"])
        e += np.einsum("nk,nk->n", x, nat["heromap"][:, map_idx].T)
        al, op = pair_features(s0, s1, K, self.allied_col, self.opp_col)
        Sn = to_csr(al, n, self.S.shape[1]); Cn = to_csr(op, n, self.C.shape[1])
        e += Sn @ nat["allied"] + r * (Sn @ nat["allied_r"]) + Cn @ nat["opp"] + r * (Cn @ nat["opp_r"])
        return e
