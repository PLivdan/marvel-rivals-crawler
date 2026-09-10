"""Stage 6: every reported quantity from one fitted model.

Conventions. r is the standardized lobby rank (mean pre-match score of the twelve starters).
Native coefficients at rank r: hero beta_h(r) = beta_h + r d_h; hero-by-map u_{h,v}; allied
S_hk(r) = S_hk + r T_hk (symmetric); opposing K_hl(r) = kappa_hl + r delta_hl stored for h<l and
antisymmetric, so Cm[h,l] = advantage to h's side when l starts on the other side.

Replacement value of hero h for incumbent g in slot (match m, side s), same role, log-odds:
  dEta = [beta_h - beta_g](r) + [u_{h,v} - u_{g,v}] + sum_{k in 5 allies}[S_hk - S_gk](r)
         + sum_{l in 6 opponents}[Cm_hl - Cm_gl](r)
and the probability change for side s is sigma(zeta + dEta) - sigma(zeta) with zeta the side's
fitted log-odds. Legality: h not already on that side, and h not banned in the match.
Four-lineup allied contrast: S_ab - mean_{a'} S_a'b - mean_{b'} S_ab' + mean_{a',b'} S_a'b' over
same-role alternatives (a' != a, b; b' != b, a). Opposing analogue on Cm.
"""
import numpy as np, pandas as pd
from scipy.special import expit


def matrices(d, phi, r):
    """Symmetric allied matrix and antisymmetric opposing matrix at standardized rank r (scalar)."""
    nat = d.native(phi); K = d.K
    Sm = np.zeros((K, K)); Cm = np.zeros((K, K))
    for j, (a, b) in enumerate(d.meta["allied_pairs"]):
        v = nat["allied"][j] + r * nat["allied_r"][j]; Sm[a, b] = Sm[b, a] = v
    for j, (a, b) in enumerate(d.meta["opposing_pairs"]):
        v = nat["opp"][j] + r * nat["opp_r"][j]; Cm[a, b] = v; Cm[b, a] = -v
    beta = nat["hero"] + r * nat["hero_r"]
    return nat, beta, Sm, Cm


def hero_rank_table(d, phi, sc, r_values):
    """Within-role hero coefficients (log-odds) at several standardized ranks, with the lobby score they correspond to."""
    nat = d.native(phi); rows = []
    for h in range(d.K):
        row = {"hero_id": d.meta["hero_ids"][h], "name": d.meta["names"][h], "role": d.meta["roles"][h], "beta": nat["hero"][h], "slope": nat["hero_r"][h]}
        for lab, r in r_values.items():
            row[f"beta_{lab}"] = nat["hero"][h] + r * nat["hero_r"][h]
            row[f"pp50_{lab}"] = 100 * (expit(row[f"beta_{lab}"]) - 0.5)          # illustrative marginal at a 50% baseline
        rows.append(row)
    return pd.DataFrame(rows)


def replacement_scores(d, phi, sc, pool, r_thirds=None, min_contexts=200):
    """Same-slot replacement values over a pool of design rows. Two declared summaries per hero:
    obs_meta_pp   = mean over legal contexts of 100*[p(side wins with h) - p(with the observed incumbent g)]
                    (alternatives = the same-role heroes actually starting in the slot);
    common_ref_pp = mean over legal contexts of 100*[p(with h) - mean over legal same-role heroes h' of p(with h')]
                    (alternative = the average legal same-role hero in that slot; same context pool for every hero,
                    coverage differs only through bans and uniqueness).
    Legality: h not already on that side, not banned in the match. Contexts by rank third also reported."""
    nat = d.native(phi); K = d.K; roles = np.array(d.meta["roles"])
    eta = d.eta(phi, sc, pool); rr = d.r_std(sc)[pool]; vmap = d.map_idx[pool]
    s0, s1 = d.s0[pool], d.s1[pool]; banned = d.bans[pool].toarray().astype(bool)
    # pair coefficient arrays (base and slope) as full matrices
    S0 = np.zeros((K, K)); S1 = np.zeros((K, K)); C0 = np.zeros((K, K)); C1 = np.zeros((K, K))
    for j, (a, b) in enumerate(d.meta["allied_pairs"]):
        S0[a, b] = S0[b, a] = nat["allied"][j]; S1[a, b] = S1[b, a] = nat["allied_r"][j]
    for j, (a, b) in enumerate(d.meta["opposing_pairs"]):
        C0[a, b] = nat["opp"][j]; C0[b, a] = -nat["opp"][j]; C1[a, b] = nat["opp_r"][j]; C1[b, a] = -nat["opp_r"][j]
    thirds = np.digitize(d.lobby[pool], r_thirds) if r_thirds is not None else np.zeros(len(pool), int)
    out = []
    # per role: contexts = (row, side, slot) where the incumbent has that role
    for role in ("Tank", "Damage", "Support"):
        heroes = [h for h in range(K) if roles[h] == role]
        ctx_rows, ctx_side, ctx_slot = [], [], []
        for side_id, side in ((0, s0), (1, s1)):
            for i in range(6):
                m = roles[side[:, i]] == role
                ctx_rows.append(np.nonzero(m)[0]); ctx_side.append(np.full(m.sum(), side_id)); ctx_slot.append(np.full(m.sum(), i))
        cr, cs, ci = np.concatenate(ctx_rows), np.concatenate(ctx_side), np.concatenate(ctx_slot)
        own = np.where(cs[:, None] == 0, s0[cr], s1[cr]); opp = np.where(cs[:, None] == 0, s1[cr], s0[cr])
        g = own[np.arange(len(cr)), ci]
        allies = np.array([np.delete(own[t], ci[t]) for t in range(len(cr))]) if len(cr) else np.zeros((0, 5), int)
        zeta = np.where(cs == 0, eta[cr], -eta[cr]); r_c = rr[cr]; v_c = vmap[cr]; th = thirds[cr]
        base_p = expit(zeta)
        # legality per hero: not already on own side, not banned
        legal = np.ones((len(cr), len(heroes)), bool)
        for jh, h in enumerate(heroes):
            legal[:, jh] = ~(own == h).any(1) & ~banned[cr, h]
        # probability of the side winning with EACH role hero in the slot (contexts x heroes), incumbent's own value = base
        gcol = g[:, None]
        P_h = np.zeros((len(cr), len(heroes))); DE = np.zeros((len(cr), len(heroes)))
        beta_g = nat["hero"][g] + r_c * nat["hero_r"][g]
        for jh, h in enumerate(heroes):
            d_eta = (nat["hero"][h] + r_c * nat["hero_r"][h]) - beta_g + nat["heromap"][h, v_c] - nat["heromap"][g, v_c]
            d_eta += ((S0[h, allies] - S0[gcol, allies]) + r_c[:, None] * (S1[h, allies] - S1[gcol, allies])).sum(1)
            d_eta += ((C0[h, opp] - C0[gcol, opp]) + r_c[:, None] * (C1[h, opp] - C1[gcol, opp])).sum(1)
            DE[:, jh] = d_eta; P_h[:, jh] = expit(zeta + d_eta)
        # declared common reference: the average legal same-role hero in the same slot and context
        legal_f = legal.astype(float); n_legal = legal_f.sum(1)
        ref_p = (P_h * legal_f).sum(1) / np.maximum(n_legal, 1)
        for jh, h in enumerate(heroes):
            L = legal[:, jh] & (g != h)
            dp = 100 * (P_h[:, jh] - base_p); d_eta = DE[:, jh]
            Lc = L & (n_legal >= 2)
            row = {"hero_id": d.meta["hero_ids"][h], "name": d.meta["names"][h], "role": role,
                   "contexts_total": int(len(cr)), "contexts_legal": int(L.sum()), "coverage": float(L.mean()),
                   "obs_meta_pp": float(dp[L].mean()) if L.sum() >= min_contexts else np.nan,
                   "obs_meta_logodds": float(d_eta[L].mean()) if L.sum() >= min_contexts else np.nan,
                   "common_ref_contexts": int(Lc.sum()),
                   "common_ref_pp": float((100 * (P_h[Lc, jh] - ref_p[Lc])).mean()) if Lc.sum() >= min_contexts else np.nan}
            for t in range(3 if r_thirds is not None else 1):
                m = L & (th == t)
                row[f"obs_meta_pp_third{t}"] = float(dp[m].mean()) if m.sum() >= min_contexts else np.nan
                row[f"n_third{t}"] = int(m.sum())
            out.append(row)
    return pd.DataFrame(out)


def pair_tables(d, phi, sc, r, support_rank):
    """Allied and opposing pair summaries at standardized rank r: raw coefficient, four-lineup contrast against
    same-role alternatives, and identifying support (distinct matches with nonzero signed feature, share by rank third)."""
    nat, beta, Sm, Cm = matrices(d, phi, r); K = d.K; roles = np.array(d.meta["roles"]); names = d.meta["names"]
    role_members = {ro: [h for h in range(K) if roles[h] == ro] for ro in ("Tank", "Damage", "Support")}
    tu = {tuple(p) for p in d.meta["teamup_pairs"]}
    def did(M, a, b):
        A = [x for x in role_members[roles[a]] if x not in (a, b)]; B = [x for x in role_members[roles[b]] if x not in (a, b)]
        if not A or not B: return np.nan
        return float(M[a, b] - M[np.ix_(A, [b])].mean() - M[np.ix_([a], B)].mean() + M[np.ix_(A, B)].mean())
    rows = []
    for j, (a, b) in enumerate(d.meta["allied_pairs"]):
        rows.append({"kind": "allied", "hero_a": names[a], "hero_b": names[b], "role_a": roles[a], "role_b": roles[b],
                     "coef": Sm[a, b], "slope": nat["allied_r"][j], "did": did(Sm, a, b), "is_teamup": (a, b) in tu,
                     "support": d.meta["allied_support"][j], **support_rank["allied"][j]})
    for j, (a, b) in enumerate(d.meta["opposing_pairs"]):
        rows.append({"kind": "opposing", "hero_a": names[a], "hero_b": names[b], "role_a": roles[a], "role_b": roles[b],
                     "coef": Cm[a, b], "slope": nat["opp_r"][j], "did": did(Cm, a, b), "is_teamup": False,
                     "support": d.meta["opposing_support"][j], **support_rank["opposing"][j]})
    return pd.DataFrame(rows)


def support_by_rank(d, thirds_edges, rows_mask=None):
    """Per pair: share of identifying matches in each lobby-rank third (and count), from the signed features."""
    thirds = np.digitize(d.lobby, thirds_edges)
    out = {}
    for name, B in (("allied", d.S), ("opposing", d.C)):
        Bn = B.copy(); Bn.data[:] = 1.0
        if rows_mask is not None:
            Bn = Bn.multiply(rows_mask.astype(float)[:, None]).tocsr()
        tot = np.asarray(Bn.sum(0)).ravel()
        per = [np.asarray(Bn.multiply((thirds == t).astype(float)[:, None]).sum(0)).ravel() for t in range(3)]
        out[name] = [{"support_dev": int(tot[j]), **{f"support_third{t}": int(per[t][j]) for t in range(3)}} for j in range(B.shape[1])]
    return out
