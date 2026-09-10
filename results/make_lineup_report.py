"""Revised report for the unified starting-lineup model -- from saved CSV/JSON only, no database access,
no refitting. Usage: python3 results/make_lineup_report.py
Figure inputs are listed in results/lineup_report_figures.md. Every number in the text is injected;
unfilled placeholders abort the build."""
import hashlib, json, os, re
import numpy as np, pandas as pd

SEL = "results/lineup_selected"                       # the selected development fit (451,204 matches)
def need(path, kind="csv"):
    try:
        return pd.read_csv(path) if kind == "csv" else json.load(open(path))
    except FileNotFoundError:
        raise SystemExit(f"{path} missing")

H = need(f"{SEL}_hero_by_rank.csv"); R = need(f"{SEL}_replacement.csv"); P = need(f"{SEL}_pairs_p50.csv"); SM = need(f"{SEL}_summary_meta.json", "json")
TUN = need("results/lineup_tuning.json", "json"); BASE = need("results/lineup_baseline_dev.json", "json")
DS = need("results/lineup_design/summary.json", "json"); COV = need("results/rank_coverage.json", "json")
CONF = need("results/lineup_confirmation.json", "json"); SNAP = need("results/dev_snapshot/filters.json", "json")
LOCK = need("results/lineup_lock.json", "json"); PUB = json.load(open("results/lineup_published.json")) if os.path.exists("results/lineup_published.json") else None
PUBR = pd.read_csv("results/lineup_published_replacement.csv") if os.path.exists("results/lineup_published_replacement.csv") else None
OLD = {w: need(f"results/apm_hero_table_{w}_full.csv") for w in ("W0", "W2", "W1")}; OLDM = need("results/apm_run_meta_W1_full.json", "json")
BOOT_REPS = int(np.load("results/lineup_bootstrap_draws.npz")["replacement_obs_meta"].shape[0]) if os.path.exists("results/lineup_bootstrap_draws.npz") else 0

def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s
def num(x): return f"{int(x):,}"
def f2(x): return "---" if pd.isna(x) else f"{x:+.2f}"
def f3(x): return "---" if pd.isna(x) else f"{x:+.3f}"
def oxford(xs):
    xs = list(xs); return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + ", and " + xs[-1]
def side_by_side(blocks, ncol):
    depth = max(len(b) for b in blocks); blank = " & " * (ncol - 1)
    return "\n".join(" & ".join(b[i] if i < len(b) else blank for b in blocks) + r" \\" for i in range(depth))
def panel(text, ncol): return rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{text}}}}}"

# ---- numbers for the prose --------------------------------------------------------------------------
Ri = R.set_index("name")
def rv(n, col="common_ref_pp"): return float(Ri.loc[n, col])
lams = SM["lams"]; best_key = ",".join(f"{l:g}" for l in lams)
evals = TUN["evaluations"]; sel = next(e for e in evals if e["key"] == best_key and e["sub"] == 1.0)
RED = TUN["reduced"]
def ev_for(k): return next(e for e in evals if e["key"] == k and e["sub"] == 1.0)
LAD = [("Heroes, composition, maps, rank imbalance", ev_for(RED["no_pairs_no_slopes_no_heromap"])),
       ("+ rank slopes and hero-by-map", ev_for(RED["no_pairs"])),
       ("+ shrunk allied and opposing pairs", ev_for(RED["no_pair_slopes"])),
       ("+ pair rank slopes (selected fit)", ev_for(RED["selected"]))]
FREE = DS["free"]; N_PAIRS = FREE["allied"] + FREE["opposing"]; N_SLOPES = FREE["hero"] + FREE["shape"]
_base = FREE["maps"] + 2 + FREE["hero"] + FREE["shape"]
PARAMS = [_base, _base + N_SLOPES + FREE["heromap"], _base + N_SLOPES + FREE["heromap"] + N_PAIRS, _base + N_SLOPES + FREE["heromap"] + 2 * N_PAIRS]
BASE_MEAN = BASE["mean_val_logloss"]; SEL_MEAN = sel["mean_val_logloss"]; GAIN_DEV = BASE_MEAN - SEL_MEAN
NOPAIR_MEAN = ev_for(RED["no_pairs"])["mean_val_logloss"]; PAIR_GAIN = NOPAIR_MEAN - ev_for(RED["no_pair_slopes"])["mean_val_logloss"]
SLOPE_GAIN = ev_for(RED["no_pair_slopes"])["mean_val_logloss"] - SEL_MEAN
sel_third_slopes = [np.mean([f["by_third"][t]["cal_slope"] for f in sel["folds"]]) for t in range(3)]
thirds = SM["thirds"]; lob = SM["lobby_at"]
top = {ro: R[R.role == ro].sort_values("common_ref_pp", ascending=False) for ro in ("Tank", "Damage", "Support")}
def tops(ro, k): return oxford(f"{esc(r['name'])} ({r.common_ref_pp:+.2f})" for _, r in top[ro].head(k).iterrows())
def bottoms(ro, k): return oxford(f"{esc(r['name'])} ({r.common_ref_pp:+.2f})" for _, r in top[ro].tail(k).iloc[::-1].iterrows())
R["chg"] = R.obs_meta_pp_third2 - R.obs_meta_pp_third0
N_MOVE1 = int((R.chg.abs() >= 1.0).sum())
MIN_SUPPORT = 500
A = P[(P.kind == "allied") & (P.support_dev >= MIN_SUPPORT)].copy(); O = P[(P.kind == "opposing") & (P.support_dev >= MIN_SUPPORT)].copy()
TU = A[A.is_teamup].sort_values("did", ascending=False); NT = A[~A.is_teamup]
def pr(r): return f"{esc(r.hero_a)} with {esc(r.hero_b)} ({r.did:+.3f})"
def find_pair(df, x, y):
    m = df[((df.hero_a == x) & (df.hero_b == y)) | ((df.hero_a == y) & (df.hero_b == x))]; return m.iloc[0] if len(m) else None
BCSM = find_pair(A, "Black Cat", "Spider-man")
def opp_phrase(r):
    a, b, dd = (r.hero_a, r.hero_b, r.did) if r.did >= 0 else (r.hero_b, r.hero_a, -r.did); return f"{esc(a)} against {esc(b)} ({dd:+.3f})"
O_TOP = O.assign(a=O.did.abs()).sort_values("a", ascending=False)
TM = find_pair(O, "The Thing", "Magik")
OLD_SD = {w: float(OLD[w].within_role_pp.std()) for w in OLD}; OLD_MAGIK = {w: float(OLD[w][OLD[w].name == "Magik"].within_role_pp.iloc[0]) for w in OLD}
PUB_NOTE = ""
if PUB and PUBR is not None:
    pubs = PUBR.set_index("name").common_ref_pp; devs = Ri.common_ref_pp.reindex(pubs.index)
    PUB_NOTE = (f"Refitting the same specification and penalties on all {num(PUB['n'])} eligible matches, including the evaluation slice, "
                f"changes the common-reference replacement values by {float((pubs - devs).abs().mean()):.2f} points on average, with a rank correlation of "
                f"{pubs.corr(devs, method='spearman'):.4f} between the two sets; the tables in this report use the development fit.")

# ---- tables for the appendix ------------------------------------------------------------------------------
Hj = H.merge(R[["hero_id", "obs_meta_pp", "common_ref_pp", "obs_meta_pp_third0", "obs_meta_pp_third1", "obs_meta_pp_third2", "coverage", "contexts_legal"]], on="hero_id")
def hero_row(r):
    return (f"{esc(r['name'])} & {f3(r.beta_p50)} & {f3(r.slope)} & {f2(r.common_ref_pp)} & {f2(r.obs_meta_pp)} & {f2(r.obs_meta_pp_third0)} & "
            f"{f2(r.obs_meta_pp_third2)} & {100*r.coverage:.0f}")
def hero_block(ro, letter):
    d = Hj[Hj.role == ro].sort_values("common_ref_pp", ascending=False)
    return [panel(f"Panel {letter}: {ro} ({len(d)} heroes)", 8)] + [hero_row(r) for _, r in d.iterrows()]
HERO_BODY = side_by_side([hero_block("Tank", "A") + hero_block("Support", "C"), hero_block("Damage", "B")], 8)
_hero_hdr = (r"Hero & $\beta_h$ & slope & \makecell[r]{Common\\ref.} & \makecell[r]{Obs.\\incumbent} & \makecell[r]{Lower\\third} & \makecell[r]{Upper\\third} & "
             r"\makecell[r]{Slots\\\%}")
_hero_span = (r"& \multicolumn{2}{c}{Coefficient} & \multicolumn{2}{c}{Replacement, pp} & \multicolumn{2}{c}{Obs.\ incumbent} & ")
HERO_HDR = (_hero_span + " & " + _hero_span + r" \\" + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
            + r"\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}" + "\n" + _hero_hdr + " & " + _hero_hdr + r" \\")
def pair_row(r, opp=False, short=False):
    if opp:
        a, b, dd = (r.hero_a, r.hero_b, r.did) if r.did >= 0 else (r.hero_b, r.hero_a, -r.did); label = f"{esc(a)} against {esc(b)}"
    else:
        label = f"{esc(r.hero_a)} + {esc(r.hero_b)}"; dd = r.did
    return f"{label} & {dd:+.3f} & {num(r.support_dev)}" + ("" if short else f" & {100*r.support_third2/max(r.support_dev,1):.0f}")
_per = (len(TU) + 2) // 3
TU_BODY = side_by_side([[pair_row(r, short=True) for r in TU.iloc[b * _per:(b + 1) * _per].itertuples()] for b in range(3)], 3)
_tu_hdr = r"Pair & Contrast & Matches"
TU_HDR = " & ".join([_tu_hdr] * 3) + r" \\"
NT_POS = NT.sort_values("did", ascending=False).head(25); NT_NEG = NT.sort_values("did").head(25)
NT_ROWS = side_by_side([[pair_row(r) for r in NT_POS.itertuples()], [pair_row(r) for r in NT_NEG.itertuples()]], 4)
O50 = O_TOP.head(50)
O_ROWS = side_by_side([[pair_row(r, True) for r in O50.iloc[:25].itertuples()], [pair_row(r, True) for r in O50.iloc[25:].itertuples()]], 4)
def tuning_rows(k=8):
    ev = sorted([e for e in evals if e["sub"] == 1.0], key=lambda e: e["mean_val_logloss"])[:k]
    return "\n".join(rf"{esc(e['key'])} & {e['mean_val_logloss']:.5f} & {e['mean_cal_slope']:.3f} & {'yes' if all(f['converged'] for f in e['folds']) else 'no'} \\" for e in ev)
def ladder_rows():
    rows = [rf"Headline spec., refitted on development rows & {BASE['k']} & " + " & ".join(f"{f['logloss']:.5f}" for f in BASE["folds"]) + rf" & {BASE_MEAN:.5f} & --- \\"]
    for (label, e), k in zip(LAD, PARAMS):
        rows.append(rf"{esc(label)} & {num(k)} & " + " & ".join(f"{f['logloss']:.5f}" for f in e["folds"]) + rf" & {e['mean_val_logloss']:.5f} & {e['mean_cal_slope']:.3f} \\")
    return "\n".join(rows)
CAL_ROWS = "\n".join([rf"Development folds (mean of three) & {np.mean([f['cal_slope'] for f in sel['folds']]):.3f} & " + " & ".join(f"{s:.3f}" for s in sel_third_slopes) + r" \\",
                      rf"Internal temporal evaluation & {CONF['cal_slope']:.3f} & " + " & ".join(f"{t['cal_slope']:.3f}" for t in CONF["by_third"]) + r" \\"])
q = COV["lobby_mean_quantiles"]; pq = COV["player_score_quantiles"]; mx = COV["lobby_max_player_quantiles"]
QK = ("0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1.0")
COV_ROWS = "\n".join([r"Lobby mean score & " + " & ".join(f"{q[k]:,.0f}" for k in QK) + r" \\",
                      r"Highest player in lobby & " + " & ".join(f"{mx[k]:,.0f}" for k in QK) + r" \\",
                      r"Individual player score & " + " & ".join(f"{pq[k]:,.0f}" for k in QK) + r" \\"])
TUNE8 = sorted([e for e in evals if e["sub"] == 1.0], key=lambda e: e["mean_val_logloss"])[:8]

TEX = open("results/lineup_report_template.tex").read()
subs = {
    "SEASON": SNAP["season_label"], "FROZEN": SNAP["frozen_at_utc"][:10],
    "N_SNAP": num(SNAP["n_matches"]), "PLAY_MIN": SNAP["play_time_min_utc"][:16].replace("T", " "), "PLAY_MAX": SNAP["play_time_max_utc"][:16].replace("T", " "),
    "N_ILLEGAL": num(DS["n_excluded_duplicate_starters"]), "N_DESIGN": num(DS["n"]), "N_DEV": num(BASE["n_dev"]), "N_CONF": num(BASE["n_conf"]),
    "CONF_CUT": "4 September 11:00", "N_FREE": num(sum(FREE.values()) + FREE["hero"] + FREE["shape"] + FREE["allied"] + FREE["opposing"] + 1),
    "N_CONSTRAINTS": str(sum(v["imposed"] for v in DS["constraints"].values())), "N_CANDIDATES": str(sum(v["candidates"] for v in DS["constraints"].values())),
    "MAX_RESID": (lambda m, e: rf"${m:.1f}\times10^{{{e}}}$")(*(lambda x: (x / 10 ** int(np.floor(np.log10(x))), int(np.floor(np.log10(x)))))(max(v['max_residual_imposed'] for v in DS['constraints'].values()))),
    "LOB_P5": f"{COV['lobby_mean_quantiles']['0.05']:,.0f}", "LOB_P95": f"{COV['lobby_mean_quantiles']['0.95']:,.0f}", "LOB_P50": f"{lob['p50']:,.0f}",
    "N_ABOVE_5000": num(COV["matches_with_lobby_mean_above"]["5000"]), "N_PL_5500": num(COV["distinct_players_above"]["5500"]), "COV_ROWS": COV_ROWS,
    "R_SD": f"{SM['scalers']['r_sd']:,.0f}", "POOL_N": num(SM["pool_size"]), "THIRD_LO": f"{thirds[0]:,.0f}", "THIRD_HI": f"{thirds[1]:,.0f}",
    "MANTIS_CR": f2(rv("Mantis")), "THING_CR": f2(rv("The Thing")), "MAGIK_CR": f2(rv("Magik")),
    "ELSA_LO": f2(rv("Elsa Bloodstone", "obs_meta_pp_third0")), "ELSA_HI": f2(rv("Elsa Bloodstone", "obs_meta_pp_third2")),
    "WOLV_LO": f2(rv("Wolverine", "obs_meta_pp_third0")), "WOLV_HI": f2(rv("Wolverine", "obs_meta_pp_third2")),
    "MANTIS_LO": f2(rv("Mantis", "obs_meta_pp_third0")), "MANTIS_HI": f2(rv("Mantis", "obs_meta_pp_third2")),
    "BP_LO": f2(rv("Black Panther", "obs_meta_pp_third0")), "BP_HI": f2(rv("Black Panther", "obs_meta_pp_third2")),
    "CD_LO": f2(rv("Cloak & Dagger", "obs_meta_pp_third0")), "CD_HI": f2(rv("Cloak & Dagger", "obs_meta_pp_third2")),
    "ULTRON_LO": f2(rv("Ultron", "obs_meta_pp_third0")), "ULTRON_HI": f2(rv("Ultron", "obs_meta_pp_third2")), "N_MOVE1": str(N_MOVE1),
    "STRANGE_CR": f2(rv("Doctor Strange")), "STRANGE_OM": f2(rv("Doctor Strange", "obs_meta_pp")),
    "BLACKCAT_COV": f"{100*rv('Black Cat', 'coverage'):.0f}\\%", "WOLV_COV": f"{100*rv('Wolverine', 'coverage'):.0f}\\%",
    "BLACKCAT_CR": f2(rv("Black Cat")), "SPIDER_CR": f2(rv("Spider-man")), "BCSM_DID": f3(BCSM.did) if BCSM is not None else "not estimated",
    "TOP_TANK": tops("Tank", 3), "BOT_TANK": bottoms("Tank", 2), "TOP_DMG": tops("Damage", 3), "BOT_DMG": bottoms("Damage", 2),
    "TOP_SUP2": oxford(f"{esc(r['name'])} ({r.common_ref_pp:+.2f})" for _, r in top["Support"].iloc[1:3].iterrows()), "BOT_SUP": bottoms("Support", 2),
    "TU_TOP3": oxford(pr(r) for r in TU.head(3).itertuples()), "NT_TOP3": oxford(pr(r) for r in NT.sort_values("did", ascending=False).head(3).itertuples()),
    "NT_BOT3": oxford(pr(r) for r in NT.sort_values("did").head(3).itertuples()), "O_TOP4": oxford(opp_phrase(r) for r in O_TOP.head(4).itertuples()),
    "TM_DID": f3(-TM.did if TM.hero_a == "Magik" else TM.did) if TM is not None else "---", "TM_N": num(TM.support_dev) if TM is not None else "---",
    "MIN_SUPPORT": num(MIN_SUPPORT), "SEL_LL": f"{SEL_MEAN:.5f}", "BASE_LL": f"{BASE_MEAN:.5f}", "GAIN_DEV": f"{GAIN_DEV:.5f}",
    "GAIN_SLOPES": f"{LAD[0][1]['mean_val_logloss'] - NOPAIR_MEAN:.5f}", "PAIR_GAIN": f"{PAIR_GAIN:.5f}", "SLOPE_GAIN": f"{SLOPE_GAIN:+.5f}",
    "CONF_D": f"{CONF['D']:+.5f}", "CONF_LO": f"{CONF['ci_boot'][0]:+.5f}", "CONF_HI": f"{CONF['ci_boot'][1]:+.5f}",
    "CONF_D0": f"{CONF['by_third'][0]['D']:+.5f}", "CONF_D1": f"{CONF['by_third'][1]['D']:+.5f}", "CONF_D2": f"{CONF['by_third'][2]['D']:+.5f}",
    "CONF_SLOPE": f"{CONF['cal_slope']:.3f}", "CONF_SLOPE2": f"{CONF['by_third'][2]['cal_slope']:.3f}", "CAL_ROWS": CAL_ROWS,
    "BOOT_REPS": str(BOOT_REPS), "PUB_NOTE": PUB_NOTE,
    "LOCK_HASH": LOCK["phi_sha256"][:12], "LOCK_TIME": LOCK["locked_at_utc"][:16].replace("T", " "), "CODE_REV": SNAP["code_revision"][:7],
    "N_EVAL": str(len(evals)), "LAMS": ", ".join(f"{l:g}" for l in lams),
    "TUNE_SPREAD": f"{max(e['mean_val_logloss'] for e in TUNE8) - min(e['mean_val_logloss'] for e in TUNE8):.5f}",
    "FINAL_ITERS": str(TUN["final"]["n_iter"]), "TUNING_ROWS": tuning_rows(), "LADDER_ROWS": ladder_rows(),
    "HERO_HDR": HERO_HDR, "HERO_BODY": HERO_BODY, "TU_HDR": TU_HDR, "TU_BODY": TU_BODY, "N_TU": num(len(TU)), "MIN_CONTEXTS": num(int(R.contexts_legal.min())), "NT_ROWS": NT_ROWS, "O_ROWS": O_ROWS,
    "N_OLD": num(OLDM["n_matches"]), "OLD_SD0": f"{OLD_SD['W0']:.2f}", "OLD_SD2": f"{OLD_SD['W2']:.2f}", "OLD_SD1": f"{OLD_SD['W1']:.2f}",
    "OLD_MAGIK0": f2(OLD_MAGIK["W0"]), "OLD_MAGIK2": f2(OLD_MAGIK["W2"]), "OLD_MAGIK1": f2(OLD_MAGIK["W1"]),
}
MATH_KEYS = {"LAMS", "MAX_RESID"}                    # substituted inside math; everything else gets a text-mode minus sign
def text_minus(v): return re.sub(r"(?<![\w{^$-])-(?=\d)", "$-$", v)
out = TEX
for _ in range(2):
    for k, v in subs.items():
        out = out.replace(f"<<{k}>>", v if k in MATH_KEYS else text_minus(v))
FIGS = "".join(open(f"results/{f}").read() for f in sorted(re.findall(r"\\input\{(figures/[^}]+\.tikz)\}", out)) if os.path.exists(f"results/{f}"))
BUILD = hashlib.sha256((out + FIGS).encode()).hexdigest()[:8]; out = out.replace("<<BUILD>>", BUILD)
left = re.findall(r"<<[A-Z0-9_]+>>", out)
if left: raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
open("results/lineup_report.tex", "w").write(out)
print(f"wrote results/lineup_report.tex ({len(out):,} bytes); build {BUILD}")
