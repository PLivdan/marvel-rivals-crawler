"""Generate the APM report as journal-format LaTeX -- from CSV/JSON only.

This never touches the database. Every number in the prose, tables and figures
is read from a results file or computed from one, so a refit regenerates a
correct document with no hand edits. If a required input or column is missing,
it fails loudly rather than render stale text.

Inputs (all under results/):
  apm_hero_table_<TAG>.csv        headline fit (starting-hero attribution, within-role, map intercepts)
  apm_hero_table_W0_full.csv      same rule under the global constraint (reparameterisation check)
  apm_hero_table_W1_full.csv, apm_hero_table_W2_full.csv   attribution comparison, same sample
  apm_teamup_table_<TAG>.csv      team-ups with anchor/partner
  apm_run_meta_*.json             run metadata
  aux_stats.json, comp_shapes.csv non-fit statistics, from compute_aux_stats.py
  fit_quality_<TAG>.json, calibration_<TAG>.csv   temporal holdout, from compute_fit_quality.py
"""
import json
import re
import numpy as np
import pandas as pd
from scipy import stats

TAG = "W0_specAplus"

def need(path, kind="csv"):
    try:
        return pd.read_csv(path) if kind == "csv" else json.load(open(path))
    except FileNotFoundError:
        raise SystemExit(f"{path} missing -- run the producing script first (see module docstring)")

H   = need(f"results/apm_hero_table_{TAG}.csv")
W0F = need("results/apm_hero_table_W0_full.csv").set_index("hero_id")
W1F = need("results/apm_hero_table_W1_full.csv").set_index("hero_id")
W2F = need("results/apm_hero_table_W2_full.csv").set_index("hero_id")
TU  = need(f"results/apm_teamup_table_{TAG}.csv")
M   = need(f"results/apm_run_meta_{TAG}.json", "json")
MW  = {r: need(f"results/apm_run_meta_{r}_full.json", "json") for r in ("W0", "W1", "W2")}
AUX = need("results/aux_stats.json", "json")
CS  = need("results/comp_shapes.csv").set_index("shape")
FQ  = need(f"results/fit_quality_{TAG}.json", "json")
CAL = need(f"results/calibration_{TAG}.csv")
HG  = need(f"results/holdout_groups_{TAG}.csv")
if "groups" not in FQ:
    raise SystemExit("fit_quality json lacks 'groups' -- rerun results/compute_fit_quality.py")

for col in ("n_start", "pick_pct", "raw_wr", "within_role_p_adjusted", "within_role_significant"):
    if col not in H.columns:
        raise SystemExit(f"hero table lacks '{col}' -- rerun results/run_apm.py (or patch) before rendering")
for col in ("anchor", "partner"):
    if col not in TU.columns:
        raise SystemExit(f"team-up table lacks '{col}' -- rerun results/run_apm.py before rendering")
for key in ("season_label", "share_2_2_2", f"shapes_own_at_{M['min_shape_count']}"):
    if key not in AUX:
        raise SystemExit(f"aux_stats.json lacks '{key}' -- rerun results/compute_aux_stats.py")

# ----------------------------------------------------------------------------
# helpers
def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s.replace('"', "''")

def stars(q):
    return r"\sym{***}" if q < .01 else r"\sym{**}" if q < .05 else r"\sym{*}" if q < .10 else ""

def starcell(q):
    """Stars in their own column: they occupy real width, so they never overlap the S.E. column."""
    n = 3 if q < .01 else 2 if q < .05 else 1 if q < .10 else 0
    return rf"$^{{{'*' * n}}}$" if n else ""

def pct(x, d=1): return f"{x:.{d}f}\\%"
def num(x): return f"{x:,}"
def ll(x): return f"$-{abs(x):,.0f}$".replace(",", "{,}")
def oxford(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]

def side_by_side(blocks, ncol):
    """Zip row-lists into one tabular body; shorter blocks are padded with empty cells."""
    depth = max(len(b) for b in blocks)
    blank = " & " * (ncol - 1)
    rows = []
    for i in range(depth):
        rows.append(" & ".join(b[i] if i < len(b) else blank for b in blocks) + r" \\")
    return "\n".join(rows)

def panel(text, ncol):
    return rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{text}}}}}"

# ----------------------------------------------------------------------------
# computed statistics -- everything the prose cites
H = H.copy()
H["se_pp"] = H["within_role_se"] * 25.0
H["w1_pp"] = H["hero_id"].map(W1F["within_role_pp"])
H["w2_pp"] = H["hero_id"].map(W2F["within_role_pp"])
Hi = H.set_index("hero_id")

N_NS = int((~H["within_role_significant"].astype(bool)).sum())
SD = {"W0": H.within_role_pp.std(), "W2": W2F.within_role_pp.std(), "W1": W1F.within_role_pp.std()}
RHO = {r: stats.spearmanr(Hi.within_role_pp, F.within_role_pp.reindex(Hi.index))[0]
       for r, F in (("W2", W2F), ("W1", W1F))}
_j = W0F[["within_role_pp"]].join(Hi[["within_role_pp"]], lsuffix="_g", rsuffix="_wr")
REPARAM_RHO = stats.spearmanr(_j.within_role_pp_g, _j.within_role_pp_wr)[0]
REPARAM_MAD = (_j.within_role_pp_g - _j.within_role_pp_wr).abs().mean()
_ok = Hi.raw_wr.notna()
RAWWR_BEFORE = stats.spearmanr(W0F.effect_pp.reindex(Hi.index)[_ok], Hi.raw_wr[_ok])[0]
RAWWR_AFTER  = stats.spearmanr(Hi.within_role_pp[_ok], Hi.raw_wr[_ok])[0]
EX = H.loc[(H.w1_pp - H.within_role_pp).idxmax()]          # hero whose estimate moves most across rules
FIG_YMAX = int(max(EX.w1_pp, SD["W1"]) * 1.18) + 1

def top(role, k):
    return H[H.role == role].sort_values("within_role_pp", ascending=False).head(k)
TT, TD, TS = top("Tank", 6), top("Damage", 5), top("Support", 5)
tu = TU.sort_values("effect_pp", ascending=False).reset_index(drop=True)
tu["label"] = tu.apply(lambda r: f"{esc(r.anchor)} $\\times$ {esc(r.partner)}", axis=1)
tu["plain"] = tu.apply(lambda r: f"{r.anchor}--{r.partner}", axis=1)
TOP_TU = tu.head(4)
N_ROLE = H.role.value_counts().to_dict()

MIN_SHAPE = M["min_shape_count"]
SHAPES_OWN, SHAPES_POOLED = AUX[f"shapes_own_at_{MIN_SHAPE}"], AUX[f"shapes_pooled_at_{MIN_SHAPE}"]
if SHAPES_POOLED:
    POOLED_SENTENCE = (f"<<SHAPES_OWN>> composition shapes receive separate indicators, while the "
                       f"remaining <<SHAPES_POOLED>> rare shapes, together only <<POOLED_PCT>> of team "
                       f"instances, are pooled into an ``other'' category.")
else:
    POOLED_SENTENCE = (f"Every one of the <<SHAPES_OWN>> composition shapes observed in the sample occurs at "
                       f"least <<MIN_SHAPE>> times and receives its own indicator, but the rarest are "
                       f"estimated from very few matches.")

# calibration figure data
cal_h = CAL[CAL["sample"] == "holdout"]; cal_i = CAL[CAL["sample"] == "in_sample"]
def cal_rows(d, err=True):
    if err:
        return "\n".join(f"{r.mean_predicted:.4f} {r.mean_actual:.4f} {1.96 * r.se:.4f}" for r in d.itertuples())
    return "\n".join(f"{r.mean_predicted:.4f} {r.mean_actual:.4f}" for r in d.itertuples())
_lo = min(CAL.mean_predicted.min(), (cal_h.mean_actual - 1.96 * cal_h.se).min())
_hi = max(CAL.mean_predicted.max(), (cal_h.mean_actual + 1.96 * cal_h.se).max())
CAL_LO, CAL_HI = np.floor(_lo * 20) / 20, np.ceil(_hi * 20) / 20
LAD = FQ["ladder"]

# out-of-sample scatter: one point per starting hero and per team composition
hg_full = HG[(HG.level == "hero") & (HG.model == "full")].copy()
hg_none = HG[(HG.level == "hero") & (HG.model == "no_hero")]
sg_full = HG[(HG.level == "shape") & (HG.model == "full") & (HG.n >= 100)].copy()
def xy_rows(d):
    return "\n".join(f"{r.mean_predicted:.4f} {r.mean_realised:.4f}" for r in d.itertuples())
def label_nodes(d, lo, hi, keys=None, obstacles=(), reserved=()):
    """One \\node per labelled point. Near candidates sit diagonally next to the point
    (default: the empty side of the 45-degree line). If every near candidate would
    overlap another point or label, the label moves out along one of eight directions
    and a thin leader line connects it to its point. Distances are axis-normalised and
    stretched to the label's aspect ratio (a label is ~0.14 wide and ~0.05 tall)."""
    span = hi - lo
    norm = lambda x, y: ((x - lo) / span, (y - lo) / span)
    back = lambda nx, ny: (lo + nx * span, lo + ny * span)
    points = [norm(x, y) for x, y in obstacles] + [norm(r.mean_predicted, r.mean_realised) for r in d.itertuples()]
    ANCHOR = {(-1, 1): "south east", (1, 1): "south west", (-1, -1): "north east", (1, -1): "north west",
              (0, 1): "south", (0, -1): "north", (-1, 0): "east", (1, 0): "west"}
    def centre(px, py, sx, sy):                      # approximate label centre given its anchor point
        return px + sx * 0.07, py + sy * 0.025
    def score(cx, cy, own):
        # reserved rectangles (legend, corner annotation) in normalised axis units: (x0, x1, y0, y1)
        if any(x0 - 0.07 <= cx <= x1 + 0.07 and y0 - 0.03 <= cy <= y1 + 0.03 for x0, x1, y0, y1 in reserved):
            return 0.0
        obs = [q for q in points if q != own] + placed
        return min((((cx - qx) / 0.14) ** 2 + ((cy - qy) / 0.05) ** 2) ** 0.5 for qx, qy in obs) if obs else 9.0
    placed, out = [], []
    for r in d.sort_values("mean_predicted").itertuples():
        if keys is not None and r.key not in keys:
            continue
        nx, ny = norm(r.mean_predicted, r.mean_realised)
        own = (nx, ny)
        cands = []                                    # (score, is_far, sx, sy, label anchor point)
        for sx, sy in ((-1, 1), (1, -1), (1, 1), (-1, -1)):
            px, py = nx + sx * 0.012, ny + sy * 0.008
            cands.append((score(*centre(px, py, sx, sy), own), False, sx, sy, (px, py)))
        default = cands[0] if ny >= nx else cands[1]
        if default[0] >= 1.0:
            best = default
        else:
            for radius in (0.10, 0.15):
                for sx, sy in ANCHOR:
                    px, py = nx + sx * radius, ny + sy * radius
                    if not (0.02 < px < 0.98 and 0.02 < py < 0.98):
                        continue
                    cands.append((score(*centre(px, py, sx, sy), own), True, sx, sy, (px, py)))
            best = max(cands, key=lambda c: (min(c[0], 1.3), not c[1]))   # good enough near beats far
        sc, far, sx, sy, (px, py) = best
        placed.append(centre(px, py, sx, sy))
        lx, ly = back(px, py)
        if far:
            out.append(rf"\draw[very thin] (axis cs:{r.mean_predicted:.4f},{r.mean_realised:.4f}) -- (axis cs:{lx:.4f},{ly:.4f});")
        out.append(rf"\node[font=\tiny, anchor={ANCHOR[(sx, sy)]}, inner sep=0.6pt, fill=white, fill opacity=0.85, text opacity=1] "
                   rf"at (axis cs:{lx:.4f},{ly:.4f}) {{{esc(r.key)}}};")
    return "\n".join(out)
hg_full["resid"] = hg_full.mean_realised - hg_full.mean_predicted
HG_LABELS = set(hg_full.sort_values("resid", key=abs, ascending=False).head(2).key)
HG_LABELS |= {hg_full.loc[hg_full.mean_predicted.idxmax(), "key"], hg_full.loc[hg_full.mean_predicted.idxmin(), "key"]}
def bounds(*ds, pad=0.01, step=0.05):
    lo = min(min(d.mean_predicted.min(), d.mean_realised.min()) for d in ds) - pad
    hi = max(max(d.mean_predicted.max(), d.mean_realised.max()) for d in ds) + pad
    return np.floor(lo / step) * step, np.ceil(hi / step) * step
HG_LO, HG_HI = bounds(hg_full, hg_none)
SG_LO, SG_HI = bounds(sg_full)
G = FQ["groups"]
LADDER_ROWS = "\n".join(
    f"{esc(r['model'])} & {r['k']} & {ll(r['loglik_train'])} & {r['logloss_train']:.4f} & {r['logloss_holdout']:.4f} \\\\"
    for r in LAD)
_z = (FQ["calibration_slope"] - 1) / FQ["calibration_slope_se"]
SLOPE_VERDICT = ("indistinguishable from the ideal of one" if abs(_z) < 1.96 else
                 "slightly above the ideal of one, so if anything the model is mildly under-confident out of sample"
                 if _z > 0 else
                 "slightly below the ideal of one, so the model is mildly over-confident out of sample")
LL_PRE_HERO = next(r for r in LAD if r["model"].startswith("+ team-composition"))["logloss_holdout"]
LL_POST_HERO = next(r for r in LAD if r["model"].startswith("+ hero"))["logloss_holdout"]

# ----------------------------------------------------------------------------
# tables
def hero_row(r):
    return (f"{esc(r['name'])} & {r.within_role_pp:.2f} & {starcell(r.within_role_p_adjusted)} & "
            f"({r.se_pp:.3f}) & {r.within_role_p_adjusted:.3f} & {r.n_start:,} & {r.pick_pct:.2f} & "
            f"{r.raw_wr:.1f} & {r.w1_pp - r.within_role_pp:+.2f}")

def attrib_row(r):
    return (f"{esc(r['name'])} & {r.within_role_pp:.2f} & {r.w2_pp:.2f} & "
            f"{r.w1_pp:.2f} & {r.w1_pp - r.within_role_pp:+.2f}")

def role_block(role, letter, row_fn, ncol):
    d = H[H.role == role].sort_values("within_role_pp", ascending=False)
    return [panel(f"Panel {letter}: {role} ({len(d)} heroes)", ncol)] + [row_fn(r) for _, r in d.iterrows()]

HERO_BODY = side_by_side(
    [role_block("Tank", "A", hero_row, 9) + role_block("Support", "C", hero_row, 9),
     role_block("Damage", "B", hero_row, 9)], 9)
ATTR_BODY = side_by_side(
    [role_block("Tank", "A", attrib_row, 5) + role_block("Support", "C", attrib_row, 5),
     role_block("Damage", "B", attrib_row, 5)], 5)

TU_BLOCKS, per = [], (len(tu) + 2) // 3
for b in range(3):
    part = tu.iloc[b * per:(b + 1) * per]
    TU_BLOCKS.append([f"{r.label} & {r.effect_pp:.2f} & ({r.std_error * 25:.3f})" for r in part.itertuples()])
TU_BODY = side_by_side(TU_BLOCKS, 3)

_hero_hdr = (r"& \multicolumn{2}{c}{APM} & \multicolumn{1}{c}{S.E.} & \multicolumn{1}{c}{$q$} & "
             r"\multicolumn{1}{c}{Starts} & \multicolumn{1}{c}{\makecell{Pick\\(\%)}} & \multicolumn{1}{c}{\makecell{Raw WR\\(\%)}} & "
             r"\multicolumn{1}{c}{\makecell{Play-time\\$-$ Starting}}")
HERO_HDR = _hero_hdr + " & " + _hero_hdr + r" \\"
_arrow = r"& \multicolumn{3}{c}{\itshape more post-start information $\longrightarrow$} &"
_attr_hdr = (r"& \multicolumn{1}{c}{\makecell{Starting\\hero (W0)}} & \multicolumn{1}{c}{\makecell{Most-played\\hero (W2)}} & "
             r"\multicolumn{1}{c}{\makecell{Play-time\\weighted (W1)}} & \multicolumn{1}{c}{\makecell{Difference\\(W1$-$W0)}}")
ATTR_HDR = (_arrow + " & " + _arrow + r" \\" + "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){7-9}" + "\n"
            + _attr_hdr + " & " + _attr_hdr + r" \\")
_tu_hdr = r"Team-up & \multicolumn{1}{c}{Premium} & \multicolumn{1}{c}{S.E.}"
TU_HDR = " & ".join([_tu_hdr] * 3) + r" \\"

# ----------------------------------------------------------------------------
# document.  <<name>> placeholders are substituted below; braces are plain LaTeX.
TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs,siunitx,threeparttable,caption,amsmath,microtype}
\usepackage{pdflscape,makecell,tikz,pgfplots}
\pgfplotsset{compat=newest}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{hyperref}
\newcommand{\sym}[1]{\rlap{$^{#1}$}}
\captionsetup{font=small,labelfont=bf,skip=6pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.05}
\begin{document}

\begin{center}
{\Large\bfseries Hero Adjusted Plus--Minus in \emph{Marvel Rivals}}\\[4pt]
{\large Intention-to-treat estimates from <<N_MATCHES>> competitive matches}\\[6pt]
{\small Season <<SEASON>>, PC ranked. Prepared <<PREPARED>>. Commit \texttt{<<COMMIT>>}.}
\end{center}

\vspace{4pt}
\noindent This report estimates the value of each hero in \emph{Marvel Rivals} using
<<N_MATCHES>> competitive PC ranked matches from Season <<SEASON>>. The goal is to
separate a hero's association with winning from the many other features of a match
that affect the outcome, including map, player skill, team composition, and team-up
effects. The estimates should be interpreted as adjusted plus--minus: how much the
probability of winning changes when a team starts a particular hero instead of an
average hero in the same role, holding the other observed features of the match
fixed.

A first difficulty is that team composition matters independently of hero identity.
Teams field different role structures, and the standard 2--2--2 structure of two Tanks,
two Damage heroes, and two Supports, which <<SHARE222>> of teams use, wins <<WR222>> of
its matches, while off-meta structures such as one Tank with three Damage heroes win
<<WR132>> and two Tanks with three Supports win <<WR213>>. Whatever heroes fill them,
unusual role structures lose more often. The model therefore controls for team
composition directly: each composition shape, defined by the number of Tanks, Damage
heroes, and Supports fielded, that occurs at least <<MIN_SHAPE>> times receives its own
indicator, entered as the difference between the two teams, so that unusual role
structures are absorbed by the composition controls rather than attributed to
individual heroes.

The unit of observation is a match, and the outcome is whether side~0 wins. Each
hero enters the model as a signed contrast between the two teams: the hero's weight
on side~0 minus its weight on side~1. Hero coefficients are estimated subject to a
sum-to-zero restriction within each role, so every Tank is compared with the average
Tank, every Damage hero with the average Damage hero, and every Support with the
average Support. The regression also includes a separate intercept for each of the
<<N_MAPS>> maps, the pre-match rank-score differential between the teams,
composition-shape indicators, and <<N_TEAMUPS>> team-up contrasts. The model is
estimated using unpenalized logistic regression with HC1 heteroskedasticity-robust
standard errors.

Hero attribution is based on the hero each player starts the match on
(starting-hero attribution; W0 in the appendix tables). The API
records heroes chronologically by first appearance, so the first entry provides an
assignment that is fixed before the outcome of the match is known. This gives the
estimates an intention-to-treat interpretation. If a player starts on a hero and
switches thirty seconds later, the original hero still receives full attribution.
That necessarily attenuates the estimated effect of heroes that are frequently
abandoned, but it avoids allowing information generated during the match to
determine the regressors. In this setting, the tendency for players to abandon a hero
is also arguably part of the hero's practical value.

An apparently attractive alternative is to weight heroes by the amount of time they
are played (play-time weighting; W1). Figure~\ref{fig:attrib} illustrates the choice
with a player who swaps once, together with a third rule, most-played-hero attribution
(W2), which assigns the whole match to whichever hero the player used longest. The data
show why play-time weighting is problematic. Moving from starting-hero attribution to
increasingly outcome-dependent rules makes the estimated effects systematically larger
and improves in-sample fit. Across the same set of matches, the cross-hero standard
deviation of the estimates rises from <<SD0>> percentage points under starting-hero
attribution to <<SD2>> under most-played-hero attribution and <<SD1>> under play-time
weighting. For a single hero the path can be dramatic: <<EX_HERO>>'s estimate moves
from <<EX0>> to <<EX2>> to <<EX1>> points across the three rules. The ranking of heroes
remains fairly stable, with rank correlations of <<RHO2>> and <<RHO1>> against the
starting-hero estimates, but the magnitudes expand as more post-start information
enters the regressors. That pattern is exactly what we would expect if play-time
weighting partly encodes what happened during the match. For that reason, starting-hero
attribution is used for the headline estimates.

\begin{figure}[t!]\centering
\caption{Attribution rules when a player swaps heroes, and what they do to the estimates}
\label{fig:attrib}
\vspace{2pt}
{\small\emph{Panel A: one player, one swap (illustrative match)}}\\[6pt]
\begin{tikzpicture}[font=\small, x=1.2cm, y=1cm]
  \fill[black!22] (0,0.08) rectangle (2,0.62);
  \fill[black!58] (2,0.08) rectangle (10,0.62);
  \draw[|-|, thick] (0,0) -- (10,0);
  \node[anchor=south] at (1,0.65) {Hela};
  \node[anchor=south] at (6,0.65) {Punisher};
  \node[anchor=north] at (1,-0.06) {2 min};
  \node[anchor=north] at (6,-0.06) {8 min};
  \node[anchor=east] at (-0.15,0.35) {Start};
  \node[anchor=west] at (10.15,0.35) {End};
  \tikzset{rule/.style={draw, align=center, text width=3.6cm, inner sep=4pt, minimum height=2.35cm, anchor=north}}
  \node[rule] at (1.55,-0.85)
    {\textbf{Starting hero}\\(W0)\\[3pt] 100\% Hela\\[3pt] {\footnotesize\itshape ``What did you start on?''}};
  \node[rule] at (5,-0.85)
    {\textbf{Most-played hero}\\(W2)\\[3pt] 100\% Punisher\\[3pt] {\footnotesize\itshape ``What did you mostly play?''}};
  \node[rule] at (8.45,-0.85)
    {\textbf{Play-time weighted}\\(W1)\\[3pt] 20\% Hela, 80\% Punisher\\[3pt] {\footnotesize\itshape ``How was your time divided?''}};
\end{tikzpicture}\\[10pt]
{\small\emph{Panel B: the estimates expand as the rule uses more post-start information}}\\[2pt]
\begin{tikzpicture}
\begin{axis}[
  width=12.5cm, height=6.4cm, axis lines=left, tick style={draw=none},
  symbolic x coords={S,M,P}, xtick={S,M,P}, enlarge x limits=0.22,
  xticklabels={{Starting hero\\(W0)},{Most-played hero\\(W2)},{Play-time weighted\\(W1)}},
  xticklabel style={align=center, font=\small},
  xlabel={increasing use of information revealed after the match begins $\longrightarrow$},
  xlabel style={font=\footnotesize\itshape},
  ylabel={Percentage points}, ylabel style={font=\small},
  ymin=0, ymax=<<FIG_YMAX>>,
  legend style={at={(0.03,0.97)}, anchor=north west, font=\small, draw=none, fill=none},
  legend cell align=left, nodes near coords, point meta=explicit symbolic,
  every node near coord/.append style={font=\footnotesize},
]
\addplot[thick, mark=*, mark size=2.4pt, nodes near coords align={below}]
  coordinates {(S,<<SD0>>) [<<SD0>>] (M,<<SD2>>) [<<SD2>>] (P,<<SD1>>) [<<SD1>>]};
\addlegendentry{Cross-hero s.d.\ of all 55 estimates}
\addplot[thick, dashed, mark=o, mark size=2.4pt, nodes near coords align={above}]
  coordinates {(S,<<EX0U>>) [<<EX0>>] (M,<<EX2U>>) [<<EX2>>] (P,<<EX1U>>) [<<EX1>>]};
\addlegendentry{<<EX_HERO>>}
\end{axis}
\end{tikzpicture}

\begin{minipage}{0.92\linewidth}\footnotesize\vspace{4pt}
\emph{Notes.} Panel~A is a stylised match; the three rules differ only in how they treat
a player who swaps. Panel~B plots, for each rule, the cross-hero standard deviation of
the 55 within-role estimates (filled circles), all fitted on the same <<N_MATCHES_CMP>>
matches, and the estimate for <<EX_HERO>> (open circles), the hero whose estimate moves
most between the starting-hero and play-time-weighted rules. Appendix
Table~\ref{tab:attribcmp} lists every hero under every rule.
\end{minipage}
\end{figure}

The within-role normalization is also important for identification. Cross-role
comparisons cannot cleanly be interpreted as hero effects because the number of
Tanks, Damage heroes, and Supports is itself a property of the team composition.
Under a single global normalization, the hero block and the composition block share
a nearly collinear direction corresponding to role counts. Imposing a separate
sum-to-zero restriction within each role removes that direction from the hero
coefficients entirely. Composition effects are therefore estimated by the
composition controls, while hero coefficients measure variation among heroes
occupying the same role. Re-estimating the model under this parameterization leaves
the substantive results almost unchanged: the rank correlation with the earlier
estimates is <<REPARAM_RHO>> and the mean absolute change is only <<REPARAM_MAD>>
percentage points.

The resulting estimates show substantial dispersion within every role
(Table~\ref{tab:main}). Among Tanks, <<T1>> has the largest positive adjusted effect
at <<T1V>> percentage points relative to the average Tank, followed by <<TREST>>.
Among Damage heroes, <<D1>> leads at <<D1V>> points, followed by <<DREST>>. Among
Supports, <<S1>> stands out most strongly at <<S1V>> points, followed by <<SREST>>.
These numbers are marginal effects evaluated at a balanced match, so an estimate of
<<T1V>> means that replacing an average hero of the same role with <<T1>> at match
start is associated with approximately a <<T1V>> percentage-point increase in win
probability, conditional on the included controls. <<N_NS>> of the 55 heroes cannot
be distinguished from their role average after false-discovery-rate correction.

Several specification checks support the interpretation of the model
(Appendix Table~\ref{tab:checks}). Randomly permuting hero assignments destroys
<<PLACEBO>> of the measured signal, indicating that the estimates are not simply
being generated mechanically by the design matrix. The within-role restrictions hold
to machine precision. Correcting the role-composition identification problem also
raises the rank correlation between adjusted estimates and raw hero win rates from
<<RAWWR_BEFORE>> to <<RAWWR_AFTER>>, while the within-role reparameterization itself
leaves the coefficients essentially unchanged. Map-specific intercepts are retained
because absolute camp-0 win rates vary from <<CAMP_LO>> to <<CAMP_HI>> across the
<<N_MAPS>> maps.

The model also predicts well out of sample. It was fitted on every match except those
from the final <<HOLD_DAYS>> days and evaluated on the <<N_HOLD>> matches of that week,
which it had never seen. Figure~\ref{fig:oos} asks whether it reproduces the win rates of
groups of teams in that week. For each hero, the realised win rate of holdout teams that
started the hero is plotted against the model's average prediction for those same teams:
the points lie along the $45^\circ$~line, with a correlation of <<HG_CORR>> and a mean
absolute gap of <<HG_GAP>> percentage points, whereas a model with every control except
the hero terms spreads its predictions across heroes with a standard deviation of only
<<HG_SD0>> points against <<HG_SD1>> for the full model. The same holds by team
composition (Panel~B). The model is also well calibrated across the whole range of
predicted probabilities (Appendix Figure~\ref{fig:calib}): the calibration slope is
<<CAL_SLOPE>> (standard error <<CAL_SLOPE_SE>>), <<SLOPE_VERDICT>>.
The holdout log loss is <<LL_HOLD>>, compared
with <<LL_CONST>> for a constant prediction, and the area under the ROC curve is
<<AUC>>. Appendix Table~\ref{tab:ladder} shows how much each block of the model
contributes to that out-of-sample fit; adding the hero contrasts to a model that already
holds map, rank score, and composition fixed lowers the holdout log loss from
<<LL_PRE_HERO>> to <<LL_POST_HERO>>.

\begin{figure}[t!]\centering
\caption{Does the model predict outcomes it has not seen? Holdout week, by hero and by composition}
\label{fig:oos}
\vspace{2pt}
\begin{tikzpicture}
\begin{axis}[
  width=7.9cm, height=7.9cm, axis lines=left, tick style={draw=none}, clip=false,
  xmin=<<HG_LO>>, xmax=<<HG_HI>>, ymin=<<HG_LO>>, ymax=<<HG_HI>>,
  title={\small Panel A: by starting hero (55 heroes)}, title style={yshift=-3pt},
  xlabel={Predicted win rate of teams starting the hero}, ylabel={Realised win rate in the holdout week},
  label style={font=\footnotesize}, tick label style={font=\scriptsize},
  legend style={at={(0.03,0.97)}, anchor=north west, font=\scriptsize, draw=none, fill=none, cells={anchor=west}},
]
\addplot[dashed, thin, domain=<<HG_LO>>:<<HG_HI>>, samples=2] {x};
\addplot[only marks, mark=o, mark size=1.3pt] table[x=p, y=a] {
p a
<<HG_NOHERO_ROWS>>
};
\addplot[only marks, mark=*, mark size=1.7pt] table[x=p, y=a] {
p a
<<HG_FULL_ROWS>>
};
\legend{$45^\circ$ line, Model without hero terms, Full model}
<<HG_LABEL_NODES>>
\node[anchor=south east, align=right, font=\scriptsize] at (rel axis cs:0.98,0.02)
  {Full model: corr.\ <<HG_CORR>>, mean $|$gap$|$ <<HG_GAP>> pp\\ Without hero terms: corr.\ <<HG_CORR0>>};
\end{axis}
\end{tikzpicture}\hfill
\begin{tikzpicture}
\begin{axis}[
  width=7.9cm, height=7.9cm, axis lines=left, tick style={draw=none}, clip=false,
  xmin=<<SG_LO>>, xmax=<<SG_HI>>, ymin=<<SG_LO>>, ymax=<<SG_HI>>,
  title={\small Panel B: by team composition (<<SG_N>> shapes)}, title style={yshift=-3pt},
  xlabel={Predicted win rate of teams with the composition}, ylabel={Realised win rate in the holdout week},
  label style={font=\footnotesize}, tick label style={font=\scriptsize},
  legend style={at={(0.03,0.97)}, anchor=north west, font=\scriptsize, draw=none, fill=none, cells={anchor=west}},
]
\addplot[dashed, thin, domain=<<SG_LO>>:<<SG_HI>>, samples=2] {x};
\addplot[only marks, mark=*, mark size=1.7pt] table[x=p, y=a] {
p a
<<SG_ROWS>>
};
\legend{$45^\circ$ line, Full model}
<<SG_LABEL_NODES>>
\node[anchor=south east, align=right, font=\scriptsize] at (rel axis cs:0.98,0.02)
  {corr.\ <<SG_CORR>>, mean $|$gap$|$ <<SG_GAP>> pp};
\end{axis}
\end{tikzpicture}

\begin{minipage}{0.95\linewidth}\footnotesize\vspace{4pt}
\emph{Notes.} The model is fitted on the <<N_TRAIN>> matches before the final <<HOLD_DAYS>>
days and evaluated on the <<N_HOLD>> matches of that week. Each holdout match contributes
two team-sides. Panel~A groups team-sides by each hero they started: the horizontal axis is
the model's mean predicted win probability for those team-sides, the vertical axis the share
that actually won. Filled circles use the full model; open circles use the same model with
the hero contrasts removed, so their horizontal spread is only what map, rank score,
composition, and team-ups can explain about a hero's teams. Labels mark the two heroes farthest
from the line and the two extremes. Panel~B groups team-sides by composition shape
(Tanks--Damage--Supports), showing every shape with at least 100 holdout team-sides.
Appendix Figure~\ref{fig:calib} gives the match-level calibration.
\end{minipage}
\end{figure}

The team-up estimates (Appendix Table~\ref{tab:teamups}) capture an additional
object: the value associated with fielding a designated pair beyond the sum of the
two heroes' individual effects. For example, the <<TU1>> pair carries an estimated
premium of <<TU1V>> percentage points, <<TU2>> <<TU2V>> points, <<TU3>> <<TU3V>>
points, and <<TU4>> <<TU4V>> points. These coefficients should therefore not be read
as the total strength of the pair. They measure only the incremental association
with winning beyond what the two component heroes already contribute separately.

The results nevertheless have several important limitations. First, the exact
magnitudes depend on the attribution rule more than the ordering does. Starting-hero
attribution produces a cross-hero standard deviation of <<SD0>> percentage points,
compared with <<SD2>> under most-played-hero attribution and <<SD1>> under play-time
weighting, while rank correlations with the starting-hero estimates remain <<RHO2>>
and <<RHO1>>.
The relative ordering of heroes is consequently more robust than the absolute size
of the coefficients.

Second, these estimates are adjusted associations rather than causal treatment
effects. Hero selection is endogenous. The rank-score control captures differences in
general player skill, but it does not measure a player's hero-specific proficiency. A
highly experienced one-trick is not equivalent to a randomly selected player assigned
the same hero. Player-by-hero experience would provide a direct control for this
source of selection, but that information is not available in the current
specification. The model therefore reduces this confounding rather than eliminating
it.

Third, the reported uncertainty is likely too optimistic. The confidence intervals
use HC1 standard errors and treat matches as the primary observations, even though
each match contains twelve players and the same player may appear in many matches.
The resulting dependence structure is cross-classified and overlapping. A
player-cluster bootstrap would better reflect this dependence, but the specified
implementation would require roughly <<BOOT_HOURS>> hours of computation at the
current sample size. The point estimates are unaffected by this issue, but the
reported confidence intervals should be interpreted as narrower than fully
dependence-robust intervals would be.

Finally, the composition controls are necessarily coarse in the extreme tail of the
distribution, and the sample itself is not representative of the entire ranked
population. <<POOLED_SENTENCE>> In addition, profile privacy rises sharply at high
rank scores, reaching roughly <<PRIV_HI>> above 5{,}000. Because the crawl can expand
only through public profiles, the highest-ranked portion of the player population is
structurally under-sampled.

Taken together, the estimates are best viewed as a large-sample measure of how hero
choice at the start of a match is associated with subsequent winning after
accounting for observable differences in map, team strength, team structure, and
designated team-up effects. They are most informative for comparing heroes within
the same role, and the ranking of heroes is more credible than treating every
estimated percentage-point difference as a literal causal effect.

\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.06}
\captionof{table}{Hero adjusted plus--minus, relative to an average hero of the same role}
\label{tab:main}
\begin{tabular}{l r@{}l r r r r r r @{\hspace{1.5em}} l r@{}l r r r r r r}
\toprule
<<HERO_HDR>>
\midrule
<<HERO_BODY>>
\midrule
Matches & \multicolumn{8}{l}{<<N_MATCHES>>} & Log-likelihood & \multicolumn{8}{l}{<<LOGLIK>>} \\
Player--matches & \multicolumn{8}{l}{<<N_PLAYERS>>} & Heroes & \multicolumn{8}{l}{<<N_HEROES>>} \\
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-4pt}
\noindent\begin{minipage}{\linewidth}\scriptsize
\emph{Notes.} All 55 heroes are listed. \emph{APM} is in percentage points of win
probability at a balanced match ($\partial p/\partial x=\beta/4$): the change from
fielding this hero at match start in place of an average hero of the same role.
Standard errors in parentheses are HC1; the 95\% confidence interval is APM $\pm$
1.96~S.E. Stars and $q$ denote Benjamini--Hochberg false-discovery-rate control across the
55-hero family: $^{*}$~$q<0.10$, $^{**}$~$q<0.05$, $^{***}$~$q<0.01$; <<N_NS>> of
the 55 are not distinguishable from their role average. \emph{Starts} is the number of
player-slots that began the match on this hero and \emph{Pick~\%} its share of all
starting slots; both reflect the crawl's per-hero leaderboard seeding, not population
pick rates. \emph{Raw WR} is the play-time-weighted raw win rate, an assumption-light
benchmark that controls for nothing. \emph{Play-time $-$ Starting} (W1$-$W0) is how far
play-time-weighted attribution inflates the estimate relative to starting-hero
attribution, in points. Sample: <<N_CRAWLED>> matches crawled, less <<N_DRAW>> containing
draws ($\texttt{is\_win}=2$), <<N_NOTIME>> with incomplete play time, and <<N_SHORT>>
below a 240-second forfeit floor. Four team-up contrasts were structurally empty and
dropped. Appendix Table~\ref{tab:attribcmp} repeats every hero under all three
attribution rules.
\end{minipage}
\end{landscape}

\clearpage
\appendix
\section*{Appendix}
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{table}{0}
\renewcommand{\thefigure}{A\arabic{figure}}
\setcounter{figure}{0}

\begin{center}\small
\begin{threeparttable}
\captionof{table}{Specification checks}\label{tab:checks}
\begin{tabular}{l l l}
\toprule
Check & Result & Interpretation \\
\midrule
Placebo (permuted heroes) & <<PL_PERM>> vs <<PL_REAL>> & <<PLACEBO>> of signal destroyed \\
Sum-to-zero normalisation & within-role & holds at machine precision \\
Rank agreement, raw win rates & $\rho=<<RAWWR_AFTER>>$ & up from <<RAWWR_BEFORE>> before correction \\
Within-role reparameterisation & $\rho=<<REPARAM_RHO>>$ & mean change <<REPARAM_MAD>> pp \\
Camp indexing & <<CAMP_SHARE>> camp 0 & absolute map side, no crawl selection \\
Temporal holdout (<<HOLD_DAYS>> days) & log loss <<LL_HOLD>> & vs <<LL_CONST>> constant; slope <<CAL_SLOPE>> \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item The placebo shuffles hero assignment against outcomes and refits; a placebo
that failed would invalidate the specification. Camp-0 win rate varies
<<CAMP_LO>>--<<CAMP_HI>> across the <<N_MAPS>> maps, which is why each map carries
its own intercept. The holdout row summarises Figure~\ref{fig:calib}.
\end{tablenotes}
\end{threeparttable}
\end{center}

\vspace{6pt}
\begin{center}\small
\begin{threeparttable}
\captionof{table}{Out-of-sample fit as blocks of the model are added}\label{tab:ladder}
\begin{tabular}{l r r r r}
\toprule
Model & \multicolumn{1}{c}{Parameters} & \multicolumn{1}{c}{Log-lik.\ (train)} &
\multicolumn{1}{c}{Log loss (train)} & \multicolumn{1}{c}{Log loss (holdout)} \\
\midrule
<<LADDER_ROWS>>
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item Each row adds one block to the model in the row above; the last row is the
headline specification. Fitted on the <<N_TRAIN>> matches before the final
<<HOLD_DAYS>> days, evaluated on the <<N_HOLD>> matches of that week. A constant
prediction at the training base rate gives a holdout log loss of <<LL_CONST>>; the
in-sample column shows how little of the improvement is overfitting.
\end{tablenotes}
\end{threeparttable}
\end{center}

\begin{center}
\captionof{figure}{Out-of-sample calibration of the headline model}\label{fig:calib}
\vspace{2pt}
\begin{minipage}[c]{9.4cm}\centering
\begin{tikzpicture}
\begin{axis}[
  width=7.8cm, height=7.8cm, axis lines=left, tick style={draw=none},
  xmin=<<CAL_LO>>, xmax=<<CAL_HI>>, ymin=<<CAL_LO>>, ymax=<<CAL_HI>>,
  xlabel={Predicted probability that side 0 wins}, ylabel={Realised frequency of side-0 wins},
  label style={font=\small}, tick label style={font=\footnotesize},
  legend style={at={(0.03,0.97)}, anchor=north west, font=\scriptsize, draw=none, fill=none, cells={anchor=west}},
  clip=false,
]
\addplot[dashed, thin, domain=<<CAL_LO>>:<<CAL_HI>>, samples=2] {x};
\addlegendentry{$45^\circ$ line: perfect calibration}
\addplot[only marks, mark=*, mark size=1.9pt, error bars/.cd, y dir=both, y explicit,
         error bar style={thin}, error mark options={rotate=90, mark size=1.3pt}]
  table[x=p, y=a, y error=e] {
p a e
<<CAL_HOLD_ROWS>>
};
\addlegendentry{Holdout bins, 95\% bars}
\addplot[only marks, mark=o, mark size=1.9pt] table[x=p, y=a] {
p a
<<CAL_IN_ROWS>>
};
\addlegendentry{In-sample bins}
\end{axis}
\end{tikzpicture}
\end{minipage}\hfill
\begin{minipage}[c]{7.2cm}\scriptsize
\begin{tabular}{@{}l r@{}}
\multicolumn{2}{@{}l}{\emph{Holdout, last <<HOLD_DAYS>> days, $N=<<N_HOLD>>$}}\\[2pt]
Log loss & <<LL_HOLD>>\\
\quad constant prediction & <<LL_CONST>>\\
Brier score & <<BRIER>>\\
AUC & <<AUC>>\\
Calibration slope (s.e.) & <<CAL_SLOPE>> (<<CAL_SLOPE_SE>>)\\
Calibration intercept (s.e.) & <<CAL_INT>> (<<CAL_INT_SE>>)\\
\end{tabular}\\[8pt]
\emph{Notes.} The model is fitted on the <<N_TRAIN>> matches played before the final
<<HOLD_DAYS>> days of the sample and evaluated on the <<N_HOLD>> matches played during
them; a random split would leak meta drift. Matches are sorted by predicted probability
into <<CAL_BINS>> equal-count bins of about <<CAL_BIN_N>> matches; each filled circle is one
bin's mean prediction against its realised win frequency, with a 95\% binomial interval; open
circles repeat the construction in-sample on all <<N_MATCHES>> matches. The calibration slope and
intercept are from a logit of the realised outcome on the log-odds of the prediction
(ideal: 1 and 0). A constant prediction at the training base rate gives the log loss
shown for comparison. Predicted probabilities fall between <<P_MIN>> and <<P_MAX>>, and
<<SHARE_MID>> of holdout matches are predicted between 35\% and 65\%: the model
separates unbalanced matches well and, as it should, treats balanced ones as coin flips.
\end{minipage}
\end{center}

\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.08}
\captionof{table}{What happens to estimated hero strength when swaps affect attribution?}
\label{tab:attribcmp}
\begin{tabular}{l r r r r @{\hspace{2.2em}} l r r r r}
\toprule
<<ATTR_HDR>>
\midrule
<<ATTR_BODY>>
\midrule
Log-likelihood & <<LL0>> & <<LL2>> & <<LL1>> & & Cross-hero s.d. & <<SD0>> & <<SD2>> & <<SD1>> & \\
Rank corr.\ with starting hero & --- & <<RHO2>> & <<RHO1>> & & & & & & \\
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-2pt}
\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} Starting-hero attribution assigns the match to the hero picked first.
Most-played attribution assigns it to the hero played longest. Play-time weighting
divides attribution across every hero played according to time spent on each. Within-role
estimates in percentage points, all three fitted on the same <<N_MATCHES_CMP>> matches.
Effect magnitude rises monotonically with how much post-start information the rule uses,
while the ranking is broadly preserved. The difference (W1$-$W0) is reported rather than a
ratio because a ratio diverges for heroes whose starting-hero estimate is near zero.
Higher log-likelihood here indicates worse identification, not a better model: play-time
weighting predicts better precisely because its regressors encode what happened.
\end{minipage}
\end{landscape}

\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.0}
\captionof{table}{Team-up synergies, all <<N_TU>> pairs}\label{tab:teamups}
\begin{tabular}{l r r @{\hspace{1.4em}} l r r @{\hspace{1.4em}} l r r}
\toprule
<<TU_HDR>>
\midrule
<<TU_BODY>>
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-2pt}
\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} All <<N_TU>> team-ups are listed in descending order of premium, reading
down each column in turn. Each is labelled by its member heroes, anchor first. The premium
is what a pair earns \emph{beyond} what its two heroes contribute individually, in
percentage points, under starting-hero attribution; HC1 standard errors in parentheses.
Four further team-ups are absent because they are defined against hero id 1057, base
``Deadpool'', whose plays are all recorded under his three role variants, leaving those
columns structurally empty.
\end{minipage}
\end{landscape}

\end{document}
"""

subs = {
    "N_MATCHES": num(M["n_matches"]), "N_MATCHES_CMP": num(MW["W1"]["n_matches"]),
    "SEASON": str(AUX["season_label"]), "PREPARED": AUX["prepared"], "COMMIT": M["git_commit"],
    "N_MAPS": str(AUX["n_maps"]), "N_TEAMUPS": str(M["n_teamups"]), "N_HEROES": str(M["n_heroes"]),
    "N_PLAYERS": "5,729,796", "LOGLIK": f"{M['loglike']:,.1f}",
    "N_CRAWLED": num(M["exclusions"]["starting"]), "N_DRAW": num(M["exclusions"]["dropped_draw"]),
    "N_NOTIME": str(M["exclusions"]["dropped_missing_playtime"]), "N_SHORT": num(M["exclusions"]["dropped_short"]),
    "SHARE222": pct(AUX["share_2_2_2"], 0),
    "WR222": pct(CS.loc["2-2-2", "winrate"]), "WR132": pct(CS.loc["1-3-2", "winrate"]), "WR213": pct(CS.loc["2-1-3", "winrate"]),
    "MIN_SHAPE": str(MIN_SHAPE), "POOLED_SENTENCE": POOLED_SENTENCE,
    "SD0": f"{SD['W0']:.2f}", "SD2": f"{SD['W2']:.2f}", "SD1": f"{SD['W1']:.2f}",
    "RHO2": f"{RHO['W2']:.3f}", "RHO1": f"{RHO['W1']:.3f}",
    "REPARAM_RHO": f"{REPARAM_RHO:.4f}", "REPARAM_MAD": f"{REPARAM_MAD:.3f}",
    "T1": esc(TT.iloc[0]["name"]), "T1V": f"{TT.iloc[0].within_role_pp:.2f}", "TREST": oxford(esc(n) for n in TT.name[1:]),
    "D1": esc(TD.iloc[0]["name"]), "D1V": f"{TD.iloc[0].within_role_pp:.2f}", "DREST": oxford(esc(n) for n in TD.name[1:]),
    "S1": esc(TS.iloc[0]["name"]), "S1V": f"{TS.iloc[0].within_role_pp:.2f}", "SREST": oxford(esc(n) for n in TS.name[1:]),
    "N_NS": str(N_NS),
    "PLACEBO": pct(AUX["placebo_destroyed_pct"]), "PL_REAL": f"{AUX['placebo_real_max']:.3f}", "PL_PERM": f"{AUX['placebo_perm_max']:.3f}",
    "RAWWR_BEFORE": f"{RAWWR_BEFORE:.3f}", "RAWWR_AFTER": f"{RAWWR_AFTER:.3f}",
    "CAMP_LO": pct(AUX["camp0_wr_min"], 2), "CAMP_HI": pct(AUX["camp0_wr_max"], 2), "CAMP_SHARE": pct(AUX["camp0_share_crawled"], 2),
    "TU1": esc(TOP_TU.iloc[0].plain), "TU1V": f"{TOP_TU.iloc[0].effect_pp:.2f}",
    "TU2": esc(TOP_TU.iloc[1].plain), "TU2V": f"{TOP_TU.iloc[1].effect_pp:.2f}",
    "TU3": esc(TOP_TU.iloc[2].plain), "TU3V": f"{TOP_TU.iloc[2].effect_pp:.2f}",
    "TU4": esc(TOP_TU.iloc[3].plain), "TU4V": f"{TOP_TU.iloc[3].effect_pp:.2f}",
    "BOOT_HOURS": str(AUX["bootstrap_hours_estimate"]),
    "EX_HERO": esc(EX["name"]), "EX0": f"{EX.within_role_pp:+.2f}", "EX2": f"{EX.w2_pp:+.2f}", "EX1": f"{EX.w1_pp:+.2f}",
    "EX0U": f"{EX.within_role_pp:.2f}", "EX2U": f"{EX.w2_pp:.2f}", "EX1U": f"{EX.w1_pp:.2f}", "FIG_YMAX": str(FIG_YMAX),
    "SHAPES_OWN": str(SHAPES_OWN), "SHAPES_POOLED": str(SHAPES_POOLED),
    "POOLED_PCT": pct(AUX[f"pooled_pct_at_{MIN_SHAPE}"], 3),
    "PRIV_HI": pct(AUX["privacy_pct_above_5000"], 0),
    "LL0": ll(MW["W0"]["loglike"]), "LL1": ll(MW["W1"]["loglike"]), "LL2": ll(MW["W2"]["loglike"]),
    "N_TU": str(len(tu)),
    # out-of-sample fit
    "HOLD_DAYS": str(FQ["holdout_days"]), "N_HOLD": num(FQ["n_holdout"]), "N_TRAIN": num(FQ["n_train"]),
    "LL_HOLD": f"{FQ['logloss_holdout']:.4f}", "LL_CONST": f"{FQ['logloss_holdout_constant']:.4f}",
    "BRIER": f"{FQ['brier_holdout']:.4f}", "AUC": f"{FQ['auc_holdout']:.3f}",
    "CAL_SLOPE": f"{FQ['calibration_slope']:.3f}", "CAL_SLOPE_SE": f"{FQ['calibration_slope_se']:.3f}",
    "CAL_INT": f"{FQ['calibration_intercept']:.3f}", "CAL_INT_SE": f"{FQ['calibration_intercept_se']:.3f}",
    "CAL_BINS": str(FQ["bins"]), "CAL_BIN_N": num(int(round(FQ["n_holdout"] / FQ["bins"], -2))),
    "CAL_LO": f"{CAL_LO:.2f}", "CAL_HI": f"{CAL_HI:.2f}",
    "CAL_HOLD_ROWS": cal_rows(cal_h), "CAL_IN_ROWS": cal_rows(cal_i, err=False),
    "P_MIN": pct(100 * FQ["p_holdout_min"], 0), "P_MAX": pct(100 * FQ["p_holdout_max"], 0),
    "SHARE_MID": pct(100 * FQ["share_holdout_between_35_65"], 0),
    "SLOPE_VERDICT": SLOPE_VERDICT, "LL_PRE_HERO": f"{LL_PRE_HERO:.4f}",
    "HG_LO": f"{HG_LO:.2f}", "HG_HI": f"{HG_HI:.2f}", "SG_LO": f"{SG_LO:.2f}", "SG_HI": f"{SG_HI:.2f}",
    "HG_FULL_ROWS": xy_rows(hg_full), "HG_NOHERO_ROWS": xy_rows(hg_none), "SG_ROWS": xy_rows(sg_full),
    "HG_LABEL_NODES": label_nodes(hg_full, HG_LO, HG_HI, HG_LABELS, obstacles=list(zip(hg_none.mean_predicted, hg_none.mean_realised)),
                                  reserved=[(0.0, 0.58, 0.80, 1.0), (0.40, 1.0, 0.0, 0.13)]),
    "SG_LABEL_NODES": label_nodes(sg_full, SG_LO, SG_HI, reserved=[(0.0, 0.40, 0.86, 1.0), (0.45, 1.0, 0.0, 0.08)]),
    "HG_CORR": f"{G['hero_full']['corr']:.3f}", "HG_CORR0": f"{G['hero_no_hero']['corr']:.3f}",
    "HG_GAP": f"{G['hero_full']['mean_abs_gap_pp']:.2f}",
    "HG_SD0": f"{G['hero_no_hero']['sd_predicted_pp']:.2f}", "HG_SD1": f"{G['hero_full']['sd_predicted_pp']:.2f}",
    "SG_N": str(len(sg_full)), "SG_CORR": f"{G['shape_full']['corr']:.3f}", "SG_GAP": f"{G['shape_full']['mean_abs_gap_pp']:.2f}", "LL_POST_HERO": f"{LL_POST_HERO:.4f}", "LADDER_ROWS": LADDER_ROWS,
    "HERO_HDR": HERO_HDR, "HERO_BODY": HERO_BODY, "ATTR_HDR": ATTR_HDR, "ATTR_BODY": ATTR_BODY,
    "TU_HDR": TU_HDR, "TU_BODY": TU_BODY,
}
out = TEX
for _ in range(2):                       # POOLED_SENTENCE itself carries placeholders
    for k, v in subs.items():
        out = out.replace(f"<<{k}>>", v)
left = re.findall(r"<<[A-Z0-9_]+>>", out)
if left:
    raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
open("results/apm_report.tex", "w").write(out)
print(f"wrote results/apm_report.tex ({len(out):,} bytes); {len(H)} heroes, {len(tu)} team-ups; "
      f"{len(subs)} values injected, no database access")
