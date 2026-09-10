"""Draft report for the unified starting-lineup model -- from CSV/JSON only, no database access.

Usage: python3 results/make_lineup_report.py <summary_prefix> [status]
  summary_prefix  prefix of the stage-6 outputs (e.g. results/lineup_selected)
  status          'provisional' (default) or 'confirmed'
Reads: <prefix>_hero_by_rank.csv, <prefix>_replacement.csv, <prefix>_pairs_p50.csv, <prefix>_summary_meta.json,
results/lineup_tuning.json, results/lineup_baseline_dev.json, results/lineup_design/summary.json,
results/rank_coverage.json, results/confirmation_protocol.json, results/dev_snapshot/filters.json,
optionally results/lineup_bootstrap_summary.json and results/lineup_confirmation.json.
Every number in the text is injected; unfilled placeholders abort the build."""
import hashlib, json, os, re, sys
import numpy as np, pandas as pd

prefix = sys.argv[1] if len(sys.argv) > 1 else "results/lineup_selected"
STATUS = sys.argv[2] if len(sys.argv) > 2 else "provisional"

def need(path, kind="csv"):
    try:
        return pd.read_csv(path) if kind == "csv" else json.load(open(path))
    except FileNotFoundError:
        raise SystemExit(f"{path} missing")

H = need(f"{prefix}_hero_by_rank.csv"); R = need(f"{prefix}_replacement.csv"); P = need(f"{prefix}_pairs_p50.csv")
SM = need(f"{prefix}_summary_meta.json", "json"); TUN = need("results/lineup_tuning.json", "json")
BASE = need("results/lineup_baseline_dev.json", "json"); DS = need("results/lineup_design/summary.json", "json")
COV = need("results/rank_coverage.json", "json"); PROT = need("results/confirmation_protocol.json", "json")
SNAP = need("results/dev_snapshot/filters.json", "json")
BOOT = json.load(open("results/lineup_bootstrap_summary.json")) if os.path.exists("results/lineup_bootstrap_summary.json") else None
CONF = json.load(open("results/lineup_confirmation.json")) if os.path.exists("results/lineup_confirmation.json") else None

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

# ---- selected fit and tuning ---------------------------------------------------------------------
lams = SM["lams"]; best_key = ",".join(f"{l:g}" for l in lams)
evals = TUN["evaluations"]; sel = next((e for e in evals if e["key"] == best_key and e["sub"] == 1.0), None) or next((e for e in evals if e["key"] == best_key), None)
if sel is None: raise SystemExit("selected penalties not found in results/lineup_tuning.json")
RED = TUN.get("reduced", {})
def ev_for(k): return next((e for e in evals if e["key"] == k and e["sub"] == 1.0), None)
LADDER = []
for name, label in (("no_pairs_no_slopes_no_heromap", "Heroes, composition, maps, rank imbalance"), ("no_pairs", "+ rank slopes and hero-by-map"),
                    ("no_pair_slopes", "+ allied and opposing pairs (no pair slopes)"), ("selected", "+ pair rank slopes (selected fit)")):
    e = ev_for(RED.get(name, "")) if RED else None
    if e: LADDER.append((label, e))
BASE_MEAN = BASE["mean_val_logloss"]
def ladder_rows():
    rows = [rf"Headline specification, refitted on development rows & {BASE['k']} & " + " & ".join(f"{f['logloss']:.5f}" for f in BASE["folds"]) + rf" & {BASE_MEAN:.5f} & --- \\"]
    for label, e in LADDER:
        rows.append(rf"{esc(label)} & --- & " + " & ".join(f"{f['logloss']:.5f}" for f in e["folds"]) + rf" & {e['mean_val_logloss']:.5f} & {e['mean_cal_slope']:.3f} \\")
    return "\n".join(rows)
# tuning table: best evaluation and its neighbours (same sub as the selection)
def tuning_rows(k=8):
    ev = sorted([e for e in evals if e["sub"] == sel["sub"]], key=lambda e: e["mean_val_logloss"])[:k]
    return "\n".join(rf"{esc(e['key'])} & {e['mean_val_logloss']:.5f} & {e['mean_cal_slope']:.3f} & {'yes' if all(f['converged'] for f in e['folds']) else 'no'} \\" for e in ev)
N_EVAL = len(evals); SUB = sel["sub"]
GAIN = BASE_MEAN - sel["mean_val_logloss"]
sel_third_slopes = [np.mean([f["by_third"][t]["cal_slope"] for f in sel["folds"]]) for t in range(3)]

# ---- hero tables -------------------------------------------------------------------------------------
lob = SM["lobby_at"]; thirds = SM["thirds"]
R = R.merge(H[["hero_id", "beta_p25", "beta_p50", "beta_p75"]], on="hero_id")
def hero_row(r):
    return (f"{esc(r['name'])} & {f3(r.beta_p25)} & {f3(r.beta_p50)} & {f3(r.beta_p75)} & {f2(r.obs_meta_pp)} & {f2(r.common_ref_pp)} & "
            f"{f2(r.obs_meta_pp_third0)} & {f2(r.obs_meta_pp_third1)} & {f2(r.obs_meta_pp_third2)} & {100*r.coverage:.0f}")
def role_block(role, letter):
    d = R[R.role == role].sort_values("common_ref_pp", ascending=False)
    return [panel(f"Panel {letter}: {role} ({len(d)} heroes)", 10)] + [hero_row(r) for _, r in d.iterrows()]
HERO_BODY = side_by_side([role_block("Tank", "A") + role_block("Support", "C"), role_block("Damage", "B")], 10)
top = {ro: R[R.role == ro].sort_values("common_ref_pp", ascending=False) for ro in ("Tank", "Damage", "Support")}
def top_phrase(ro, k=3): return oxford(f"{esc(r['name'])} ({r.common_ref_pp:+.1f})" for _, r in top[ro].head(k).iterrows())
GRAD = R.assign(g=R.obs_meta_pp_third2 - R.obs_meta_pp_third0).dropna(subset=["g"]).sort_values("g")
RANK_GAIN = oxford(f"{esc(r['name'])} ({r.obs_meta_pp_third0:+.1f} to {r.obs_meta_pp_third2:+.1f})" for _, r in GRAD.tail(3).iloc[::-1].iterrows())
RANK_LOSS = oxford(f"{esc(r['name'])} ({r.obs_meta_pp_third0:+.1f} to {r.obs_meta_pp_third2:+.1f})" for _, r in GRAD.head(2).iterrows())
COV_MED = float(R.coverage.median())

# ---- pair tables ---------------------------------------------------------------------------------------
MIN_SUPPORT = 500
A = P[P.kind == "allied"].copy(); O = P[P.kind == "opposing"].copy()
A_ok = A[A.support_dev >= MIN_SUPPORT]; O_ok = O[O.support_dev >= MIN_SUPPORT]
TU = A_ok[A_ok.is_teamup].sort_values("did", ascending=False)
NT = A_ok[~A_ok.is_teamup]
def pair_row(r, opp=False):
    if opp:
        a, b, c, dd = (r.hero_a, r.hero_b, r.coef, r.did) if r.did >= 0 else (r.hero_b, r.hero_a, -r.coef, -r.did)
        label = f"{esc(a)} over {esc(b)}"
    else:
        label = f"{esc(r.hero_a)} $\\times$ {esc(r.hero_b)}"; c, dd = r.coef, r.did
    return f"{label} & {c:+.3f} & {dd:+.3f} & {num(r.support_dev)} & {num(r.support_third0)}/{num(r.support_third1)}/{num(r.support_third2)}"
TU_ROWS = side_by_side([[pair_row(r) for r in TU.iloc[:(len(TU) + 1) // 2].itertuples()], [pair_row(r) for r in TU.iloc[(len(TU) + 1) // 2:].itertuples()]], 5)
NT_POS = NT.sort_values("did", ascending=False).head(25); NT_NEG = NT.sort_values("did").head(25)
NT_ROWS = side_by_side([[pair_row(r) for r in NT_POS.itertuples()], [pair_row(r) for r in NT_NEG.itertuples()]], 5)
O_top = O_ok.assign(a=O_ok.did.abs()).sort_values("a", ascending=False).head(50)
O_ROWS = side_by_side([[pair_row(r, True) for r in O_top.iloc[:25].itertuples()], [pair_row(r, True) for r in O_top.iloc[25:].itertuples()]], 5)
TU_TOP = oxford(f"{esc(r.hero_a)} with {esc(r.hero_b)} ({r.did:+.2f})" for r in TU.head(3).itertuples())
NT_TOP = oxford(f"{esc(r.hero_a)} with {esc(r.hero_b)} ({r.did:+.2f})" for r in NT_POS.head(3).itertuples())
NT_BOT = oxford(f"{esc(r.hero_a)} with {esc(r.hero_b)} ({r.did:+.2f})" for r in NT_NEG.head(3).itertuples())
def opp_phrase(r):
    a, b, dd = (r.hero_a, r.hero_b, r.did) if r.did >= 0 else (r.hero_b, r.hero_a, -r.did); return f"{esc(a)} over {esc(b)} ({dd:+.2f})"
O_TOP = oxford(opp_phrase(r) for r in O_top.head(4).itertuples())
SD = {"tu": float(TU.did.std()), "nt": float(NT.did.std()), "opp": float(O_ok.did.std())}

# ---- bootstrap and confirmation (optional) -------------------------------------------------------------
if BOOT:
    BOOT_TEXT = (f"Across <<BOOT_REPS>> replications, the replacement scores have a median across-replication standard deviation of "
                 f"<<BOOT_REPL_SD>> percentage points, and <<BOOT_TOP_STABLE>> of the ten largest common-reference scores keep their sign in every replication. "
                 f"Among the highlighted pairs, <<BOOT_PAIR_STABLE>> of the <<BOOT_PAIR_N>> listed allied contrasts and <<BOOT_OPP_STABLE>> of the "
                 f"<<BOOT_OPP_N>> listed opposing contrasts keep their sign in at least 95\\% of replications. These are stability statements "
                 f"conditional on the selected specification and penalties, not frequentist confidence intervals.")
else:
    BOOT_TEXT = "The bootstrap stability analysis has not been run for this fit; no stability claim is made."
if CONF:
    verdict = CONF["verdict"]
    CONF_TEXT = (f"The locked evaluation compares the candidate and the refitted headline model on the same <<CONF_N>> eligible confirmation matches. "
                 f"The paired log-loss improvement is <<CONF_D>> (player-multiplicity bootstrap 95\\% interval <<CONF_LO>> to <<CONF_HI>>), the candidate's "
                 f"calibration slope is <<CONF_SLOPE>> overall, and by rank third <<CONF_SLOPES>>. Verdict under the predeclared criterion: <<CONF_VERDICT>>.")
else:
    CONF_TEXT = ("The confirmation slice (<<N_CONF>> matches played in the final four days of the snapshot) has not been opened. Its outcomes were excluded "
                 "from every design build, fold, tuning run, diagnostic and decision above; the evaluation is run once, after the candidate is locked.")

TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs,caption,amsmath,microtype,pdflscape,makecell,titlesec,fancyhdr,float,placeins}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{16pt plus 4pt minus 2pt}{6pt}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\thepage}\fancyfoot[R]{\scriptsize build \texttt{<<BUILD>>}}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=6pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.05}
\widowpenalty=10000 \clubpenalty=10000
\begin{document}

\begin{center}
{\Large\bfseries A Unified Starting-Lineup Model for \emph{Marvel Rivals}}\\[4pt]
{\large <<STATUS_LINE>>}\\[10pt]
{\normalsize Clocktock (Philip Livdan)}\\[2pt]
{\small McCombs School of Business, The University of Texas at Austin}\\
{\small\texttt{plivdan@utexas.edu}}\\[8pt]
{\small Season <<SEASON>>, PC ranked, balance regime 7 August to 11 September 2026. Development snapshot frozen <<FROZEN>>. Report build \texttt{<<BUILD>>}.}
\end{center}

\vspace{4pt}
\noindent This report estimates one penalized match-level model of the probability that side~0 wins,
conditional on the recorded starting lineups and pre-match information, and derives every reported
quantity from that single fit. It replaces separate treatment of designated team-ups, rank bands, and
pairwise interactions with one specification in which hero effects, composition effects, hero-by-map
deviations, allied pairs, and opposing pairs are estimated jointly, each with a pooled rank slope, under
five ridge penalties tuned on chronological development folds. The estimand is an adjusted
starting-lineup association: what happens after a team starts a hero, including whatever swaps and
responses follow, in the sampled competitive environment. It is not the effect of a hero's abilities
in uninterrupted play, and starting attribution alone does not make it a causal estimate.

\section*{Sample, regime, and protocol}

The development snapshot holds <<N_SNAP>> competitive PC ranked matches played between
<<PLAY_MIN>> and <<PLAY_MAX>> UTC, after the published filters (twelve players, six per side, no draws,
scores present, every player with positive play time, at least 240 seconds). The starting record must
also be legal: <<N_ILLEGAL>> matches in which two teammates share the same first-appearance hero are
excluded, because a shared start cannot be simultaneous and the record is unverifiable. That leaves
<<N_DESIGN>> matches. The official patch record places the whole snapshot inside one balance regime,
from the Season 9.5 patch of 7 August to the Season 10 patch of 11 September; the five intermediate
updates were cosmetic. A weekly win-rate check flags only The Hood, released on 7 August, whose drift
is a learning curve rather than a patch signature.

Collection was stopped at the user's instruction, so the locked confirmation sample is internal: the
<<N_CONF>> matches played in the final four days of the snapshot, after <<CONF_CUT>> UTC. The remaining
<<N_DEV>> development matches feed every design build, fold, tuning run, and table below. The
confirmation outcomes are read once, after the candidate is locked, against a predeclared criterion.
Because the published headline model was estimated on the whole snapshot, the baseline for that
comparison is the identical headline specification refitted on the development rows.

Lobby rank, the mean pre-match score of the twelve starters, runs from <<LOB_P5>> at the 5th percentile
to <<LOB_P95>> at the 95th, with a median of <<LOB_P50>>. Rank-dependent quantities are evaluated inside
that range and no claim is made above 5{,}200. <<N_ABOVE_5000>> matches have a lobby mean above 5{,}000,
and no individual player in the sample scores above 5{,}750, so the top of the ranked ladder is outside
the data. That is direct coverage evidence.

\section*{Specification}

Let $A_{mh}$ and $B_{mh}$ indicate whether hero $h$ starts on side~0 or side~1 of match $m$, $r_m$ the
standardized lobby rank, $\Delta R_m$ the standardized difference in team mean pre-match score, $v_m$ the
map, and $c(\cdot)$ the composition (Tank--Damage--Support counts). With
$x_{mh}=A_{mh}-B_{mh}$, $z_{mhk}=A_{mh}A_{mk}-B_{mh}B_{mk}$ and $w_{mhk}=A_{mh}B_{mk}-A_{mk}B_{mh}$ for $h<k$,
\begin{align*}
\operatorname{logit}p_m ={}& \alpha_{v_m}+(\theta_0+\theta_1 r_m)\Delta R_m
 +(\gamma_0+\gamma_1 r_m)^\top[c(A_m)-c(B_m)]
 +\sum_h(\beta_h+d_h r_m+u_{h,v_m})\,x_{mh}\\
&+\sum_{h<k}(S_{hk}+T_{hk}r_m)\,z_{mhk}+\sum_{h<k}(C_{hk}+D_{hk}r_m)\,w_{mhk}.
\end{align*}
Every block uses the same starting record. The allied block is symmetric and the opposing block
antisymmetric, so swapping the two camps negates every feature except the map intercept, which the
implementation checks numerically. Every observed composition shape receives its own coefficient, every
observed allied and opposing pair its own coefficient, and every hero its own deviation on every map;
nothing is removed by a frequency threshold. The blocks are then aliased with each other in known ways:
summing a hero's allied features over its five teammates gives five times its own contrast, summing its
opposing features over six opponents gives six times it, summing allied features within a role pair gives
a function of the composition contrast, and summing hero-by-map deviations over maps gives the hero
contrast again. Each candidate centering direction was tested against the data and imposed only when
its image lies exactly in the span of the lower-order blocks; <<N_CONSTRAINTS>> of <<N_CANDIDATES>>
candidates did, with the largest residual at <<MAX_RESID>>. The reduced design has <<N_FREE>> free
coefficients.

Five ridge groups shrink the coefficients on their native scale: hero and composition averages
($\lambda_1$), allied averages ($\lambda_2$), opposing averages ($\lambda_3$), hero rank slopes,
composition rank slopes and hero-by-map deviations ($\lambda_4$), and pair rank slopes ($\lambda_5$).
Map intercepts and the two rank-imbalance terms are unpenalized. The rank basis and $\Delta R$ are
scaled with training rows only. Designated team-ups are labels attached after estimation; they receive
no separate regressors, constraints, or penalties.

\section*{Tuning}

Penalties were tuned on three chronological development folds, training on the first 60, 73 and 87 per
cent of development matches by play time and validating on the following slice of about 60{,}000
matches each. A bounded coordinate search on $\log_{10}\lambda$ with warm starts ran <<N_EVAL>>
evaluations<<SUB_CLAUSE>>. The selected penalties are
$\lambda=(<<LAMS>>)$, with mean validation log loss <<SEL_LL>> and mean calibration slope
<<SEL_SLOPE>>, against <<BASE_LL>> for the refitted headline specification on the same folds, a gain of
<<GAIN>>. Calibration by supported rank third is <<SEL_THIRDS>>.

\begin{center}\small
\begin{tabular}{l r r r}
\toprule
$\lambda_1,\lambda_2,\lambda_3,\lambda_4,\lambda_5$ & Mean validation log loss & Mean calibration slope & Converged \\
\midrule
<<TUNING_ROWS>>
\bottomrule
\end{tabular}
\captionof{table}{Best evaluations of the penalty search (lowest validation log loss first)}
\end{center}

\begin{center}\small
\begin{tabular}{l r r r r r r}
\toprule
Model & Parameters & Fold 1 & Fold 2 & Fold 3 & Mean & Cal.\ slope \\
\midrule
<<LADDER_ROWS>>
\bottomrule
\end{tabular}
\captionof{table}{Development ladder: genuine reduced fits at the selected penalties}\label{tab:ladder}
\end{center}
\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} Validation log loss on the same three chronological folds. Reduced fits set the omitted
blocks' penalties to $10^9$, which drives those coefficients to zero, so each row is a genuine reduced
model rather than a post-hoc zeroing. The headline row is the published specification (187 parameters,
starting-hero attribution, within-role constraint, map intercepts, designated team-ups) refitted on the
development rows.
\end{minipage}

\section*{Hero values}

Table~\ref{tab:hero} reports three things per hero from the selected fit. The coefficient $\beta_h(r)$ is
the within-role log-odds contrast evaluated at the 25th, 50th and 75th percentiles of lobby rank. The
replacement scores are exact predicted probability differences, in percentage points, from placing the
hero into a starting slot in place of another hero of the same role, recomputing all five allied and six
opposing relationships, the hero-by-map deviation, and the rank slope, and keeping the slot's player,
rank, map, side, and the other eleven starters fixed. Two references are declared. The observed-meta
score replaces the hero that actually started in the slot, averaged over <<POOL_N>> development matches.
The common-reference score compares the hero with the average legal same-role hero in the same slot, so
every hero of a role is scored on the same context pool. Legality respects bans and hero uniqueness;
coverage is the share of same-role slots in which the hero was a legal replacement, with a median of
<<COV_MED>>.

By the common reference, the strongest Tanks are <<TOP_TANK>>, the strongest Damage heroes
<<TOP_DAMAGE>>, and the strongest Supports <<TOP_SUPPORT>>. The observed-meta scores by rank third show
which heroes' values move with the environment: <<RANK_GAIN>> gain the most from the bottom to the top
third, while <<RANK_LOSS>> lose the most. These are pooled slopes evaluated at observed contexts, not
separate rank models.

\section*{Pairs}

Pair effects are reported as four-lineup contrasts on the log-odds scale: the pair's coefficient minus the
average coefficient of each member with the same-role alternatives to the other, plus the average over
alternative pairs. This cancels the hero main effects and each hero's other relationships and does not
depend on where additive hero content is placed. Pairs are shown only with at least <<MIN_SUPPORT>>
identifying matches, that is, distinct development matches in which the signed feature is nonzero.
Designated team-ups are labelled; the largest by contrast are <<TU_TOP>>. Among pairs that are not
designated team-ups, the largest positive contrasts are <<NT_TOP>> and the largest negative <<NT_BOT>>.
The largest opposing matchups are <<O_TOP>>. The standard deviation of the contrasts is <<SD_TU>> across
team-up-eligible pairs, <<SD_NT>> across other allied pairs, and <<SD_OPP>> across opposing pairs.

A team-up-eligible pair's coefficient combines ordinary kit complementarity, any mechanic actually used,
player selection, coordination, and subsequent decisions; the data do not record activation. A
starting-matchup coefficient compares matches that begin with that configuration and includes whatever
adaptation follows; it is not the effect of an opponent switching to the counter mid-match.

\section*{Stability}

<<BOOT_TEXT>>

\section*{Confirmation}

<<CONF_TEXT>>

\FloatBarrier
\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}
\captionof{table}{Hero coefficients by lobby rank and same-slot replacement values from the selected fit}\label{tab:hero}
\begin{tabular}{l r r r r r r r r r @{\hspace{1.4em}} l r r r r r r r r r}
\toprule
& \multicolumn{3}{c}{$\beta_h(r)$ at lobby-rank percentile} & \multicolumn{2}{c}{Replacement, pp} & \multicolumn{3}{c}{Observed meta by third} & &
& \multicolumn{3}{c}{$\beta_h(r)$ at lobby-rank percentile} & \multicolumn{2}{c}{Replacement, pp} & \multicolumn{3}{c}{Observed meta by third} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){12-14}\cmidrule(lr){15-16}\cmidrule(lr){17-19}
& 25th & 50th & 75th & Obs.\ meta & Common ref. & Low & Mid & High & Cov.\ \% &
& 25th & 50th & 75th & Obs.\ meta & Common ref. & Low & Mid & High & Cov.\ \% \\
\midrule
<<HERO_BODY>>
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-2pt}
\noindent\begin{minipage}{\linewidth}\scriptsize
\emph{Notes.} Coefficients are within-role log-odds contrasts at lobby scores of <<LOB_P25>>, <<LOB_P50>>
and <<LOB_P75>>. Replacement values are exact predicted probability differences in percentage points over
<<POOL_N>> development matches: observed meta replaces the actual same-role incumbent, common reference
compares with the average legal same-role hero in the same slot. Rank thirds are split at lobby scores
<<THIRD_LO>> and <<THIRD_HI>>. Coverage is the share of same-role slots where the hero was legal (not on
that side already, not banned). Heroes are sorted by the common-reference score within role. Cells with
fewer than 200 legal contexts are blank.
\end{minipage}
\end{landscape}

\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.0}
\captionof{table}{Team-up-eligible starting pairs from the selected fit}\label{tab:teamups}
\begin{tabular}{l r r r r @{\hspace{1.5em}} l r r r r}
\toprule
Pair & Coef. & Contrast & Matches & By third & Pair & Coef. & Contrast & Matches & By third \\
\midrule
<<TU_ROWS>>
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-2pt}
\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} Allied pairs designated as team-ups in the Season 9.5 catalogue, with at least
<<MIN_SUPPORT>> identifying matches, sorted by contrast. \emph{Coef.} is $S_{hk}(r)$ at the median lobby
rank under the uniform centering convention (each hero's allied deviations sum to zero across its
partners and within each role pair). \emph{Contrast} is the four-lineup log-odds contrast against
same-role alternatives. \emph{Matches} counts distinct development matches with a nonzero signed
feature, split by lobby-rank third. Read as the synergy of a team-up-eligible starting pair.
\end{minipage}
\end{landscape}

\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.0}
\captionof{table}{Other allied pairs: largest positive and negative contrasts}\label{tab:allied}
\begin{tabular}{l r r r r @{\hspace{1.5em}} l r r r r}
\toprule
\multicolumn{5}{l}{\emph{Largest positive}} & \multicolumn{5}{l}{\emph{Largest negative}} \\
Pair & Coef. & Contrast & Matches & By third & Pair & Coef. & Contrast & Matches & By third \\
\midrule
<<NT_ROWS>>
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-2pt}
\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} Allied pairs that are not designated team-ups, with at least <<MIN_SUPPORT>> identifying
matches. Columns as in Table~\ref{tab:teamups}. A negative contrast means the pair does worse together
than its members' other pairings predict.
\end{minipage}
\end{landscape}

\begin{landscape}
\begin{center}
{\scriptsize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.0}
\captionof{table}{Opposing matchups: largest contrasts}\label{tab:opposing}
\begin{tabular}{l r r r r @{\hspace{1.5em}} l r r r r}
\toprule
Matchup & Coef. & Contrast & Matches & By third & Matchup & Coef. & Contrast & Matches & By third \\
\midrule
<<O_ROWS>>
\bottomrule
\end{tabular}
}
\end{center}
\vspace{-2pt}
\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} \emph{A over B} means A's side wins more often when the two start on opposite sides than
the two heroes' other matchups predict; the coefficient and contrast are oriented so that both are
positive for A. At least <<MIN_SUPPORT>> identifying matches. Starting matchups include whatever adaptation
followed.
\end{minipage}
\end{landscape}

\end{document}
"""

subs = {
    "STATUS_LINE": "Provisional development results, confirmation locked" if STATUS == "provisional" else "Confirmed results",
    "SEASON": SNAP["season_label"], "FROZEN": SNAP["frozen_at_utc"][:10],
    "N_SNAP": num(SNAP["n_matches"]), "PLAY_MIN": SNAP["play_time_min_utc"][:16].replace("T", " "), "PLAY_MAX": SNAP["play_time_max_utc"][:16].replace("T", " "),
    "N_ILLEGAL": num(DS["n_excluded_duplicate_starters"]), "N_DESIGN": num(DS["n"]),
    "N_CONF": num(BASE["n_conf"]), "N_DEV": num(BASE["n_dev"]), "CONF_CUT": "2026-09-04 11:00",
    "LOB_P5": f"{COV['lobby_mean_quantiles']['0.05']:,.0f}", "LOB_P95": f"{COV['lobby_mean_quantiles']['0.95']:,.0f}", "LOB_P50": f"{COV['lobby_mean_quantiles']['0.5']:,.0f}",
    "LOB_P25": f"{lob['p25']:,.0f}", "LOB_P75": f"{lob['p75']:,.0f}", "N_ABOVE_5000": num(COV["matches_with_lobby_mean_above"]["5000"]),
    "N_CONSTRAINTS": str(sum(v["imposed"] for v in DS["constraints"].values())), "N_CANDIDATES": str(sum(v["candidates"] for v in DS["constraints"].values())),
    "MAX_RESID": f"{max(v['max_residual_imposed'] for v in DS['constraints'].values()):.1e}",
    "N_FREE": num(sum(DS["free"].values()) + DS["free"]["hero"] + DS["free"]["shape"] + DS["free"]["allied"] + DS["free"]["opposing"] + 1),
    "N_EVAL": str(N_EVAL), "SUB_CLAUSE": f", the coarse passes on a {int(100*SUB)} per cent subsample of each fold's training rows" if SUB < 1 else "",
    "LAMS": ", ".join(f"{l:g}" for l in lams), "SEL_LL": f"{sel['mean_val_logloss']:.5f}", "SEL_SLOPE": f"{sel['mean_cal_slope']:.3f}",
    "BASE_LL": f"{BASE_MEAN:.5f}", "GAIN": f"{GAIN:+.5f}", "SEL_THIRDS": oxford(f"{s:.3f}" for s in sel_third_slopes),
    "TUNING_ROWS": tuning_rows(), "LADDER_ROWS": ladder_rows(),
    "POOL_N": num(SM["pool_size"]), "COV_MED": f"{100*COV_MED:.0f}\\%", "THIRD_LO": f"{thirds[0]:,.0f}", "THIRD_HI": f"{thirds[1]:,.0f}",
    "TOP_TANK": top_phrase("Tank"), "TOP_DAMAGE": top_phrase("Damage"), "TOP_SUPPORT": top_phrase("Support"),
    "RANK_GAIN": RANK_GAIN, "RANK_LOSS": RANK_LOSS, "HERO_BODY": HERO_BODY,
    "MIN_SUPPORT": num(MIN_SUPPORT), "TU_TOP": TU_TOP, "NT_TOP": NT_TOP, "NT_BOT": NT_BOT, "O_TOP": O_TOP,
    "SD_TU": f"{SD['tu']:.3f}", "SD_NT": f"{SD['nt']:.3f}", "SD_OPP": f"{SD['opp']:.3f}",
    "TU_ROWS": TU_ROWS, "NT_ROWS": NT_ROWS, "O_ROWS": O_ROWS, "BOOT_TEXT": BOOT_TEXT, "CONF_TEXT": CONF_TEXT,
}
if BOOT:
    subs.update({"BOOT_REPS": str(BOOT["reps"]), "BOOT_REPL_SD": f"{BOOT['replacement_sd_median']:.2f}", "BOOT_TOP_STABLE": str(BOOT["top10_sign_stable"]),
                 "BOOT_PAIR_STABLE": str(BOOT["allied_listed_sign_stable"]), "BOOT_PAIR_N": str(BOOT["allied_listed_n"]),
                 "BOOT_OPP_STABLE": str(BOOT["opposing_listed_sign_stable"]), "BOOT_OPP_N": str(BOOT["opposing_listed_n"])})
if CONF:
    subs.update({"CONF_N": num(CONF["n"]), "CONF_D": f"{CONF['D']:+.5f}", "CONF_LO": f"{CONF['ci_boot'][0]:+.5f}", "CONF_HI": f"{CONF['ci_boot'][1]:+.5f}",
                 "CONF_SLOPE": f"{CONF['cal_slope']:.3f}", "CONF_SLOPES": oxford(f"{s:.3f}" for s in CONF["cal_slope_thirds"]), "CONF_VERDICT": CONF["verdict"]})
out = TEX
for _ in range(2):
    for k, v in subs.items():
        out = out.replace(f"<<{k}>>", v)
BUILD = hashlib.sha256(out.encode()).hexdigest()[:8]; out = out.replace("<<BUILD>>", BUILD)
left = re.findall(r"<<[A-Z0-9_]+>>", out)
if left: raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
open("results/lineup_report.tex", "w").write(out)
print(f"wrote results/lineup_report.tex ({len(out):,} bytes); build {BUILD}; status {STATUS}")
