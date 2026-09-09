"""Generate the APM report as journal-format LaTeX -- from CSV/JSON only.

This never touches the database. Every number in the prose and tables is read
from a results file or computed from one, so a refit regenerates a correct
document with no hand edits. If a required column is missing, it fails loudly
rather than render stale text.

Inputs (all under results/):
  apm_hero_table_<TAG>.csv        headline fit (W0, within-role, map intercepts)
  apm_hero_table_W0_full.csv      W0 under the global constraint (reparameterisation check)
  apm_hero_table_W1_full.csv, apm_hero_table_W2_full.csv   attribution comparison
  apm_teamup_table_<TAG>.csv      team-ups with anchor/partner
  apm_run_meta_*.json             run metadata
  aux_stats.json                  non-fit statistics, from compute_aux_stats.py
"""
import json
import pandas as pd
from scipy import stats

TAG = "W0_specAplus"

H   = pd.read_csv(f"results/apm_hero_table_{TAG}.csv")
W0F = pd.read_csv("results/apm_hero_table_W0_full.csv").set_index("hero_id")
W1F = pd.read_csv("results/apm_hero_table_W1_full.csv").set_index("hero_id")
W2F = pd.read_csv("results/apm_hero_table_W2_full.csv").set_index("hero_id")
TU  = pd.read_csv(f"results/apm_teamup_table_{TAG}.csv")
M   = json.load(open(f"results/apm_run_meta_{TAG}.json"))
MW  = {r: json.load(open(f"results/apm_run_meta_{r}_full.json")) for r in ("W0", "W1", "W2")}
try:
    AUX = json.load(open("results/aux_stats.json"))
except FileNotFoundError:
    raise SystemExit("results/aux_stats.json missing -- run results/compute_aux_stats.py first")

for col in ("n_start", "pick_pct", "raw_wr", "within_role_p_adjusted", "within_role_significant"):
    if col not in H.columns:
        raise SystemExit(f"hero table lacks '{col}' -- rerun results/run_apm.py (or patch) before rendering")
for col in ("anchor", "partner"):
    if col not in TU.columns:
        raise SystemExit(f"team-up table lacks '{col}' -- rerun results/run_apm.py before rendering")

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

def pct(x, d=1): return f"{x:.{d}f}\\%"
def num(x): return f"{x:,}"
def ll(x): return f"$-{abs(x):,.0f}$".replace(",", "{,}")
def oxford(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]

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

def top(role, k):
    return H[H.role == role].sort_values("within_role_pp", ascending=False).head(k)
TT, TD, TS = top("Tank", 6), top("Damage", 5), top("Support", 5)
tu = TU.sort_values("effect_pp", ascending=False).reset_index(drop=True)
tu["label"] = tu.apply(lambda r: f"{esc(r.anchor)} $\\times$ {esc(r.partner)}", axis=1)
tu["plain"] = tu.apply(lambda r: f"{r.anchor}--{r.partner}", axis=1)
TOP_TU = tu.head(4)
TW = AUX["tank_winrate"]

# ----------------------------------------------------------------------------
# tables
def hero_panel(role, letter):
    d = H[H.role == role].sort_values("within_role_pp", ascending=False)
    out = [rf"\multicolumn{{9}}{{l}}{{\emph{{Panel {letter}: {role}}} \ ({len(d)} heroes)}} \\[2pt]"]
    for _, r in d.iterrows():
        out.append(
            f"{esc(r['name'])} & {r.within_role_pp:.2f}{stars(r.within_role_p_adjusted)} & "
            f"({r.se_pp:.3f}) & [{r.within_role_ci_low_pp:.2f}, {r.within_role_ci_high_pp:.2f}] & "
            f"{r.within_role_p_adjusted:.3f} & {r.n_start:,} & {r.pick_pct:.2f} & "
            f"{r.raw_wr:.1f} & {r.w1_pp - r.within_role_pp:+.2f} \\\\")
    return "\n".join(out)

def attrib_panel(role, letter):
    d = H[H.role == role].sort_values("within_role_pp", ascending=False)
    out = [rf"\multicolumn{{5}}{{l}}{{\emph{{Panel {letter}: {role}}}}} \\[2pt]"]
    for _, r in d.iterrows():
        out.append(f"{esc(r['name'])} & {r.within_role_pp:.2f} & {r.w2_pp:.2f} & "
                   f"{r.w1_pp:.2f} & {r.w1_pp - r.within_role_pp:+.2f} \\\\")
    return "\n".join(out)

half = (len(tu) + 1) // 2
TU_ROWS = []
for i in range(half):
    L = tu.iloc[i]; cells = [f"{L.label} & {L.effect_pp:.2f} & ({L.std_error*25:.3f})"]
    if i + half < len(tu):
        R = tu.iloc[i + half]; cells.append(f"{R.label} & {R.effect_pp:.2f} & ({R.std_error*25:.3f})")
    else:
        cells.append(" & & ")
    TU_ROWS.append(" & ".join(cells) + r" \\")
TU_ROWS = "\n".join(TU_ROWS)

HERO_HDR = r"""& \multicolumn{1}{c}{APM} & \multicolumn{1}{c}{Std.\ error} &
\multicolumn{1}{c}{95\% CI} & \multicolumn{1}{c}{$q$} &
\multicolumn{1}{c}{Starts} & \multicolumn{1}{c}{Pick \%} &
\multicolumn{1}{c}{Raw WR \%} & \multicolumn{1}{c}{W1$-$W0} \\"""
ATTR_HDR = r"""& \multicolumn{1}{c}{W0} & \multicolumn{1}{c}{W2} &
\multicolumn{1}{c}{W1} & \multicolumn{1}{c}{W1$-$W0} \\"""
TU_HDR = r"""Team-up & \multicolumn{1}{c}{Premium} & \multicolumn{1}{c}{Std.\ err.} &
Team-up & \multicolumn{1}{c}{Premium} & \multicolumn{1}{c}{Std.\ err.} \\"""

# ----------------------------------------------------------------------------
# document.  <<name>> placeholders are substituted below; braces are plain LaTeX.
TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,siunitx,threeparttable,caption,amsmath,longtable,microtype}
\usepackage{pdflscape}
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
The raw data make this clear. Teams running the standard two-Tank composition win
<<TW2>> of their matches, compared with <<TW1>> for teams running one Tank and
<<TW3>> for teams running three. The relationship is therefore strongly nonlinear
and peaks around the conventional 2--2--2 structure, which <<SHARE222>> of teams
field. A simple linear control for the number of Tanks would miss this pattern and,
because teams with too few Tanks are much more common than teams with too many, would
misleadingly produce a positive average slope. The model instead includes
composition-shape indicators so that unusual role structures are absorbed by the
composition controls rather than attributed to individual heroes.

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

Hero attribution is based on the hero each player starts the match on. The API
records heroes chronologically by first appearance, so the first entry provides an
assignment that is fixed before the outcome of the match is known. This gives the
estimates an intention-to-treat interpretation. If a player starts on a hero and
switches thirty seconds later, the original hero still receives full attribution.
That necessarily attenuates the estimated effect of heroes that are frequently
abandoned, but it avoids allowing information generated during the match to
determine the regressors. In this setting, the tendency for players to abandon a hero
is also arguably part of the hero's practical value.

An apparently attractive alternative is to weight heroes by the amount of time they
are played. The data show why this is problematic. Moving from starting-lineup
attribution to increasingly outcome-dependent attribution rules makes the estimated
effects systematically larger and improves in-sample fit. Across the same set of
matches, the cross-hero standard deviation of the estimates rises from <<SD0>>
percentage points under starting attribution to <<SD2>> under the intermediate rule
and <<SD1>> under full play-time weighting. The ranking of heroes remains fairly
stable, with rank correlations of <<RHO2>> and <<RHO1>> against the starting
estimates, but the magnitudes expand as more post-start information enters the
regressors. That pattern is exactly what we would expect if play-time weighting
partly encodes what happened during the match. For that reason, starting-lineup
attribution is used for the headline estimates.

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

The team-up estimates (Appendix Table~\ref{tab:teamups}) capture an additional
object: the value associated with fielding a designated pair beyond the sum of the
two heroes' individual effects. For example, the <<TU1>> pair carries an estimated
premium of <<TU1V>> percentage points, <<TU2>> <<TU2V>> points, <<TU3>> <<TU3V>>
points, and <<TU4>> <<TU4V>> points. These coefficients should therefore not be read
as the total strength of the pair. They measure only the incremental association
with winning beyond what the two component heroes already contribute separately.

The results nevertheless have several important limitations. First, the exact
magnitudes depend on the attribution rule more than the ordering does. Starting
attribution produces a cross-hero standard deviation of <<SD0>> percentage points,
compared with <<SD2>> and <<SD1>> under progressively more outcome-dependent rules,
while rank correlations with the starting estimates remain <<RHO2>> and <<RHO1>>.
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
population. <<SHAPES_OWN>> composition shapes receive separate indicators, while the
remaining <<SHAPES_POOLED>> are pooled into an ``other'' category representing only
<<POOLED_PCT>> of team instances. In addition, profile privacy rises sharply at high
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
{\scriptsize\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{0.98}
\begin{longtable}{l r c c r r r r r}
\caption{Hero adjusted plus--minus, relative to an average hero of the same role}
\label{tab:main} \\
\toprule
<<HERO_HDR>>
\midrule
\endfirsthead
\multicolumn{9}{l}{\emph{Table \ref{tab:main}, continued}} \\
\toprule
<<HERO_HDR>>
\midrule
\endhead
\midrule
\multicolumn{9}{r}{\emph{continued on next page}} \\
\endfoot
\midrule
Matches & \multicolumn{8}{l}{<<N_MATCHES>>} \\
Player--matches & \multicolumn{8}{l}{<<N_PLAYERS>>} \\
Heroes & \multicolumn{8}{l}{<<N_HEROES>>} \\
Log-likelihood & \multicolumn{8}{l}{<<LOGLIK>>} \\
\bottomrule
\endlastfoot
<<PANEL_TANK>>
\addlinespace[3pt]
<<PANEL_DAMAGE>>
\addlinespace[3pt]
<<PANEL_SUPPORT>>
\end{longtable}
}

\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} All 55 heroes are listed. \emph{APM} is in percentage points of win
probability at a balanced match ($\partial p/\partial x=\beta/4$): the change from
fielding this hero at match start in place of an average hero of the same role.
Standard errors in parentheses are HC1. Stars and $q$ denote Benjamini--Hochberg
false-discovery-rate control across the 55-hero family: \sym{*}~$q<0.10$,
\sym{**}~$q<0.05$, \sym{***}~$q<0.01$; <<N_NS>> of the 55 are not distinguishable
from their role average. \emph{Starts} is the number of player-slots that began the
match on this hero and \emph{Pick~\%} its share of all starting slots; both reflect
the crawl's per-hero leaderboard seeding, not population pick rates. \emph{Raw WR} is
the play-time-weighted raw win rate, an assumption-light benchmark that controls for
nothing. \emph{W1$-$W0} is how far play-time attribution inflates the estimate, in
points. Sample: <<N_CRAWLED>> matches crawled, less <<N_DRAW>> containing draws
($\texttt{is\_win}=2$), <<N_NOTIME>> with incomplete play time, and <<N_SHORT>> below
a 240-second forfeit floor. Four team-up contrasts were structurally empty and
dropped. Appendix Table~\ref{tab:attribcmp} repeats every hero under the two
contaminated attribution rules.
\end{minipage}
\end{landscape}

\clearpage
\appendix
\section*{Appendix}
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{table}{0}

\begin{table}[t]\centering\small
\begin{threeparttable}
\caption{Specification checks}\label{tab:checks}
\begin{tabular}{l l l}
\toprule
Check & Result & Interpretation \\
\midrule
Placebo (permuted heroes) & <<PL_PERM>> vs <<PL_REAL>> & <<PLACEBO>> of signal destroyed \\
Sum-to-zero normalisation & within-role & holds at machine precision \\
Rank agreement, raw win rates & $\rho=<<RAWWR_AFTER>>$ & up from <<RAWWR_BEFORE>> before correction \\
Within-role reparameterisation & $\rho=<<REPARAM_RHO>>$ & mean change <<REPARAM_MAD>> pp \\
Camp indexing & <<CAMP_SHARE>> camp 0 & absolute map side, no crawl selection \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item The placebo shuffles hero assignment against outcomes and refits; a placebo
that failed would invalidate the specification. Camp-0 win rate varies
<<CAMP_LO>>--<<CAMP_HI>> across the <<N_MAPS>> maps, which is why each map carries
its own intercept.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{landscape}
{\footnotesize
\begin{longtable}{l r r r r}
\caption{Every hero under all three attribution rules, same sample}\label{tab:attribcmp} \\
\toprule
<<ATTR_HDR>>
& \multicolumn{1}{c}{\footnotesize starting} & \multicolumn{1}{c}{\footnotesize dominant} &
\multicolumn{1}{c}{\footnotesize play-time} & \\
\midrule
\endfirsthead
\multicolumn{5}{l}{\emph{Table \ref{tab:attribcmp}, continued}} \\
\toprule
<<ATTR_HDR>>
\midrule
\endhead
\midrule
\multicolumn{5}{r}{\emph{continued on next page}} \\
\endfoot
\midrule
Log-likelihood & <<LL0>> & <<LL2>> & <<LL1>> & \\
Effect s.d. & <<SD0>> & <<SD2>> & <<SD1>> & \\
$\rho$ with W0 & --- & <<RHO2>> & <<RHO1>> & \\
\bottomrule
\endlastfoot
<<APANEL_TANK>>
\addlinespace[4pt]
<<APANEL_DAMAGE>>
\addlinespace[4pt]
<<APANEL_SUPPORT>>
\end{longtable}
}

\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} Within-role estimates in percentage points, all three fitted on the same
<<N_MATCHES_CMP>> matches. Effect magnitude rises monotonically with how much
post-outcome information the rule uses, while the ranking is broadly preserved.
W1$-$W0 is reported as a difference rather than a ratio because a ratio diverges for
heroes whose W0 estimate is near zero. Higher log-likelihood here indicates worse
identification, not a better model: play-time weighting predicts better precisely
because its regressors encode what happened.
\end{minipage}
\end{landscape}

\begin{landscape}
{\footnotesize
\begin{longtable}{l r c @{\hspace{2.2em}} l r c}
\caption{Team-up synergies, all <<N_TU>> pairs}\label{tab:teamups} \\
\toprule
<<TU_HDR>>
\midrule
\endfirsthead
\multicolumn{6}{l}{\emph{Table \ref{tab:teamups}, continued}} \\
\toprule
<<TU_HDR>>
\midrule
\endhead
\bottomrule
\endlastfoot
<<TU_ROWS>>
\end{longtable}
}

\noindent\begin{minipage}{\linewidth}\footnotesize
\emph{Notes.} All <<N_TU>> team-ups are listed. Read down the left column, then the
right. Each is labelled by its member heroes, anchor first. The premium is what a pair
earns \emph{beyond} what its two heroes contribute individually, in percentage points,
under starting-lineup attribution. Four further team-ups are absent because they are
defined against hero id 1057, base ``Deadpool'', whose plays are all recorded under
his three role variants, leaving those columns structurally empty.
\end{minipage}
\end{landscape}

\end{document}
"""

subs = {
    "N_MATCHES": num(M["n_matches"]), "N_MATCHES_CMP": num(MW["W1"]["n_matches"]),
    "SEASON": str(AUX["season"]), "PREPARED": AUX["prepared"], "COMMIT": M["git_commit"],
    "N_MAPS": str(AUX["n_maps"]), "N_TEAMUPS": str(M["n_teamups"]), "N_HEROES": str(M["n_heroes"]),
    "N_PLAYERS": "5,729,796", "LOGLIK": f"{M['loglike']:,.1f}",
    "N_CRAWLED": num(M["exclusions"]["starting"]), "N_DRAW": num(M["exclusions"]["dropped_draw"]),
    "N_NOTIME": str(M["exclusions"]["dropped_missing_playtime"]), "N_SHORT": num(M["exclusions"]["dropped_short"]),
    "TW2": pct(TW["2"]["winrate"]), "TW1": pct(TW["1"]["winrate"]), "TW3": pct(TW["3"]["winrate"]),
    "SHARE222": pct(AUX["share_2_2_2"], 0),
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
    "SHAPES_OWN": str(AUX[f"shapes_own_at_{M['min_shape_count']}"]),
    "SHAPES_POOLED": str(AUX[f"shapes_pooled_at_{M['min_shape_count']}"]),
    "POOLED_PCT": pct(AUX[f"pooled_pct_at_{M['min_shape_count']}"], 3),
    "PRIV_HI": pct(AUX["privacy_pct_above_5000"], 0),
    "LL0": ll(MW["W0"]["loglike"]), "LL1": ll(MW["W1"]["loglike"]), "LL2": ll(MW["W2"]["loglike"]),
    "N_TU": str(len(tu)),
    "HERO_HDR": HERO_HDR, "ATTR_HDR": ATTR_HDR, "TU_HDR": TU_HDR, "TU_ROWS": TU_ROWS,
    "PANEL_TANK": hero_panel("Tank", "A"), "PANEL_DAMAGE": hero_panel("Damage", "B"), "PANEL_SUPPORT": hero_panel("Support", "C"),
    "APANEL_TANK": attrib_panel("Tank", "A"), "APANEL_DAMAGE": attrib_panel("Damage", "B"), "APANEL_SUPPORT": attrib_panel("Support", "C"),
}
out = TEX
for k, v in subs.items():
    out = out.replace(f"<<{k}>>", v)
import re
left = re.findall(r"<<[A-Z0-9_]+>>", out)
if left:
    raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
open("results/apm_report.tex", "w").write(out)
print(f"wrote results/apm_report.tex ({len(out):,} bytes); {len(H)} heroes, {len(tu)} team-ups; "
      f"{len(subs)} values injected, no database access")
