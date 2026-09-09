"""Generate the APM results report as journal-format LaTeX.

Wide tables run in landscape (AER style) so that every row is printed -- no
elided entries -- and the extra width carries more columns rather than fewer
rows. Team-ups are labelled by their hero pair, since "SQUIRREL MISSILE" tells
a reader nothing that "Squirrel Girl x The Punisher" does not.
"""
import json
import sqlite3

import pandas as pd

TAG = "W0_specAplus"
H = pd.read_csv(f"results/apm_hero_table_{TAG}.csv")
W1 = pd.read_csv("results/apm_hero_table_W1_full.csv").set_index("hero_id")
W2 = pd.read_csv("results/apm_hero_table_W2_full.csv").set_index("hero_id")
TU = pd.read_csv(f"results/apm_teamup_table_{TAG}.csv")
M = json.load(open(f"results/apm_run_meta_{TAG}.json"))
MW1 = json.load(open("results/apm_run_meta_W1_full.json"))

conn = sqlite3.connect("file:data/rivals.db?mode=ro", uri=True, timeout=300)
NAMES = dict(conn.execute("SELECT hero_id, name FROM hero_info"))
PAIRS = {}
for tid, hid, anc in conn.execute(
        "SELECT teamup_id, hero_id, is_anchor FROM teamup_heroes ORDER BY teamup_id, is_anchor DESC"):
    PAIRS.setdefault(tid, []).append(NAMES.get(hid, f"hero {hid}"))


def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s.replace('"', "''")


def stars(q):
    return r"\sym{***}" if q < .01 else r"\sym{**}" if q < .05 else r"\sym{*}" if q < .10 else ""


H = H.copy()
H["se_pp"] = H["within_role_se"] * 25.0
H["w1_pp"] = H["hero_id"].map(W1["within_role_pp"])
H["w2_pp"] = H["hero_id"].map(W2["within_role_pp"])
N_NS = int((~H["within_role_significant"].astype(bool)).sum())


def hero_panel(role, letter, ncols=5):
    d = H[H.role == role].sort_values("within_role_pp", ascending=False)
    out = [rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{Panel {letter}: {role}}} \ ({len(d)} heroes)}} \\[2pt]"]
    for _, r in d.iterrows():
        out.append(
            f"{esc(r['name'])} & {r.within_role_pp:.2f}{stars(r.within_role_p_adjusted)} & "
            f"({r.se_pp:.3f}) & [{r.within_role_ci_low_pp:.2f}, {r.within_role_ci_high_pp:.2f}] & "
            f"{r.within_role_p_adjusted:.3f} \\\\")
    return "\n".join(out)


def attrib_panel(role, letter):
    """Appendix: the same heroes under all three attribution rules."""
    d = H[H.role == role].sort_values("within_role_pp", ascending=False)
    out = [rf"\multicolumn{{5}}{{l}}{{\emph{{Panel {letter}: {role}}}}} \\[2pt]"]
    for _, r in d.iterrows():
        out.append(f"{esc(r['name'])} & {r.within_role_pp:.2f} & {r.w2_pp:.2f} & "
                   f"{r.w1_pp:.2f} & {r.w1_pp - r.within_role_pp:+.2f} \\\\")
    return "\n".join(out)


# Team-ups: every one of the 100, labelled by hero pair, in two side-by-side
# blocks so the full list fits a single landscape page.
tu = TU.sort_values("effect_pp", ascending=False).reset_index(drop=True)
tu["label"] = tu["teamup_id"].map(
    lambda t: " $\\times$ ".join(esc(x) for x in PAIRS.get(t, ["?", "?"])))
half = (len(tu) + 1) // 2
tu_rows = []
for i in range(half):
    left = tu.iloc[i]
    cells = [f"{left.label} & {left.effect_pp:.2f} & ({left.std_error*25:.3f})"]
    j = i + half
    if j < len(tu):
        right = tu.iloc[j]
        cells.append(f"{right.label} & {right.effect_pp:.2f} & ({right.std_error*25:.3f})")
    else:
        cells.append(" & & ")
    tu_rows.append(" & ".join(cells) + r" \\")
TU_ROWS = "\n".join(tu_rows)

tanks = [(0, "1,990", 18.49), (1, "89,534", 39.44), (2, "533,702", 52.30),
         (3, "24,456", 41.28), (4, "288", 34.72)]
TANK_ROWS = "\n".join(
    (rf"\textbf{{{n}}} & \textbf{{{c}}} & \textbf{{{w:.2f}}} & \textbf{{---}} \\"
     if n == 2 else f"{n} & {c} & {w:.2f} & {w-52.30:+.2f} \\\\")
    for n, c, w in tanks)

tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,siunitx,threeparttable,caption,amsmath,longtable,microtype}}
\usepackage{{pdflscape}}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{{hyperref}}
\newcommand{{\sym}}[1]{{\rlap{{$^{{#1}}$}}}}
\captionsetup{{font=small,labelfont=bf,skip=6pt}}
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{document}}

\begin{{center}}
{{\Large\bfseries Hero Adjusted Plus--Minus in \emph{{Marvel Rivals}}}}\\[4pt]
{{\large Intention-to-treat estimates from {M['n_matches']:,} competitive matches}}\\[6pt]
{{\small Season 19, PC ranked. Prepared 8 September 2026. Commit \texttt{{{M['git_commit']}}}.}}
\end{{center}}

\vspace{{4pt}}
\noindent\textbf{{Specification.}} One observation per match; the outcome is victory for
side~0. Each hero enters as a signed contrast (own-side weight less opposing-side
weight) under \emph{{starting-lineup}} attribution (W0), reduced onto a basis that
sums to zero \emph{{within each role}}, so every hero receives both a coefficient and
a standard error. Controls: a separate intercept per map, the pre-match rank-score
differential, composition-shape indicators, and {M['n_teamups']} team-up contrasts.
Estimated by unpenalised logit with heteroskedasticity-robust (HC1) standard errors.

\vspace{{6pt}}
\noindent\textbf{{The estimand is intention-to-treat.}} W0 is the hero each player
\emph{{started}} on, recovered from entry~1 of the API hero array, which is
chronological by first appearance. It is the only attribution fixed before any
outcome information exists. A player who starts on a hero and abandons it after
thirty seconds still carries full weight, so these effects are diluted by
non-compliance --- but how often a hero gets abandoned is part of its value, not an
error to remove.

\vspace{{6pt}}
\noindent\textbf{{Why not play-time weighting.}} Appendix Table~\ref{{tab:attribcmp}} shows effect
magnitude rising monotonically with how much post-outcome information the
attribution uses. The best-\emph{{fitting}} rule is the most contaminated: play-time
weighting predicts better precisely because its regressors encode what happened.

\vspace{{6pt}}
\noindent\textbf{{Why estimates are within role.}} Cross-role comparisons are not a
hero property but a composition property. The composition-shape block can reproduce
the role-count differential \emph{{exactly}}, so under a single global sum-to-zero
constraint the hero and shape blocks are collinear in that direction and are
separated only by the 0.16\% of teams pooled into an ``other'' category. Constraining
$\beta$ to sum to zero within each role makes it orthogonal to every role indicator,
so composition can only land in the shape block, where the full sample identifies it.
Refitting under this constraint leaves the estimates numerically unchanged (Spearman
0.9999, mean absolute difference 0.010~pp), confirming that the earlier post-hoc
projection was already removing exactly the ill-conditioned direction.



\begin{{table}}[t]\centering\small
\begin{{threeparttable}}
\caption{{Raw win rate by number of Tanks fielded}}\label{{tab:comp}}
\begin{{tabular}}{{c r r r}}
\toprule
Tanks & \multicolumn{{1}}{{c}}{{Teams}} & \multicolumn{{1}}{{c}}{{Win rate (\%)}} &
\multicolumn{{1}}{{c}}{{vs.\ two tanks}} \\
\midrule
{TANK_ROWS}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Composition from each player's play-time-dominant hero. The relationship is an
inverted~U peaking at the 2--2--2 standard that 76\% of teams field. A term linear in
tank counts cannot represent this shape; because teams fielding too few Tanks
outnumber those fielding too many by roughly four to one, a fitted slope is dragged
positive. This is the artifact the within-role constraint removes.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\begin{{table}}[t]\centering\small
\begin{{threeparttable}}
\caption{{Specification checks}}\label{{tab:checks}}
\begin{{tabular}}{{l l l}}
\toprule
Check & Result & Interpretation \\
\midrule
Placebo (permuted heroes) & $0.023$ vs $0.881$ & 97.4\% of signal destroyed \\
Sum-to-zero normalisation & $-1.1\times10^{{-16}}$ & holds at machine precision \\
Rank agreement, raw win rates & $\rho=0.949$ & up from $0.703$ before correction \\
Within-role reparameterisation & $\rho=0.9999$ & estimates numerically unchanged \\
Camp indexing & 49.87\% camp 0 & absolute map side, no crawl selection \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item The placebo shuffles hero assignment against outcomes and refits; a placebo
that failed would invalidate the specification. It was run under play-time
attribution on the same pipeline. Camp-0 win rate varies 49.42\%--52.52\% across the
16 maps, which is why each map carries its own intercept.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

{{\footnotesize
\begin{{longtable}}{{l r c c r}}
\caption{{Hero adjusted plus--minus, relative to an average hero of the same role}}
\label{{tab:main}} \\
\toprule
& \multicolumn{{1}}{{c}}{{APM}} & \multicolumn{{1}}{{c}}{{Std.\ error}} &
\multicolumn{{1}}{{c}}{{95\% CI}} & \multicolumn{{1}}{{c}}{{$q$}} \\
\midrule
\endfirsthead
\multicolumn{{5}}{{l}}{{\emph{{Table \ref{{tab:main}}, continued}}}} \\
\toprule
& \multicolumn{{1}}{{c}}{{APM}} & \multicolumn{{1}}{{c}}{{Std.\ error}} &
\multicolumn{{1}}{{c}}{{95\% CI}} & \multicolumn{{1}}{{c}}{{$q$}} \\
\midrule
\endhead
\midrule
\multicolumn{{5}}{{r}}{{\emph{{continued on next page}}}} \\
\endfoot
\midrule
Matches & \multicolumn{{4}}{{l}}{{{M['n_matches']:,}}} \\
Player--matches & \multicolumn{{4}}{{l}}{{5,729,796}} \\
Heroes & \multicolumn{{4}}{{l}}{{{M['n_heroes']}}} \\
Log-likelihood & \multicolumn{{4}}{{l}}{{{M['loglike']:,.1f}}} \\
\bottomrule
\endlastfoot
{hero_panel('Tank','A')}
\addlinespace[4pt]
{hero_panel('Damage','B')}
\addlinespace[4pt]
{hero_panel('Support','C')}
\end{{longtable}}
}}

\noindent\begin{{minipage}}{{\linewidth}}\footnotesize
\emph{{Notes.}} All 55 heroes are listed. Units are percentage points of win
probability at a balanced match ($\partial p/\partial x=\beta/4$): the change from
fielding this hero at match start in place of an average hero of the same role.
Standard errors in parentheses are HC1. Stars and $q$ denote Benjamini--Hochberg
false-discovery-rate control across the 55-hero family: \sym{{*}}~$q<0.10$,
\sym{{**}}~$q<0.05$, \sym{{***}}~$q<0.01$; {N_NS} of the 55 are not distinguishable from
their role average. Sample: {M['exclusions']['starting']:,} matches crawled, less
{M['exclusions']['dropped_draw']:,} containing draws ($\texttt{{is\_win}}=2$),
{M['exclusions']['dropped_missing_playtime']} with incomplete play time, and
{M['exclusions']['dropped_short']:,} below a 240-second forfeit floor. Four team-up
contrasts were structurally empty and dropped. Appendix Table~\ref{{tab:attribcmp}}
repeats every hero under the two contaminated attribution rules.
\end{{minipage}}

\clearpage
\appendix
\section*{{Appendix}}
\renewcommand{{\thetable}}{{A\arabic{{table}}}}
\setcounter{{table}}{{0}}

\begin{{landscape}}
{{\footnotesize
\begin{{longtable}}{{l r r r r}}
\caption{{Every hero under all three attribution rules, same sample}}\label{{tab:attribcmp}} \\
\toprule
& \multicolumn{{1}}{{c}}{{W0}} & \multicolumn{{1}}{{c}}{{W2}} &
\multicolumn{{1}}{{c}}{{W1}} & \multicolumn{{1}}{{c}}{{W1$-$W0}} \\
& \multicolumn{{1}}{{c}}{{\footnotesize starting}} & \multicolumn{{1}}{{c}}{{\footnotesize dominant}} &
\multicolumn{{1}}{{c}}{{\footnotesize play-time}} & \\
\midrule
\endfirsthead
\multicolumn{{5}}{{l}}{{\emph{{Table \ref{{tab:attribcmp}}, continued}}}} \\
\toprule
& \multicolumn{{1}}{{c}}{{W0}} & \multicolumn{{1}}{{c}}{{W2}} &
\multicolumn{{1}}{{c}}{{W1}} & \multicolumn{{1}}{{c}}{{W1$-$W0}} \\
\midrule
\endhead
\midrule
\multicolumn{{5}}{{r}}{{\emph{{continued on next page}}}} \\
\endfoot
\midrule
Log-likelihood & $-320{{,}}068$ & $-305{{,}}869$ & $-297{{,}}648$ & \\
Effect s.d. & 2.60 & 4.27 & 6.38 & \\
$\rho$ with W0 & --- & 0.937 & 0.891 & \\
\bottomrule
\endlastfoot
{attrib_panel('Tank','A')}
\addlinespace[4pt]
{attrib_panel('Damage','B')}
\addlinespace[4pt]
{attrib_panel('Support','C')}
\end{{longtable}}
}}

\noindent\begin{{minipage}}{{\linewidth}}\footnotesize
\emph{{Notes.}} Within-role estimates in percentage points, all three fitted on the same
{MW1['n_matches']:,} matches. Effect magnitude rises monotonically with how much
post-outcome information the rule uses, while the ranking is broadly preserved.
W1$-$W0 is reported as a difference rather than a ratio because a ratio diverges for
heroes whose W0 estimate is near zero. Higher log-likelihood here indicates worse
identification, not a better model: play-time weighting predicts better precisely
because its regressors encode what happened.
\end{{minipage}}
\end{{landscape}}

\begin{{landscape}}
{{\footnotesize
\begin{{longtable}}{{l r c @{{\hspace{{2.2em}}}} l r c}}
\caption{{Team-up synergies, all {len(tu)} pairs}}\label{{tab:teamups}} \\
\toprule
Team-up & \multicolumn{{1}}{{c}}{{Premium}} & \multicolumn{{1}}{{c}}{{Std.\ err.}} &
Team-up & \multicolumn{{1}}{{c}}{{Premium}} & \multicolumn{{1}}{{c}}{{Std.\ err.}} \\
\midrule
\endfirsthead
\multicolumn{{6}}{{l}}{{\emph{{Table \ref{{tab:teamups}}, continued}}}} \\
\toprule
Team-up & \multicolumn{{1}}{{c}}{{Premium}} & \multicolumn{{1}}{{c}}{{Std.\ err.}} &
Team-up & \multicolumn{{1}}{{c}}{{Premium}} & \multicolumn{{1}}{{c}}{{Std.\ err.}} \\
\midrule
\endhead
\bottomrule
\endlastfoot
{TU_ROWS}
\end{{longtable}}
}}

\noindent\begin{{minipage}}{{\linewidth}}\footnotesize
\emph{{Notes.}} All {len(tu)} team-ups are listed. Read down the left column, then the
right. Each is labelled by its member heroes, anchor first. The premium is what a pair
earns \emph{{beyond}} what its two heroes contribute individually, in percentage points,
under W0 attribution. Four further team-ups are absent because they are defined against
hero id 1057, base ``Deadpool'', whose plays are all recorded under his three role
variants, leaving those columns structurally empty.
\end{{minipage}}
\end{{landscape}}

\clearpage
\noindent\textbf{{Limitations.}}
\begin{{enumerate}}\setlength{{\itemsep}}{{2pt}}
\item \emph{{Magnitudes depend on the attribution rule; the ranking is far more
stable.}} Across W0, W2 and W1 on an identical sample the within-role standard
deviation runs 2.60, 4.27 and 6.38 points while rank agreement with W0 holds at
$\rho=0.937$ and $0.891$. W0 is reported because it is the only rule fixed before any
outcome information exists, but it is intention-to-treat and diluted by mid-match
abandonment. Read the ordering as the firmer result.
\item \emph{{These are adjusted associations, not causal effects.}} Hero choice is not
random. Pre-match rank score controls for \emph{{general}} player skill but not for
skill \emph{{on that specific hero}}, and a one-trick is not a random player assigned
that hero. Per-player hero playtime would measure this confounder directly; it was
costed and declined, so it is bounded rather than eliminated.
\item \emph{{Intervals are analytic, not bootstrapped.}} Matches are not independent
draws --- each contains twelve players and so belongs to twelve overlapping player
clusters. The specified cluster bootstrap would need roughly 656 hours at this scale
under the current implementation, so HC1 intervals are reported instead. They ignore
cross-classified dependence and are therefore \emph{{too narrow}}; point estimates are
unaffected.
\item \emph{{Composition control is coarse in the tail.}} Twelve shapes carry their own
indicator and the remainder share one, covering 0.16\% of team-instances whose win
rates span 0.0--53.8\%.
\item \emph{{The sample is rank-selected.}} Profile privacy rises from 0\% below
4{{,}}000 rank score to 70\% above 5{{,}}000, and the crawl can only expand through
public players, so high-elo matches are structurally under-sampled.
\end{{enumerate}}

\end{{document}}
"""
open("results/apm_report.tex", "w").write(tex)
print(f"wrote results/apm_report.tex ({len(tex):,} bytes); {len(H)} heroes, {len(tu)} team-ups, none omitted")
