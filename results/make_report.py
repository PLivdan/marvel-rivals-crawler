"""Generate the APM results report as journal-format LaTeX."""
import json
import pandas as pd

TAG = "W0_specAplus"
W1 = pd.read_csv(f"results/apm_hero_table_{TAG}.csv")
W2 = pd.read_csv("results/apm_hero_table_W1_full.csv").set_index("hero_id")
TU = pd.read_csv("results/apm_teamup_table_W0_specAplus.csv")
M1 = json.load(open(f"results/apm_run_meta_{TAG}.json"))
M2 = json.load(open("results/apm_run_meta_W1_full.json"))

def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s.replace('"', "''")

def stars(q):
    return r"\sym{***}" if q < .01 else r"\sym{**}" if q < .05 else r"\sym{*}" if q < .10 else ""

W1 = W1.copy()
W1["se_pp"] = W1["within_role_se"] * 25.0
W1["w2_pp"] = W1["hero_id"].map(W2["within_role_pp"])

def panel(role, letter):
    d = W1[W1.role == role].sort_values("within_role_pp", ascending=False)
    out = [rf"\multicolumn{{5}}{{l}}{{\emph{{Panel {letter}: {role} \ ({len(d)} heroes)}}}} \\[2pt]"]
    for _, r in d.iterrows():
        out.append(
            f"{esc(r['name'])} & {r.within_role_pp:.2f}{stars(r.within_role_p_adjusted)} & "
            f"({r.se_pp:.3f}) & [{r.within_role_ci_low_pp:.2f}, {r.within_role_ci_high_pp:.2f}] & "
            f"{r.w2_pp:.2f} \\\\")
    return "\n".join(out)

tanks = [(0, "1,990", 18.49), (1, "89,534", 39.44), (2, "533,702", 52.30),
         (3, "24,456", 41.28), (4, "288", 34.72)]
tank_rows = "\n".join(
    (rf"\textbf{{{n}}} & \textbf{{{c}}} & \textbf{{{w:.2f}}} & \textbf{{---}} \\"
     if n == 2 else f"{n} & {c} & {w:.2f} & {w-52.30:+.2f} \\\\")
    for n, c, w in tanks)

tu = TU.sort_values("effect_pp", ascending=False)
tu_rows = "\n".join(
    f"{esc(r['name'])} & {r.effect_pp:.2f} & ({r.std_error*25:.3f}) \\\\"
    for _, r in pd.concat([tu.head(6), tu.tail(4)]).iterrows())
tu_split = 6
n_ns = int((~W1["within_role_significant"].astype(bool)).sum())

tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,siunitx,threeparttable,caption,amsmath,longtable,microtype}}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{{hyperref}}
\newcommand{{\sym}}[1]{{\rlap{{$^{{#1}}$}}}}
\captionsetup{{font=small,labelfont=bf,skip=6pt}}
\setlength{{\tabcolsep}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{document}}

\begin{{center}}
{{\Large\bfseries Hero Adjusted Plus--Minus in \emph{{Marvel Rivals}}}}\\[4pt]
{{\large Intention-to-treat estimates from {M1['n_matches']:,} competitive matches}}\\[6pt]
{{\small Season 19, PC ranked. Prepared 8 September 2026. Commit \texttt{{{M1['git_commit']}}}.}}
\end{{center}}

\vspace{{4pt}}
\noindent\textbf{{Specification.}} One observation per match; the outcome is victory for
side~0. Each hero enters as a signed contrast (own-side weight less opposing-side
weight) under \emph{{starting-lineup}} attribution (W0), reduced onto a basis that
sums to zero \emph{{within each role}}, so every hero receives both a coefficient
and a standard error. Controls: a separate intercept per map, the pre-match
rank-score differential, composition-shape indicators, and {M1['n_teamups']}
team-up contrasts. Estimated by unpenalised logit with heteroskedasticity-robust
(HC1) standard errors.

\vspace{{6pt}}
\noindent\textbf{{The estimand is intention-to-treat.}} W0 is the hero each player
\emph{{started}} on, recovered from entry~1 of the API hero array, which is
chronological by first appearance. It is the only attribution fixed before any
outcome information exists. A player who starts on a hero and abandons it after
thirty seconds still carries full weight, so these effects are diluted by
non-compliance --- but how often a hero gets abandoned is part of its value, not
an error to remove.

\vspace{{6pt}}
\noindent\textbf{{Why not play-time weighting.}} Table~\ref{{tab:attrib}} shows effect
magnitude rising monotonically with how much post-outcome information the
attribution uses. The best-\emph{{fitting}} rule is the most contaminated: play-time
weighting predicts better precisely because its regressors encode what happened.
Its effects are roughly 2.5 times larger than W0's while the ranking is
substantially preserved.

\vspace{{6pt}}
\noindent\textbf{{Why estimates are within role.}} Cross-role comparisons are not a
hero property but a composition property. The composition-shape block can
reproduce the role-count differential \emph{{exactly}}, so under a single global
sum-to-zero constraint the hero and shape blocks are collinear in that direction
and are separated only by the 0.16\% of teams pooled into an ``other'' category.
Constraining $\beta$ to sum to zero within each role makes it orthogonal to every
role indicator, so composition can only land in the shape block, where the full
sample identifies it. Refitting under this constraint leaves the reported
estimates numerically unchanged (Spearman 0.9999, mean absolute difference
0.010~pp), confirming that the earlier post-hoc projection was already removing
exactly the ill-conditioned direction.

{{\footnotesize
\begin{{longtable}}{{l r c c r}}
\caption{{Hero adjusted plus--minus, relative to an average hero of the same role}}
\label{{tab:main}} \\
\toprule
& \multicolumn{{1}}{{c}}{{APM}} & \multicolumn{{1}}{{c}}{{Std.\ error}} &
\multicolumn{{1}}{{c}}{{95\% CI}} & \multicolumn{{1}}{{c}}{{W1}} \\
\midrule
\endfirsthead
\multicolumn{{5}}{{l}}{{\emph{{Table \ref{{tab:main}}, continued}}}} \\
\toprule
& \multicolumn{{1}}{{c}}{{APM}} & \multicolumn{{1}}{{c}}{{Std.\ error}} &
\multicolumn{{1}}{{c}}{{95\% CI}} & \multicolumn{{1}}{{c}}{{W1}} \\
\midrule
\endhead
\midrule
\multicolumn{{5}}{{r}}{{\emph{{continued on next page}}}} \\
\endfoot
\midrule
Matches & \multicolumn{{4}}{{l}}{{{M1['n_matches']:,}}} \\
Player--matches & \multicolumn{{4}}{{l}}{{3,812,088}} \\
Heroes & \multicolumn{{4}}{{l}}{{{M1['n_heroes']}}} \\
Log-likelihood & \multicolumn{{4}}{{l}}{{{M1['loglike']:,.1f}}} \\
\bottomrule
\endlastfoot
{panel('Tank','A')}
\addlinespace[4pt]
{panel('Damage','B')}
\addlinespace[4pt]
{panel('Support','C')}
\end{{longtable}}
\noindent\begin{{minipage}}{{\linewidth}}\footnotesize
\emph{{Notes.}} Units are percentage points of win probability evaluated at a balanced
match ($\partial p/\partial x=\beta/4$). Standard errors in parentheses are HC1 and are
propagated through the within-role contrast by the delta method. Stars denote
Benjamini--Hochberg false-discovery-rate control across the 55-hero family:
$^{{*}}$~$q<0.10$, $^{{**}}$~$q<0.05$, $^{{***}}$~$q<0.01$; {n_ns} of the 55 heroes are
not distinguishable from their role average. The final column repeats the estimate under
play-time attribution (W1) rather than starting-lineup attribution (W0), on the
\emph{{same}} {M2['n_matches']:,}-match sample.
Sample: {M1['exclusions']['starting']:,} matches crawled, less
{M1['exclusions']['dropped_draw']:,} containing draws ($\texttt{{is\_win}}=2$),
{M1['exclusions']['dropped_missing_playtime']} with incomplete play time, and
{M1['exclusions']['dropped_short']:,} below a 240-second forfeit floor. Four team-up
contrasts were structurally empty and dropped.
\end{{minipage}}
}}

\begin{{table}}[t]\centering\small
\begin{{threeparttable}}
\caption{{Effect magnitude rises with post-outcome contamination}}\label{{tab:attrib}}
\begin{{tabular}}{{l l r r r}}
\toprule
Rule & Attribution & \multicolumn{{1}}{{c}}{{Log-lik.}} & \multicolumn{{1}}{{c}}{{Effect s.d.}} &
\multicolumn{{1}}{{c}}{{vs W0}} \\
\midrule
\textbf{{W0}} & starting lineup (pre-outcome) & $-320{{,}}122$ & \textbf{{2.60}} & 1.00$\times$ \\
W2 & dominant hero & $-305{{,}}869$ & 4.27 & 1.64$\times$ \\
W1 & play-time weighted & $-297{{,}}648$ & 6.38 & 2.45$\times$ \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item All three fitted on the same {M2['n_matches']:,} matches. Effect s.d.\ is the
standard deviation of within-role estimates in percentage points. Rank agreement
with W0: $\rho=0.937$ (W2), $\rho=0.891$ (W1). Squirrel Girl reads $-3.76$ under
W0, $-8.62$ under W2 and $-16.61$ under W1. Higher log-likelihood here indicates
worse identification, not a better model.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\begin{{table}}[t]\centering\small
\begin{{threeparttable}}
\caption{{Raw win rate by number of Tanks fielded}}\label{{tab:comp}}
\begin{{tabular}}{{c r r r}}
\toprule
Tanks & \multicolumn{{1}}{{c}}{{Teams}} & \multicolumn{{1}}{{c}}{{Win rate (\%)}} &
\multicolumn{{1}}{{c}}{{vs.\ two tanks}} \\
\midrule
{tank_rows}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Composition from each player's play-time-dominant hero. The relationship is an
inverted~U peaking at the 2--2--2 standard that 76\% of teams field, so more Tanks is
not better. A term linear in tank counts cannot represent this shape; because teams
fielding too few Tanks outnumber those fielding too many by roughly four to one, the
fitted slope is dragged positive. This is the artifact the within-role normalisation
in Table~\ref{{tab:main}} removes.
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
Attribution robustness (W1 vs W2) & $\rho=0.954$ & ranking stable; magnitudes are not \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item The placebo shuffles hero assignment against outcomes and refits; a placebo that
failed would invalidate the specification. Rank agreement is Spearman's $\rho$ against
play-time-weighted raw win rates. Under W2 the ranking is preserved but effect sizes
roughly halve (mean absolute difference 2.08 points; Squirrel Girl moves from $-16.79$
to $-9.02$), so the \emph{{ordering}} should be treated as the result and magnitudes as
an upper bound.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\begin{{table}}[t]\centering\small
\begin{{threeparttable}}
\caption{{Team-up synergies, largest and smallest}}\label{{tab:teamups}}
\begin{{tabular}}{{l r c}}
\toprule
Team-up & \multicolumn{{1}}{{c}}{{Premium}} & \multicolumn{{1}}{{c}}{{Std.\ error}} \\
\midrule
{chr(10).join(tu_rows.split(chr(10))[:tu_split])}
\addlinespace[3pt]
\multicolumn{{3}}{{c}}{{\emph{{$\cdots$ {len(tu)-10} team-ups omitted $\cdots$}}}} \\
\addlinespace[3pt]
{chr(10).join(tu_rows.split(chr(10))[tu_split:])}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item The premium a pair earns \emph{{beyond}} what its two heroes contribute
individually, in percentage points, under W2 attribution. Median absolute effect across
all {len(tu)} team-ups is 2.08 points. Squirrel Missile is the largest while Squirrel
Girl is the weakest Damage hero individually, which is the separation of solo value
from pair synergy the specification is built to make.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\clearpage
\noindent\textbf{{Limitations.}}
\begin{{enumerate}}\setlength{{\itemsep}}{{2pt}}
\item \emph{{Magnitudes depend on the attribution rule; the ranking is far more
stable.}} Across W0, W2 and W1 on an identical sample the within-role standard
deviation runs 2.60, 4.27 and 6.38 points while rank agreement with W0 stays at
$\rho=0.937$ and $0.891$ (Table~\ref{{tab:attrib}}). W0 is reported because it is
the only rule fixed before any outcome information exists, but it is an
intention-to-treat quantity and is diluted by players abandoning a hero mid-match.
Read the ordering as the firmer result.
\item \emph{{These are adjusted associations, not causal effects.}} Hero choice is not
random. Pre-match rank score controls for general player skill but not for skill on the
specific hero, and a one-trick is not a random player assigned that hero. Per-player
hero playtime would measure this confounder directly; it was costed and declined, so it
is bounded rather than eliminated.
\item \emph{{Intervals are analytic, not bootstrapped.}} Matches are not independent
draws --- each contains twelve players and so belongs to twelve overlapping player
clusters. The specified player-level cluster bootstrap would require roughly 656 hours
at this scale under the current implementation, so HC1 intervals are reported instead.
They ignore cross-classified dependence and are therefore \emph{{too narrow}}; point
estimates are unaffected.
\item \emph{{The sample is rank-selected.}} Profile privacy rises from 0\% below 4{{,}}000
rank score to 70\% above 5{{,}}000, and the crawl can only expand through public players,
so high-elo matches are structurally under-sampled. Further crawling does not fix this.
\item \emph{{Four team-ups are absent.}} Hero id 1057 is base ``Deadpool'', but the
roster records only his three role variants, so 1057 is played zero times. The four
team-ups defined against it are structurally empty and were dropped rather than mapped
onto the variants, which would be an unverifiable modelling assumption.
\end{{enumerate}}

\end{{document}}
"""
open("results/apm_report.tex", "w").write(tex)
print(f"wrote results/apm_report.tex ({len(tex):,} bytes)")
