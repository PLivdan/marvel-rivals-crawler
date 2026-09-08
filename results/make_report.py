"""Generate the APM results report as journal-format LaTeX."""
import json
import pandas as pd

W1 = pd.read_csv("results/apm_hero_table_W1.csv")
W2 = pd.read_csv("results/apm_hero_table_W2.csv").set_index("hero_id")
TU = pd.read_csv("results/apm_teamup_table_W2.csv")
M1 = json.load(open("results/apm_run_meta_W1.json"))
M2 = json.load(open("results/apm_run_meta_W2.json"))

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
{{\large Estimates from {M1['n_matches']:,} competitive matches}}\\[6pt]
{{\small Season 19, PC ranked. Prepared 8 September 2026. Commit \texttt{{{M1['git_commit']}}}.}}
\end{{center}}

\vspace{{4pt}}
\noindent\textbf{{Specification.}} One observation per match; the outcome is victory for
side~0. Each hero enters as a signed contrast (own-side weight less opposing-side
weight) under play-time attribution, reduced onto a sum-to-zero basis so every hero
receives both a coefficient and a standard error. Controls: a side intercept, the
pre-match rank-score differential, composition-shape indicators, and
{M1['n_teamups']} team-up contrasts. Estimated by unpenalised logit with
heteroskedasticity-robust (HC1) standard errors.

\vspace{{6pt}}
\noindent\textbf{{Why estimates are reported within role.}} The raw coefficients are
dominated by a role-composition effect: role alone explains 64.6\% of their variance,
and the twelve highest-ranked heroes are all Tanks. This is structural. A team's tank
count is the \emph{{sum}} of its tank heroes' indicators, so it is perfectly collinear
with the hero block; the composition-shape controls absorb only the categorical part,
and the linear part loads onto the individual hero coefficients. The implied direction
is also wrong (Table~\ref{{tab:comp}}). Subtracting each hero's own-role mean is a linear
contrast whose covariance follows exactly by the delta method, and it restores
agreement with an assumption-light benchmark: rank correlation with play-time-weighted
raw win rates rises from 0.70 to 0.95.

{{\footnotesize
\begin{{longtable}}{{l r c c r}}
\caption{{Hero adjusted plus--minus, relative to an average hero of the same role}}
\label{{tab:main}} \\
\toprule
& \multicolumn{{1}}{{c}}{{APM}} & \multicolumn{{1}}{{c}}{{Std.\ error}} &
\multicolumn{{1}}{{c}}{{95\% CI}} & \multicolumn{{1}}{{c}}{{W2}} \\
\midrule
\endfirsthead
\multicolumn{{5}}{{l}}{{\emph{{Table \ref{{tab:main}}, continued}}}} \\
\toprule
& \multicolumn{{1}}{{c}}{{APM}} & \multicolumn{{1}}{{c}}{{Std.\ error}} &
\multicolumn{{1}}{{c}}{{95\% CI}} & \multicolumn{{1}}{{c}}{{W2}} \\
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
$^{{*}}$~$q<0.10$, $^{{**}}$~$q<0.05$, $^{{***}}$~$q<0.01$; six heroes are not
distinguishable from their role average. The final column repeats the estimate under
dominant-hero attribution (W2) rather than play-time attribution (W1); that run used
{M2['n_matches']:,} matches rather than {M1['n_matches']:,}, because the crawl continued
between runs, so the two columns are not estimated on an identical sample.
Sample: {M1['exclusions']['starting']:,} matches crawled, less
{M1['exclusions']['dropped_draw']:,} containing draws ($\texttt{{is\_win}}=2$),
{M1['exclusions']['dropped_missing_playtime']} with incomplete play time, and
{M1['exclusions']['dropped_short']:,} below a 240-second forfeit floor. Four team-up
contrasts were structurally empty and dropped.
\end{{minipage}}
}}

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
\item \emph{{Magnitudes are attribution-dependent.}} Ordering is stable across W1 and W2
($\rho=0.954$) but effect sizes roughly halve under W2. Play-time attribution does not
\emph{{condition}} on mid-match hero swaps, but it does \emph{{weight}} by them, and a
losing player abandoning a hero reduces that hero's weight --- so it remains exposed to
the outcome indirectly.
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
