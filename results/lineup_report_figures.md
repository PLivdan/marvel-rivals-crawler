# lineup_report.pdf: figure and table inputs, and wording corrections

Build pipeline (no model refits, no retuning, no new collection, no new bootstraps):

```
PYTHONPATH=. python3 results/lineup_fit_scatter.py   # results/lineup_fit_scatter.json (evaluation-slice aggregates; no fitting)
python3 results/lineup_figures.py        # results/figures/lineup_*.tikz (+ standalone .pdf copies)
python3 results/make_lineup_report.py    # results/lineup_report.tex from results/lineup_report_template.tex
cd results && latexmk -pdf lineup_report.tex
```

Figures are pgfplots pictures in the style of Figure A2 of the headline report (axis lines left, no tick marks,
dashed zero line, open/filled circles joined by thin segments, italic group labels, legend inside without a
frame, Computer Modern at the document's sizes; hero names and value labels at 9 pt). Each picture is sized
in two compile passes so that its full extent, labels included, equals the text width. The report `\input`s
the `.tikz` sources; the `.pdf` copies are the same pictures compiled standalone. The build id in the footer
is the SHA-256 of the substituted document text plus the `.tikz` sources.

## Figures

| Figure | File | Saved inputs | Construction |
|---|---|---|---|
| 1. Three ways to credit a swap | TikZ in `lineup_report_template.tex`; `figures/icon-spider-man.png`, `figures/icon-elsa-bloodstone.png` | none (illustrative) | Timeline 2 min Spider-Man, 8 min Elsa Bloodstone; boxes "Starting hero (used in this report, W0)", "Most-played hero (W2)", "Time-weighted (W1)". |
| 2. Replacement values, all 55 heroes | `figures/lineup_replacement.tikz` | `lineup_selected_replacement.csv`: `common_ref_pp`, `coverage`, `role` | One axis, three role sections, sorted within role by value; value printed beside each dot; "Eligible slots (%)" column = 100 x `coverage`. |
| 3. Largest changes between lobby-rank thirds | `figures/lineup_rank_highlight.tikz` | `lineup_selected_replacement.csv`: `obs_meta_pp_third0`, `obs_meta_pp_third2`; `lineup_selected_summary_meta.json`: `thirds` | Selection = four largest increases and four largest decreases of third2 - third0; sorted by change; both endpoints printed. |
| 4. Pair contrasts | `figures/lineup_pairs.tikz` | `lineup_selected_pairs_p50.csv`: `kind`, `did`, `support_dev`, `is_teamup`, `hero_a`, `hero_b` | Pairs with `support_dev` >= 500. Team-ups: 10 largest positive + 5 largest negative `did`; other allied: 8 + 8; opposing: 14 largest by absolute `did`, oriented so the contrast is positive ("A against B"). Label = contrast (identifying matches). |
| 5. Prediction | `figures/lineup_prediction.tikz` | `lineup_tuning.json` (`evaluations` with `sub == 1.0` for the keys in `reduced`: selected, no_pairs, no_pairs_no_slopes_no_heromap), `lineup_baseline_dev.json` (`folds`), `lineup_confirmation.json` (`D`, `ci_iid`, `ci_boot`, `by_third`, `n`) | Left: per-fold validation log loss. Right: paired improvement, all matches with the i.i.d. (thin) and player-resampling (thick) 95% intervals; by-third points without intervals. |
| 6. Predicted against realised win rates on the evaluation slice | `figures/lineup_fit.tikz` | `lineup_fit_scatter.json` (written by `results/lineup_fit_scatter.py` from `lineup_selected.npz`, the cached design rows after the confirmation cut, and the headline predictions saved in `lineup_baseline_dev.npz`; no fitting) | Panel A: one point per starting hero, mean predicted win probability of its team-sides against the share that won; labels = two farthest from the line + two extremes. Panel B: by composition shape with at least 100 team-sides. Stats box: correlation and mean absolute gap for the unified model and the refitted headline. |
| 7 (Appendix). Lower third against upper third, all 55 heroes | `figures/lineup_rank_all.tikz` | as Figure 3 | Remake of Figure A2 of the headline report: within role, sorted by third2 - third0; open = lower third, filled = upper third. |

## Tables

| Table | Saved inputs |
|---|---|
| 1. Doctor Strange benchmarks | `lineup_selected_replacement.csv`: `common_ref_pp`, `obs_meta_pp` |
| 2. Calibration slopes | `lineup_tuning.json` (selected evaluation, `folds[].cal_slope`, `folds[].by_third[].cal_slope`), `lineup_confirmation.json` (`cal_slope`, `by_third[].cal_slope`) |
| 3. Rank coverage | `rank_coverage.json` |
| 4. Penalty evaluations | `lineup_tuning.json`: eight best full-row evaluations by mean validation log loss |
| 5. Development ladder | `lineup_tuning.json` (`reduced`), `lineup_baseline_dev.json`, `lineup_design/summary.json` (`free`, for parameter counts) |
| 6. Hero coefficients and replacement values (landscape) | `lineup_selected_hero_by_rank.csv` (`beta_p50`, `slope`), `lineup_selected_replacement.csv` (`common_ref_pp`, `obs_meta_pp`, `obs_meta_pp_third0`, `obs_meta_pp_third2`, `coverage`). Middle-third and context-count columns are omitted for width; the minimum context count is stated in the notes. |
| 7-9. Pair tables (landscape) | `lineup_selected_pairs_p50.csv`, same support rule as Figure 4; Table 7 lists all 100 team-up-eligible pairs in three columns, Tables 8-9 two blocks of 25. |

Other text inputs: `dev_snapshot/filters.json` (snapshot), `lineup_design/summary.json` (exclusions, aliases,
free coefficients), `lineup_lock.json` (lock hash and time), `lineup_bootstrap_draws.npz` (completed
replication count only), `lineup_published.json` / `lineup_published_replacement.csv` (full-data refit
comparison sentence), `apm_hero_table_W{0,2,1}_full.csv` and `apm_run_meta_W1_full.json` (Appendix D only).

## Wording corrections carried into this revision

- The 19,804-match slice is described as an **internal temporal evaluation**, with the disclosure that it
  overlaps the final week examined in the earlier report; "confirmed results" and "PROMOTE" do not appear.
- The **bootstrap interval** is attached only to the paired log-loss difference. Hero and pair estimates carry
  no uncertainty; the stability bootstrap is reported only as a count of completed replications and no figure
  carries confidence bands or significance marks.
- **The Hood**: the weekly win-rate audit is described as a drift with no identified balance change, not a
  learning curve.
- **Attribution illustration**: plain labels (starting hero, most-played hero, time-weighted); no
  intention-to-treat or attenuation claims. Earlier-specification attribution numbers (dispersions 2.60 /
  4.27 / 6.38, Magik +3.48 / +7.94 / +14.57 on 477,483 matches) appear only in Appendix D, labelled as coming
  from the earlier specification.
- **Pair contrasts** are stated as four-draft log-odds contrasts, not percentage-point contributions of a
  pair to a lineup; orientation "A against B" is defined in the caption and table notes.
- **Averaging rule**: each hero is averaged over its own eligible contexts (legal = not already on that side
  and not banned), so coverage differs across heroes (Black Cat 29% of Damage slots, Wolverine 94%); the
  common-reference and observed-incumbent benchmarks are distinguished with the Doctor Strange example
  (+0.06 against -1.41).
- **Rank-change figure**: the selection rule (four largest increases, four largest decreases) is stated, no
  stars are used, and lines are described as joining two estimates from the same fit rather than as intervals.
- Sample counts stated as 471,008 eligible matches, 451,204 development rows, 19,804 evaluation matches.
- The evaluation slice's outcomes are described as first read for the Figure 5 comparison, with Figure 6 a
  descriptive view of the same slice (the earlier "read once" wording no longer holds).
