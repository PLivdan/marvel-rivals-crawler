# APM follow-up plan: a controlled estimate of hero strength

Companion to *Marvel Rivals: Data, Nuances, and the Current Model* (8 Sep 2026).
Replaces the earlier draft. The goal is one number per hero that is right, with
an interval that is right, and a clear statement of what it does and does not
hold fixed. Nothing here is built for coaching.

────────

## 0. Estimand and identification

**Estimand.** β_h is the change in log-odds of winning from fielding hero *h* at
match start in place of an average same-role hero, holding fixed: the twelve
players' general skill, each player's proficiency on the hero they field, the
rest of both lineups, map, and side. Reported as a win-probability contrast in a
2-2-2 at equal skill and a common proficiency level (§4). Relative to same-role
average because cross-role differences are not separable from composition (§7).

**Ideal experiment.** Same twelve players, same map, same lineups, except one
player starts on *h* instead of *h'*, and is equally proficient on both.

**Confounders and the control for each.**

| confounder | mechanism | control |
|---|---|---|
| swap endogeneity | final / time-weighted hero is selected on the outcome (§2.2–2.4 of the data doc) | W0 initial lineup (§2) |
| player general skill | strong players cluster on some heroes; mean rank score is a coarse proxy | all twelve players' effects `a_i` (§3) |
| player-hero skill | specialists and fill; `E[b_ih | i plays h] > 0` and hero-dependent | proficiency channel (§4) |
| lineup context | own composition, team-ups, enemy lineup | δ, τ, and γ as a robustness block (§3) |
| side, map | | α by map; map identity differences out |
| time / patch | balance changes inside the window | patch-aligned bins, stability test (§5) |
| sample | BFS from leaderboards, rank-selected privacy | rank-band refits; population stated (§7) |

**Identifying assumption under W0.** At hero select the enemy lineup is not
visible (verify: only bans are). The starting hero therefore depends on bans,
teammates' picks, map, and the player's own preferences. Bans and map are
match-level and difference out; teammates' picks are in the composition
controls; preferences are the proficiency confound, handled in §4. What remains
is day-level form on a hero — the player who starts on *h* because it went well
yesterday. That is a within-player-hero, time-varying term and it is the
residual we cannot remove; §5 tests whether it is large.

────────

## 1. Two checks before anything else

**Hero-list chronology.** For player-matches with ≥2 heroes, test whether the
last entry in `match_player_heroes` equals `cur_hero_id`, and whether the list is
sorted by `play_time` or `hero_id`. Match in ~100% of cases and no sort ⇒
chronological, first entry is the starting hero ⇒ W0 exists. If sorted, W3 (§2).

**Camp indexing.** Absolute (map side) or relative to the discovering player?
Share of discovering players in camp 0; camp-0 win rate by map and mode. If
relative, the 2.2-point side advantage is crawl selection and must be corrected
before α means anything.

### Results (run 8 Sep 2026, 482,997-match database)

**Hero-list chronology — W0 EXISTS.** The plan's stated criterion ("last entry
== `cur_hero_id` in ~100% of cases") is the *wrong test for this schema* and
fails: the last entry matches only 74.4% of the time. But the array is
chronological anyway, ordered **by first appearance**. Evidence:

| test | result | reading |
|---|---|---|
| `hero_id` ascending | 46.60% | not sorted by id (chance ≈ 50%) |
| `play_time` non-increasing | 58.24% | not sorted by playtime |
| `argmax(play_time) == cur` | tracks 1/k (50.9 / 37.0 / 29.9 / 24.5 / 18.4 vs 50 / 33 / 25 / 20 / 17) | `cur_hero_id` is **not** the longest-played hero |
| `last == cur` by k=2..6 | 73.6 / 74.3 / 75.9 / 76.5 / 77.8% — **flat** | a real ordering rule; chance would collapse with 1/k |
| `first == cur` by k=2..6 | 25.4 / 13.2 / 8.4 / 6.7 / 4.8% — declining | more swaps ⇒ less likely to end where you started |
| when `last != cur`, is `cur` in the array? | **97.08%** | yes — it was used earlier |
| position of `cur` among those | position 1 in ~63% | **swap-back**: A → B → A |

The mechanism: the primary key is `(match_uid, player_uid, hero_id)`, so a
player who returns to an earlier hero gets **no second row** — that row already
exists and holds their *total* time on it. The last array entry is therefore the
last *newly-introduced* hero, which diverges from `cur_hero_id` precisely when
someone swaps back. That is the entire 25% gap.

**Consequence: entry 1 (by `rowid`) is the starting hero, and W0 is
constructible.** `match_player_heroes` is not `WITHOUT ROWID`, so its implicit
rowid preserves API array order; `ROW_NUMBER() OVER (PARTITION BY match_uid,
player_uid ORDER BY rowid)` recovers it.

Two caveats for the W0 builder:
- Do **not** filter `play_time > 0` when locating entry 1. That filter is why
  `cur_hero_id` was missing from the array in the residual 2.92% of cases — a
  hero swapped to in the final seconds carries ~0 recorded time.
- The corrected acceptance criterion is "`cur_hero_id` is recoverable from the
  array", not "`cur_hero_id` is last".

**Camp indexing — ABSOLUTE (map side), no crawl selection.** Crawled (`done`)
players sit 49.87% in camp 0, so camp is not assigned relative to the
discovering player. Camp-0 win rate varies across the 16 maps from 49.42% to
52.52% (3.10 pp spread), with two maps favouring camp 1. The ~2.2-point side
advantage is therefore **real and map-varying**, `alpha_map(m)` is justified, and
Spec A's single intercept is mildly misspecified in pooling that spread.

────────

────────

## 2. Attribution

Primary: **W0**, initial lineup. The estimand is then "starting on *h*", an
intention-to-treat quantity that includes whatever swapping follows. That is a
property of the hero (how often it gets abandoned is part of its value) and it is
the only attribution decided before outcome information exists.

Fallback: **W3**, expected lineup — the player's pre-match modal hero from
lifetime shares where fetched (§4), leave-one-out in-window shares elsewhere.

Sensitivity: W1, W2, and `w_ih ∝ t_ih^kappa`, κ ∈ [1, ∞), which runs from W1 to
W2. Plot each hero's rank against κ; heroes whose rank moves are the ones whose
raw estimate depends on swap attribution.

Effect sizes are compared only as standardised contrasts, never as raw
coefficients across attribution rules; the W1→W2 halving is largely regressor
scale.

────────

## 3. Specification C

Match-level logit, one row per match, twelve player effects per row.

```
logit P(camp0 wins) =
    alpha_map(m)
  + sum_{i in camp0} a_i  -  sum_{i in camp1} a_i      player effects, ridge
  + sum_h      beta_h  x_h                             hero, within-role sum-to-zero
  + sum_h      phi_h   q_h                             proficiency channel (§4)
  + sum_s      delta_s c_s                             shape, min count 100
  + sum_t      tau_t   z_t                             team-ups
  + theta' skill_diff                                  prior mean for a
  [ + sum_{h<k} gamma_hk M_hk ]                        matchups, robustness block
```

**Why twelve player effects rather than one-per-row Spec B.** A single `a_i` per
row absorbs one twelfth of the heterogeneity; the other eleven players' skill
stays in the error, correlated with lineup. Spec B as drafted also dropped
teammates' and opponents' heroes, so its A-vs-B gap compared different
estimands. C keeps A's regressors exactly and adds the player block.

**Estimation.** Sparse design, ~885k columns, ~80 nonzeros per row. Ridge on `a`
only, λ by temporal CV on a five-point grid; β, δ, τ, θ, φ unpenalised (φ shrunk
toward a common slope). IRLS with sparse CG or L-BFGS; LPM by sparse least
squares for the λ grid, logit once at the end. One-match players shrink to zero.
Confirm the player graph is one component.

**Constraints, as contrast reparameterisations.**

- β: sum-to-zero **within each role** — three constraints. Role-count
  differentials are linear combinations of both the hero columns and the shape
  dummies; with one constraint that direction is identified only off the 0.16%
  "other" bucket and the W1/W2 discrepancy, which is why twelve tanks topped the
  raw table. Three constraints put all of composition into δ, estimated from the
  full sample, and make within-role reporting structural rather than a
  post-hoc projection.
- γ (when included): `sum_k gamma_hk = 0` for every *h*. Because
  `sum_k M_hk = 6 x_h` the matchup block nests the main effects; without the
  constraint the β/γ split is set by λ. With it, β is the transitive (Hodge
  gradient) part and γ the cyclic part. Low-rank form, if used:
  `gamma_hk = u_h'v_k − u_k'v_h`.
- δ: sum-to-zero across shapes.

────────

### Result of the §3 feasibility spike (8 Sep 2026)

One fit at λ=1 on 477,483 matches with 879,317 player effects, L-BFGS.

| | |
|---|---|
| design build | 657 s |
| optimisation | 1,068 s, **did not converge** (2,000-iteration ceiling) |
| log-likelihood | −320,068 (A+) → −142,970 — overfitting at this λ, not fit |
| hero effects | sd 2.60 → 4.06, Spearman 0.989 with A+ |

The uniform inflation is the logit-scale artifact — adding variance-absorbing
regressors rescales every coefficient — not confounding removal. **Dividing C by
the sd ratio (1.558) leaves a mean absolute shift of 0.26 pp; 48 of 55 heroes are
within ±0.5 pp of A+.** Only Deadpool (Duelist) moves materially (−0.67 → −2.31).

**Reading.** The "+a" step of the §5 A→C decomposition is small: general player
skill beyond the rank-score differential is not a first-order confounder here.
This is what §4 predicted — `a_i` cannot touch hero-specific skill `b_ih`, and
that is the confounder that would matter. With the §4 proficiency fetch declined,
Spec C's marginal value is lower than this plan assumed: it is a robustness
table confirming A+, not a correction to it.

**If C is built anyway**, three things the spike settles: λ=1 is far too weak
and must come from the temporal CV the plan specifies; L-BFGS needs either the
LPM-for-the-λ-grid route the plan already names or ~5× more iterations
(≈ 90 min per converged fit); and A-vs-C must be compared on average partial
effects or standardised contrasts, never on raw β, or the scale artifact will be
read as a finding.

**Recommended reprioritisation.** The report's most-cited limitation is that the
intervals ignore cross-classified player dependence. Making the bootstrap
feasible (§6: build the design once, row-replicate) fixes a stated weakness of
the headline table. Spec C confirms something we now already know. Bootstrap
first; C as a later appendix.

────────

## 4. The specialist control

**The problem, exactly.** In Spec A, β_h = hero value + `E[b_ih | i plays h]`,
where `b_ih` is player *i*'s skill on *h* beyond their general level. Players
play what they are best at, so the term is positive for every hero and larger
for heroes with narrow specialist playerbases. The mirror image is fill: flex
players pushed onto support carry negative `b_ih`, biasing support effects down.

`a_i` does not fix this: a one-trick's `a_i` and their contribution to `x_h`
are collinear across their matches, and the ridge splits the difference.
Player-hero effects cannot be estimated alongside β_h: β_h is their mean, and
the mean is precisely what selection contaminates.

**The control.** Measure proficiency pre-match and hold it fixed. Exposure is
the measure because it carries no outcome information:

```
e1 = log(1 + lifetime minutes on h before the window)   API heroes_ranked, less in-window observed
e2 = lifetime share of h in i's play                     API
e3 = lagged in-window minutes on h, prior matches only   DB, divided by observed matches
```

`prof_ih` is e1 (or the first PC of e1–e3 if all three are available), measured
on a **common scale across heroes** and centred at a **common reference** — the
pooled median of minutes-on-fielded-hero. Centring at hero-specific medians would
compare heroes at different proficiencies and reintroduce the confound through
the reference level. Enter as

```
q_h = sum_{i in camp0} w_ih * prof_ih  -  sum_{i in camp1} w_ih * prof_ih
```

β_h is then "*h* at the reference proficiency"; φ_h is the hero's slope in
proficiency. Report β_h at two common levels (e.g. 5 h and 50 h) so the
interaction is visible; the reference-level number is the headline.

**Identification of φ.** φ_h comes from cross-player proficiency differences,
usable because `a` is shrunk; under unpenalised player FE it would be identified
only from within-window growth in e3. Mechanically, the model now has a channel
to attribute a specialist's outperformance to specialisation rather than to the
hero, and fill underperformance to low proficiency rather than to the role.

**Fetch, prioritised.** Rank players by player-rows in the fit sample, fetch
until 85–90% coverage — tens of thousands of calls, not 883k. Unfetched players:
e3 plus a missing indicator. Include account age / total matches so a new
account's lifetime ≈ window is not read as specialisation.

**Panel-only measures.** For non-crawled players, "prior observed minutes" is
intersections with a rank-selected crawl. Fallback, not primary.

A performance-based proficiency (shrunk running residual on *h* from a fit
without φ) is a robustness variant only; its residuals come from a full-sample
fit.

────────

## 5. Specification tests

**Ban test of the specialist control.** A ban of *h* changes only the team with
an *h*-main, so it is a within-player shifter of the fielded hero that is decided
before the match. Define main `M_i` (pre-window share ≥ 0.4),
`D_im = 1[M_i banned]`, and add

```
sum_{i in camp0} psi_{M_i} D_im  -  sum_{i in camp1} psi_{M_i} D_im
```

to C with `x`, `q`, and `a` present. If proficiency is correctly modelled,
ψ_h ≈ 0: the loss from forcing an *h*-main off *h* is fully accounted for by the
hero substitution and the proficiency drop. ψ_h < 0 is residual specialist
confounding in *h* that e1–e3 miss. This is the falsification test for §4.

**A→C decomposition.** For each hero, β under A, then +`a`, then +`q`, then +γ.
The successive changes attribute each hero's raw reputation to its players'
skill, to specialisation, and to matchup structure. Heroes whose estimate moves
most are the ones Spec A got most wrong.

**Stability.** Patch-aligned time bins from the S19 notes; β_h per bin; joint
test of equality. If it fails, the pooled number is a window average and is
reported as such. Split-half by time, compared to √2·SE, also calibrates the
intervals.

**Heterogeneity.** Hero × map deviations, shrunk, with a joint test. Rank-band
refit at ≥4,750 as a separate table, not a pooled interaction — that band is
under-sampled by the crawl and may have a different meta.

**Attribution sensitivity.** Rank-vs-κ plot from §2; W0 vs W3 vs W1 vs W2 table.

**Additivity.** C with and without the γ block. Under the divergence constraint
β should barely move; if it does, the additive model was absorbing cyclic
matchup structure into main effects.

────────

## 6. Inference and reporting

Design built once, row-replicated. Match-level bootstrap with `a` re-estimated
per draw; player heterogeneity is now in `a` rather than the error, so this is
far closer to valid than HC1 was. Report per hero: β_h, 95% interval, median
rank, 80% rank interval, P(top-3 in role).

Two point estimates: the unpenalised β_h with its interval, and the empirical
Bayes posterior mean under a normal prior across heroes. The first is right for
any one hero; the second is right for the table as a whole, and the gap between
them is the winner's-curse correction at the top of the ranking. BH at q=0.05
stays for the 55 "different from role average" tests.

Standardised contrast: swap an average same-role hero for *h* in a 2-2-2 at
equal skill and reference proficiency; report the win-probability change.

────────

## 7. What the model does not identify

- **Cross-role comparisons.** "Is Mantis better than Magik" is not a hero
  property; it is a composition property, and it lives in δ.
- **Anything beyond starting on *h*.** The estimand is intention-to-treat; the
  value of swapping to *h* mid-match is a different quantity and is not
  estimable from this data.
- **Day-level form.** §0. The ban test bounds it but does not remove it.
- **The population.** Diamond+ PC competitive, S19 through the last patch in the
  window, players reachable by BFS from the leaderboards. Effects at the top of
  the ladder come from the ≥4,750 refit and carry wider intervals.

────────

## 8. Dropped from the earlier draft

Choice model, ban value as a scouting quantity, p90 "pro prior", the
coaching framing throughout; ρ-continuum (wrong endpoint); `u_h'v_k` (not
antisymmetric); one-player-per-row Spec B.

────────

## 9. Order

1 checks → 2 attribution → 3 C without φ → 4 fetch and φ → 5 ban test and A→C
table → 5 stability and heterogeneity → 3 γ robustness → 6 bootstrap and final
table.

The ban test runs immediately after φ because it decides whether the specialist
control is adequate before anything is built on top of it.
