# Hero Adjusted Plus-Minus (Character Value) — Design

Date: 2026-09-07
Status: approved for planning
Data source: `data/rivals.db`, built by the rivalsmeta crawler (see `2026-09-07-rivalsmeta-crawler-design.md`)

## 1. Research question

For each of the 55 Marvel Rivals heroes, estimate the change in win probability
attributable to fielding that hero, net of who is playing it, which five
teammates accompany it, which six heroes oppose it, the team's role
composition, the side of the map, and the team-up abilities the lineup unlocks.

This is the adjusted plus-minus (APM) framing familiar from basketball, adapted
to a 6v6 draft game: the "adjustment" is that all twelve heroes enter the
estimating equation simultaneously, so a hero's coefficient is identified net
of the company it keeps.

The chosen estimand is **balance diagnosis** — how strong is the hero itself,
not how valuable it is to pick in the current meta. That choice dictates
maximal control for selection into hero choice.

## 2. Estimand and normalization

Let `beta_h` denote the coefficient on hero `h`.

**Definition.** `beta_h` is the change in the log-odds of victory from replacing
an average hero with hero `h` on a team, holding fixed the opposing lineup, the
remaining own lineup, players' general skill, composition shape, side, and
team-up structure.

**Normalization.** Every team fields exactly six heroes, so the hero contrast
columns sum to zero identically (Section 5.2). The hero block of the design
matrix is therefore rank-deficient by exactly one, and `beta` is identified only
up to an additive constant. We resolve this with an explicit **sum-to-zero
constraint**, making every coefficient an effect *relative to the average
hero*. This is a substantive choice, not a numerical convenience: "average hero"
is the natural replacement level for a game where a roster slot is always
filled.

Implementation is a contrast basis rather than a dropped reference category. Let
`C` be the 55 x 54 matrix whose columns span `{b : sum(b) = 0}`. We estimate 54
free parameters `gamma` and recover `beta = C gamma`, which satisfies the
constraint by construction. The covariance follows as `Var(beta) = C Var(gamma)
C'`. This yields a valid standard error for *every* hero including the one that
a dropped-category coding would have left without one.

**Reporting units.** Coefficients are reported in log-odds and converted to
marginal win-probability points. For the logit specification the marginal effect
at a balanced match is `dP/dx = beta * p * (1 - p) = beta / 4` at `p = 0.5`; the
linear probability model returns probability points directly. Both are tabulated
so the two specifications are comparable on a common scale.

## 3. Why regularization is excluded

The obvious approach to a rank-deficient, highly collinear hero design is ridge
regression, which is standard in the games-analytics literature. We reject it.

Ridge coefficients are biased by construction and do not admit valid standard
errors without debiasing machinery, which is fatal for a study whose entire
output is a ranked table of effects with confidence intervals. The statistical
motivation for shrinkage is also absent: the specification carries roughly 170
parameters against 171,000 matches and rising, and the least-covered hero
already appears in tens of thousands of player-rows. The collinearity that
motivates ridge is *structural* (columns sum to zero) rather than incidental,
and a structural rank deficiency is properly handled by a constraint, not a
penalty.

Ridge is retained in exactly one place: as a robustness check reported in the
appendix, to confirm the unpenalized rankings are not driven by
near-collinearity among frequently co-occurring heroes.

## 4. Sample construction

Population: PC (`device=1`) competitive matches in season 19, as surfaced by the
crawler's breadth-first traversal of the player graph.

Inclusion criteria, applied in order, each logged with the count it removes so
the exclusion table can be reported:

1. Match has exactly 12 player rows and 6 per side.
2. No player in the match has `is_win = 2`. This value is neither a win nor a
   loss (it spans whole matches and is almost certainly a draw or an abandon);
   filtering to `is_win IN (0,1)` moves the overall win rate from 50.69% to
   49.989%, i.e. to the exactly-50% a symmetric game must produce. Currently 762
   matches.
3. Every player has non-null `new_score` and `add_score`, so pre-match rank is
   computable. Currently 100% of rows.
4. Every player has at least one `match_player_heroes` row with `play_time > 0`.
5. Match duration is finite and above a forfeit floor, set from the duration
   distribution (p1 = 240s, median = 709s). The floor is a *reported robustness
   parameter*, not a silent default: results are shown with no floor and with
   floors at p1 and p5.

Ban rows with `hero_id = 0` are empty ban slots, not heroes, and are dropped
before any ban-derived variable is constructed (currently 18,200 rows).

**Sampling bias, stated not hidden.** The sample is a breadth-first crawl seeded
from global and per-hero leaderboards, not a random draw of matches. Two
distortions are documented and carried into the limitations section: (a)
per-hero leaderboard seeding deliberately over-samples mains of rare heroes,
which is desirable for precision but means pick-rate figures from this sample
are not population pick rates; (b) profile privacy is strongly rank-selected
(0% private below 3,999 rank score, rising monotonically to 70.2% above 5,000),
and the crawl can only expand through public players, so high-elo matches are
structurally under-sampled. Private players' match rows *are* captured through
public co-players, so no outcome data is missing — the bias is in match
selection, not in measurement.

## 5. Variable construction

### 5.1 Hero attribution — the central measurement problem

There is no draft data. `dynamic_fields.ban_pick_info` returns exactly six
entries per match, all with `is_pick = 0`, verified against the live API and not
merely inferred from the database. The pick order the crawler's design spec
assumed exists is not served by rivalsmeta. Consequently **a player's hero is
never observed pre-match**, and roughly 55% of player-matches involve a
mid-match swap.

This matters more than it first appears, because swapping is endogenous to the
match state. Measured across the full table, players who used a single hero all
match win 65.6% while players who swapped win 37.3%. Swapping is a marker of
already losing. Two consequences follow, and both are traps:

- Using `cur_hero_id` (the hero at match *end*) attributes losses to whoever was
  swapped *to*. Doctor Strange pools to a 36.0% win rate this way — an artifact,
  not a finding.
- "Fixing" this by restricting to non-swap player-matches conditions on the
  outcome in the opposite direction and pushes every hero above 54%.

The only attribution that does not condition on the outcome is **play-time
weighting**: hero `h` receives weight equal to the share of that player's match
time spent on it, so each team's weights sum to 6. Applied to raw win rates this
produces a distribution centred exactly where a symmetric game requires — median
50.0%, ranging from 43.1% (Squirrel Girl) to 57.0% (Mantis).

Play-time weighting is the primary attribution rule. It remains partly
post-treatment (a losing team swaps, changing the weights), which cannot be
repaired without pick data. We therefore treat attribution as a **specification
axis rather than a settled choice**, reporting the full hero table under:

- **W1 (primary)** play-time-weighted fractional membership;
- **W2** dominant-hero ("main") membership, integer-valued, which defines
  team-ups and composition unambiguously at the cost of discarding swap
  information;
- **W3** W1 estimated within match-duration strata, since swap opportunity rises
  with match length — stability of `beta` across strata bounds how much
  contamination the primary estimates carry.

Divergence between W1 and W2 beyond confidence bounds is itself a reportable
finding about the severity of swap contamination.

### 5.2 Hero contrasts

For hero `h`, `x_h = w_h(camp 0) - w_h(camp 1)`, where `w` is the attribution
weight from 5.1. Because both teams' weights sum to 6, `sum_h x_h = 0` for every
match — this is the structural rank deficiency handled in Section 2. Mirror
picks give `x_h = 0` and correctly contribute no information about `h`.

### 5.3 Skill controls

Pre-match rank score is `new_score - add_score`, available for 100% of rows and
genuinely pre-treatment (`add_score` is this match's delta, so subtracting it
recovers the score the player entered with).

Primary control is the team mean differential. Because matchmaking is imperfect
— the mean within-match spread in pre-match score is 155 points, with a maximum
of 691 — this control does real work rather than absorbing nothing. Secondary
terms (minimum pre-score differential, capturing a weakest-link effect; a
quadratic or spline in the mean differential, since rank-to-win-probability is
unlikely to be linear) are selected by out-of-sample log-loss on the validation
split, and the selection is reported.

### 5.4 Composition

Role composition is `(#Tank, #Damage, #Support)` per team, computed from
`hero_info.role`. 2-2-2 accounts for 75.9% of teams, with 1-3-2 (9.5%), 2-1-3
(5.8%), 1-2-3 (3.8%) and 3-1-2 (3.4%) making up most of the remainder — enough
variation to identify shape effects.

**Role counts cannot be entered as controls.** The number of tanks on a team is
the sum of the tank heroes' indicators, so a role-count differential is a linear
combination of the hero contrast columns and is perfectly collinear with them.
Composition is already absorbed by the hero block, and a separate role-count
coefficient is not identified.

What *is* identified is a **non-linear** function of the counts. We enter an
indicator for each composition shape, contrasted across teams: `c_s = 1[camp 0
is shape s] - 1[camp 1 is shape s]`. Since each team has exactly one shape,
`sum_s c_s = 0` identically, so the shape block carries the same structural rank
deficiency as the hero block and receives the same sum-to-zero contrast-basis
treatment. Shapes below a frequency floor are pooled into an "other" category.

### 5.5 Team-ups

The 104 two-hero team-up abilities are mechanically real pair synergies, which
makes them enormously better conditioned than estimating all C(55,2) = 1,485
arbitrary hero pairs: we estimate the 104 interactions the game actually
implements and ignore the rest.

For team-up `t` with members `{a, b}`, `z_t = 1[both on camp 0] - 1[both on camp
1]`. This is a product of hero indicators, hence non-linear in them and
separately identified.

**Interpretation caveat, to be stated in the results.** Team-up terms are highly
correlated with their constituent heroes' main effects, so the estimator splits
credit between them. `beta_h` is therefore the hero's value *excluding* its
team-up bonuses. Two quantities are reported:

- `solo_value` = `beta_h`, the hero net of team-ups;
- `total_value` = `beta_h` + the expected team-up contribution given how often
  the hero is actually paired in this sample.

Under W1, team-up activation is approximate, because per-hero play-time totals
do not establish that two heroes were on the field *simultaneously*. Team-up
terms are therefore constructed under W2 (dominant hero) even in the primary
specification, and this asymmetry is documented.

### 5.6 Hero-specific experience — a validation subsample, not a universal control

`GET /api/player/{uid}?season={N}` returns `heroes_ranked` and
`heroes_unranked`: dicts keyed by hero id giving that player's `matches`,
`win`, `play_time`, K/D/A, damage, heal and `session_hit_rate` on each hero,
separately for competitive and quickplay. This is a direct measure of the
confounder Section 7 names as dominant, and the crawler does not currently store
it.

It is nevertheless **not** adopted as a universal control, for three compounding
reasons:

1. **Competitive hero-hours are post-treatment.** The totals are season-to-date
   at fetch time, so the very matches being modelled are inside them, and the
   feedback runs the wrong way: winning on a hero causes more play on that hero.
   Conditioning on it is a textbook bad control and would bias hero effects
   toward zero.
2. **Quickplay hours are the clean version but are missing non-randomly.**
   Unranked play never enters our competitive sample, so quickplay hero-hours
   measure familiarity without the mechanical feedback loop — the single best
   available proxy. But 36% of sampled players have no unranked data at all, and
   that missingness plausibly correlates with competitive seriousness, which is
   the very trait being proxied.
3. **Acquisition cost is a second full crawl.** Profiles exist for ~2,400
   crawled players against ~400k distinct players appearing in matches, and
   `visibility.career_stats` gates the field — so the rank-selected private
   population (70% above 5,000 rank score) is unreachable, reintroducing
   selection into the control itself.

There are also no lifetime totals to fall back on: `season=0` and `season=all`
both return empty, and omitting the parameter defaults to the current season, so
no pre-season baseline exists that would be cleanly pre-determined.

**Design decision.** Collect these stats for a *random subsample* of roughly
20-30k players (on the order of 30 minutes of crawling, versus 6+ hours for a
full sweep) and use them to run the headline specification with hero-experience
controls added, restricted to matches where they are observed. If hero
coefficients do not move relative to the unrestricted fit, the untestable
assumption in Section 7 has been converted into a **tested** one against a
directly measured confounder — a materially stronger result than an Oster bound,
which only asks how large an unobserved confounder would have to be. If they do
move, that is the headline finding and the estimates must be reported with the
control.

The subsample must be drawn at random from players appearing in the analysis
matches, not from the crawl frontier, and the snapshot timestamp recorded, since
these totals drift.

### 5.7 Bans, and why they mostly cancel

Bans are structurally clean: exactly 6 per match, 3 per side, and zero instances
of a banned hero being played in that match. They are the only genuinely
pre-outcome draft-stage variable in the dataset.

They are nevertheless **almost useless as controls in a differential
specification**, and the spec says so explicitly rather than including them for
appearance. A ban removes a hero from both teams' pools, so the banned set is a
match-level constant that differences out exactly as map identity does. Ban main
effects cannot enter. For the same reason bans cannot serve as an instrument for
hero presence: they shift no team's lineup relative to the other's.

Bans are used in three legitimate ways instead:

1. **External validation.** Ban rate is a revealed-preference measure of
   perceived strength, produced by players rather than by our model. The rank
   correlation between estimated `beta_h` and ban rate is a genuine
   out-of-model check; agreement is evidence the model recovers something real,
   and sharp disagreement for a specific hero is a finding worth reporting.
2. **Availability context.** Interactions between a hero's effect and whether
   its strongest counter was banned are a stated extension, not v1 scope.
3. **Sample description.** Ban frequencies characterise the meta the estimates
   describe.

### 5.8 Side

The outcome is camp-0 victory and all hero terms are differentials, so the
intercept identifies the side advantage. This is not negligible: camp 0 wins
51.10% against camp 1's 48.88%. Omitting the intercept would push a real ~2.2
point asymmetry into the hero coefficients.

Map identity differences out for the same reason bans do; hero-by-map
interactions are an explicit non-goal for v1.

## 6. Specifications

### 6.1 Specification A (headline) — match-level differential logit

Unit of observation: the match. Outcome: camp 0 wins.

```
logit P(camp0 wins) = alpha
                    + sum_h beta_h * x_h            (hero contrasts, sum-to-zero)
                    + sum_t tau_t * z_t             (team-up contrasts)
                    + sum_s delta_s * c_s           (composition shape, sum-to-zero)
                    + theta' * skill_diff           (Section 5.3)
```

Estimated unpenalized by maximum likelihood on the 54 + 104 + (shapes - 1) +
skill free parameters. This is the natural structure for "who beat whom" and
produces the headline table.

Its limitation is precisely the conditional independence assumption in Section
7: it controls for players' *general* skill but not for their skill *on the
specific hero*.

### 6.2 Specification B (co-headline) — player-match LPM with player fixed effects

Unit of observation: the player-match (~1.9M rows). Outcome: the player's team
won.

```
won_ip = mu_i                                   (player fixed effect)
       + sum_h beta_h * own_hero_ih             (sum-to-zero)
       + sum_h phi_h * teammate_heroes_ih
       + sum_h psi_h * opponent_heroes_ih
       + theta' * skill_terms
       + side + e_ip
```

Player fixed effects absorb every time-invariant attribute of the player,
including general skill and playstyle, so hero effects are identified purely
from **within-player variation** — the same person's results across the
different heroes they play. This directly attacks the dominant confounder, that
strong players main particular heroes.

Two deliberate choices require justification:

- **Linear probability model, not logit.** Fixed-effects logit is inconsistent
  when the number of observations per fixed effect is small (the incidental
  parameters problem); here players average only ~5.6 matches. LPM with
  absorbed fixed effects is consistent and is the standard applied-econometrics
  response. Predicted probabilities outside [0,1] are a known LPM cost, reported
  as a diagnostic rather than ignored.
- **One-way absorption is exact.** With a single fixed-effect dimension the
  within-transformation (demeaning by player) recovers the OLS estimates
  exactly via Frisch-Waugh-Lovell, so no iterative absorption library
  (`pyfixest`, `linearmodels`) is required — neither is installed, and neither
  needs to be.

Player fixed effects remain unable to remove *hero-specific* human capital. The
spec claims only what the design delivers.

**The comparison of A against B is a primary result, not a robustness check.** A
hero whose estimated value collapses once player fixed effects are introduced is
a hero whose reputation is largely its playerbase — which is exactly the
question a balance diagnosis is asked to answer.

### 6.3 Deferred — hierarchical Bayesian

A partially-pooled model with hero effects drawn from role-level hyperpriors
would give principled shrinkage and exact posterior intervals. It is explicitly
deferred, not silently dropped: at this sample size pooling is not binding, and
the computational cost on 500k matches buys little the cluster bootstrap does
not already provide.

## 7. Identification and threats

**Assumption (CIA).** Conditional on players' general skill, the opposing
lineup, the remaining own lineup, composition shape, team-up structure and side,
hero deployment is independent of unobserved determinants of victory.

This assumption is *not* fully credible, and the design's job is to bound the
damage rather than assert it away. The threats, and what each specification does
about them:

| Threat | Mechanism | Treatment |
|---|---|---|
| Hero-specific human capital | A one-trick is not a random player assigned that hero | Partially addressed by Spec B (removes general skill only). Bounded by Oster (2019), and **directly measured on a validation subsample** (Section 5.6) rather than only bounded. |
| Post-treatment swapping | Losing teams swap, so play-time weights respond to the outcome | W1/W2/W3 attribution axis (5.1); duration strata bound the contamination |
| Selection into sample | BFS crawl, rank-selected privacy | Documented (Section 4); rank-band refits as robustness |
| Simultaneity of team-ups | Play-time totals do not prove co-presence | Team-ups built under W2 (5.5) |
| Balance patch mid-window | 31-day sample may straddle a patch | Structural-break tests (Section 9) |

**Sensitivity.** Oster (2019) coefficient-stability bounds quantify how strong
unobserved selection would have to be, relative to observed selection on rank
score, to overturn each hero's sign. This converts "there may be confounders"
from a hand-wave into a number.

## 8. Inference

**Dependence structure.** Matches are the unit in Spec A, but matches share
players: a given match belongs to twelve overlapping player-clusters. This is
cross-classified, not nested, so conventional one-way clustering does not apply.

**Reported inference** is a **cluster bootstrap over players**: resample players
with replacement, take their matches, refit, and take percentile intervals over
replications. This respects the actual sampling process (the crawl enumerates
players and harvests their matches) rather than pretending matches are
independent draws. Heteroskedasticity-robust match-level standard errors are
reported alongside as a lower bound, with the gap between them shown so a reader
can see what the dependence costs.

For Spec B, two-way clustering by match and by player (Cameron-Gelbach-Miller)
is available in closed form and is reported in addition to the bootstrap.

**Multiple testing.** 55 hero coefficients are tested simultaneously.
Benjamini-Hochberg FDR control at q = 0.05 is applied across the hero family,
and both raw and adjusted p-values are reported. Unadjusted significance stars
on 55 simultaneous tests would be indefensible.

**Bootstrap replications:** 1,000 for reported intervals; convergence of
interval width in the number of replications is checked and reported.

## 9. Robustness and falsification battery

Every item produces a table or figure in the output; nothing here is optional.

1. **Placebo — permuted assignment.** Randomly permute hero contrasts across
   matches. Coefficients must collapse to zero and the FDR-adjusted discovery
   count must fall to approximately its nominal rate. A placebo that fails
   invalidates the pipeline.
2. **Mirror matchups.** Restricting to matches where a hero appears on both
   sides, that hero's contrast is identically zero; verifying the estimator
   returns no effect is a machine-checkable invariant.
3. **Attribution sensitivity.** Full hero table under W1, W2, W3 (5.1).
4. **Duration strata.** Estimates by match-length quantile, bounding swap
   contamination.
5. **Temporal stability.** Split-half by date plus a Chow-type structural break
   test scanning candidate break dates, to detect a mid-sample balance patch.
6. **Rank-band refits.** Re-estimate on high-elo matches only, given the
   documented rank-selected sampling.
7. **Forfeit floor sensitivity.** No floor, p1 floor, p5 floor.
8. **Ridge appendix.** Confirms rankings are not artifacts of near-collinearity.
9. **Oster bounds.** Per Section 7.
10. **Hero-experience validation subsample.** Headline specification refit with
    measured hero-specific experience controls on the Section 5.6 subsample,
    against the same specification without them on the identical matches, so the
    comparison isolates the control rather than the sample. Quickplay hours are
    the preferred control (not post-treatment); competitive hours are reported
    separately and read as a lower bound, since they are contaminated in the
    known direction.

## 10. Model evaluation

Nested comparison on a **temporal holdout** (final week held out, so evaluation
is genuinely out-of-sample rather than a random split that leaks meta drift):

| Model | Purpose |
|---|---|
| Intercept only | Side advantage floor |
| + skill differential | How much is just matchmaking |
| + hero contrasts | The quantity of interest |
| + team-ups | Marginal value of pair synergies |
| + composition shape | Marginal value of the shape control |

Metrics: out-of-sample log-loss (primary), AUC, and a calibration curve with
Brier decomposition.

**Expectation setting.** In team games hero choice explains far less variance
than player skill. An AUC in the 0.55-0.62 range with hero terms adding perhaps
0.01-0.03 over a skill-only baseline is the anticipated result and is not a
failure — with 171k+ matches, coefficients can be estimated precisely even when
aggregate predictive power is modest. The deliverable is the coefficient table,
not a match predictor.

## 11. Outputs

**Results written back into `data/rivals.db`** so they are queryable alongside
the source data, in tables `apm_runs` (one row per estimation run: timestamp,
specification, attribution rule, sample filters, n, git commit) and
`apm_hero_effects` / `apm_teamup_effects` (coefficients keyed to a run).
Persisting the run metadata alongside the coefficients is what makes a number in
a table reproducible six months later.

**Every reported table carries**, per the project's reporting standards: sample
period, n (matches and player-matches), the exclusion counts from Section 4,
attribution rule, control set, estimator, clustering/bootstrap scheme, raw and
FDR-adjusted p-values, significance stars keyed to adjusted values, and the
coefficient in both log-odds and win-probability points.

**Headline table columns:** hero, role, `solo_value` (pp), `total_value` (pp),
bootstrap 95% CI, adjusted p-value, matches played, pick rate, ban rate, and the
Spec A vs Spec B gap.

## 12. Architecture

A new `apm/` package, kept separate from the crawler's flat modules because it
is a distinct subsystem with a different dependency footprint and lifecycle:

- `apm/sample.py` — inclusion criteria, exclusion accounting, holdout splits.
- `apm/features.py` — design-matrix construction; the attribution rule (W1/W2/W3)
  is a parameter, not a branch, so specifications differ only by argument.
- `apm/contrasts.py` — sum-to-zero contrast bases and the `beta = C gamma`
  recovery with its covariance transform.
- `apm/estimate.py` — Spec A (logit) and Spec B (within-transformed LPM).
- `apm/inference.py` — player cluster bootstrap, two-way clustering, BH FDR.
- `apm/diagnostics.py` — the Section 9 battery.
- `apm/report.py` — result tables, DB persistence, formatted output.
- `apm_main.py` — CLI, mirroring `main.py`'s argument style.

**One upstream prerequisite.** The Section 5.6 validation subsample needs data
the crawler does not currently store: a `player_hero_stats` table
(`player_uid`, `hero_id`, `mode` in {ranked, unranked}, `matches`, `wins`,
`play_time`, K/D/A, `session_hit_rate`, `snapshot_at`), populated by a
subsample profile sweep. The profile endpoint is already called by
`crawl_player`, so this is a new ingest path and a bounded sweep script rather
than new fetching machinery. It is sequenced *after* the headline model works,
so the model is never blocked on it.

Dependencies are limited to what is already installed: numpy, scipy, pandas,
statsmodels, matplotlib. Sparse design matrices (`scipy.sparse`) and float32 keep
the 1.9M-row Spec B within memory.

Testing follows the crawler's existing pattern (pytest, no network in the
default suite). Estimator correctness is verified against **synthetic data with
known coefficients** — the only way to test a statistical estimator honestly —
alongside unit tests for contrast-basis algebra (verifying `sum(beta) = 0` and
covariance transformation), sample filters, and the mirror-matchup invariant.

## 13. Non-goals for v1

- Hero-by-map interactions (map differences out; interactions are a large model).
- Draft recommendation or pick-order optimisation.
- Match outcome prediction as a product.
- Cross-season estimation (the sample is one season, and team-ups are reworked
  between seasons).
- Causal claims beyond what Section 7 supports. The output is an *adjusted
  association* with quantified sensitivity to its central assumption.
