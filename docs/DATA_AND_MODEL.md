# Marvel Rivals: Data, Nuances, and the Current Model

Reference document for the `marvel-rivals-crawler` project. Covers what data we
hold, the traps in it, and the specification currently being estimated.

Last updated 8 September 2026. Database snapshot: 482,997 matches.

---

## 1. Data

### 1.1 Source and collection

| | |
|---|---|
| Source | `rivalsmeta.com` first-party JSON endpoints |
| Auth | None. Unauthenticated, plain JSON over HTTPS, Cloudflare-fronted |
| Scope | Season code 19 (Season 9.5), PC (`device=1`), **competitive only** (`game_mode_id=2`) |
| Method | Breadth-first crawl of the player graph, seeded from the global top-500 leaderboard plus one per-hero leaderboard for each hero |
| Collector | `main.py` (see `docs/superpowers/specs/2026-09-07-rivalsmeta-crawler-design.md`) |

The per-hero leaderboard seeding is deliberate: it counteracts pick-rate skew so
rarely-played heroes still accumulate enough observations. It also means **pick
rates in this sample are not population pick rates.**

### 1.2 Scale

| Table | Rows | Contents |
|---|---:|---|
| `matches` | 482,997 | One row per match: timestamp, duration, map, mode, season, MVP/SVP |
| `match_players` | 5,795,964 | 12 per match: hero at end, K/D/A, damage, healing, win flag, rank score |
| `match_player_heroes` | 11,679,961 | Per-hero breakdown within a player-match, with `play_time` |
| `match_bans` | 2,886,120 | 6 per match (3 per side) |
| `players` | 883,512 | Crawl frontier; `crawl_status` *is* the queue |
| `hero_info` | 55 | Hero id → name, role, difficulty, real name |
| `teamups` / `teamup_heroes` | 104 / 208 | Team-up abilities and their two members each |
| `apm_runs` / `apm_hero_effects` | 2 / 110 | Model results with full run metadata |

Database is ~3.4 GB. Matches span **2026-08-07 to 2026-09-08 (32 days)**, all
season code 19 (Season 9.5). 16 distinct maps. Every match has exactly 12
player rows.

Crawl state: 10,189 players fully crawled, 866,662 pending, 4,323 below the
Diamond floor, 2,324 private, 14 errored. The crawler has since been stopped
at 482,997 matches; these are a snapshot, not a moving target.

### 1.3 What each match gives us

Per player: hero(es) played with per-hero playtime, K/D/A, damage dealt/taken,
healing, win/loss, and **rank score before and after** (`new_score - add_score`
is the pre-match value). Per match: full 6-ban sequence with side, map, mode,
duration, timestamp.

---

## 2. Nuances and traps

These are empirical findings, not schema documentation. Several were expensive
to discover and are easy to re-introduce.

### 2.1 `is_win` is ternary, not boolean

Values are `0`, `1`, and **`2`** (25,629 rows). The `2` spans whole matches — a
draw or abandon. Consequences:

- `AVG(is_win)` is **not** a win rate. It reads 50.69%; filtering to
  `is_win IN (0,1)` gives 49.989%, i.e. the exactly-50% a symmetric game must
  produce.
- **Always filter `is_win IN (0,1)`** before averaging or fitting.

### 2.2 `cur_hero_id` is endogenous — never use it as "the hero played"

It is the hero at match *end*, and ~55% of player-matches involve a mid-match
swap. Players swap **when losing**, so the final hero is selected on the outcome.

Doctor Strange measured 66.9% win rate when he was the player's only hero, but
29.5% when swapped onto — pooling to a meaningless 36.0%.

### 2.3 Conditioning on swapping is biased in *both* directions

The obvious fix — keep only non-swappers — is equally wrong:

| | n | win rate |
|---|---:|---:|
| Used one hero all match | 832k | **65.6%** |
| Swapped (2+ heroes) | 1.03M | **37.3%** |

Not swapping is itself a marker of already winning. Restricting to non-swappers
pushes every hero above 54%.

### 2.4 Play-time weighting is the only non-conditioning attribution

`SUM(play_time * is_win) / SUM(play_time)` per hero. Produces a sane
distribution: median exactly 50.0%, range 42.6% (Squirrel Girl) to 56.9%
(Mantis). This is play-time weighting (**W1**): the match is split across the
heroes a player used, in proportion to time played.

It is still not clean: it does not *condition* on swapping but it does
*weight* by it, and a losing player abandoning a hero cuts that hero's weight.
That is precisely why play-time weighting is **not** the headline attribution
rule: the headline instead uses starting-hero attribution (W0, §3.1), fixed
before the outcome is known, and play-time weighting is reported only as a
comparison (§4.2). See also §4.4.

### 2.5 There is no draft/pick data

`dynamic_fields.ban_pick_info` returns exactly 6 entries per match and
**`is_pick` is always 0** — verified against the live API, not just the database.
The field is bans-only; the pick order the original design assumed does not
exist.

Consequence: **hero choice can never be made exogenous here.** Hero effects are
descriptive, not causal.

### 2.6 Bans are clean but mostly cancel

6 per match, 3 per side, zero cases of a banned hero being played. They are the
only genuinely pre-outcome draft-stage variable.

But a ban removes a hero from *both* pools, so the banned set is a match-level
constant that differences out of any differential specification, exactly as map
identity does. Bans also **cannot serve as an instrument** for hero presence,
because they shift no team's lineup relative to the other's. Use them for
external validation (ban rate as revealed preference) instead.

`hero_id = 0` in `match_bans` is an **empty ban slot**, not a hero (46,680 rows).
Drop it before constructing any ban variable.

### 2.7 Privacy is strongly rank-selected

Among players we attempted:

| rank score | attempted | private |
|---|---:|---:|
| 3,750–3,999 | 66 | 0.0% |
| 4,250–4,499 | 486 | 6.4% |
| 4,500–4,749 | 486 | 18.7% |
| 4,750–4,999 | 219 | 39.7% |
| 5,000+ | 47 | **70.2%** |

Two consequences:

- **No outcome data is lost.** A private player's rows inside a match are fully
  captured via public co-players — 97.9% of private players appear in matches,
  100% complete on hero, score, win and K/D/A.
- **Match selection is biased.** The crawl can only *expand through* public
  players, so high-elo matches are structurally under-sampled. More crawling
  does not fix this.

### 2.8 Hero id 1057 is base "Deadpool" and is never played

The roster records only his three role variants — 10571 (Vanguard), 10572
(Duelist), 10573 (Strategist). Hero 1057 has **0 plays and 0 bans** in 482,997
matches, yet four team-ups are defined against it: `1047002`, `1057001`,
`1057002`, `1059002`. Their columns are structurally all-zero and will
singularize a logit fit. They are dropped dynamically at fit time.

The `heroes` table also contains a row for hero_id `0`, which `hero_info` does
not — crawler bookkeeping, not a real hero.

### 2.9 There is a real side advantage

Camp 0 wins **51.06%**, camp 1 wins 48.88%. Any specification must carry an
intercept or this ~2.2-point asymmetry is pushed into the hero coefficients.

### 2.10 Role counts are collinear with hero indicators

**This is the single most important modelling nuance.** A team's tank count is
the *sum* of its tank heroes' indicators, so a role-count differential is a
linear combination of the hero contrast columns and is **not separately
identified**. Only a *non-linear* function of the counts — a composition-shape
indicator — is estimable.

The practical damage: composition loads onto individual hero coefficients. Role
alone explains **64.6%** of the variance in the raw hero estimates, and the
twelve highest-ranked heroes come out all Tanks. See §4.2.

### 2.11 Tank count has an inverted-U relationship with winning

| tanks | teams | win rate |
|---:|---:|---:|
| 0 | 2,992 | 19.25% |
| 1 | 133,441 | 39.68% |
| **2** | **789,192** | **52.29%** |
| 3 | 35,649 | 40.67% |
| 4 | 409 | 35.21% |
| 5 | 14 | 50.00% |

(Over 961,697 team-instances in the full crawl; `results/tank_winrate.csv`, from
`results/compute_aux_stats.py`. The five-Tank row is 14 teams and means nothing.)

More tanks is **not** better; two is optimal. A term linear in tank counts
cannot represent this, and since too-few-tank teams outnumber too-many by ~4:1,
a fitted linear slope is dragged positive — a straight line through the left arm
of a hill.

### 2.12 Two players on one team can share a most-played hero ("dominant" hero)

Hero uniqueness binds only at a moment in time. With sequential non-overlapping
swaps, two teammates can each spend most of their match on the same hero, i.e.
share a most-played-hero attribution (W2). Naively de-duplicating a lineup
into a `set` collapses it to five heroes for **2.06%** of team-instances
(11,388 to five, 128 to four, 2 to three).

This matters more than the rate suggests: the collision is driven by swap
chatter, which §2.3 shows is correlated with losing. It is therefore
**outcome-correlated measurement error**, not noise. Fixed by preserving
player identity in the lineup-roster construction (`_lineup_rosters`), which
builds rosters under whichever attribution rule is active, not only
most-played-hero.

### 2.13 Writing results while the crawler runs

SQLite returns `SQLITE_BUSY` **immediately** on a read→write transaction
upgrade and does **not** honour `busy_timeout` for it. A connection that runs a
`SELECT` first and then an `INSERT` will fail with "database is locked" no
matter how long the timeout.

Rules: do reads on a read-only handle, use autocommit plus a long
`busy_timeout` on the writer, and **always save results to disk before touching
the database.** The first production run completed the entire fit and then lost
it to this.

---

## 3. Current model

### 3.1 Specification A+ (the headline: starting-hero attribution)

Match-level differential logit. One observation per match; outcome is victory
for camp 0. The headline specification ("Spec A+") uses starting-hero
attribution (W0): the whole match is attributed to the hero the player picked
first (first entry by `MIN(rowid)` in `match_player_heroes`, chronological by
first appearance; no `play_time>0` filter). This is fixed before the outcome
is known — intention-to-treat — unlike the play-time (W1) and most-played-hero
(W2) rules reported for comparison in §4.2.

```
logit P(camp0 wins) = alpha_m                     per-map intercept (16 maps)
                    + sum_h  beta_h  * x_h        hero contrasts (within-role sum-to-zero)
                    + sum_t  tau_t   * z_t        100 team-up contrasts (104 defined)
                    + sum_s  delta_s * c_s        composition shape (sum-to-zero)
                    + theta' * skill_diff         pre-match rank differential
```

- `alpha_m` = one intercept per map (16 maps), replacing a single pooled
  side-advantage intercept; camp-0 win rate varies 49.42%–52.52% across maps.
  Disable with the runner's `--no-map-intercepts` flag.
- `x_h` = own-side weight of hero *h* minus opposing-side weight. The hero
  block carries **three** within-role sum-to-zero constraints (Tank, Damage,
  Support), not one global constraint — see §3.2.
- `z_t` = +1 if both team-up members are on camp 0's lineup, −1 if on camp 1.
  Team-composition shapes and team-up contrasts are built from the *same
  rule's* lineup rosters as the hero block, so under the headline (W0) they
  use the STARTING lineups, not dominant heroes.
- `c_s` = composition-shape indicator, contrasted across sides.
- `skill_diff` = mean `(new_score - add_score)` differential.

The headline fit has log-likelihood **−320,067.7** over the 477,483-match
sample (§3.3), with HC1 standard errors and Benjamini–Hochberg FDR control
(q = 0.05) across the 55 hero tests. Of the 104 defined team-up contrasts,
100 enter the fit; the other 4, defined against the never-played hero id 1057
(base "Deadpool", §2.8), are structurally empty and are dropped dynamically.

### 3.2 Estimation choices

| Choice | Value | Why |
|---|---|---|
| Penalty | **None** | Ridge coefficients are biased and carry no valid standard errors, which is fatal when the output is a ranked table with intervals |
| Collinearity | **Within-role** sum-to-zero contrast basis (three constraints: Tank, Damage, Support), `contrasts.within_role_basis(roles)` | Under one global constraint the composition-shape block reproduces the role-count differential exactly (δ_s = tankcount(s) ⇒ Σ δ_s c_s = θ′x), so hero and shape blocks are collinear except through the tiny pooled "other" shape. Constraining within role removes the role-count direction from the hero block entirely, so composition effects are estimated by the shape controls instead. See §3.4 |
| Normalisation | `sum(beta_role) = 0` for each of Tank/Damage/Support | Effects are relative to an average hero of the same role. Falsified against a globally-constrained fit: Spearman **0.9999**, mean \|difference\| **0.010 pp** (§3.4) |
| Std. errors | **HC1** heteroskedasticity-robust | Spec §8 |
| Multiple testing | Benjamini-Hochberg FDR, q = 0.05 | 55 simultaneous hero tests |
| Attribution | **Starting-hero (W0)** — headline. Play-time (W1) and most-played-hero (W2) reported as comparisons | §2.4, §4.2 |
| Team-ups / shape | Built from the **same rule's** lineup rosters as the hero block — under the headline (W0) this means the STARTING lineup, not dominant heroes; the W1/W2 comparison runs keep the most-played-hero (W2) roster | They need a definite set of six heroes; fractional weights cannot establish co-presence. Pairing a W0 hero block with a W2 lineup would reintroduce post-match information through the controls |
| Forfeit floor | 240s | Short matches are the *least* swap-contaminated data (1.49 heroes/player under 5 min vs 2.40 over 15), so an aggressive floor discards the cleanest observations |
| Map intercepts | 16 own dummies (default), disable with `--no-map-intercepts` | Camp-0 win rate varies 49.42%–52.52% across maps; a single pooled intercept would mask this |

### 3.3 Sample construction

Filters applied in order, each counted (see `apm/sample.py`):

1. Exactly 12 player rows, 6 per side
2. No `is_win = 2` in the match
3. `add_score` and `new_score` present for every player
4. Every player has ≥1 hero row with `play_time > 0`
5. Duration ≥ 240s

On the 477,483-match fit: 482,997 starting, less 2,139 draws, 19 incomplete
play-time, 3,356 short.

### 3.4 Reporting: within-role normalisation

**The raw coefficients are not reported.** Because of §2.10 they measure
composition, not hero strength. The within-role structure is now imposed
directly in estimation (§3.2) — `contrasts.within_role_basis(roles)` fits
three separate sum-to-zero constraints, one per role, rather than one global
constraint — so the reported coefficients are the fitted parameters
themselves, not a post-hoc projection.

Historically, within-role effects were instead obtained after a
globally-constrained fit by subtracting each hero's own-role mean — a linear
contrast, so its covariance followed exactly by the delta method. That
delta-method result stands as a historical validation: estimates from the
global and within-role parameterisations agree with Spearman rank correlation
**0.9999** and mean absolute difference **0.010 pp**, confirming that moving
the constraint into estimation only removes the composition-collinear
direction and does not otherwise change what is estimated.

Validation: rank correlation against play-time-weighted raw win rates rises from
**0.70 (raw) to 0.95 (within-role)**.

### 3.5 Temporal holdout

`results/compute_fit_quality.py` fits the headline specification on all
matches except the final 7 days (437,453 training matches) and evaluates it
on the 40,030 holdout matches of that week: calibration in 25 equal-count
bins, log loss against a constant-prediction baseline, Brier score, AUC, Cox
calibration slope/intercept, and a nested ladder of holdout log losses as
blocks are added (map intercepts → + rank-score differential → + composition
shapes → + hero contrasts → + team-up contrasts). The resulting numbers are
reported in the PDF (Figure 2 and Appendix Table A2), not reproduced here, and
are stored in `results/fit_quality_W0_specAplus.json` and
`results/calibration_W0_specAplus.csv`.

---

## 4. Known limitations, ordered by severity

### 4.1 Composition control: 18 shapes, everything else pooled

Only shapes appearing at least `min_shape_count` times (default **100**, was
500) get their own dummy; the rest are pooled into a single `other` bucket.
Measured over 961,697 team-instances, 27 distinct compositions occur, of which
**18 get their own control and 9 are pooled**. Full table in
`results/comp_shapes.csv` (produced by `results/compute_aux_stats.py`):

| T-D-S | teams | share | win rate | control |
|---|---:|---:|---:|---|
| 2-2-2 | 723,226 | 75.20% | **53.53%** | own dummy |
| 1-3-2 | 92,111 | 9.58% | 42.20% | own dummy |
| 2-1-3 | 57,549 | 5.98% | 40.44% | own dummy |
| 1-2-3 | 38,473 | 4.00% | 35.09% | own dummy |
| 3-1-2 | 30,698 | 3.19% | 42.19% | own dummy |
| 2-3-1 | 7,866 | 0.82% | 27.73% | own dummy |
| 3-2-1 | 3,153 | 0.33% | 26.51% | own dummy |
| 3-0-3 | 1,758 | 0.18% | 40.22% | own dummy |
| 1-4-1 | 1,736 | 0.18% | 20.51% | own dummy |
| 0-4-2 | 1,558 | 0.16% | 21.63% | own dummy |
| 0-3-3 | 1,051 | 0.11% | 17.22% | own dummy |
| 1-1-4 | 1,016 | 0.11% | 19.88% | own dummy |
| 2-0-4 | 407 | 0.04% | 23.10% | own dummy |
| 4-0-2 | 278 | 0.03% | 36.33% | own dummy |
| 0-5-1 | 171 | 0.02% | 12.87% | own dummy |
| 2-4-0 | 144 | 0.01% | 4.86% | own dummy |
| 0-2-4 | 144 | 0.01% | 15.97% | own dummy |
| 4-1-1 | 127 | 0.01% | 33.07% | own dummy |
| *9 others* | *231* | *0.024%* | *0.0–53.8%* | **pooled** |

**The pooling is far less damaging than it first appears: it covers 0.024% of
team-instances (231 teams).** The 18 dummies span 99.976% of the sample.
Extreme comps that sound alarming are genuinely almost nonexistent — 5 Damage
and 1 Support (`0-5-1`) occurs 171 times with a 12.87% win rate (own dummy at
this threshold); all-Damage (`0-6-0`) occurs 48 times and remains pooled.

The residual concern is heterogeneity, not coverage: win rates inside the
pooled bucket still span 0.0% to 53.8%, so one coefficient represents `5-1-0`
(0.0%), `1-5-0` (7.1%) and `5-0-1` (53.8%) alike. This is exactly what
lowering `min_shape_count` from 500 to its current default of 100 buys: six
more own-dummy rows (`2-0-4`, `4-0-2`, `0-5-1`, `2-4-0`, `0-2-4`, `4-1-1`) at
negligible added cost, shrinking the pooled bucket from 15 shapes / 0.16% to
9 shapes / 0.024%.

Two structural caveats remain regardless of the threshold:

- Per §2.10 the shape dummies capture only the **non-linear residual**; the
  linear part of composition lives inside the hero coefficients and cannot be
  separated from them. This is why results are reported within role.
- The shape is computed from the **same rule's lineup roster** as the hero
  block (`_lineup_rosters`, §3.1–3.2) — under the headline starting-hero rule
  (W0) this is the STARTING lineup, so a team that swapped through several
  compositions is represented by its shape at kickoff, not by its most-played
  shape. The W1/W2 comparison runs still use the most-played-hero (W2) roster.

Note how strongly these figures corroborate §2.11: every deviation from 2-2-2
loses, and the worst comps are those dropping a role entirely — `2-4-0` (no
Support) wins 4.86%, `1-5-0` wins 7.06%.

### 4.2 Magnitudes are attribution-dependent; only the ranking is stable

All three attribution rules — starting-hero (W0), most-played-hero (W2), and
play-time weighted (W1) — are fitted on the **identical** 477,483-match sample
(within-role constraint, map intercepts):

| rule | cross-hero s.d. of estimates (pp) | log-likelihood | Spearman ρ with starting-hero |
|---|---:|---:|---:|
| Starting hero (W0) | 2.60 | −320,122 | — |
| Most-played hero (W2) | 4.27 | −305,869 | 0.937 |
| Play-time weighted (W1) | 6.38 | −297,648 | 0.890 |

Magik moves +3.47 → +7.94 → +14.57 pp across the three rules in that order.
Magnitude scales with how much post-start information the rule uses: a rule
that lets outcome-correlated swapping (§2.3) into the attribution mechanically
inflates the spread of the estimates it produces. The higher log-likelihood of
W1 reflects regressors that encode more of what happened during the match, not
a better model of hero strength.

**The headline specification therefore uses starting-hero attribution (W0)**:
it is fixed before the outcome is known (intention-to-treat) and yields the
smallest, most conservative estimates. Treat the ordering as the result and
magnitudes as attribution-rule-dependent upper bounds.

### 4.3 These are adjusted associations, not causal effects

Pre-match rank controls for *general* skill but not skill *on that specific
hero*. A one-trick is not a random player assigned that hero. Per-player hero
playtime (available from `/api/player/{uid}` as `heroes_ranked` /
`heroes_unranked`) would measure this directly; it was costed and declined, so
the confounder is **bounded, never measured**.

### 4.4 Intervals are analytic, not bootstrapped

The design calls for a player-level cluster bootstrap — matches are not
independent draws, since each contains twelve players and belongs to twelve
overlapping clusters. The naive implementation would need **~656 hours**:
`build_design` re-queries the database every replication, and
`attribution_weights` has no `match_uid` predicate, so it pulls the whole
`match_player_heroes` join (65s per call, twice per build).

A fast implementation now exists. `apm/estimate.py` has
`fit_logit_weighted(X, y, weights, beta0=None)` — a frequency-weighted,
warm-started IRLS solver — and `results/run_bootstrap.py` uses it to run the
player-cluster bootstrap by building the design **once**, resampling players
with replacement, converting the resample to frequency weights on the fixed
rows, and refitting from the full-sample solution: roughly 15s per
replication, so ~4 hours for 1,000 replications versus the ~656 hours of the
naive design. **It has not been run** — the analytic HC1 intervals below were
judged sufficient for now.

The reported intervals therefore remain analytic HC1. They ignore
cross-classified player dependence and are **too narrow**. Point estimates are
unaffected.

### 4.5 The sample is not random

BFS crawl seeded from leaderboards, with rank-selected privacy (§2.7).
High-elo under-sampled; pick rates are not population pick rates.

---

## 5. What is not yet built

Deferred to a follow-up plan:

- **Specification B** wired end to end — player-match LPM absorbing ~345k player
  fixed effects, identifying hero effects from *within-player* variation. The
  estimator exists and is tested (`apm/estimate.py:fit_absorbed_lpm`); only the
  CLI wiring and player-match feature builder are missing. The A-vs-B gap is the
  headline diagnostic: a hero whose value collapses under player fixed effects
  is one whose reputation is mostly its playerbase.
- Two-way (match × player) clustered standard errors
- Running the cluster bootstrap (§4.4) — the fast implementation
  (`results/run_bootstrap.py`) exists but has not been executed
- Robustness battery: attribution sensitivity tables, duration strata, temporal
  stability / structural break tests, rank-band refits at ≥4,750, forfeit-floor
  sensitivity, ridge appendix
- `total_value` = solo effect plus expected team-up contribution
- Ban-rate correlation as external validation

---

## 6. Running it

```bash
# crawler (resumable; crawl_status in the DB is the queue)
python3 main.py --db-path data/rivals.db --workers 8
python3 main.py --status                      # progress, no network

# refresh hero/team-up reference data (after a season change)
python3 refresh_reference.py --db-path data/rivals.db

# fit (DB)
python3 results/run_apm.py --attribution W0 --constraint within_role --min-shape-count 100 --tag W0_specAplus

# non-fit statistics -> results/aux_stats.json, comp_shapes.csv (DB)
python3 results/compute_aux_stats.py

# temporal holdout -> fit_quality_*.json, calibration_*.csv (DB)
python3 results/compute_fit_quality.py

# validation: placebo, sum-to-zero, agreement with raw win rates
python3 results/placebo_check.py

# CSV/JSON -> results/apm_report.tex, no DB access
python3 results/make_report.py
latexmk -pdf -interaction=nonstopmode -outdir=results results/apm_report.tex
```

Everything downstream of the fit is CSV/JSON → LaTeX; `make_report.py` never
touches the database. Other runner flags: `--attribution {W0,W1,W2}`,
`--constraint {global,within_role}`, `--no-map-intercepts`,
`--min-shape-count N`, `--forfeit-floor S`, `--tag NAME`. Results are saved to
CSV/JSON before any database write, so a lock (§2.13) cannot lose a run.

Stop the crawler with plain `kill <pid>` (SIGTERM). It drains gracefully and
hands claimed players back; `kill -9` strands them until the stale-claim sweep
reaps them an hour into the next run.
