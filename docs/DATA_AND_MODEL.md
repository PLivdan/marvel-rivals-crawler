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
| Scope | Season 19, PC (`device=1`), **competitive only** (`game_mode_id=2`) |
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
season 19. 16 distinct maps. Every match has exactly 12 player rows.

Crawl state: 10,189 players fully crawled, 866,662 pending, 4,323 below the
Diamond floor, 2,324 private, 14 errored.

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
(Mantis). This is the default attribution rule (**W1**).

It is still not clean: it does not *condition* on swapping but it does *weight*
by it, and a losing player abandoning a hero cuts that hero's weight. See §4.4.

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
| 0 | 1,990 | 18.49% |
| 1 | 89,534 | 39.44% |
| **2** | **533,702** | **52.30%** |
| 3 | 24,456 | 41.28% |
| 4 | 288 | 34.72% |

More tanks is **not** better; two is optimal. A term linear in tank counts
cannot represent this, and since too-few-tank teams outnumber too-many by ~4:1,
a fitted linear slope is dragged positive — a straight line through the left arm
of a hill.

### 2.12 Two players on one team can share a "dominant" hero

Hero uniqueness binds only at a moment in time. With sequential non-overlapping
swaps, two teammates can each spend most of their match on the same hero.
Naively de-duplicating a lineup into a `set` collapses it to five heroes for
**2.06%** of team-instances (11,388 to five, 128 to four, 2 to three).

This matters more than the rate suggests: the collision is driven by swap
chatter, which §2.3 shows is correlated with losing. It is therefore
**outcome-correlated measurement error**, not noise. Fixed by preserving player
identity through the W2 aggregation.

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

### 3.1 Specification A (the one being estimated)

Match-level differential logit. One observation per match; outcome is victory
for camp 0.

```
logit P(camp0 wins) = alpha                       side advantage
                    + sum_h  beta_h  * x_h        hero contrasts (sum-to-zero)
                    + sum_t  tau_t   * z_t        104 team-up contrasts
                    + sum_s  delta_s * c_s        composition shape (sum-to-zero)
                    + theta' * skill_diff         pre-match rank differential
```

- `x_h` = own-side weight of hero *h* minus opposing-side weight.
- `z_t` = +1 if both team-up members are on camp 0's lineup, −1 if on camp 1.
- `c_s` = composition-shape indicator, contrasted across sides.
- `skill_diff` = mean `(new_score - add_score)` differential.

### 3.2 Estimation choices

| Choice | Value | Why |
|---|---|---|
| Penalty | **None** | Ridge coefficients are biased and carry no valid standard errors, which is fatal when the output is a ranked table with intervals |
| Collinearity | Sum-to-zero contrast basis | The rank deficiency is *structural* (every team fields six heroes), so it is removed by a constraint, not a penalty. Every hero gets a coefficient **and** a standard error |
| Normalisation | `sum(beta) = 0` | Effects are relative to an average hero; verified at machine precision |
| Std. errors | **HC1** heteroskedasticity-robust | Spec §8 |
| Multiple testing | Benjamini-Hochberg FDR, q = 0.05 | 55 simultaneous hero tests |
| Attribution | **W1** play-time (primary), **W2** dominant-hero (robustness) | §2.4 |
| Team-ups / shape | Always built under **W2** | They need a definite set of six heroes; fractional weights cannot establish co-presence |
| Forfeit floor | 240s | Short matches are the *least* swap-contaminated data (1.49 heroes/player under 5 min vs 2.40 over 15), so an aggressive floor discards the cleanest observations |

### 3.3 Sample construction

Filters applied in order, each counted (see `apm/sample.py`):

1. Exactly 12 player rows, 6 per side
2. No `is_win = 2` in the match
3. `add_score` and `new_score` present for every player
4. Every player has ≥1 hero row with `play_time > 0`
5. Duration ≥ 240s

On the 317,674-match fit: 321,337 starting, less 1,423 draws, 11 incomplete
play-time, 2,229 short.

### 3.4 Reporting: within-role normalisation

**The raw coefficients are not reported.** Because of §2.10 they measure
composition, not hero strength. Each hero's own-role mean is subtracted — a
linear contrast, so its covariance follows exactly by the delta method.

Validation: rank correlation against play-time-weighted raw win rates rises from
**0.70 (raw) to 0.95 (within-role)**.

---

## 4. Known limitations, ordered by severity

### 4.1 Composition control: 12 shapes, everything else pooled

Only shapes appearing at least `min_shape_count` times (default 500) get their
own dummy; the rest are pooled into a single `other` bucket. Measured over
961,697 team-instances, 27 distinct compositions occur, of which **12 get their
own control and 15 are pooled**:

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
| *15 others* | *1,502* | *0.16%* | *0.0–53.8%* | **pooled** |

**The pooling is far less damaging than it first appears: it covers 0.16% of
team-instances.** The 12 dummies span 99.84% of the sample. Extreme comps that
sound alarming are genuinely almost nonexistent — 5 Damage and 1 Support
(`0-5-1`) occurs 171 times with a 12.87% win rate; all-Damage (`0-6-0`) occurs
48 times.

The residual concern is heterogeneity, not coverage: win rates inside the pooled
bucket span 0.0% to 53.8%, so one coefficient represents `2-0-4` (23.1%),
`4-1-1` (33.1%) and `2-4-0` (4.9%) alike. Lowering `min_shape_count` to ~100
would give own dummies to `2-0-4`, `4-0-2` and `0-5-1` at negligible cost.

Two structural caveats remain regardless of the threshold:

- Per §2.10 the shape dummies capture only the **non-linear residual**; the
  linear part of composition lives inside the hero coefficients and cannot be
  separated from them. This is why results are reported within role.
- The shape is computed from **dominant heroes (W2)**, so a team that swapped
  through several compositions is represented by one summary shape.

Note how strongly these figures corroborate §2.11: every deviation from 2-2-2
loses, and the worst comps are those dropping a role entirely — `2-4-0` (no
Support) wins 4.86%, `1-5-0` wins 7.06%.

### 4.2 Magnitudes are attribution-dependent; only the ranking is stable

W1 vs W2: Spearman **0.954**, but effect sizes roughly **halve** under W2
(Squirrel Girl −16.79 → −9.02; Magik +14.65 → +7.97; mean absolute difference
2.08 points). **Treat the ordering as the result and magnitudes as an upper
bound.**

Note the two runs are not on an identical sample (317,674 vs 329,572 matches)
because the crawl continued between them.

### 4.3 These are adjusted associations, not causal effects

Pre-match rank controls for *general* skill but not skill *on that specific
hero*. A one-trick is not a random player assigned that hero. Per-player hero
playtime (available from `/api/player/{uid}` as `heroes_ranked` /
`heroes_unranked`) would measure this directly; it was costed and declined, so
the confounder is **bounded, never measured**.

### 4.4 Intervals are analytic, not bootstrapped

The design calls for a player-level cluster bootstrap — matches are not
independent draws, since each contains twelve players and belongs to twelve
overlapping clusters. The current implementation would need **~656 hours**:
`build_design` re-queries the database every replication, and
`attribution_weights` has no `match_uid` predicate, so it pulls the whole
`match_player_heroes` join (65s per call, twice per build).

Bootstrap is therefore **default-off** and the reported intervals are analytic
HC1. They ignore cross-classified dependence and are **too narrow**. Point
estimates are unaffected. Fix: build the design once, then row-replicate
prebuilt arrays per replication.

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
- The cluster bootstrap, made feasible per §4.4
- Robustness battery: attribution sensitivity tables, duration strata, temporal
  stability / structural break tests, rank-band refits at ≥4,750, forfeit-floor
  sensitivity, ridge appendix
- `total_value` = solo effect plus expected team-up contribution
- Ban-rate correlation as external validation
- Nested out-of-sample evaluation on a temporal holdout

---

## 6. Running it

```bash
# crawler (resumable; crawl_status in the DB is the queue)
python3 main.py --db-path data/rivals.db --workers 8
python3 main.py --status                      # progress, no network

# refresh hero/team-up reference data (after a season change)
python3 refresh_reference.py --db-path data/rivals.db

# fit and persist; saves CSV before touching the DB
python3 results/run_apm.py --attribution W1 --tag W1

# validation: placebo, sum-to-zero, agreement with raw win rates
python3 results/placebo_check.py

# regenerate the LaTeX report
python3 results/make_report.py && cd results && latexmk -pdf apm_report.tex
```

Stop the crawler with plain `kill <pid>` (SIGTERM). It drains gracefully and
hands claimed players back; `kill -9` strands them until the stale-claim sweep
reaps them an hour into the next run.
