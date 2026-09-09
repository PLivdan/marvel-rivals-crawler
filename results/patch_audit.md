# Patch audit and rank coverage (development snapshot, frozen 2026-09-09)

## Balance regime

Official patch record (marvelrivals.com, fetched 2026-09-09; details in the research log below):

| Date (UTC) | Version | Type | Balance-relevant? |
|---|---|---|---|
| 2026-07-30 09:00 | v20260730 | mode/event, hero bug fixes | no |
| 2026-08-07 09:00 | v20260807 | **Season 9.5 start**; balance post of 08-04 goes live; new hero The Hood | **yes (regime start)** |
| 2026-08-13 09:00 | v20260813 | cosmetics, bug fixes | no |
| 2026-08-20 09:00 | v20260820 | cosmetics, VFX fixes (Loki x Hela team-up VFX only) | no |
| 2026-08-27 09:00 | v20260827 | cosmetics, VFX fixes | no |
| 2026-09-03 09:00 | v20260903 | cosmetics, map bug fix | no |
| 2026-09-08 | balance post | announced changes, **effective with v20260911** | not yet live |
| 2026-09-11 09:00 | v20260911 | **Season 10 start**; ~30+ hero changes, Scarlet Witch rework, new hero Gorr, team-up tweaks | **yes (regime end)** |

**Regime the model describes:** Season 9.5 balance state, 2026-08-07 09:00 UTC to 2026-09-11 09:00 UTC.
The development snapshot (play time 2026-08-07 11:12 to 2026-09-08 11:00 UTC) lies entirely inside it,
with no balance patch in between. No development matches are removed, so the baseline is not refitted.

**Predeclared rule applied:** the confirmation window ends at 2026-09-11 09:00 UTC. Eligible confirmation
matches are those PLAYED after 2026-09-08 11:00:32 UTC and before 2026-09-11 09:00:00 UTC, harvested by
the crawler until the stopping date. Matches played after the Season 10 patch are a different regime and
are excluded; any later candidate for Season 10 needs its own development and confirmation samples.

Secondary-source items (not relied on): The Hood's specific team-up partners, Gorr's team-up status, a
reported Loki restriction, the 09-25 map addition.

## Weekly win-rate check (supporting evidence only)

Per-hero win rates by starting week over 55 heroes, chi-square heterogeneity across weeks with
Bonferroni alpha 0.00091. Heroes below the corrected threshold:
The Hood (p=4.9e-10, range 1.3 pp).
The Hood is the hero released on 2026-08-07, so a drift as players learn him is expected and is not a patch
signature. No other hero shows a corrected-significant break; the largest weekly ranges are Human Torch
(3.2 pp, p=0.04) and Hawkeye
(2.5 pp). Consistent with the official record: one balance regime.

## Rank coverage

Lobby mean pre-match score (12 players) across 477,483 matches; individual scores across
5,729,796 player-rows and 879,317 distinct players.

| Statistic | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| Lobby mean score | 3898 | 4141 | 4379 | 4548 | 4707 | 4944 | 5126 | 5525 |
| Highest player in lobby | 3969 | 4216 | 4463 | 4627 | 4784 | 5023 | 5217 | 5657 |
| Individual player score | 3890 | 4133 | 4376 | 4547 | 4713 | 4950 | 5135 | 5657 |

Matches with lobby mean above 4,800 / 5,000 / 5,200 / 5,400: 70,045 / 14,722 / 2,410 / 128.
Distinct players above 5,000 / 5,250 / 5,500 / 5,750: 26,977 / 3,129 / 123 / 0.

**Supported range (declared):** lobby mean score 4141 to 4944 (5th to 95th percentile) is where
rank-dependent summaries are evaluated and reported. Lobby means above 5126 (top 1%, 14,722 matches above 5,000)
are thinly supported and no claim is made above 5,200. Individual players above 5,500 number 123 and none above 5,750,
so the top of the ranked ladder is outside the sample: this is direct coverage evidence, not an inference from privacy rates.
