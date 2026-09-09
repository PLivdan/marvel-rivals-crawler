"""Stage 2 data pieces (DB read on the frozen snapshot only):
(a) rank coverage: lobby mean score and individual player score quantiles, counts near the top;
(b) weekly hero win-rate break check as SUPPORTING evidence for the patch audit (cannot establish patch dates).
Writes results/rank_coverage.json, results/rank_coverage.csv, results/weekly_hero_winrate.csv, results/weekly_break_summary.json."""
import json, sys, time
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
import db
from apm.features import _starting_hero_per_player
t0 = time.time()
snap = pd.read_csv("results/dev_snapshot/matches.csv"); uids = list(snap.match_uid); uset = set(uids)
conn = db.connect_readonly("data/rivals.db")
mp = pd.DataFrame(conn.execute("SELECT match_uid, player_uid, camp, new_score-add_score AS pre FROM match_players").fetchall(),
                  columns=["match_uid", "player_uid", "camp", "pre"])
mp = mp[mp.match_uid.isin(uset)]
lobby = mp.groupby("match_uid").pre.agg(["mean", "min", "max"]).loc[uids]
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
cov = {"n_matches": len(uids), "n_player_rows": int(len(mp)), "n_distinct_players": int(mp.player_uid.nunique()),
       "lobby_mean_quantiles": {str(p): float(lobby["mean"].quantile(p)) for p in q},
       "lobby_max_player_quantiles": {str(p): float(lobby["max"].quantile(p)) for p in q},
       "player_score_quantiles": {str(p): float(mp.pre.quantile(p)) for p in q},
       "matches_with_lobby_mean_above": {str(t): int((lobby["mean"] > t).sum()) for t in (4800, 5000, 5200, 5400)},
       "matches_with_any_player_above": {str(t): int((lobby["max"] > t).sum()) for t in (5000, 5250, 5500, 5750, 6000)},
       "player_rows_above": {str(t): int((mp.pre > t).sum()) for t in (5000, 5250, 5500, 5750, 6000)},
       "distinct_players_above": {str(t): int(mp[mp.pre > t].player_uid.nunique()) for t in (5000, 5250, 5500, 5750, 6000)},
       "dev_tercile_boundaries_lobby_mean": [float(lobby["mean"].quantile(1/3)), float(lobby["mean"].quantile(2/3))]}
json.dump(cov, open("results/rank_coverage.json", "w"), indent=1)
rows = [{"statistic": "lobby mean score", **{f"p{int(100*p)}": round(cov["lobby_mean_quantiles"][str(p)]) for p in q}},
        {"statistic": "highest player in lobby", **{f"p{int(100*p)}": round(cov["lobby_max_player_quantiles"][str(p)]) for p in q}},
        {"statistic": "individual player score", **{f"p{int(100*p)}": round(cov["player_score_quantiles"][str(p)]) for p in q}}]
pd.DataFrame(rows).to_csv("results/rank_coverage.csv", index=False)
print(json.dumps({k: cov[k] for k in ("lobby_mean_quantiles", "matches_with_lobby_mean_above", "matches_with_any_player_above", "distinct_players_above")}, indent=1))

# ---- weekly win rates by starting hero -----------------------------------------------------
per = _starting_hero_per_player(conn, uids)
y = snap.set_index("match_uid").camp0_win; ts = snap.set_index("match_uid").timestamp
per["win"] = np.where(per.camp == 0, per.match_uid.map(y), 1 - per.match_uid.map(y))
per["day"] = ((per.match_uid.map(ts) - ts.min()) // 86400).astype(int); per["week"] = per.day // 7
names = dict(conn.execute("SELECT hero_id, name FROM hero_info"))
g = per.groupby(["hero_id", "week"]).win.agg(["size", "mean"]).reset_index()
g["name"] = g.hero_id.map(names); g.to_csv("results/weekly_hero_winrate.csv", index=False)
# chi-square heterogeneity of win rate across weeks per hero, and max week-to-week jump in pp
summ = []
for h, d in g.groupby("hero_id"):
    d = d[d["size"] >= 500]
    if len(d) < 3: continue
    wins = (d["size"] * d["mean"]).to_numpy(); tot = d["size"].to_numpy()
    chi2, p, *_ = stats.chi2_contingency(np.array([wins, tot - wins]))
    summ.append({"hero_id": int(h), "name": names[h], "weeks": int(len(d)), "starts": int(tot.sum()), "chi2": float(chi2), "p": float(p),
                 "max_abs_weekly_shift_pp": float(100 * np.max(np.abs(np.diff(d["mean"].to_numpy())))),
                 "range_pp": float(100 * (d["mean"].max() - d["mean"].min()))})
S = pd.DataFrame(summ).sort_values("p")
S.to_csv("results/weekly_break_summary.csv", index=False)
daily = per.groupby("day").win.size()
out = {"heroes_tested": int(len(S)), "bonferroni_alpha": 0.05 / max(len(S), 1),
       "heroes_with_heterogeneity_p_below_bonferroni": S[S.p < 0.05 / max(len(S), 1)][["name", "p", "range_pp", "max_abs_weekly_shift_pp"]].to_dict("records"),
       "largest_ranges": S.sort_values("range_pp", ascending=False).head(8)[["name", "range_pp", "p", "starts"]].to_dict("records"),
       "starts_per_day": {int(k): int(v) for k, v in daily.items()}}
json.dump(out, open("results/weekly_break_summary.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("heroes_tested", "heroes_with_heterogeneity_p_below_bonferroni", "largest_ranges")}, indent=1))
print(f"[{time.time()-t0:.0f}s] done")
