"""Descriptive diagnostics for one hero (DB step): win rate, pick share, ban rate and
swap-off share by rank band, and win rate by week. Usage: python3 results/hero_diagnostic.py 1059 [benchmark ids...]"""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import db
from apm import sample
from apm.features import _starting_hero_per_player
t0 = time.time()
ids = [int(x) for x in sys.argv[1:]] or [1059]
conn = db.connect_readonly("data/rivals.db")
frame, _ = sample.build_sample(conn, 240)
uids = list(frame.match_uid); uset = set(uids)
names = dict(conn.execute("SELECT hero_id, name FROM hero_info"))
per = _starting_hero_per_player(conn, uids)                     # match_uid, camp, player_uid, hero_id
score = pd.DataFrame(conn.execute("SELECT match_uid, AVG(new_score-add_score), MIN(new_score-add_score), MAX(new_score-add_score) FROM match_players GROUP BY match_uid").fetchall(),
                     columns=["match_uid", "mean_score", "min_score", "max_score"]).set_index("match_uid")
score = score.loc[uids]
edges = np.quantile(score.mean_score, [0, .2, .4, .6, .8, 1.0]); edges[-1] += 1
score["band"] = np.digitize(score.mean_score, edges[1:-1])
band_label = {b: f"{edges[b]:.0f}-{edges[b+1]:.0f}" for b in range(5)}
y = frame.set_index("match_uid").camp0_win
per["win"] = np.where(per.camp == 0, per.match_uid.map(y), 1 - per.match_uid.map(y))
per["band"] = per.match_uid.map(score.band)
per["ts"] = per.match_uid.map(frame.set_index("match_uid").timestamp)
per["week"] = ((per.ts - per.ts.min()) // (7 * 86400)).astype(int)
# swap-off: starters with more than one hero row
nrows = pd.DataFrame(conn.execute("SELECT match_uid, player_uid, COUNT(*) FROM match_player_heroes GROUP BY match_uid, player_uid").fetchall(),
                     columns=["match_uid", "player_uid", "n_heroes"])
per = per.merge(nrows, on=["match_uid", "player_uid"], how="left")
bans = pd.DataFrame(conn.execute("SELECT match_uid, hero_id FROM match_bans WHERE hero_id>0").fetchall(), columns=["match_uid", "hero_id"])
bans = bans[bans.match_uid.isin(uset)]
matches_with_bans = bans.match_uid.nunique()
slots_by_band = per.groupby("band").size()
print(f"[{time.time()-t0:.0f}s] sample {len(uids):,} matches; matches with any ban {matches_with_bans:,} ({100*matches_with_bans/len(uids):.1f}%)")
print("rank bands (match mean pre-match score, quintiles):", band_label)
for h in ids:
    d = per[per.hero_id == h]
    print(f"\n=== {names[h]} (id {h}): {len(d):,} starts, raw side win rate {100*d.win.mean():.1f}%, swapped off {100*(d.n_heroes>1).mean():.1f}% of starts")
    g = d.groupby("band").agg(starts=("win", "size"), winrate=("win", "mean"), swap_off=("n_heroes", lambda s: (s > 1).mean()))
    g["pick_share_pct"] = 100 * g.starts / slots_by_band
    bm = bans[bans.hero_id == h].groupby(bans.match_uid.map(score.band)).match_uid.nunique()
    g["ban_rate_pct"] = 100 * bm.reindex(g.index).fillna(0) / score.groupby("band").size()
    g["winrate"] = 100 * g.winrate; g["swap_off"] = 100 * g.swap_off
    g.index = [band_label[b] for b in g.index]
    print(g.round(1).to_string())
    w = d.groupby("week").agg(starts=("win", "size"), winrate=("win", "mean")); w["winrate"] = 100 * w.winrate
    print("by week:", w.round(1).to_dict("index"))
print(f"[{time.time()-t0:.0f}s] done")
