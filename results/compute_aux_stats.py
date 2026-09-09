"""Compute every non-fit statistic the report needs, ONCE, and save it.

The report generator (make_report.py) must never touch the database: it reads
CSV/JSON and emits TeX. But several numbers in the prose are not products of a
model fit -- win rate by tank count, the composition-shape distribution, the
camp check, the privacy-by-rank table. Those come from here.

Run after any change to the underlying data:  python3 results/compute_aux_stats.py
Outputs: results/comp_shapes.csv, results/tank_winrate.csv, results/aux_stats.json
"""
import json, re, sqlite3, sys, time
import pandas as pd

t0 = time.time()
c = sqlite3.connect("file:data/rivals.db?mode=ro", uri=True, timeout=1800)

# --- composition shapes (dominant hero per player), complete table ---
shapes = pd.DataFrame(c.execute("""
WITH dom AS (
  SELECT h.match_uid, h.player_uid, h.hero_id,
         ROW_NUMBER() OVER (PARTITION BY h.match_uid,h.player_uid
                            ORDER BY h.play_time DESC, h.hero_id) rn
  FROM match_player_heroes h WHERE h.play_time>0),
team AS (
  SELECT mp.match_uid, mp.camp,
         SUM(hi.role='Tank') t, SUM(hi.role='Damage') d, SUM(hi.role='Support') s,
         COUNT(*) n, MAX(mp.is_win) won
  FROM match_players mp
  JOIN dom ON dom.match_uid=mp.match_uid AND dom.player_uid=mp.player_uid AND dom.rn=1
  JOIN hero_info hi ON hi.hero_id=dom.hero_id
  WHERE mp.is_win IN (0,1) GROUP BY mp.match_uid, mp.camp)
SELECT t||'-'||d||'-'||s AS shape, t AS tanks, COUNT(*) AS teams, 100.0*AVG(won) AS winrate
FROM team WHERE n=6 GROUP BY shape ORDER BY teams DESC""").fetchall(),
    columns=["shape", "tanks", "teams", "winrate"])
shapes.to_csv("results/comp_shapes.csv", index=False)
print(f"[{time.time()-t0:.0f}s] {len(shapes)} shapes over {shapes.teams.sum():,} team-instances", flush=True)

tank = shapes.groupby("tanks").apply(
    lambda g: pd.Series({"teams": int(g.teams.sum()),
                         "winrate": float((g.teams * g.winrate).sum() / g.teams.sum())})).reset_index()
tank.to_csv("results/tank_winrate.csv", index=False)

# --- camp indexing ---
done_by_camp = dict(c.execute("""SELECT mp.camp, COUNT(*) FROM match_players mp
    JOIN players p ON p.uid=mp.player_uid WHERE p.crawl_status='done' GROUP BY mp.camp"""))
camp_map = pd.DataFrame(c.execute("""SELECT m.map_id, COUNT(*) n,
    100.0*AVG(CASE WHEN mp.is_win=1 THEN 1.0 ELSE 0.0 END) wr
    FROM matches m JOIN match_players mp ON mp.match_uid=m.match_uid AND mp.camp=0
    WHERE mp.is_win IN (0,1) GROUP BY m.map_id HAVING n>2000""").fetchall(),
    columns=["map_id", "matches", "camp0_wr"])

# --- privacy by rank band (attempted players only) ---
priv = pd.DataFrame(c.execute("""SELECT CAST(latest_known_score/250 AS INT)*250 band,
    COUNT(*) n, 100.0*SUM(crawl_status='skipped_private')/COUNT(*) pct
    FROM players WHERE crawl_status IN ('done','skipped_private') AND latest_known_score IS NOT NULL
    GROUP BY band HAVING n>=15 ORDER BY band""").fetchall(), columns=["band", "attempted", "private_pct"])

total = int(shapes.teams.sum())
aux = json.load(open("results/aux_stats.json")) if __import__("os").path.exists("results/aux_stats.json") else {}
aux.update({
    "team_instances": total,
    "share_2_2_2": round(100 * float(shapes.loc[shapes.shape == "2-2-2", "teams"].sum()) / total, 1),
    "tank_winrate": {str(int(r.tanks)): {"teams": int(r.teams), "winrate": round(r.winrate, 2)}
                     for _, r in tank.iterrows()},
    "camp0_share_crawled": round(100 * done_by_camp.get(0, 0) / max(sum(done_by_camp.values()), 1), 2),
    "camp0_wr_min": round(float(camp_map.camp0_wr.min()), 2),
    "camp0_wr_max": round(float(camp_map.camp0_wr.max()), 2),
    "n_maps": int(len(camp_map)),
    "privacy_pct_above_5000": round(float(priv.loc[priv.band >= 5000, "private_pct"].mean()), 1),
    "privacy_pct_below_4000": round(float(priv.loc[priv.band < 4000, "private_pct"].mean()), 1),
})
for thresh in (500, 100):
    own = shapes[shapes.teams >= thresh]; pooled = shapes[shapes.teams < thresh]
    aux[f"shapes_own_at_{thresh}"] = int(len(own))
    aux[f"shapes_pooled_at_{thresh}"] = int(len(pooled))
    aux[f"pooled_teams_at_{thresh}"] = int(pooled.teams.sum())
    aux[f"pooled_pct_at_{thresh}"] = round(100 * float(pooled.teams.sum()) / total, 3)
    aux[f"pooled_wr_min_at_{thresh}"] = round(float(pooled.winrate.min()), 1) if len(pooled) else None
    aux[f"pooled_wr_max_at_{thresh}"] = round(float(pooled.winrate.max()), 1) if len(pooled) else None
json.dump(aux, open("results/aux_stats.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] aux_stats.json written: "
      f"{aux['shapes_own_at_100']} shapes own at 100, pooled {aux['pooled_pct_at_100']}%; "
      f"camp0 {aux['camp0_wr_min']}-{aux['camp0_wr_max']}% over {aux['n_maps']} maps")
