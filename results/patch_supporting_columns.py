"""One-time migration: add the supporting columns to CSVs produced BEFORE
run_apm.py emitted them natively. Same SQL as run_apm.py, applied to existing
files so no refit is needed. Safe to re-run; it overwrites the same columns."""
import sqlite3, sys, time
import pandas as pd
t0 = time.time()
c = sqlite3.connect("file:data/rivals.db?mode=ro", uri=True, timeout=1800)
names = dict(c.execute("SELECT hero_id, name FROM hero_info"))
starts = dict(c.execute("""SELECT h.hero_id, COUNT(*) FROM match_player_heroes h
  JOIN (SELECT match_uid, player_uid, MIN(rowid) AS first_rid FROM match_player_heroes
        GROUP BY match_uid, player_uid) f ON f.first_rid = h.rowid GROUP BY h.hero_id"""))
total = sum(starts.values()) or 1
raw = dict(c.execute("""SELECT h.hero_id, 100.0*SUM(h.play_time*mp.is_win)/NULLIF(SUM(h.play_time),0)
  FROM match_player_heroes h JOIN match_players mp ON mp.match_uid=h.match_uid AND mp.player_uid=h.player_uid
  WHERE mp.is_win IN (0,1) AND h.play_time>0 GROUP BY h.hero_id"""))
pairs = {}
for tid, hid, anc in c.execute("SELECT teamup_id, hero_id, is_anchor FROM teamup_heroes ORDER BY teamup_id, is_anchor DESC"):
    pairs.setdefault(tid, []).append(names.get(hid, f"hero {hid}"))
print(f"[{time.time()-t0:.0f}s] statistics computed", flush=True)
for tag in sys.argv[1:] or ("W0_specAplus", "W0_full", "W1_full", "W2_full"):
    p = f"results/apm_hero_table_{tag}.csv"; h = pd.read_csv(p)
    h["n_start"] = h.hero_id.map(starts).fillna(0).astype(int)
    h["pick_pct"] = 100.0 * h.n_start / total
    h["raw_wr"] = h.hero_id.map(raw)
    h.to_csv(p, index=False)
    tp = f"results/apm_teamup_table_{tag}.csv"
    try:
        t = pd.read_csv(tp)
        t["anchor"] = t.teamup_id.map(lambda x: (pairs.get(x) or [None])[0])
        t["partner"] = t.teamup_id.map(lambda x: pairs[x][1] if len(pairs.get(x, [])) > 1 else None)
        t.to_csv(tp, index=False)
    except FileNotFoundError:
        pass
    print(f"  patched {tag}", flush=True)
print(f"[{time.time()-t0:.0f}s] done")
