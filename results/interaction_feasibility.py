"""How identifiable are pairwise hero interactions? (DB step, run once.)

Counts, on STARTING lineups (W0), how often each unordered hero pair appears
(a) on the same team and (b) on opposite teams, and the raw win-rate signal for
two user-named examples. Writes results/interaction_feasibility.json and the
count matrices to results/interaction_counts.npz.
"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import db
from apm import sample
from apm.features import _starting_hero_per_player

t0 = time.time()
conn = db.connect_readonly("data/rivals.db")
frame, _ = sample.build_sample(conn, 240)
per = _starting_hero_per_player(conn, list(frame.match_uid))
info = {h: (n, r) for h, n, r in conn.execute("SELECT hero_id, name, role FROM hero_info")}
heroes = sorted(per.hero_id.unique()); idx = {h: i for i, h in enumerate(heroes)}; K = len(heroes)
names = [info[h][0] for h in heroes]; roles = [info[h][1] for h in heroes]
per = per.sort_values(["match_uid", "camp"])
g = per.groupby(["match_uid", "camp"]).hero_id.apply(list)
assert (g.map(len) == 6).all(), "every starting side must have six heroes"
uids = frame.match_uid.to_numpy()
side0 = np.array([[idx[h] for h in g[(u, 0)]] for u in uids]); side1 = np.array([[idx[h] for h in g[(u, 1)]] for u in uids])
y = frame.set_index("match_uid").loc[uids, "camp0_win"].to_numpy().astype(float)
print(f"[{time.time()-t0:.0f}s] {len(uids):,} matches, {K} heroes", flush=True)

same = np.zeros((K, K), int); same_w = np.zeros((K, K))         # co-occurrence and wins, same team
cross = np.zeros((K, K), int); cross_w = np.zeros((K, K))       # cross[a,b]: a on a side, b on the other; wins for a's side
for side, win in ((side0, y), (side1, 1 - y)):
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = side[:, i], side[:, j]
            np.add.at(same, (a, b), 1); np.add.at(same, (b, a), 1)
            np.add.at(same_w, (a, b), win); np.add.at(same_w, (b, a), win)
for i in range(6):
    for j in range(6):
        a, b = side0[:, i], side1[:, j]
        np.add.at(cross, (a, b), 1); np.add.at(cross_w, (a, b), y)          # a on side 0 vs b on side 1
        np.add.at(cross, (b, a), 1); np.add.at(cross_w, (b, a), 1 - y)      # same pair, b's side perspective
iu = np.triu_indices(K, 1)
same_pairs = same[iu]; cross_pairs = cross[iu]      # cross is symmetric in total count
def summary(v):
    return {"pairs": int(len(v)), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "ge_1000": int((v >= 1000).sum()), "ge_2500": int((v >= 2500).sum()), "ge_10000": int((v >= 10000).sum()),
            "ge_40000": int((v >= 40000).sum()), "share_of_instances_in_pairs_ge_2500": float(v[v >= 2500].sum() / v.sum())}
def find(name):
    return names.index(name)
ex = {}
bp, thing = find("Black Panther"), find("The Thing")
jeff, dino = find("Jeff The Land Shark"), find("Devil Dinosaur")
# raw signal: BP's side win rate when The Thing starts on the enemy side vs when it does not
bp0 = (side0 == bp).any(1); bp1 = (side1 == bp).any(1)
th0 = (side0 == thing).any(1); th1 = (side1 == thing).any(1)
bp_vs_thing = np.concatenate([y[bp0 & th1], (1 - y)[bp1 & th0]]); bp_no_thing = np.concatenate([y[bp0 & ~th1], (1 - y)[bp1 & ~th0]])
ex["black_panther_vs_thing"] = {"n_matches": int(len(bp_vs_thing)), "bp_side_winrate": float(bp_vs_thing.mean()),
                                "n_without_thing": int(len(bp_no_thing)), "bp_side_winrate_without": float(bp_no_thing.mean())}
jf0 = (side0 == jeff).any(1); jf1 = (side1 == jeff).any(1); dn0 = (side0 == dino).any(1); dn1 = (side1 == dino).any(1)
with_d = np.concatenate([y[jf0 & dn0], (1 - y)[jf1 & dn1]]); without = np.concatenate([y[jf0 & ~dn0], (1 - y)[jf1 & ~dn1]])
ex["jeff_with_dino"] = {"n_team_instances": int(len(with_d)), "winrate": float(with_d.mean()),
                        "n_without": int(len(without)), "winrate_without": float(without.mean())}
# implied precision for a pairwise contrast with n co-occurrences: SE ~ 2/sqrt(n) logit ~ 50/sqrt(n) pp
out = {"n_matches": int(len(uids)), "n_heroes": K,
       "same_team": summary(same_pairs), "cross_team": summary(cross_pairs),
       "se_pp_at": {str(n): round(50 / np.sqrt(n), 2) for n in (500, 1000, 2500, 10000, 40000)},
       "examples": ex,
       "top_same": [(names[a], names[b], int(same[a, b])) for a, b in zip(*iu) for _ in [0]][:0]}
order = np.argsort(-same_pairs)[:10]; out["top_same"] = [(names[iu[0][k]], names[iu[1][k]], int(same_pairs[k])) for k in order]
order = np.argsort(-cross_pairs)[:10]; out["top_cross"] = [(names[iu[0][k]], names[iu[1][k]], int(cross_pairs[k])) for k in order]
# by role combination
rc = {}
for k in range(len(iu[0])):
    key = "-".join(sorted([roles[iu[0][k]], roles[iu[1][k]]]))
    rc.setdefault(key, []).append(int(same_pairs[k]))
out["same_team_by_role_pair"] = {k: {"pairs": len(v), "median": float(np.median(v)), "ge_2500": int(sum(x >= 2500 for x in v))} for k, v in rc.items()}
json.dump(out, open("results/interaction_feasibility.json", "w"), indent=1)
np.savez("results/interaction_counts.npz", same=same, same_w=same_w, cross=cross, cross_w=cross_w, hero_ids=np.array(heroes), names=np.array(names), roles=np.array(roles))
print(json.dumps({k: v for k, v in out.items() if k not in ("top_same", "top_cross")}, indent=1))
print("top same-team:", out["top_same"][:5]); print("top cross-team:", out["top_cross"][:5])
print(f"[{time.time()-t0:.0f}s] done", flush=True)
