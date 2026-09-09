"""Freeze the development snapshot (DB read, run once). Writes results/dev_snapshot/:
matches.csv (match_uid, play timestamp, camp0_win), filters.json, manifest.json (sha256 of every
file), and copies of the headline fit. The confirmation sample is defined against this snapshot's
maximum PLAY time, never against download time."""
import hashlib, json, os, shutil, subprocess, sys, time
import pandas as pd
sys.path.insert(0, ".")
import db
from apm import sample
out = "results/dev_snapshot"
conn = db.connect_readonly("data/rivals.db")
frame, rep = sample.build_sample(conn, 240)
frame[["match_uid", "timestamp", "camp0_win", "duration"]].to_csv(f"{out}/matches.csv", index=False)
fetched = dict(conn.execute("SELECT match_uid, fetched_at FROM matches"))
frame["fetched_at"] = frame.match_uid.map(fetched)
info = {
    "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "n_matches": int(len(frame)), "n_crawled_at_freeze": int(conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]),
    "filters": {"players_per_match": 12, "six_per_side": True, "no_draw_is_win_2": True, "scores_present": True,
                "every_player_positive_play_time": True, "forfeit_floor_seconds": 240},
    "exclusions": rep.__dict__ if hasattr(rep, "__dict__") else str(rep),
    "play_time_min_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(frame.timestamp.min()))),
    "play_time_max_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(frame.timestamp.max()))),
    "play_time_max_unix": int(frame.timestamp.max()),
    "download_time_max_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(frame.fetched_at.max()))),
    "season_code": 19, "season_label": "9.5",
    "headline_fit": {"tag": "W0_specAplus", "model_fit_commit": json.load(open("results/apm_run_meta_W0_specAplus.json"))["git_commit"]},
}
for f in ["apm_hero_table_W0_specAplus.csv", "apm_teamup_table_W0_specAplus.csv", "apm_run_meta_W0_specAplus.json", "apm_cov_W0_specAplus.npz"]:
    shutil.copy(f"results/{f}", f"{out}/{f}")
json.dump(info, open(f"{out}/filters.json", "w"), indent=1)
manifest = {f: hashlib.sha256(open(f"{out}/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir(out)) if f != "manifest.json"}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "exclusions"}, indent=1))
