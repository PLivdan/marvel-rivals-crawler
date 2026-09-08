"""Analysis-sample construction with per-rule exclusion accounting.

Filters are applied in a fixed order and each one's cost is counted, because a
published table has to state what was thrown away and why. Rules follow the
design spec section 4.
"""

from dataclasses import dataclass

import pandas as pd

DEFAULT_FORFEIT_FLOOR_SECONDS = 240


@dataclass
class ExclusionReport:
    starting: int
    dropped_roster: int
    dropped_draw: int
    dropped_missing_score: int
    dropped_missing_playtime: int
    dropped_short: int
    remaining: int


def build_sample(conn, forfeit_floor_seconds=DEFAULT_FORFEIT_FLOOR_SECONDS):
    """Return (match-level frame, exclusion report)."""
    all_uids = {r[0] for r in conn.execute("SELECT match_uid FROM matches")}

    complete = {
        r[0]
        for r in conn.execute(
            "SELECT match_uid FROM match_players GROUP BY match_uid "
            "HAVING COUNT(*)=12 AND SUM(camp=0)=6 AND SUM(camp=1)=6"
        )
    }
    drew = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT match_uid FROM match_players WHERE is_win=2")
    }
    no_score = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT match_uid FROM match_players "
            "WHERE add_score IS NULL OR new_score IS NULL")
    }
    # A match qualifies only if every one of its 12 players has positive
    # play time; counting distinct players with play time is cheaper than
    # a correlated per-player existence check.
    timed = {
        r[0] for r in conn.execute(
            "SELECT match_uid FROM match_player_heroes WHERE play_time>0 "
            "GROUP BY match_uid HAVING COUNT(DISTINCT player_uid)=12")
    }
    long_enough = {
        r[0] for r in conn.execute(
            "SELECT match_uid FROM matches WHERE match_play_duration IS NOT NULL "
            "AND match_play_duration>=?", (forfeit_floor_seconds,))
    }

    after_roster = all_uids & complete
    after_draw = after_roster - drew
    after_score = after_draw - no_score
    after_time = after_score & timed
    kept = after_time & long_enough

    report = ExclusionReport(
        starting=len(all_uids),
        dropped_roster=len(all_uids) - len(after_roster),
        dropped_draw=len(after_roster) - len(after_draw),
        dropped_missing_score=len(after_draw) - len(after_score),
        dropped_missing_playtime=len(after_score) - len(after_time),
        dropped_short=len(after_time) - len(kept),
        remaining=len(kept),
    )

    rows = conn.execute(
        "SELECT m.match_uid, m.match_time_stamp, m.match_play_duration, "
        "       MAX(CASE WHEN p.camp=0 THEN p.is_win END) "
        "FROM matches m JOIN match_players p ON p.match_uid=m.match_uid "
        "GROUP BY m.match_uid"
    ).fetchall()
    frame = pd.DataFrame(
        [r for r in rows if r[0] in kept],
        columns=["match_uid", "timestamp", "duration", "camp0_win"],
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["match_uid", "camp0_win", "duration", "timestamp"]], report


def temporal_holdout(sample, holdout_days=7):
    """Split on time, never at random: a random split leaks meta drift."""
    if sample.empty:
        return sample.copy(), sample.copy()
    cutoff = sample["timestamp"].max() - holdout_days * 86_400
    train = sample[sample["timestamp"] <= cutoff].reset_index(drop=True)
    test = sample[sample["timestamp"] > cutoff].reset_index(drop=True)
    return train, test
