"""Hero attribution: turning a player-match into hero weights.

No pick data exists (is_pick is always 0), and roughly 55% of player-matches
involve a mid-match swap whose timing is endogenous to the match state. So
"which hero did this player play" is a modelling choice, not an observation,
and it is exposed here as a parameter rather than baked into the feature code.

W1 (primary) splits a player's slot across heroes in proportion to play time.
It is the only rule that does not condition on the outcome: taking the final
hero blames losses on whoever was swapped to, and keeping only non-swappers
selects on winning (65.6% vs 37.3% in the full data).

W2 assigns the whole slot to the longest-played hero. It discards swap
information but yields integer lineups, which team-up and composition features
require because those need a definite set of six.
"""

import pandas as pd

RULES = ("W1", "W2")


def attribution_weights(conn, match_uids, rule):
    if rule not in RULES:
        raise ValueError(f"unknown attribution rule {rule!r}; expected one of {RULES}")
    if not match_uids:
        return pd.DataFrame(columns=["match_uid", "camp", "hero_id", "weight"])

    rows = conn.execute(
        "SELECT h.match_uid, p.camp, h.player_uid, h.hero_id, h.play_time "
        "FROM match_player_heroes h "
        "JOIN match_players p ON p.match_uid=h.match_uid "
        "                    AND p.player_uid=h.player_uid "
        "WHERE h.play_time>0"
    ).fetchall()
    frame = pd.DataFrame(
        rows, columns=["match_uid", "camp", "player_uid", "hero_id", "play_time"]
    )
    frame = frame[frame["match_uid"].isin(match_uids)]
    if frame.empty:
        return pd.DataFrame(columns=["match_uid", "camp", "hero_id", "weight"])

    keys = ["match_uid", "player_uid"]
    if rule == "W1":
        total = frame.groupby(keys)["play_time"].transform("sum")
        frame = frame.assign(weight=frame["play_time"] / total)
    else:
        best = frame.groupby(keys)["play_time"].transform("max")
        # A tie would otherwise give a player two full slots and break the
        # sums-to-six invariant, so keep exactly one row per player.
        frame = frame[frame["play_time"] == best]
        frame = frame.drop_duplicates(subset=keys, keep="first").assign(weight=1.0)

    return (
        frame.groupby(["match_uid", "camp", "hero_id"], as_index=False)["weight"]
        .sum()
    )
