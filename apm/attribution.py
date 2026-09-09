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

RULES = ("W0", "W1", "W2")


def attribution_weights(conn, match_uids, rule):
    if rule not in RULES:
        raise ValueError(f"unknown attribution rule {rule!r}; expected one of {RULES}")
    if rule == "W0":
        return _starting_lineup_weights(conn, match_uids)
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


def _starting_lineup_weights(conn, match_uids):
    """W0: the hero each player STARTED on, weight 1.0.

    `match_player_heroes` is not WITHOUT ROWID, so its implicit rowid preserves
    the order the API returned the hero array in, and that array is
    chronological by FIRST APPEARANCE. Entry 1 is therefore the starting hero.
    (Verified over 482,997 matches: last-entry == cur_hero_id sits flat at
    73.6-77.8% across 2..6 heroes while the chance baseline collapses 50%->17%,
    and where they differ cur_hero_id is still in the array 97.08% of the time,
    at position 1 in ~63% of those -- the A->B->A swap-back that the
    (match_uid, player_uid, hero_id) primary key cannot give a second row.)

    Two things differ from W1/W2 and both are deliberate:

    - **No `play_time > 0` filter.** That filter is right for W1/W2, where a
      hero never played was never deployed. It is wrong here: a player who
      swaps away in the opening seconds records ~0 time on the hero they
      started on, and dropping it would report the swap *target* as the start,
      reintroducing exactly the outcome-dependence W0 exists to remove.
    - **MIN(rowid) rather than a window function.** Same first row, without
      sorting 11.7M rows.

    W0 also makes a team's lineup six genuinely distinct heroes, because hero
    uniqueness binds at match start. W2 has no such guarantee -- two teammates
    can each spend most of a match on the same hero via sequential swaps, which
    collapses a W2 roster to five for ~2% of team-instances.
    """
    if not match_uids:
        return pd.DataFrame(columns=["match_uid", "camp", "hero_id", "weight"])

    rows = conn.execute(
        "SELECT h.match_uid, p.camp, h.player_uid, h.hero_id "
        "FROM match_player_heroes h "
        "JOIN (SELECT match_uid, player_uid, MIN(rowid) AS first_rid "
        "      FROM match_player_heroes GROUP BY match_uid, player_uid) f "
        "  ON f.first_rid = h.rowid "
        "JOIN match_players p ON p.match_uid = h.match_uid "
        "                    AND p.player_uid = h.player_uid"
    ).fetchall()
    frame = pd.DataFrame(rows, columns=["match_uid", "camp", "player_uid", "hero_id"])
    frame = frame[frame["match_uid"].isin(match_uids)]
    if frame.empty:
        return pd.DataFrame(columns=["match_uid", "camp", "hero_id", "weight"])

    frame = frame.assign(weight=1.0)
    return (
        frame.groupby(["match_uid", "camp", "hero_id"], as_index=False)["weight"].sum()
    )
