"""Design-matrix assembly for the adjusted plus-minus model.

Column layout, in order: intercept, hero free parameters, team-up contrasts,
composition free parameters, skill differential. The hero and composition
blocks are rank-deficient (each sums to zero across levels within a row) and so
are reduced onto sum-to-zero bases; the recovered per-level effects come back
via apm.contrasts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apm import attribution, contrasts


@dataclass
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    column_names: list
    hero_ids: list
    hero_basis: np.ndarray
    hero_slice: slice
    teamup_ids: list
    match_uids: list


def hero_contrast_frame(weights):
    """Pivot attribution weights to match x hero, camp 0 minus camp 1."""
    signed = weights.assign(
        signed=np.where(weights["camp"] == 0, weights["weight"], -weights["weight"])
    )
    return signed.pivot_table(
        index="match_uid", columns="hero_id", values="signed",
        aggfunc="sum", fill_value=0.0,
    )


def _dominant_hero_per_player(conn, match_uids):
    """Each player's single most-played hero, one row per player.

    This mirrors apm.attribution's W2 rule up to, but not including, its
    final match/camp/hero aggregation (attribution.py's closing
    `.groupby(["match_uid","camp","hero_id"])["weight"].sum()`).
    attribution_weights needs that aggregation -- its own consumer, the hero
    contrast columns, needs weights that sum to 6.0 per team regardless of
    how many distinct heroes are on it, and that is correct for its purpose.
    But it is exactly what destroys player identity when two teammates share
    a dominant hero: two players' 1.0-weight rows for the same hero_id merge
    into one row worth 2.0, and a team of six real player-slots reads back
    as five (or fewer) distinct heroes. Composition shape and team-up
    membership both need all six player-slots intact, so this stops one
    step short of that collapse and returns one row per player instead. The
    query and tie-break duplicate attribution.py's W2 branch deliberately,
    to keep attribution_weights' own behaviour (pinned by its tests) alone.
    """
    if not match_uids:
        return pd.DataFrame(columns=["match_uid", "camp", "player_uid", "hero_id"])

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
        return pd.DataFrame(columns=["match_uid", "camp", "player_uid", "hero_id"])

    keys = ["match_uid", "player_uid"]
    best = frame.groupby(keys)["play_time"].transform("max")
    # Same tie-break as attribution.py's W2 branch: a play-time tie must
    # still resolve to exactly one hero per player, or a roster could come
    # back with more than six entries.
    frame = frame[frame["play_time"] == best]
    frame = frame.drop_duplicates(subset=keys, keep="first")
    return frame[["match_uid", "camp", "player_uid", "hero_id"]]


def _lineup_rosters(conn, match_uids):
    """Six player-slot heroes per (match, camp), duplicates preserved.

    Team-ups and composition need a definite lineup, so these always use W2
    regardless of the hero rule. Two players sharing a dominant hero must
    still occupy two slots here -- a set would silently drop one, so this
    returns a list with one entry per player. Callers that only need
    presence (team-up membership) build a set from the list; callers that
    need counts (composition shape) iterate the list directly so a
    duplicated hero counts twice.
    """
    per_player = _dominant_hero_per_player(conn, match_uids)
    return {
        key: list(group["hero_id"])
        for key, group in per_player.groupby(["match_uid", "camp"])
    }


def build_design(conn, sample, attribution_rule="W1", min_shape_count=500):
    # `order` may legitimately contain the same match twice: a cluster
    # bootstrap resamples players with replacement, so their matches repeat.
    # Every row-wise loop below must therefore index by position, never by a
    # {uid: row} dict, which would keep only the last occurrence.
    match_uids = set(sample["match_uid"])
    order = list(sample["match_uid"])
    n = len(order)

    weights = attribution.attribution_weights(conn, match_uids, attribution_rule)
    hero_frame = hero_contrast_frame(weights).reindex(order, fill_value=0.0)
    hero_ids = [int(h) for h in hero_frame.columns]
    hero_raw = hero_frame.to_numpy(dtype=float)
    hero_basis = contrasts.sum_to_zero_basis(len(hero_ids))
    hero_block = contrasts.reduce_design(hero_raw, hero_basis)

    rosters = _lineup_rosters(conn, match_uids)
    lineup_sets = {key: set(heroes) for key, heroes in rosters.items()}
    roles = dict(conn.execute("SELECT hero_id, role FROM hero_info"))

    teamups = {}
    for tid, hid in conn.execute("SELECT teamup_id, hero_id FROM teamup_heroes"):
        teamups.setdefault(tid, set()).add(hid)
    teamup_ids = sorted(teamups)
    teamup_block = np.zeros((n, len(teamup_ids)))
    for j, tid in enumerate(teamup_ids):
        members = teamups[tid]
        for row, uid in enumerate(order):
            in0 = members <= lineup_sets.get((uid, 0), set())
            in1 = members <= lineup_sets.get((uid, 1), set())
            teamup_block[row, j] = float(in0) - float(in1)

    def shape_of(uid, camp):
        # A list, not a set: two player-slots sharing a hero must both count
        # toward the role total, or a true 2-2-2 misreads as e.g. 1-2-2.
        heroes = rosters.get((uid, camp), [])
        counts = [0, 0, 0]
        for hero in heroes:
            role = roles.get(hero)
            if role == "Tank":
                counts[0] += 1
            elif role == "Damage":
                counts[1] += 1
            elif role == "Support":
                counts[2] += 1
        return "{}-{}-{}".format(*counts)

    shapes0 = [shape_of(uid, 0) for uid in order]
    shapes1 = [shape_of(uid, 1) for uid in order]
    counts = pd.Series(shapes0 + shapes1).value_counts()
    common = sorted(counts[counts >= min_shape_count].index)
    labels = common + ["other"]
    canon = lambda s: s if s in common else "other"

    shape_raw = np.zeros((n, len(labels)))
    for i, (s0, s1) in enumerate(zip(shapes0, shapes1)):
        shape_raw[i, labels.index(canon(s0))] += 1.0
        shape_raw[i, labels.index(canon(s1))] -= 1.0
    if len(labels) >= 2:
        shape_basis = contrasts.sum_to_zero_basis(len(labels))
        shape_block = contrasts.reduce_design(shape_raw, shape_basis)
        shape_names = [f"shape_free_{i}" for i in range(shape_block.shape[1])]
    else:
        shape_block = np.zeros((n, 0))
        shape_names = []

    score_rows = conn.execute(
        "SELECT match_uid, camp, AVG(new_score-add_score) FROM match_players "
        "WHERE add_score IS NOT NULL GROUP BY match_uid, camp"
    ).fetchall()
    pre = {(r[0], r[1]): r[2] for r in score_rows}
    skill = np.array(
        [[pre.get((uid, 0), 0.0) - pre.get((uid, 1), 0.0)] for uid in order]
    )

    intercept = np.ones((n, 1))
    X = np.hstack([intercept, hero_block, teamup_block, shape_block, skill])
    names = (
        ["intercept"]
        + [f"hero_free_{i}" for i in range(hero_block.shape[1])]
        + [f"teamup_{t}" for t in teamup_ids]
        + shape_names
        + ["skill_diff"]
    )
    hero_slice = slice(1, 1 + hero_block.shape[1])
    y = sample["camp0_win"].to_numpy(dtype=int)
    return DesignMatrix(X, y, names, hero_ids, hero_basis, hero_slice,
                        teamup_ids, order)
