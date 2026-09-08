import numpy as np
import pytest

import db
from apm import features


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1011,'A','Tank')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1022,'B','Tank')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1033,'C','Damage')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1044,'D','Damage')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1055,'E','Support')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1066,'F','Support')")
    conn.commit()
    return conn


def add_match(conn, uid, camp0_heroes, camp1_heroes, camp0_win=1,
              camp0_score=4500.0, camp1_score=4500.0, timestamp=1_787_000_000):
    conn.execute(
        "INSERT INTO matches (match_uid, match_time_stamp, match_play_duration) "
        "VALUES (?,?,700.0)", (uid, timestamp))
    for i, hero in enumerate(camp0_heroes + camp1_heroes):
        camp = 0 if i < 6 else 1
        score = camp0_score if camp == 0 else camp1_score
        conn.execute(
            "INSERT INTO match_players (match_uid, player_uid, camp, cur_hero_id,"
            " is_win, add_score, new_score) VALUES (?,?,?,?,?,?,?)",
            (uid, hash((uid, i)) % 10**6, camp, hero,
             camp0_win if camp == 0 else 1 - camp0_win, 10.0, score + 10.0))
        conn.execute(
            "INSERT INTO match_player_heroes (match_uid, player_uid, hero_id,"
            " play_time) VALUES (?,?,?,600.0)",
            (uid, hash((uid, i)) % 10**6, hero))
    conn.commit()


SIX_A = [1011, 1022, 1033, 1044, 1055, 1066]
SIX_B = [1022, 1011, 1044, 1033, 1066, 1055]


def test_hero_block_is_reduced_by_one_column_and_recovers_zero_sum_effects():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_B)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    # Six heroes present -> five free hero parameters.
    assert len(design.hero_ids) == 6
    assert design.hero_basis.shape == (6, 5)
    assert design.X.shape[0] == 1


def test_mirror_lineups_give_a_zero_hero_contrast():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_A)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    hero_block = design.X[:, design.hero_slice]
    np.testing.assert_allclose(hero_block, np.zeros_like(hero_block), atol=1e-12)


def test_skill_control_is_the_pre_match_score_differential():
    conn = make_conn()
    # new_score - add_score is the pre-match score, so camp0 enters 200 higher.
    add_match(conn, "m1", SIX_A, SIX_B, camp0_score=4700.0, camp1_score=4500.0)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    idx = design.column_names.index("skill_diff")
    assert design.X[0, idx] == pytest.approx(200.0)


def test_intercept_column_is_present_for_the_side_advantage():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_B)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    idx = design.column_names.index("intercept")
    assert design.X[0, idx] == pytest.approx(1.0)


def test_outcome_vector_matches_the_sample():
    conn = make_conn()
    add_match(conn, "m1", SIX_A, SIX_B, camp0_win=0)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)
    assert design.y.tolist() == [0]


def test_composition_shape_block_is_reduced_when_two_shapes_are_common():
    # camp0 is 2-2-2 (SIX_A); camp1 swaps one damage hero out for a third
    # damage hero, giving a genuinely different 1-3-2 shape rather than
    # relabelling the same shape "other". With min_shape_count=1 both shapes
    # clear the threshold, so the degenerate all-"other" branch is not taken
    # and the sum-to-zero reduction on the shape block actually runs.
    conn = make_conn()
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1077,'G','Damage')")
    conn.commit()
    camp1 = [1011, 1033, 1044, 1077, 1055, 1066]
    add_match(conn, "m1", SIX_A, camp1)
    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame, min_shape_count=1)

    shape_cols = [c for c in design.column_names if c.startswith("shape_free_")]
    # Three labels survive (1-3-2, 2-2-2, other) -> two free parameters.
    assert shape_cols == ["shape_free_0", "shape_free_1"]
    idx = [design.column_names.index(c) for c in shape_cols]
    shape_block = design.X[:, idx]
    assert shape_block.shape == (1, 2)
    assert not np.allclose(shape_block, 0.0)


def test_team_up_membership_is_plus_one_minus_one_or_zero():
    # A team-up needs both its heroes fielded by the same camp. Cover all
    # three outcomes: the pair together on camp0, together on camp1, and
    # split across camps -- the split case is what distinguishes real
    # subset-membership logic from a check that only ever fires on one side.
    conn = make_conn()
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1077,'G','Damage')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1088,'H','Support')")
    conn.execute(
        "INSERT INTO teamups (teamup_id, name, anchor_hero_id) VALUES (1,'Duo',1011)")
    conn.execute(
        "INSERT INTO teamup_heroes (teamup_id, hero_id, is_anchor) VALUES (1,1011,1)")
    conn.execute(
        "INSERT INTO teamup_heroes (teamup_id, hero_id, is_anchor) VALUES (1,1022,0)")
    conn.commit()

    no_pair = [1033, 1044, 1055, 1066, 1077, 1088]
    add_match(conn, "pair_camp0", SIX_A, no_pair, timestamp=1_787_000_000)
    add_match(conn, "pair_camp1", no_pair, SIX_B, timestamp=1_787_000_100)
    add_match(
        conn, "pair_split",
        [1011, 1033, 1044, 1055, 1066, 1077],
        [1022, 1033, 1044, 1055, 1066, 1088],
        timestamp=1_787_000_200,
    )

    from apm import sample as sample_mod
    frame, _ = sample_mod.build_sample(conn)
    design = features.build_design(conn, frame)

    assert design.teamup_ids == [1]
    idx = design.column_names.index("teamup_1")
    by_match = dict(zip(design.match_uids, design.X[:, idx]))
    assert by_match["pair_camp0"] == pytest.approx(1.0)
    assert by_match["pair_camp1"] == pytest.approx(-1.0)
    assert by_match["pair_split"] == pytest.approx(0.0)
