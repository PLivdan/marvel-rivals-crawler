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
