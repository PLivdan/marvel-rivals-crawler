import numpy as np
import pytest

import apm_main
import db


def build_fixture(path, n_matches=400, n_short_matches=0):
    """A database large enough for the logit to converge.

    n_short_matches adds extra, fully-rostered matches with duration 100s
    (below the default 240s forfeit floor). They inflate the whole
    match_players table but must not survive apm.sample.build_sample's
    filters, which is what test_cli_scopes_n_players_to_the_sample below
    exercises. Default 0 leaves the original fixture (and the two tests
    that predate this parameter) unchanged.
    """
    conn = db.connect(path)
    db.init_schema(conn)
    heroes = [1011, 1022, 1033, 1044, 1055, 1066, 1077, 1088]
    roles = ["Tank", "Tank", "Damage", "Damage", "Support", "Support",
             "Damage", "Support"]
    for hero, role in zip(heroes, roles):
        conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (?,?,?)",
                     (hero, f"H{hero}", role))
    rng = np.random.default_rng(0)

    def insert_match(index, uid, duration):
        conn.execute(
            "INSERT INTO matches (match_uid, match_time_stamp,"
            " match_play_duration) VALUES (?,?,?)",
            (uid, 1_787_000_000 + index * 600, duration))
        pick0 = rng.choice(heroes, size=6, replace=False)
        pick1 = rng.choice(heroes, size=6, replace=False)
        win0 = int(rng.random() < 0.5)
        for i, hero in enumerate(list(pick0) + list(pick1)):
            camp = 0 if i < 6 else 1
            player = index * 12 + i
            # new_score varies per player: build_design's skill_diff is the
            # camp-level average of (new_score - add_score), and a literal
            # constant here would make that column identically zero, leaving
            # the design matrix rank-deficient and the Newton-method logit's
            # Hessian singular.
            new_score = 4510.0 + float(rng.normal(0, 100))
            conn.execute(
                "INSERT INTO match_players (match_uid, player_uid, camp,"
                " cur_hero_id, is_win, add_score, new_score)"
                " VALUES (?,?,?,?,?,?,?)",
                (uid, player, camp, int(hero),
                 win0 if camp == 0 else 1 - win0, 10.0, new_score))
            conn.execute(
                "INSERT INTO match_player_heroes (match_uid, player_uid,"
                " hero_id, play_time) VALUES (?,?,?,600.0)",
                (uid, player, int(hero)))

    for m in range(n_matches):
        insert_match(m, f"m{m}", 700.0)
    for m in range(n_short_matches):
        insert_match(n_matches + m, f"short{m}", 100.0)

    conn.commit()
    conn.close()


def test_cli_runs_end_to_end_and_persists_results(tmp_path):
    path = str(tmp_path / "t.db")
    build_fixture(path)
    assert apm_main.main(["--db-path", path, "--bootstrap-reps", "0"]) == 0
    conn = db.connect(path)
    runs = conn.execute("SELECT COUNT(*) FROM apm_runs").fetchone()[0]
    effects = conn.execute("SELECT COUNT(*) FROM apm_hero_effects").fetchone()[0]
    assert runs == 1
    assert effects == 8


def test_cli_records_the_attribution_rule_it_was_given(tmp_path):
    path = str(tmp_path / "t.db")
    build_fixture(path)
    apm_main.main(["--db-path", path, "--attribution", "W2",
                   "--bootstrap-reps", "0"])
    conn = db.connect(path)
    assert conn.execute(
        "SELECT attribution_rule FROM apm_runs").fetchone()[0] == "W2"


def test_cli_scopes_n_players_to_the_sample_not_the_whole_table(tmp_path):
    # n_matches survive the default 240s forfeit floor; n_short_matches (100s
    # duration) do not, so the whole match_players table and the sample the
    # run is actually fit on diverge -- exactly what n_players must track.
    path = str(tmp_path / "t.db")
    build_fixture(path, n_matches=400, n_short_matches=10)
    apm_main.main(["--db-path", path, "--bootstrap-reps", "0"])
    conn = db.connect(path)
    total_players = conn.execute(
        "SELECT COUNT(*) FROM match_players").fetchone()[0]
    recorded = conn.execute(
        "SELECT n_players FROM apm_runs").fetchone()[0]
    assert total_players == 400 * 12 + 10 * 12
    assert recorded == 400 * 12
    assert recorded < total_players


def test_hero_adjusted_p_values_quarantines_a_degenerate_standard_error():
    # Hero 1033's exactly-zero variance would otherwise NaN z, then raw_p,
    # then (via benjamini_hochberg's cumulative minimum) EVERY hero's
    # adjusted p-value, not just its own.
    hero_ids = [1011, 1022, 1033]
    hero_effects = {1011: 0.5, 1022: -0.5, 1033: 0.0}
    hero_cov = np.diag([0.01, 0.01, 0.0])

    with pytest.warns(RuntimeWarning, match="1033"):
        adjusted_p = apm_main._hero_adjusted_p_values(
            hero_ids, hero_effects, hero_cov)

    assert np.isnan(adjusted_p[1033])
    assert np.isfinite(adjusted_p[1011])
    assert np.isfinite(adjusted_p[1022])
