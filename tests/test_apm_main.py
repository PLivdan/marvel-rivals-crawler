import numpy as np

import apm_main
import db


def build_fixture(path, n_matches=400):
    """A database large enough for the logit to converge."""
    conn = db.connect(path)
    db.init_schema(conn)
    heroes = [1011, 1022, 1033, 1044, 1055, 1066, 1077, 1088]
    roles = ["Tank", "Tank", "Damage", "Damage", "Support", "Support",
             "Damage", "Support"]
    for hero, role in zip(heroes, roles):
        conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (?,?,?)",
                     (hero, f"H{hero}", role))
    rng = np.random.default_rng(0)
    for m in range(n_matches):
        uid = f"m{m}"
        conn.execute(
            "INSERT INTO matches (match_uid, match_time_stamp,"
            " match_play_duration) VALUES (?,?,700.0)",
            (uid, 1_787_000_000 + m * 600))
        pick0 = rng.choice(heroes, size=6, replace=False)
        pick1 = rng.choice(heroes, size=6, replace=False)
        win0 = int(rng.random() < 0.5)
        for i, hero in enumerate(list(pick0) + list(pick1)):
            camp = 0 if i < 6 else 1
            player = m * 12 + i
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
