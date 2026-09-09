import numpy as np
import pytest

import apm_main
import db
from apm import contrasts
from apm.features import DesignMatrix


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
            " match_play_duration, map_id) VALUES (?,?,?,?)",
            (uid, 1_787_000_000 + index * 600, duration, 1231 + (index % 3)))
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


def _design_with_extra_columns(extra, n=20, n_hero=3):
    """A minimal design (intercept + hero block + named extra columns) for
    exercising _drop_zero_variance_extra_columns without a database.

    `extra` is a list of (column_name, values) pairs appended after the
    hero block, exactly where team-up and shape_free columns sit in a real
    design (see apm/features.py's column layout).
    """
    basis = contrasts.sum_to_zero_basis(n_hero)
    rng = np.random.default_rng(0)
    raw = rng.choice([-1.0, 0.0, 1.0], size=(n, n_hero))
    hero_block = contrasts.reduce_design(raw, basis)
    intercept = np.ones((n, 1))
    extra_cols = (np.column_stack([v for _, v in extra]) if extra
                  else np.zeros((n, 0)))
    X = np.hstack([intercept, hero_block, extra_cols])
    names = (["intercept"] + [f"hero_free_{i}" for i in range(hero_block.shape[1])]
             + [name for name, _ in extra])
    hero_slice = slice(1, 1 + hero_block.shape[1])
    teamup_ids = [int(name[len("teamup_"):]) for name, _ in extra
                  if name.startswith("teamup_")]
    return DesignMatrix(X, np.zeros(n, dtype=int), names, list(range(n_hero)),
                        basis, hero_slice, teamup_ids, [f"m{i}" for i in range(n)])


def test_drop_zero_variance_extra_columns_drops_and_reports(capsys):
    zeros = np.zeros(20)
    varying = np.array([1.0, -1.0] * 10)
    design = _design_with_extra_columns(
        [("teamup_1", zeros), ("shape_free_0", varying)])

    new_design = apm_main._drop_zero_variance_extra_columns(design)

    assert "teamup_1" not in new_design.column_names
    assert "shape_free_0" in new_design.column_names
    assert new_design.teamup_ids == []
    assert new_design.hero_slice == design.hero_slice
    assert new_design.X.shape[1] == design.X.shape[1] - 1

    err = capsys.readouterr().err
    assert "teamup_1" in err
    assert "dropping zero-variance" in err


def test_drop_zero_variance_extra_columns_is_a_no_op_when_nothing_is_degenerate():
    design = _design_with_extra_columns([("teamup_1", np.array([1.0, -1.0] * 10))])
    new_design = apm_main._drop_zero_variance_extra_columns(design)
    assert new_design is design


def test_drop_zero_variance_extra_columns_raises_on_a_degenerate_hero_column():
    # The hero block is the estimand and must never be silently dropped:
    # a degenerate hero column has to surface loudly instead.
    n = 20
    hero_block = np.zeros((n, 2))
    intercept = np.ones((n, 1))
    X = np.hstack([intercept, hero_block])
    names = ["intercept", "hero_free_0", "hero_free_1"]
    design = DesignMatrix(X, np.zeros(n, dtype=int), names, [10, 20, 30],
                          contrasts.sum_to_zero_basis(3), slice(1, 3), [],
                          [f"m{i}" for i in range(n)])
    with pytest.raises(RuntimeError, match="hero_free_0"):
        apm_main._drop_zero_variance_extra_columns(design)


def test_cli_drops_a_structurally_zero_teamup_column_and_still_fits(tmp_path, capsys):
    # Reproduces the real hero_id 1057 "Deadpool" bug: a team-up references
    # a hero that is defined but never actually played (all its plays are
    # recorded under role-variant hero ids instead), so the team-up column
    # can never be non-zero and would otherwise singularize the Newton fit.
    path = str(tmp_path / "t.db")
    build_fixture(path)
    conn = db.connect(path)
    conn.execute(
        "INSERT INTO teamups (teamup_id, name, anchor_hero_id) "
        "VALUES (99,'Ghost',1011)")
    conn.execute(
        "INSERT INTO teamup_heroes (teamup_id, hero_id, is_anchor) "
        "VALUES (99,1011,1)")
    conn.execute(
        "INSERT INTO teamup_heroes (teamup_id, hero_id, is_anchor) "
        "VALUES (99,9999,0)")  # hero_id 9999 never appears in any match
    conn.commit()
    conn.close()

    assert apm_main.main(["--db-path", path, "--bootstrap-reps", "0"]) == 0

    err = capsys.readouterr().err
    assert "teamup_99" in err
    assert "dropping zero-variance" in err

    conn = db.connect(path)
    effects = conn.execute(
        "SELECT effect_logodds FROM apm_hero_effects").fetchall()
    assert len(effects) == 8
    assert sum(e[0] for e in effects) == pytest.approx(0.0, abs=1e-6)


def test_bootstrap_reps_defaults_to_zero(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    build_fixture(path)
    captured = {}
    real = apm_main.run_specification_a

    def spy(conn, args):
        captured["bootstrap_reps"] = args.bootstrap_reps
        return real(conn, args)

    monkeypatch.setattr(apm_main, "run_specification_a", spy)
    assert apm_main.main(["--db-path", path]) == 0
    assert captured["bootstrap_reps"] == 0


def test_nonzero_bootstrap_reps_prints_a_cost_warning(tmp_path, capsys):
    path = str(tmp_path / "t.db")
    build_fixture(path)
    assert apm_main.main(["--db-path", path, "--bootstrap-reps", "1"]) == 0
    err = capsys.readouterr().err
    assert "bootstrap-reps=1" in err
    assert "65s" in err
    assert "hours" in err


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


def test_cli_defaults_to_the_identified_specification(tmp_path, monkeypatch):
    """A bare `apm_main.py --db-path X` must produce the good spec, not the one
    it superseded. Every default here was chosen from a measured result: W0 is
    the only pre-outcome attribution; within_role removes an exact hero/shape
    collinearity; map intercepts follow a measured 3.10pp spread across maps;
    100 gives the rarer compositions their own dummy at 0.16% cost."""
    seen = {}

    def spy(conn, args):
        seen.update(vars(args))
        return 1

    monkeypatch.setattr(apm_main, "run_specification_a", spy)
    monkeypatch.setattr(apm_main.report, "hero_table",
                        lambda conn, run_id: __import__("pandas").DataFrame())
    path = str(tmp_path / "t.db")
    build_fixture(path, n_matches=40)
    apm_main.main(["--db-path", path])
    assert seen["attribution"] == "W0"
    assert seen["constraint"] == "within_role"
    assert seen["map_intercepts"] is True
    assert seen["min_shape_count"] == 100
    assert seen["bootstrap_reps"] == 0
