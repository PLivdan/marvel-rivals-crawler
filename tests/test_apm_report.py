import numpy as np
import pytest

import db
from apm import report
from apm.estimate import FitResult


def make_conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1011,'Hulk','Tank')")
    conn.execute("INSERT INTO hero_info (hero_id, name, role) VALUES (1022,'Cap','Tank')")
    conn.commit()
    return conn


def test_schema_creates_the_result_tables():
    conn = make_conn()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"apm_runs", "apm_hero_effects", "apm_teamup_effects"} <= tables


def test_run_metadata_round_trips():
    conn = make_conn()
    run_id = report.record_run(
        conn, spec="A", attribution_rule="W1", forfeit_floor=240,
        n_matches=1000, n_players=12000, exclusions="{}", git_commit="abc123")
    row = conn.execute(
        "SELECT spec, attribution_rule, forfeit_floor, n_matches, git_commit "
        "FROM apm_runs WHERE run_id=?", (run_id,)).fetchone()
    assert row == ("A", "W1", 240, 1000, "abc123")


def test_hero_effects_are_written_and_join_to_names():
    conn = make_conn()
    run_id = report.record_run(
        conn, spec="A", attribution_rule="W1", forfeit_floor=240,
        n_matches=10, n_players=120, exclusions="{}", git_commit="abc")
    fit = FitResult(
        params=np.zeros(2), cov=np.eye(2), column_names=["a", "b"],
        hero_effects={1011: 0.20, 1022: -0.20},
        hero_cov=np.diag([0.0004, 0.0004]), hero_ids=[1011, 1022],
        loglike=-1.0, n=10)
    report.write_hero_effects(
        conn, run_id, fit,
        intervals={1011: (0.15, 0.25), 1022: (-0.25, -0.15)},
        adjusted_p={1011: 0.001, 1022: 0.001})
    table = report.hero_table(conn, run_id)
    assert set(table["name"]) == {"Hulk", "Cap"}
    hulk = table[table["name"] == "Hulk"].iloc[0]
    assert hulk["effect_logodds"] == pytest.approx(0.20)
    assert hulk["effect_pp"] == pytest.approx(5.0)


def test_probability_point_conversion_uses_the_balanced_match_margin():
    # dP/dx = beta * p * (1-p); at p = 0.5 that is beta/4.
    assert report.to_probability_points(0.4) == pytest.approx(10.0)


def test_runs_are_independent():
    conn = make_conn()
    a = report.record_run(conn, spec="A", attribution_rule="W1",
                          forfeit_floor=240, n_matches=1, n_players=1,
                          exclusions="{}", git_commit="x")
    b = report.record_run(conn, spec="B", attribution_rule="W2",
                          forfeit_floor=0, n_matches=2, n_players=2,
                          exclusions="{}", git_commit="y")
    assert a != b
