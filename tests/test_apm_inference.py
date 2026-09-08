import numpy as np
import pytest

from apm import inference


def test_bh_rejects_nothing_when_all_pvalues_are_uniform_noise():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    _, rejected = inference.benjamini_hochberg(p, q=0.05)
    # Under the null, expected rejections are far below 5% of tests.
    assert rejected.sum() <= 5


def test_bh_rejects_obvious_signal_and_is_less_strict_than_bonferroni():
    p = np.array([1e-8, 1e-7, 1e-6, 0.4, 0.6, 0.9])
    adjusted, rejected = inference.benjamini_hochberg(p, q=0.05)
    assert rejected[:3].all()
    assert not rejected[3:].any()
    assert np.all(adjusted >= p)
    assert adjusted[2] < p[2] * len(p)


def test_bh_adjusted_values_are_monotone():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    adjusted, _ = inference.benjamini_hochberg(p)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_percentile_interval_brackets_the_truth():
    rng = np.random.default_rng(1)
    draws = rng.normal(loc=0.3, scale=0.1, size=(2000, 1))
    lo, hi = inference.percentile_interval(draws)
    assert lo[0] < 0.3 < hi[0]


def test_cluster_bootstrap_returns_one_row_per_replication():
    rng = np.random.default_rng(2)

    class FakeDesign:
        match_uids = [f"m{i}" for i in range(50)]
        hero_ids = [1, 2, 3]

    clusters = {f"m{i}": [i % 10] for i in range(50)}

    def fake_fit(match_uids):
        return {h: rng.normal() for h in FakeDesign.hero_ids}

    draws = inference.cluster_bootstrap(
        FakeDesign(), fake_fit, clusters, n_reps=25, seed=0)
    assert draws.shape == (25, 3)


def test_player_clusters_maps_matches_to_players_and_excludes_unwanted():
    class FakeConn:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, query):
            # A fake connection satisfies "pure synthetic, no database": it
            # only needs to support the same `.execute()` -> row-iterable
            # protocol a real DB-API connection does.
            return self._rows

    rows = [
        ("m1", 10),
        ("m1", 11),
        ("m2", 12),
        ("m3", 13),  # m3 is not in the requested set and must be excluded.
    ]
    conn = FakeConn(rows)

    clusters = inference.player_clusters(conn, ["m1", "m2"])

    assert clusters == {"m1": [10, 11], "m2": [12]}
    assert "m3" not in clusters


def test_cluster_bootstrap_resamples_players_not_matches():
    """Two matches share a player found nowhere else; a third match holds a
    player who is otherwise unique. Under genuine player-level resampling,
    the shared player is either drawn or not *as a unit*, so the two matches
    that hinge on them must always co-occur: both present or both absent in
    every replication. Naive match-level resampling (drawing match_uids
    directly, ignoring player structure) would make their presence
    independent, so this assertion would eventually fail under it -- that
    failure mode is exactly the regression this test exists to catch."""
    rng = np.random.default_rng(2)

    class FakeDesign:
        hero_ids = [1, 2, 3]

    clusters = {f"solo{i}": [f"solo_player{i}"] for i in range(20)}
    clusters["shared_a"] = ["shared_player"]
    clusters["shared_b"] = ["shared_player"]

    captured = []

    def fake_fit(match_uids):
        captured.append(set(match_uids))
        return {h: rng.normal() for h in FakeDesign.hero_ids}

    inference.cluster_bootstrap(
        FakeDesign(), fake_fit, clusters, n_reps=200, seed=3)

    for uids in captured:
        assert ("shared_a" in uids) == ("shared_b" in uids)
    # Both outcomes must actually occur, or the assertion above would be
    # vacuously true (e.g. if the shared player were never resampled at all).
    assert any("shared_a" in uids for uids in captured)
    assert any("shared_a" not in uids for uids in captured)


def test_percentile_interval_warns_and_returns_nan_for_all_nan_column():
    draws = np.array([
        [0.10, np.nan, 1.0],
        [0.20, np.nan, 2.0],
        [0.30, np.nan, 3.0],
    ])

    with pytest.warns(RuntimeWarning, match=r"1 of 3 column"):
        lo, hi = inference.percentile_interval(draws)

    assert np.isnan(lo[1]) and np.isnan(hi[1])
    assert not np.isnan(lo[0]) and not np.isnan(hi[0])
    assert not np.isnan(lo[2]) and not np.isnan(hi[2])
