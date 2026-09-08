import numpy as np

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
