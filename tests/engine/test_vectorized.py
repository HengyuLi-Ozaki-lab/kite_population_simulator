import numpy as np

from kite.engine import vectorized as vec


def setup(n: int = 5000, effect: float = 0.1):
    probs = np.array([[0.6, 0.25, 0.15], [0.5, 0.25, 0.25], [0.6 - effect, 0.25, 0.15 + effect], [0.5 - effect, 0.25, 0.25 + effect]])
    types = np.random.default_rng(0).integers(0, 2, n)
    return probs, {"control": types, "treated": types + 2}


def test_uniforms_are_deterministic_and_uniform():
    a = vec.event_uniforms(1, np.arange(100000), 0, 0)
    assert np.array_equal(a, vec.event_uniforms(1, np.arange(100000), 0, 0))
    assert not np.array_equal(a, vec.event_uniforms(2, np.arange(100000), 0, 0))
    assert not np.array_equal(a, vec.event_uniforms(1, np.arange(100000), 1, 0))
    assert not np.array_equal(a, vec.event_uniforms(1, np.arange(100000), 0, 1))
    assert 0 <= a.min() and a.max() < 1
    assert abs(a.mean() - 0.5) < 0.005 and abs(a.var() - 1 / 12) < 0.002
    hist, _ = np.histogram(a, bins=20, range=(0, 1))
    assert hist.min() > 4500 and hist.max() < 5500


def test_uniforms_do_not_depend_on_the_batch():
    whole = vec.event_uniforms(3, np.arange(1000), 2, 1)
    parts = np.concatenate([vec.event_uniforms(3, np.arange(i, i + 100), 2, 1) for i in range(0, 1000, 100)])
    assert np.array_equal(whole, parts)


def test_zero_effect_gives_exactly_zero_contrast():
    probs, states = setup(effect=0.0)
    counts = vec.run(vec.cumulative_table(probs), states, seed=5)
    assert np.array_equal(counts["control"], counts["treated"])


def test_common_random_numbers_and_expectation():
    probs, states = setup(n=20000, effect=0.02)
    c = vec.cumulative_table(probs)
    paired = np.array([vec.run(c, states, seed=s)["treated"][2] - vec.run(c, states, seed=s)["control"][2] for s in range(40)])
    independent = np.array([vec.run(c, states, seed=s)["treated"][2] - vec.run(c, states, seed=s + 500)["control"][2] for s in range(40)])
    e = vec.expectation(probs, states)
    truth = e["treated"][2] - e["control"][2]
    assert abs(paired.mean() - truth) < 4 * paired.std(ddof=1) / np.sqrt(40) + 1e-9
    # inverse-CDF coupling: the paired contrast varies only where the two CDFs differ (sd ~ sqrt(n * 0.02)), the
    # independent one with the full response variance (sd ~ sqrt(2 * n * 0.2 * 0.8))
    assert paired.std(ddof=1) < 0.5 * independent.std(ddof=1)
    sampled = np.mean([vec.run(c, states, seed=s)["control"] for s in range(50)], axis=0)
    assert np.allclose(sampled, e["control"], rtol=0.02)


def test_weights_and_degenerate_rows():
    probs = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    states = {"only": np.array([0, 0, 1])}
    counts = vec.run(vec.cumulative_table(probs), states, np.array([2.0, 3.0, 1.0]), seed=0)
    assert np.array_equal(counts["only"], np.array([1.0, 5.0, 0.0]))
