import numpy as np
import pytest

from kite.engine import Table, event_generator, event_uniform, expectation, run

OPTIONS = ("ignore", "like", "share")


def population(n: int, seed: int = 0):
    """n agents over four persona types; the treated condition gets a different state key per type."""
    rng = np.random.default_rng(seed)
    types = rng.integers(0, 4, n)
    return [(f"agent-{i}", 1.0, {"control": f"type{t}|control", "treated": f"type{t}|treated"}) for i, t in enumerate(types)]


def table(effect: float = 0.1) -> Table:
    rows = {}
    for t in range(4):
        base = np.array([0.6 - 0.05 * t, 0.25, 0.15 + 0.05 * t])
        rows[f"type{t}|control"] = base
        rows[f"type{t}|treated"] = base + np.array([-effect, 0.0, effect])
    return Table(OPTIONS, rows)


def test_event_uniform_is_a_pure_function_of_its_key():
    assert event_uniform(1, "a", 2, "decide") == event_uniform(1, "a", 2, "decide")
    assert event_uniform(1, "a", 2, "decide") != event_uniform(2, "a", 2, "decide")
    assert event_uniform(1, "a", 2, "decide") != event_uniform(1, "a", 3, "decide")
    draws = np.array([event_uniform(0, i) for i in range(20000)])
    assert 0 <= draws.min() and draws.max() < 1
    assert abs(draws.mean() - 0.5) < 0.01 and abs(draws.var() - 1 / 12) < 0.005


def test_event_generator_is_reproducible():
    assert event_generator(3, "x").normal() == event_generator(3, "x").normal()


def test_inverse_cdf_draw():
    t = Table(OPTIONS, {"s": [0.2, 0.3, 0.5]})
    assert [t.draw("s", u) for u in (0.0, 0.19, 0.2, 0.49, 0.5, 0.999)] == [0, 0, 1, 1, 2, 2]
    degenerate = Table(OPTIONS, {"s": [0.0, 1.0, 0.0]})
    assert {degenerate.draw("s", u) for u in np.linspace(0, 0.999, 50)} == {1}


def test_table_validates_rows():
    with pytest.raises(ValueError):
        Table(OPTIONS, {"s": [0.5, 0.5]})
    with pytest.raises(ValueError):
        Table(OPTIONS, {"s": [-0.1, 0.6, 0.5]})


def test_zero_effect_intervention_gives_exactly_zero_contrast():
    counts = run(table(effect=0.0), population(500), ["control", "treated"], seed=7)
    assert np.array_equal(counts["control"], counts["treated"])


def test_common_random_numbers_couple_the_conditions():
    """With CRN the treated-minus-control contrast has far less variance than two independent runs would."""
    t = table(effect=0.05)
    agents = population(400)
    paired = [run(t, agents, ["control", "treated"], seed=s) for s in range(30)]
    paired_diff = np.array([c["treated"][2] - c["control"][2] for c in paired])
    independent_diff = np.array(
        [run(t, agents, ["treated"], seed=s)["treated"][2] - run(t, agents, ["control"], seed=s + 1000)["control"][2] for s in range(30)]
    )
    expected = expectation(t, agents, ["control", "treated"])
    truth = expected["treated"][2] - expected["control"][2]
    assert abs(paired_diff.mean() - truth) < 3 * paired_diff.std(ddof=1) / np.sqrt(30) + 1e-9
    assert paired_diff.std(ddof=1) < 0.5 * independent_diff.std(ddof=1)


def test_result_does_not_depend_on_order_or_batching():
    t = table()
    agents = population(300)
    whole = run(t, agents, ["control", "treated"], seed=11)
    reordered = run(t, list(reversed(agents)), ["control", "treated"], seed=11)
    chunked = {c: sum(run(t, agents[i : i + 50], ["control", "treated"], seed=11)[c] for i in range(0, 300, 50)) for c in ("control", "treated")}
    for c in ("control", "treated"):
        assert np.array_equal(whole[c], reordered[c])
        assert np.array_equal(whole[c], chunked[c])


def test_expectation_is_the_mean_of_runs():
    t = table()
    agents = population(200)
    expected = expectation(t, agents, ["control"])["control"]
    sampled = np.mean([run(t, agents, ["control"], seed=s)["control"] for s in range(200)], axis=0)
    assert np.allclose(sampled, expected, atol=2.0)  # 200 agents x 200 seeds: standard error well under 1 agent
    assert abs(expected.sum() - 200) < 1e-9
