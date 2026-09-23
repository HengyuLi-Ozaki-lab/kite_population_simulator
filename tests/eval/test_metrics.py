import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from kite.eval import metrics


def test_as_dist_normalizes_and_rejects_bad_input():
    assert metrics.as_dist([1, 3]).tolist() == [0.25, 0.75]
    with pytest.raises(ValueError):
        metrics.as_dist([0, 0])
    with pytest.raises(ValueError):
        metrics.as_dist([1, -1])
    # NaN is neither negative nor a non-positive sum, so it used to pass both guards and come back
    # as [nan, nan] - a metric computed from it is a number-shaped hole in the table
    with pytest.raises(ValueError, match="NaN"):
        metrics.as_dist([float("nan"), 1.0])
    with pytest.raises(ValueError, match="infinity"):
        metrics.as_dist([float("inf"), 1.0])


def test_opinionqa_alignment_matches_the_original_formula():
    """Reference: tatsu-lab/opinions_qa helpers.get_max_wd and process_results.ipynb."""
    rng = np.random.default_rng(0)
    for ordinal in ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0], [4.0, 3.0, 2.0, 1.0, 0.0]):
        p, q = rng.dirichlet(np.ones(len(ordinal))), rng.dirichlet(np.ones(len(ordinal)))
        d0, d1 = np.zeros(len(ordinal)), np.zeros(len(ordinal))
        d0[np.argmax(ordinal)] = 1
        d1[np.argmin(ordinal)] = 1
        max_wd = wasserstein_distance(ordinal, ordinal, d0, d1)
        expected = 1 - wasserstein_distance(ordinal, ordinal, p, q) / max_wd
        assert metrics.opinionqa_alignment(p, q, ordinal) == pytest.approx(expected)


def test_alignment_extremes():
    assert metrics.opinionqa_alignment([1, 0, 0], [1, 0, 0], [1, 2, 3]) == pytest.approx(1.0)
    assert metrics.opinionqa_alignment([1, 0, 0], [0, 0, 1], [1, 2, 3]) == pytest.approx(0.0)


def test_wasserstein_unit_rescales_to_zero_one():
    assert metrics.wasserstein_unit([1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]) == pytest.approx(1.0)
    assert metrics.wasserstein_unit([1, 0, 0], [0, 1, 0]) == pytest.approx(0.5)


def test_observed_range_positions_uses_the_answers_people_gave():
    # nobody chose level 0 or level 4, so the used range is levels 1..3 and level 2 sits at its middle
    assert metrics.observed_range_positions([0, 3, 0, 1, 0]).tolist() == [-0.5, 0.0, 0.5, 1.0, 1.5]
    # a full-width cell is the same as the stated-scale convention
    assert metrics.observed_range_positions([1, 1, 1, 1, 1]).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    # everyone gave the same answer: no range to divide by, so positions count whole levels from it
    assert metrics.observed_range_positions([0, 5, 0]).tolist() == [-1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        metrics.observed_range_positions([0, 0, 0])


def test_wasserstein_observed_matches_a_hand_computation():
    # on a five-level scale the humans split evenly between levels 1 and 3, so positions are
    # (arange(5) - 1) / 2 = [-0.5, 0, 0.5, 1, 1.5] and their mass sits at 0 and 1
    human = [0, 2, 0, 2, 0]
    assert metrics.wasserstein_observed([0, 0, 0, 1, 0], human) == pytest.approx(0.5)  # all at 1: 0.5 * 1 + 0.5 * 0
    assert metrics.wasserstein_observed([0, 1, 0, 0, 0], human) == pytest.approx(0.5)  # all at 0: 0.5 * 0 + 0.5 * 1
    # the same pair under the stated-scale convention is half as large: that divisor is the whole scale
    assert metrics.wasserstein_unit([0, 0, 0, 1, 0], human) == pytest.approx(0.25)
    # a prediction on a level nobody used sits outside [0, 1]: all at -0.5 is 0.5 * 0.5 + 0.5 * 1.5
    assert metrics.wasserstein_observed([1, 0, 0, 0, 0], human) == pytest.approx(1.0)
    assert metrics.wasserstein_unit([1, 0, 0, 0, 0], human) == pytest.approx(0.5)


def test_total_variation_js_entropy():
    assert metrics.total_variation([1, 0], [0, 1]) == pytest.approx(1.0)
    assert metrics.total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert metrics.js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    assert metrics.js_divergence([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0, abs=1e-12)
    assert metrics.entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert metrics.entropy([1, 0]) == 0.0


def test_entropy_ratio():
    assert metrics.entropy_ratio([1, 0], [0.5, 0.5]) == 0.0
    assert metrics.entropy_ratio([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    assert metrics.entropy_ratio([0.5, 0.5], [1, 0]) is None


def test_the_entropy_ratio_headline_survives_one_near_unanimous_cell():
    """The exact failure mode: an unbounded ratio whose mean is set by the worst denominator."""
    # 999 cells where the model matches people, and one where the humans were all but unanimous
    nearly = [9000, 1]  # H = 0.0016 bits
    assert metrics.entropy(nearly) == pytest.approx(0.0016, abs=1e-4)
    assert metrics.entropy(nearly) < metrics.LOW_ENTROPY_BITS
    outlier = metrics.entropy_ratio([0.5, 0.5], nearly)
    assert outlier > 600
    ratios = [1.0] * 999 + [outlier]
    summary = metrics.describe_entropy_ratios(ratios)
    assert summary["entropy_ratio_mean"] > 1.6  # the mean is moved by one cell in a thousand
    assert summary["entropy_ratio_median"] == pytest.approx(1.0)  # the median is not
    assert (summary["entropy_ratio_p25"], summary["entropy_ratio_p75"]) == (1.0, 1.0)


def test_describe_entropy_ratios_reports_quartiles_and_handles_emptiness():
    summary = metrics.describe_entropy_ratios([0.5, 1.0, 1.5, 2.0, 2.5])
    assert summary["entropy_ratio_median"] == pytest.approx(1.5)
    assert (summary["entropy_ratio_p25"], summary["entropy_ratio_p75"]) == (1.0, 2.0)
    assert summary["entropy_ratio_mean"] == pytest.approx(1.5)
    assert set(metrics.describe_entropy_ratios([]).values()) == {None}


def test_mean_position_rescales_to_zero_one():
    assert metrics.mean_position([1, 0, 0, 0, 0]) == 0.0
    assert metrics.mean_position([0, 0, 0, 0, 1]) == 1.0
    assert metrics.mean_position([0.5, 0, 0, 0, 0.5]) == pytest.approx(0.5)
    assert metrics.mean_position([1, 1, 1, 1]) == pytest.approx(0.5)
    assert metrics.mean_position([3, 1]) == pytest.approx(0.25)  # mean level 0.25 of 1
    with pytest.raises(ValueError):
        metrics.mean_position([1.0])


def test_regrid_keeps_the_mean_position_and_is_the_identity_at_equal_length():
    assert metrics.regrid([1, 2, 3], 3).tolist() == [1, 2, 3]
    # a 3-level histogram onto 5 levels: positions 0, 0.5, 1 land on levels 0, 2, 4
    assert metrics.regrid([1, 0, 0], 5).tolist() == [1, 0, 0, 0, 0]
    assert metrics.regrid([0, 1, 0], 5).tolist() == [0, 0, 1, 0, 0]
    # a 2-level histogram onto 4 levels: position 1 lands on level 3, position 0 on level 0
    assert metrics.regrid([3, 1], 4).tolist() == [3, 0, 0, 1]
    # 5 levels onto 4: level 1 sits at 0.25 of 3 = 0.75, split three-quarters onto level 1
    assert metrics.regrid([0, 4, 0, 0, 0], 4) == pytest.approx([1.0, 3.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    for source, target in [(3, 7), (7, 3), (2, 11), (11, 2), (6, 6)]:
        weights = rng.random(source) + 0.01
        assert metrics.mean_position(metrics.regrid(weights, target)) == pytest.approx(metrics.mean_position(weights))
        assert metrics.regrid(weights, target).sum() == pytest.approx(weights.sum())
    with pytest.raises(ValueError):
        metrics.regrid([1.0], 4)


def test_condition_sensitivity_recovers_a_planted_relationship():
    # two tasks; within each, the human mean moves at half the predicted movement, plus a fixed offset
    by_task = {
        ("s1", 0): [(0.2, 0.5), (0.4, 0.6), (0.6, 0.7)],
        ("s2", 1): [(0.1, 0.15), (0.5, 0.35)],
    }
    fit = metrics.condition_sensitivity(by_task, n_boot=200, seed=0)
    assert fit["r"] == pytest.approx(1.0)
    assert fit["slope"] == pytest.approx(0.5)  # one unit of predicted movement is half a unit of human movement
    assert fit["spread_ratio"] == pytest.approx(1 / 0.5)  # the prediction moves twice as far as people do
    assert (fit["n_tasks"], fit["n_cells"]) == (2, 5)
    assert 0 < fit["ci_low"] <= fit["r"] <= fit["ci_high"]
    # the task-level offsets are removed, so adding a constant to one task changes nothing
    shifted = {("s1", 0): [(p, h + 10) for p, h in by_task[("s1", 0)]], ("s2", 1): by_task[("s2", 1)]}
    assert metrics.condition_sensitivity(shifted, n_boot=200, seed=0)["r"] == pytest.approx(fit["r"])


def test_condition_sensitivity_is_none_for_a_condition_blind_predictor():
    blind = {("s1", 0): [(0.5, 0.2), (0.5, 0.8)], ("s2", 0): [(0.5, 0.1), (0.5, 0.4)]}
    fit = metrics.condition_sensitivity(blind, n_boot=50)
    assert fit["r"] is None and fit["slope"] is None
    assert fit["spread_ratio"] == 0.0
    assert fit["ci_low"] is None and fit["ci_high"] is None
    # rounding noise in a flat prediction must not be reported as faint sensitivity
    jittery = {key: [(0.5 + i * 1e-16, h) for i, (_, h) in enumerate(pairs)] for key, pairs in blind.items()}
    assert metrics.condition_sensitivity(jittery, n_boot=50)["r"] is None
    # but a real, small movement is still measured
    assert metrics.condition_sensitivity({("s1", 0): [(0.5, 0.2), (0.5 + 1e-3, 0.8)]}, n_boot=50)["r"] == pytest.approx(1.0)


def test_condition_sensitivity_is_none_when_the_humans_do_not_move_either():
    fit = metrics.condition_sensitivity({("s1", 0): [(0.2, 0.5), (0.8, 0.5)]}, n_boot=50)
    assert fit["r"] is None and fit["slope"] is None and fit["spread_ratio"] is None


def test_condition_sensitivity_drops_single_condition_tasks():
    assert metrics.condition_sensitivity({("s1", 0): [(0.3, 0.4)]}, n_boot=10) == {
        "r": None,
        "slope": None,
        "spread_ratio": None,
        "n_tasks": 0,
        "n_cells": 0,
        "ci_low": None,
        "ci_high": None,
        "null_p95": None,
        "permutation_p": None,
    }
    mixed = {("s1", 0): [(0.3, 0.4)], ("s2", 0): [(0.1, 0.2), (0.5, 0.9)]}
    assert metrics.condition_sensitivity(mixed, n_boot=50)["n_tasks"] == 1


def test_condition_sensitivity_is_deterministic_under_a_fixed_seed():
    by_task = {(f"s{i}", 0): [(0.1 * i, 0.2), (0.4, 0.3 + 0.01 * i), (0.7, 0.9)] for i in range(8)}
    first = metrics.condition_sensitivity(by_task, n_boot=200, seed=7)
    assert first == metrics.condition_sensitivity(by_task, n_boot=200, seed=7)
    assert first["ci_low"] != metrics.condition_sensitivity(by_task, n_boot=200, seed=8)["ci_low"]


def test_median_index():
    assert metrics.median_index([0.2, 0.3, 0.5]) == 1
    assert metrics.median_index([0.6, 0.4]) == 0
    assert metrics.median_index([0.0, 0.0, 1.0]) == 2
    assert metrics.median_index([0.5, 0.5]) == 0


def test_absolute_error_helpers():
    assert metrics.normalized_abs_error(0, 6, 7) == pytest.approx(1.0)
    assert metrics.normalized_abs_error(3, 4, 7) == pytest.approx(1 / 6)
    # true answer at the low end of a 1-7 scale: mean |k - 0| over k = 0..6 is 3, normalized by 6
    assert metrics.uniform_expected_abs_error(0, 7) == pytest.approx(0.5)
    assert metrics.uniform_expected_abs_error(3, 7) == pytest.approx((3 + 2 + 1 + 0 + 1 + 2 + 3) / 7 / 6)


def test_the_two_accuracy_conventions_differ_on_a_uniform_prediction():
    flat = [1 / 7] * 7
    # a uniform guess drawn at random, averaged over every true answer on a 7-level scale
    drawn = [1 - metrics.random_draw_abs_error(flat, true) for true in range(7)]
    assert float(np.mean(drawn)) == pytest.approx(1 - 112 / 42 / 7)  # 0.6190
    # the median of that same uniform distribution is the middle level, a far better point prediction
    middle = [1 - metrics.normalized_abs_error(metrics.median_index(flat), true, 7) for true in range(7)]
    assert float(np.mean(middle)) == pytest.approx(1 - 12 / 6 / 7)  # 0.7143
    assert np.mean(middle) - np.mean(drawn) == pytest.approx(0.0952, abs=5e-4)
    # the drawing convention agrees with the analytic uniform helper, level by level
    for true in range(7):
        assert metrics.random_draw_abs_error(flat, true) == pytest.approx(metrics.uniform_expected_abs_error(true, 7))
    # and a confident prediction is scored the same either way
    assert metrics.random_draw_abs_error([0, 0, 1, 0], 2) == 0.0


def test_brier():
    assert metrics.brier([1, 0, 0], 0) == 0.0
    assert metrics.brier([0, 1, 0], 0) == pytest.approx(2.0)
    assert metrics.brier([0.5, 0.5], 0) == pytest.approx(0.5)


def test_ece_perfectly_calibrated_and_overconfident():
    confidences = [0.75] * 4
    value, bins = metrics.ece(confidences, [True, True, True, False])
    assert value == pytest.approx(0.0)
    assert len(bins) == 1
    assert (bins[0]["n"], bins[0]["confidence"], bins[0]["accuracy"]) == (4, 0.75, 0.75)
    value, _ = metrics.ece([0.95] * 4, [True, False, False, False])
    assert value == pytest.approx(0.70)


def _movement(rng, n_tasks, n_conditions, signal):
    """Tasks whose predicted movement tracks the human movement by `signal`, plus noise."""
    by_task = {}
    for task in range(n_tasks):
        human = rng.normal(0, 0.15, n_conditions)
        predicted = signal * human + (1 - signal) * rng.normal(0, 0.15, n_conditions)
        by_task[("study", task)] = list(zip(predicted, human, strict=True))
    return by_task


def test_permutation_null_centres_on_zero_and_is_wide():
    """The bootstrap says how precisely r is known; the permutation null says whether r beats chance."""
    rng = np.random.default_rng(0)
    blind = metrics.condition_sensitivity(_movement(rng, 150, 3, signal=0.0), n_boot=200, n_perm=400)
    assert abs(blind["r"]) < 0.2
    assert blind["permutation_p"] > 0.05  # indistinguishable from chance
    assert blind["null_p95"] > 0.05  # and the null really is wide, not a point at zero

    real = metrics.condition_sensitivity(_movement(rng, 150, 3, signal=0.6), n_boot=200, n_perm=400)
    assert real["r"] > 0.5
    assert real["permutation_p"] < 0.01
    assert real["r"] > real["null_p95"]


def test_permutation_p_is_never_reported_as_exactly_zero():
    rng = np.random.default_rng(1)
    strong = metrics.condition_sensitivity(_movement(rng, 60, 4, signal=1.0), n_boot=100, n_perm=100)
    assert strong["permutation_p"] > 0  # (hits + 1) / (n + 1), so a finite run cannot claim p = 0
    assert strong["permutation_p"] <= 1 / 101 + 1e-12


def test_a_blind_predictor_has_no_permutation_p():
    flat = {("s", 0): [(0.5, 0.2), (0.5, 0.8)], ("s", 1): [(0.5, 0.3), (0.5, 0.6)]}
    assert metrics.condition_sensitivity(flat, n_boot=50, n_perm=50)["permutation_p"] is None
