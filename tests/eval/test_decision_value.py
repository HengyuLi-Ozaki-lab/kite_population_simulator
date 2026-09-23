import zlib

import numpy as np
import pandas as pd
import pytest

from kite.eval import decision_value as dv
from kite.eval.scales import Scale
from kite.eval.task import Prediction

SEVEN = Scale(lo=1, hi=7)


def rows_for(means, *, study="s1", task=0, people=400, repeats=1, seed=0):
    """One cell per condition mean: `people` respondents, each answering `repeats` times around a personal level."""
    rng = np.random.default_rng(seed)
    out = []
    for condition, mean in enumerate(means):
        levels = np.clip(rng.normal(mean, 1.2, people), 1, 7)
        for participant, level in enumerate(levels):
            for _ in range(repeats):
                out.append(
                    {
                        "study_id": study,
                        "participant": participant + 10_000 * condition,
                        "condition_num": condition,
                        "task_num": task,
                        "response": int(np.clip(round(level + rng.normal(0, 0.3)), 1, 7)),
                    }
                )
    return pd.DataFrame(out)


def scales_for(n_conditions, *, study="s1", task=0, scale=SEVEN):
    return {f"{study}|{condition}|{task}": scale for condition in range(n_conditions)}


def test_comparable_tasks_need_one_scale_across_conditions():
    same = scales_for(3)
    varying = {"s2|0|0": Scale(lo=1, hi=5, labels={1: "Brian", 5: "Matt"}), "s2|1|0": Scale(lo=1, hi=5, labels={1: "Amy", 5: "Jennifer"})}
    alone = {"s3|0|0": SEVEN}
    assert dv.comparable_tasks({**same, **varying, **alone}) == {("s1", 0): [0, 1, 2]}


def test_standard_errors_cluster_on_the_respondent():
    """Four rows from one person are not four independent answers; row-wise errors would be about half as wide."""
    once = dv.human_cells(rows_for([4.0], repeats=1), scales_for(1))
    four = dv.human_cells(rows_for([4.0], repeats=4), scales_for(1))
    assert four.n_people[0] == once.n_people[0] == 400
    assert four.se[0] == pytest.approx(once.se[0], rel=0.15)  # more rows, same people: barely more precise
    naive = np.sqrt(np.var(rows_for([4.0], repeats=4)["response"].sub(1).div(6), ddof=1) / 1600)
    assert four.se[0] > 1.7 * naive


def test_halves_put_each_respondent_wholly_in_one_half_and_carry_their_own_errors():
    human = dv.human_cells(rows_for([3.0, 5.0], repeats=3), scales_for(2))
    in_a = np.random.default_rng(5).integers(0, 2, human.n_persons)[human.group_person].astype(bool)
    (mean_a, se_a), (mean_b, se_b) = human.halves(np.random.default_rng(5))
    for cell in range(2):
        rows_a = human.group_count[(human.group_cell == cell) & in_a].sum()
        rows_b = human.group_count[(human.group_cell == cell) & ~in_a].sum()
        assert rows_a % 3 == 0 and rows_b % 3 == 0  # whole people, three rows each
        assert mean_a[cell] * rows_a + mean_b[cell] * rows_b == pytest.approx(human.mean[cell] * (rows_a + rows_b))
        assert se_a[cell] == pytest.approx(human.se[cell] * np.sqrt(2), rel=0.2)  # half the people, errors about sqrt(2) wider


def blocks_for(means, predicted):
    human = dv.human_cells(rows_for(means), scales_for(len(means)))
    keys = [f"s1|{c}|0" for c in range(len(means))]
    return human, dv.build_blocks(dv.comparable_tasks(scales_for(len(means))), human, dict(zip(keys, predicted, strict=True)))


def test_a_simulation_that_orders_conditions_like_people_scores_one_and_the_reverse_scores_minus_one():
    means = [2.5, 3.5, 4.5, 5.5]
    _, right = blocks_for(means, [0.2, 0.4, 0.6, 0.8])
    assert dv.evaluate(right).summary() == {"sign_accuracy": 1.0, "captured_gain": pytest.approx(1.0)}
    _, wrong = blocks_for(means, [0.8, 0.6, 0.4, 0.2])
    assert dv.evaluate(wrong).summary() == {"sign_accuracy": 0.0, "captured_gain": pytest.approx(-1.0)}


def test_a_simulation_that_cannot_tell_conditions_apart_scores_chance_not_whatever_comes_first():
    _, flat = blocks_for([2.5, 3.5, 4.5, 5.5], [0.5, 0.5, 0.5, 0.5])
    assert dv.evaluate(flat).summary() == {"sign_accuracy": 0.5, "captured_gain": pytest.approx(0.0)}


def test_only_reliable_human_contrasts_count_towards_sign_accuracy():
    """Conditions 0 and 1 are the same population, so their contrast is noise and must not be scored."""
    _, blocks = blocks_for([4.0, 4.0, 5.5], [0.9, 0.1, 0.95])
    parts = dv.evaluate(blocks)
    assert parts.sign_n[0] == 2  # (0, 2) and (1, 2); the (0, 1) contrast is within noise
    assert parts.summary()["sign_accuracy"] == 1.0


def test_the_permutation_null_sits_at_chance_and_a_real_signal_clears_it():
    rng = np.random.default_rng(0)
    frames, scales, predicted = [], {}, {}
    for task in range(30):
        means = rng.uniform(2.5, 5.5, 4)
        frames.append(rows_for(means, task=task, people=300, seed=task))
        scales.update(scales_for(4, task=task))
        for condition, mean in enumerate(means):
            predicted[f"s1|{condition}|{task}"] = (mean - 1) / 6 + rng.normal(0, 0.03)
    human = dv.human_cells(pd.concat(frames), scales)
    blocks = dv.build_blocks(dv.comparable_tasks(scales), human, predicted)
    observed = dv.evaluate(blocks).summary()
    null = dv.permutation_null(blocks, n_perm=300, rng=np.random.default_rng(1))
    assert abs(null["sign_accuracy"].mean() - 0.5) < 0.05 and abs(null["captured_gain"].mean()) < 0.08
    assert observed["sign_accuracy"] > np.percentile(null["sign_accuracy"], 99)
    assert observed["captured_gain"] > np.percentile(null["captured_gain"], 99)

    comparison = dv.half_sample_comparison(blocks, human, n_splits=20, rng=np.random.default_rng(2))
    assert comparison["pilot_sign_accuracy"] > 0.95  # effects this large replicate across halves
    assert comparison["simulation_sign_accuracy"] > 0.85

    low, high = dv.bootstrap_by_study(dv.evaluate(blocks), n_boot=50, rng=np.random.default_rng(3))["captured_gain"]
    assert low == pytest.approx(high)  # one study: resampling studies cannot move anything, and says so honestly

    curve = dv.effect_curve(blocks, human, n_splits=10, rng=np.random.default_rng(4))
    assert 175 < curve["contrasts_per_split"].sum() <= 30 * 6  # a judged difference of exactly zero is not scored
    assert curve.iloc[-1]["pilot"] > curve.iloc[0]["pilot"]  # bigger effects replicate better

    by_reliability = dv.reliability_curve(blocks)
    assert by_reliability["contrasts"].sum() == 30 * 6
    assert (by_reliability["simulation"] <= by_reliability["ceiling"] + 0.15).all()


def test_binning_by_the_judge_keeps_the_pilot_at_chance_where_there_is_no_effect():
    """Binning on the full sample would force two halves of a null effect to disagree; binning on the judge must not."""
    frames, scales, predicted = [], {}, {}
    for task in range(40):
        frames.append(rows_for([4.0, 4.0, 4.0], task=task, people=200, seed=100 + task))
        scales.update(scales_for(3, task=task))
        predicted.update({f"s1|{c}|{task}": 0.5 + 0.01 * c for c in range(3)})
    human = dv.human_cells(pd.concat(frames), scales)
    blocks = dv.build_blocks(dv.comparable_tasks(scales), human, predicted)
    curve = dv.effect_curve(blocks, human, n_splits=40, rng=np.random.default_rng(0))
    weighted = np.average(curve["pilot"], weights=curve["contrasts_per_split"])
    assert weighted == pytest.approx(0.5, abs=0.06)
    assert curve.iloc[0]["pilot"] > 0.4  # the smallest bin is not pushed below chance


def prediction(item_id, probs, condition=0):
    return Prediction(item_id=item_id, probs=probs, meta={"study_id": "s1", "condition_num": condition, "task_num": 0})


def test_predicted_means_nest_smaller_experiments_in_larger_ones_and_accept_both_id_formats():
    samples = [str(i) for i in range(6)]
    probs = {s: [1 - i / 10, i / 10] for i, s in enumerate(samples)}
    predictions = [prediction(f"s1:{s}", probs[s]) for s in samples]
    everyone = dv.predicted_means(predictions)["s1|0|0"]
    assert everyone == pytest.approx(np.mean([p[1] for p in probs.values()]))

    first_two = sorted(samples, key=lambda s: zlib.crc32(f"0:{s}".encode()))[:2]
    expected = np.mean([probs[s][1] for s in first_two])
    assert dv.predicted_means(predictions, n_per_cell=2)["s1|0|0"] == pytest.approx(expected)
    bare = [prediction(s, probs[s]) for s in samples]  # runs made before ids carried the study
    assert dv.predicted_means(bare, n_per_cell=2)["s1|0|0"] == pytest.approx(expected)
