import asyncio
import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kite.eval import metrics
from kite.eval.runner import read_predictions, run_task
from kite.eval.socsci210 import SocSci210Task, dev_studies, prepare, split_studies
from kite.eval.task import Prediction
from kite.kernel.mock import MockKernel

TAIL = " Only return an integer from 1 to 5, where 1 means Strongly disagree and 5 means Strongly agree, nothing else."


def demographic(age):
    return {
        "age": age,
        "education": "Bachelor's degree",
        "employment": "Retired",
        "ethnicity": None,
        "gender": "Male",
        "household_size": 2,
        "housing_ownership": "Owned",
        "housing_type": "A one-family house",
        "ideology": "Moderate",
        "income": "75-99K",
        "internet_access": "Internet Household",
        "location": "Iowa",
        "marital_status": "Married",
        "metro_status": "Metro Area",
        "party_id": "Lean Democrat",
        "phone_service": "Cellphone only",
    }


def make_raw(root):
    rows, sample_id = [], 0
    # study "aaaaa": 2 conditions x 1 task, 1-5 scale; condition 0 answers low, condition 1 answers high
    for condition, responses in [(0, [1, 1, 2, 2]), (1, [4, 5, 5, 5])]:
        for participant, response in enumerate(responses):
            rows.append(
                dict(
                    sample_id=sample_id,
                    participant=participant,
                    demographic=demographic(30 + participant),
                    stimuli=f'You read "vignette {condition}" and then were asked: "I support the policy."' + TAIL,
                    response=response,
                    condition_num=condition,
                    task_num=0,
                    prompt="long prompt",
                    reasoning="long reasoning",
                    study_id="aaaaa",
                )
            )
            sample_id += 1
    # study "bbbbb": no parseable range -> observed range 0..3; one out-of-scale study "ccccc" is never requested
    for participant, response in enumerate([0, 1, 3, 3]):
        rows.append(
            dict(
                sample_id=sample_id,
                participant=participant,
                demographic=demographic(50),
                stimuli="How many? Only return an integer, nothing else.",
                response=response,
                condition_num=0,
                task_num=0,
                prompt="p",
                reasoning="r",
                study_id="bbbbb",
            )
        )
        sample_id += 1
    rows.append(
        dict(
            sample_id=sample_id,
            participant=0,
            demographic=demographic(40),
            stimuli="x" + TAIL,
            response=3,
            condition_num=0,
            task_num=0,
            prompt="p",
            reasoning="r",
            study_id="ccccc",
        )
    )
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "train-00000-of-00001.parquet")
    (root / "metadata").mkdir()
    (root / "metadata" / "participant_mapping.json").write_text(json.dumps({"seen": ["ccccc"], "unseen": ["aaaaa", "bbbbb"]}))


def make_varying_raw(root):
    """A study whose conditions state different scales: the shape that a per-task scale silently corrupts.

    Modelled on the real data: `3ydty` swaps the option labels between conditions (the labels *are*
    the manipulation) and `53kjy` task 0 offers 0-4 in one condition and 0-5 in the other.
    """
    rows, sample_id = [], 0
    instructions = {
        (0, 0): " Only return an integer from 1 to 2, where 1 means Brian and 2 means Matt, nothing else.",
        (0, 1): " Only return an integer from 1 to 2, where 1 means Amy and 2 means Jennifer, nothing else.",
        (1, 0): " Only return an integer from 0 to 4, nothing else.",
        (1, 1): " Only return an integer from 0 to 5, nothing else.",
        # task 2 states no range at all, in either condition, and the conditions answer far apart
        (2, 0): " Only return an integer, nothing else.",
        (2, 1): " Only return an integer, nothing else.",
    }
    answers = {(0, 0): [1, 1, 2], (0, 1): [1, 2, 2], (1, 0): [0, 2, 4], (1, 1): [0, 3, 5], (2, 0): [0, 1, 2], (2, 1): [4, 5, 6]}
    for (task, condition), tail in instructions.items():
        for participant, response in enumerate(answers[(task, condition)]):
            rows.append(
                dict(
                    sample_id=sample_id,
                    participant=participant,
                    demographic=demographic(30 + participant),
                    stimuli="Whose name should the couple use?" + tail,
                    response=response,
                    condition_num=condition,
                    task_num=task,
                    prompt="p",
                    reasoning="r",
                    study_id="ddddd",
                )
            )
            sample_id += 1
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "train-00000-of-00001.parquet")


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw)
    summary = prepare(raw, tmp_path / "prepared", split_studies(raw)["unseen"])
    return tmp_path / "prepared", summary


@pytest.fixture
def varying(tmp_path):
    make_varying_raw(tmp_path / "raw")
    summary = prepare(tmp_path / "raw", tmp_path / "prepared", ["ddddd"])
    return tmp_path / "prepared", summary


def test_split_and_dev_studies(tmp_path):
    make_raw(tmp_path / "raw")
    assert split_studies(tmp_path / "raw")["unseen"] == ["aaaaa", "bbbbb"]
    seen = [f"s{i:03d}" for i in range(170)]
    assert dev_studies(seen) == dev_studies(list(reversed(seen)))
    assert len(dev_studies(seen)) == 20


def _one_hot(meta):
    probs = [0.0] * meta["n_levels"]
    probs[meta["true_index"]] = 1.0
    return probs


def test_prepare_filters_studies_and_tabulates_cells(prepared):
    root, summary = prepared
    assert (summary["n_rows"], summary["n_studies"], summary["n_tasks"], summary["n_cells"]) == (12, 2, 2, 3)
    assert summary["n_cells_observed_range"] == 1
    scales = json.loads((root / "scales.json").read_text())
    # one scale per (study, condition, task), the same key as cells.json
    assert sorted(scales) == ["aaaaa|0|0", "aaaaa|1|0", "bbbbb|0|0"]
    assert not (root / "tasks.json").exists()  # the old per-task file must not survive a re-prepare
    assert scales["aaaaa|0|0"]["lo"] == 1 and scales["aaaaa|0|0"]["hi"] == 5
    assert (scales["bbbbb|0|0"]["lo"], scales["bbbbb|0|0"]["hi"], scales["bbbbb|0|0"]["source"]) == (0, 3, "observed")
    cells = json.loads((root / "cells.json").read_text())
    assert cells["aaaaa|0|0"] == [2, 2, 0, 0, 0]
    assert cells["aaaaa|1|0"] == [0, 0, 0, 1, 3]
    # these conditions agree, so nothing is flagged
    assert (summary["n_tasks_range_varies"], summary["n_tasks_labels_vary"], summary["n_tasks_conditions_disagree"]) == (0, 0, 0)


def test_each_condition_keeps_its_own_range_and_labels(varying):
    root, summary = varying
    scales = json.loads((root / "scales.json").read_text())
    # task 0: same 1-2 range, but the labels are the manipulation
    assert scales["ddddd|0|0"]["labels"] == {"1": "Brian", "2": "Matt"}
    assert scales["ddddd|1|0"]["labels"] == {"1": "Amy", "2": "Jennifer"}
    # task 1: the conditions state different ranges
    assert (scales["ddddd|0|1"]["lo"], scales["ddddd|0|1"]["hi"]) == (0, 4)
    assert (scales["ddddd|1|1"]["lo"], scales["ddddd|1|1"]["hi"]) == (0, 5)
    assert json.loads((root / "cells.json").read_text())["ddddd|1|1"] == [1, 0, 0, 1, 0, 1]  # six levels, not five
    assert (summary["n_tasks"], summary["n_cells"]) == (3, 6)
    assert (summary["n_tasks_range_varies"], summary["n_tasks_labels_vary"], summary["n_tasks_conditions_disagree"]) == (1, 1, 2)


def test_the_no_range_fallback_is_read_across_the_task_not_within_the_condition(varying):
    """Otherwise each cell is handed the exact span of the answers it is scored against."""
    root, summary = varying
    scales = json.loads((root / "scales.json").read_text())
    # task 2 states no range; its conditions answered 0-2 and 4-6, so both get the task's 0-6
    assert (scales["ddddd|0|2"]["lo"], scales["ddddd|0|2"]["hi"], scales["ddddd|0|2"]["source"]) == (0, 6, "observed")
    assert (scales["ddddd|1|2"]["lo"], scales["ddddd|1|2"]["hi"]) == (0, 6)
    assert summary["n_cells_observed_range"] == 2
    cells = json.loads((root / "cells.json").read_text())
    # on the shared grid the two conditions are visibly different; a per-condition range would make
    # both [1, 1, 1] and erase the condition effect that G1b exists to measure
    assert cells["ddddd|0|2"] == [1, 1, 1, 0, 0, 0, 0]
    assert cells["ddddd|1|2"] == [0, 0, 0, 0, 1, 1, 1]
    assert metrics.mean_position(cells["ddddd|0|2"]) != pytest.approx(metrics.mean_position(cells["ddddd|1|2"]))


def test_items_are_asked_with_their_own_conditions_options(varying):
    task = SocSci210Task(varying[0])
    asked = {(item.meta["condition_num"], item.meta["task_num"]): list(item.request.questions["answer"].criteria) for item in task.items()}
    assert asked[(0, 0)] == ["1: Brian", "2: Matt"]
    assert asked[(1, 0)] == ["1: Amy", "2: Jennifer"]  # not condition 0's names
    assert len(asked[(0, 1)]) == 5 and len(asked[(1, 1)]) == 6
    # and the answer levels line up with the condition's own scale
    levels = {(item.meta["condition_num"], item.meta["task_num"]): item.meta["n_levels"] for item in task.items()}
    assert levels[(0, 1)] == 5 and levels[(1, 1)] == 6


def test_pooled_baseline_merges_conditions_that_state_different_ranges_on_the_shared_axis(varying):
    task = SocSci210Task(varying[0])
    by_cell = {(p.meta["condition_num"], p.meta["task_num"]): tuple(p.probs) for p in task.baseline("pooled")}
    # task 0's conditions share the 1-2 range, so they pool the obvious way
    assert by_cell[(0, 0)] == by_cell[(1, 0)] == pytest.approx((3 / 6, 3 / 6))
    # task 1's conditions state 0-4 and 0-5; each cell still gets the merged shape, on its own grid
    assert len(by_cell[(0, 1)]) == 5 and len(by_cell[(1, 1)]) == 6
    # and the baseline stays condition-blind: one mean position for the whole task
    assert metrics.mean_position(by_cell[(0, 1)]) == pytest.approx(metrics.mean_position(by_cell[(1, 1)]))
    assert metrics.mean_position(by_cell[(0, 0)]) == pytest.approx(metrics.mean_position(by_cell[(1, 0)]))


def test_condition_blind_baselines_stay_blind_when_the_conditions_state_different_ranges(varying):
    task = SocSci210Task(varying[0])
    for kind in ("uniform", "pooled"):
        scores = task.score(task.baseline(kind))
        assert scores["condition_sensitivity_r"] in (None, 0.0), kind
        assert scores["condition_spread_ratio"] == pytest.approx(0.0, abs=1e-12), kind
        assert scores["condition_sensitivity_n_tasks"] == 3, kind


def test_items_carry_persona_question_and_described_options(prepared):
    task = SocSci210Task(prepared[0])
    item = next(task.items())
    assert item.item_id == "aaaaa:0"  # sample ids restart per study, so the study is part of the id
    assert item.request.state["respondent"]["age"] == "in their 30s"
    assert item.request.state["survey"]["question"].endswith('"I support the policy.')
    assert "Only return" not in item.request.state["survey"]["question"]
    criteria = list(item.request.questions["answer"].criteria)
    assert criteria[0] == "1: Strongly disagree" and criteria[4] == "5: Strongly agree"
    assert item.meta == {
        "study_id": "aaaaa",
        "condition_num": 0,
        "task_num": 0,
        "participant": 0,
        "true_index": 0,
        "n_levels": 5,
    }


def test_max_per_cell_subsamples_deterministically(prepared):
    first = [item.item_id for item in SocSci210Task(prepared[0], max_per_cell=2).items()]
    second = [item.item_id for item in SocSci210Task(prepared[0], max_per_cell=2).items()]
    assert first == second
    assert len(first) == 6  # three cells, two rows each


def test_perfect_predictions_score_perfectly(prepared):
    task = SocSci210Task(prepared[0])
    perfect = [
        Prediction(item_id=item.item_id, probs=task.target(Prediction(item_id="", probs=[], meta=item.meta)), meta=item.meta) for item in task.items()
    ]
    scores = task.score(perfect)
    assert scores["accuracy_micro"] == pytest.approx(1.0)
    assert scores["distribution"] == pytest.approx(0.0)
    assert scores["brier"] == pytest.approx(0.0)
    assert (scores["n_predictions"], scores["n_cells"], scores["n_studies"]) == (12, 3, 2)


def test_pooled_baseline_ignores_the_condition_and_uniform_is_flat(prepared):
    task = SocSci210Task(prepared[0])
    pooled = {p.item_id: p for p in task.baseline("pooled")}
    assert pooled["aaaaa:0"].probs == pytest.approx([2 / 8, 2 / 8, 0, 1 / 8, 3 / 8])  # both conditions merged
    assert pooled["aaaaa:0"].probs == pooled["aaaaa:7"].probs
    uniform = task.baseline("uniform")
    assert uniform[0].probs == [0.2] * 5
    # the pooled baseline cannot tell the two conditions apart, so it is far from each cell's own distribution
    assert task.score(list(pooled.values()))["distribution"] > 0.2
    with pytest.raises(ValueError):
        task.baseline("nope")


def test_condition_sensitivity_is_perfect_for_predictions_that_are_each_cells_own_distribution(prepared):
    """The cell-wise oracle tracks every condition exactly, so r is 1 and the slope is 1."""
    task = SocSci210Task(prepared[0])
    cells = json.loads((prepared[0] / "cells.json").read_text())
    predictions = [
        Prediction(item_id=item.item_id, probs=cells["{study_id}|{condition_num}|{task_num}".format(**item.meta)], meta=item.meta)
        for item in task.items()
    ]
    scores = task.score(predictions)
    assert scores["condition_sensitivity_r"] == pytest.approx(1.0)
    assert scores["condition_sensitivity_slope"] == pytest.approx(1.0)
    assert scores["condition_spread_ratio"] == pytest.approx(1.0)
    # only study aaaaa has two conditions; bbbbb has one and contributes nothing
    assert (scores["condition_sensitivity_n_tasks"], scores["condition_sensitivity_n_cells"]) == (1, 2)
    assert scores["condition_sensitivity_ci_low"] == pytest.approx(1.0)


def test_condition_blind_baselines_have_no_condition_sensitivity(prepared):
    """G1b's floor: uniform and pooled cannot tell conditions apart, so their r must not be a number."""
    task = SocSci210Task(prepared[0])
    for kind in ("uniform", "pooled"):
        scores = task.score(task.baseline(kind))
        assert scores["condition_sensitivity_r"] in (None, 0.0), kind
        assert scores["condition_sensitivity_slope"] in (None, 0.0), kind
        assert scores["condition_spread_ratio"] == 0.0, kind  # the prediction does not move at all
        assert scores["condition_sensitivity_n_tasks"] == 1, kind  # it is measured, and it is flat


def test_score_reports_the_distribution_under_both_rescaling_conventions(prepared):
    task = SocSci210Task(prepared[0])
    scores = task.score(task.baseline("uniform"))
    assert scores["distribution"] > 0 and scores["distribution_published_convention"] > 0
    # the conventions differ: our cells' humans do not span their whole stated scale
    assert scores["distribution_published_convention"] != pytest.approx(scores["distribution"])
    # a perfect prediction is zero under both
    cells = json.loads((prepared[0] / "cells.json").read_text())
    oracle = [
        Prediction(item_id=item.item_id, probs=cells["{study_id}|{condition_num}|{task_num}".format(**item.meta)], meta=item.meta)
        for item in task.items()
    ]
    assert task.score(oracle)["distribution_published_convention"] == pytest.approx(0.0)


def test_score_reports_both_accuracy_conventions_and_parity_uses_the_drawing_one(prepared):
    """A paper table must not put the paper's drawn 0.612 next to our median-rule number."""
    task = SocSci210Task(prepared[0])
    scores = task.score(task.baseline("uniform"))
    # the median rule beats drawing, because the middle level is a better guess than a random one
    assert scores["accuracy_micro"] > scores["accuracy_micro_random_draw"]
    assert scores["accuracy_macro"] > scores["accuracy_macro_random_draw"]
    # and the drawing convention is exactly what parity_uniform reports, so the two are comparable
    parity = task.parity_uniform()
    assert scores["accuracy_micro_random_draw"] == pytest.approx(parity["accuracy_micro"])
    assert scores["accuracy_macro_random_draw"] == pytest.approx(parity["accuracy_macro"])
    # a prediction that puts all its mass on the right answer is perfect under both
    perfect = [Prediction(item_id=item.item_id, probs=_one_hot(item.meta), meta=item.meta) for item in task.items()]
    assert task.score(perfect)["accuracy_micro_random_draw"] == pytest.approx(1.0)
    assert task.score(perfect)["accuracy_micro"] == pytest.approx(1.0)


def test_score_reports_a_robust_entropy_ratio_and_counts_near_unanimous_cells(prepared):
    task = SocSci210Task(prepared[0])
    scores = task.score(task.baseline("uniform"))
    assert scores["entropy_ratio_median"] is not None
    assert scores["entropy_ratio_p25"] <= scores["entropy_ratio_median"] <= scores["entropy_ratio_p75"]
    assert len(scores["entropy_ratios"]) == scores["n_cells"]  # the full list is still persisted
    # the fixture's humans are spread out, so no cell is near-unanimous
    assert scores["n_low_human_entropy"] == 0


def test_parity_uniform_matches_a_hand_computation(prepared):
    parity = SocSci210Task(prepared[0]).parity_uniform()
    assert parity["distribution_published_convention"] > 0
    # study aaaaa, responses 1,1,2,2,4,5,5,5 on a 1-5 scale; expected |k - r| / 4 under a uniform guess:
    # r=1 -> 0.5, r=2 -> 0.35, r=4 -> 0.35, r=5 -> 0.5
    study_a = 1 - (0.5 + 0.5 + 0.35 + 0.35 + 0.35 + 0.5 + 0.5 + 0.5) / 8
    # study bbbbb, responses 0,1,3,3 on a 0-3 scale: r=0 -> 0.5, r=1 -> 1/3, r=3 -> 0.5
    study_b = 1 - (0.5 + 1 / 3 + 0.5 + 0.5) / 4
    assert parity["accuracy_macro"] == pytest.approx((study_a + study_b) / 2)
    assert 0 < parity["distribution"] < 0.5


def test_end_to_end_with_the_mock_kernel(prepared, tmp_path):
    task = SocSci210Task(prepared[0], phrasing="p1", primitive="score")
    summary = asyncio.run(run_task(task, MockKernel(), tmp_path / "run"))
    assert summary.n_predicted == 12 and summary.n_failed == 0
    predictions = read_predictions(tmp_path / "run")
    assert all(len(p.probs) == p.meta["n_levels"] for p in predictions)
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert 0 <= metrics["accuracy_macro"] <= 1 and metrics["distribution"] >= 0
    assert len(task.probe_cases(3)) == 3


@pytest.fixture
def panel(tmp_path):
    """Four people per condition, each answering three tasks - a panel you can correlate."""
    rows, sample_id = [], 0
    for condition in (0, 1):
        for participant in range(4):
            for task in range(3):
                rows.append(
                    dict(
                        sample_id=sample_id,
                        participant=participant,
                        demographic=demographic(30 + participant),
                        stimuli=f'You were asked: "Statement {task}."' + TAIL,
                        response=1 + (participant + task) % 5,
                        condition_num=condition,
                        task_num=task,
                        prompt="p",
                        reasoning="r",
                        study_id="panel",
                    )
                )
                sample_id += 1
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), raw / "data" / "train-00000-of-00001.parquet")
    (raw / "metadata").mkdir()
    (raw / "metadata" / "participant_mapping.json").write_text(json.dumps({"seen": [], "unseen": ["panel"]}))
    prepare(raw, tmp_path / "prepared", ["panel"])
    return tmp_path / "prepared"


def test_max_participants_keeps_whole_people_across_tasks(panel):
    """Inter-item correlation needs the same person on every task; row subsampling would not give that."""
    by_person = defaultdict(set)
    for item in SocSci210Task(panel, max_participants=2).items():
        m = item.meta
        by_person[(m["condition_num"], m["participant"])].add(m["task_num"])

    assert len(by_person) == 4  # two conditions x two people
    assert all(tasks == {0, 1, 2} for tasks in by_person.values())  # each person brings every task

    # the row-wise option cannot do this: each task gets its own slice of the panel
    rowwise = defaultdict(set)
    for item in SocSci210Task(panel, max_per_cell=2).items():
        m = item.meta
        rowwise[(m["condition_num"], m["task_num"])].add(m["participant"])
    assert any(rowwise[(0, 0)] != rowwise[(0, task)] for task in (1, 2))


def test_max_participants_is_deterministic_and_excludes_max_per_cell(panel):
    first = [item.item_id for item in SocSci210Task(panel, max_participants=2).items()]
    second = [item.item_id for item in SocSci210Task(panel, max_participants=2).items()]
    assert first == second
    with pytest.raises(ValueError, match="at most one"):
        SocSci210Task(panel, max_per_cell=2, max_participants=2)


def test_item_ids_are_unique_across_studies(prepared):
    """Sample ids restart in every study; resume keys on the item id, so a clash would skip real work."""
    ids = [item.item_id for item in SocSci210Task(prepared[0]).items()]
    assert len(ids) == len(set(ids)) == 12
    assert {i.split(":")[0] for i in ids} == {"aaaaa", "bbbbb"}
