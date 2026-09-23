import asyncio
import json

import pandas as pd
import pytest

from kite.eval.opinionqa import OpinionQATask, check_layout, human_distributions, load_wave, wave_dir
from kite.eval.runner import run_task
from kite.eval.task import Prediction
from kite.kernel.mock import MockKernel

WAVE = 26


def make_wave(root, wave=WAVE):
    """One wave of fixture data. Any wave number: the test split needs the waves it claims to cover."""
    folder = wave_dir(root, wave)
    folder.mkdir(parents=True)
    economy, guns = f"ECON_W{wave}", f"GUNS_W{wave}"
    pd.DataFrame(
        {
            "key": [economy, guns],
            "question": ["How is the economy?", "Should gun laws be stricter?"],
            "references": ["['Good', 'Fair', 'Poor', 'Refused']", "['Yes', 'No', 'Refused']"],
            "option_ordinal": ["[1.0, 2.0, 3.0]", "[1.0, 2.0]"],
        }
    ).to_csv(folder / "info.csv", index=False)
    pd.DataFrame({"key": ["POLPARTY"], "options": ["['Democrat', 'Republican']"]}).to_csv(folder / "metadata.csv", index=False)
    pd.DataFrame(
        {
            economy: ["Good", "Good", "Poor", "Poor", "Fair", "Refused"],
            guns: ["Yes", "Yes", "No", "No", None, "Yes"],
            "POLPARTY": ["Democrat", "Democrat", "Republican", "Republican", "Democrat", "Other"],
            f"WEIGHT_W{wave}": [1.0, 1.0, 1.0, 3.0, 2.0, 5.0],
        }
    ).to_csv(folder / "responses.csv", index=False)


@pytest.fixture
def root(tmp_path):
    make_wave(tmp_path)
    return tmp_path


def test_check_layout_lists_missing_files(root):
    assert check_layout(root, waves=[WAVE]) == []
    missing = check_layout(root, waves=[WAVE, 27])
    assert len(missing) == 3 and "W27" in missing[0]


def test_load_wave_drops_refusals_and_keeps_ordinals(root):
    questions, groups, _ = load_wave(root, WAVE)
    assert [q.key for q in questions] == ["ECON_W26", "GUNS_W26"]
    assert questions[0].options == ["Good", "Fair", "Poor"] and questions[0].ordinal == [1.0, 2.0, 3.0]
    assert groups == {"POLPARTY": ["Democrat", "Republican"]}


def test_human_distributions_are_weighted_and_exclude_refusals(root):
    questions, groups, responses = load_wave(root, WAVE)
    dists = human_distributions(questions, groups, responses, WAVE)
    # ECON overall: Good 1+1, Fair 2, Poor 1+3; the refusal (weight 5) is excluded
    assert dists[("ECON_W26", "Overall", "Overall")].tolist() == pytest.approx([2 / 8, 2 / 8, 4 / 8])
    assert dists[("ECON_W26", "POLPARTY", "Democrat")].tolist() == pytest.approx([2 / 4, 2 / 4, 0.0])
    assert dists[("ECON_W26", "POLPARTY", "Republican")].tolist() == pytest.approx([0.0, 0.0, 1.0])
    # GUNS overall: Yes 1+1+5, No 1+3; the missing answer is ignored
    assert dists[("GUNS_W26", "Overall", "Overall")].tolist() == pytest.approx([7 / 11, 4 / 11])


def test_default_mode_asks_each_question_once_without_a_persona(root):
    task = OpinionQATask(root, waves=[WAVE], mode="default")
    items = list(task.items())
    assert [item.item_id for item in items] == ["W26|ECON_W26|Overall|Overall", "W26|GUNS_W26|Overall|Overall"]
    assert isinstance(items[0].request.state["respondent"], str)
    assert list(items[0].request.questions["answer"].criteria) == ["Good", "Fair", "Poor"]


def test_steered_mode_asks_once_per_group(root):
    task = OpinionQATask(root, waves=[WAVE], mode="steered")
    items = list(task.items())
    assert len(items) == 4
    assert items[0].request.state["respondent"] == {"political_party": "Democrat"}
    assert items[0].meta == {"wave": 26, "key": "ECON_W26", "attribute": "POLPARTY", "group": "Democrat"}


def test_score_perfect_and_pooled(root):
    task = OpinionQATask(root, waves=[WAVE], mode="steered")
    perfect = [
        Prediction(item_id=item.item_id, probs=task.target(Prediction(item_id="", probs=[], meta=item.meta)), meta=item.meta) for item in task.items()
    ]
    scores = task.score(perfect)
    assert scores["alignment"] == pytest.approx(1.0)
    assert scores["js_mean"] == pytest.approx(0.0, abs=1e-9)
    assert set(scores["alignment_by_group"]) == {"POLPARTY|Democrat", "POLPARTY|Republican"}
    pooled = task.score(task.baseline("pooled"))
    assert pooled["alignment"] < 1.0
    # Republicans: ECON target [0, 0, 1] vs pooled [.25, .25, .5] -> 1 - 0.75 / 2; GUNS target [0, 1] vs [7/11, 4/11] -> 4/11
    assert pooled["alignment_by_group"]["POLPARTY|Republican"] == pytest.approx((0.625 + 4 / 11) / 2)


def test_default_mode_reports_alignment_with_every_group(root):
    task = OpinionQATask(root, waves=[WAVE], mode="default")
    scores = task.score(task.baseline("pooled"))
    assert scores["alignment"] == pytest.approx(1.0)  # pooled is the overall distribution itself
    assert set(scores["alignment_by_group"]) == {"POLPARTY|Democrat", "POLPARTY|Republican"}
    assert scores["alignment_by_group"]["POLPARTY|Republican"] < 1.0


def test_end_to_end_with_the_mock_kernel(root, tmp_path):
    task = OpinionQATask(root, waves=[WAVE], mode="steered", phrasing="p1", primitive="score")
    summary = asyncio.run(run_task(task, MockKernel(), tmp_path / "run"))
    assert (summary.n_predicted, summary.n_failed) == (4, 0)
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert 0.0 <= metrics["alignment"] <= 1.0
    assert len(task.probe_cases(3)) == 3
