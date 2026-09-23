import json

import numpy as np
import pytest

from kite.eval.protocol import ProtocolError, RunConfig, Split, TaskConfig
from kite.eval.recalibrate import apply_temperature, apply_to_run, fit_from_run, fit_temperature, load_fit_for, read_fit
from kite.eval.runner import read_predictions, write_predictions
from kite.eval.task import Prediction


def test_temperature_one_is_the_identity():
    assert apply_temperature([0.2, 0.3, 0.5], 1.0) == pytest.approx([0.2, 0.3, 0.5])


def test_high_temperature_flattens_and_low_temperature_sharpens():
    flat = apply_temperature([0.1, 0.9], 4.0)
    sharp = apply_temperature([0.1, 0.9], 0.25)
    assert 0.5 < flat[1] < 0.9 < sharp[1]
    assert sum(flat) == pytest.approx(1.0) and sum(sharp) == pytest.approx(1.0)


def test_zero_probabilities_do_not_break_the_logarithm():
    scaled = apply_temperature([0.0, 1.0], 2.0)
    assert sum(scaled) == pytest.approx(1.0) and scaled[0] < 0.01


def test_fit_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(5), size=300)
    too_sharp = [apply_temperature(row, 0.5) for row in truth]  # a model twice as sharp as the truth
    assert fit_temperature(too_sharp, truth.tolist()) == pytest.approx(2.0, rel=0.02)


class OneHotTask:
    name = "toy"

    def target(self, prediction):
        one_hot = [0.0, 0.0, 0.0]
        one_hot[prediction.meta["true_index"]] = 1.0
        return one_hot

    def score(self, predictions):
        return {"n_predictions": len(predictions), "mean_top": float(np.mean([max(p.probs) for p in predictions]))}


def make_run(path, split, *, n=400, seed=1, phrasing="p1", commit="abc1234"):
    """A run directory as `eval` writes one: predictions plus a config that says what they are."""
    rng = np.random.default_rng(seed)
    predictions = []
    for index in range(n):
        truth = rng.dirichlet(np.ones(3))
        predictions.append(Prediction(item_id=str(index), probs=apply_temperature(truth, 0.5), meta={"true_index": int(rng.choice(3, p=truth))}))
    write_predictions(path, predictions)
    task = TaskConfig(name="opinionqa", split=split, phrasing=phrasing, primitive="choice", mode="steered", members=["W26"])
    RunConfig(task=task, kernel={"backend": "mock", "model_id": "mock-1"}, git_commit=commit).write(path)
    return predictions


def test_fit_from_run_and_apply_to_run(tmp_path):
    predictions = make_run(tmp_path / "dev", Split.dev)
    make_run(tmp_path / "test", Split.test, n=200, seed=2)

    temperature = fit_from_run(OneHotTask(), tmp_path / "dev")
    assert temperature == pytest.approx(2.0, rel=0.25)  # noisier: fitted on sampled one-hot answers
    assert json.loads((tmp_path / "dev" / "temperature.json").read_text())["n_fit"] == 400

    fit, target = load_fit_for(tmp_path / "dev", tmp_path / "test")
    assert (fit.temperature, fit.task.split, fit.git_commit) == (temperature, Split.dev, "abc1234")
    assert target.task.split is Split.test

    metrics = apply_to_run(OneHotTask(), tmp_path / "test", tmp_path / "out", fit)
    assert metrics["temperature"] == temperature
    # the recalibrated run names its fit source, that source's split and its commit
    assert metrics["temperature_fitted_on"] == str(tmp_path / "dev")
    assert (metrics["temperature_fit_split"], metrics["temperature_fit_commit"], metrics["temperature_n_fit"]) == ("dev", "abc1234", 400)
    assert metrics["mean_top"] < np.mean([max(p.probs) for p in predictions])  # flattened
    assert len(read_predictions(tmp_path / "out")) == 200


def test_fitting_on_a_test_run_is_refused(tmp_path):
    make_run(tmp_path / "test", Split.test, n=50)
    with pytest.raises(ProtocolError, match="split=test"):
        fit_from_run(OneHotTask(), tmp_path / "test")
    assert not (tmp_path / "test" / "temperature.json").exists()


def test_fitting_needs_a_run_config(tmp_path):
    write_predictions(tmp_path / "bare", [Prediction(item_id="0", probs=[0.5, 0.5], meta={"true_index": 0})])
    with pytest.raises(ProtocolError, match="does not say what it is"):
        fit_from_run(OneHotTask(), tmp_path / "bare")


def test_applying_to_the_fit_source_or_to_a_second_dev_run_is_refused(tmp_path):
    make_run(tmp_path / "dev", Split.dev, n=50)
    make_run(tmp_path / "dev2", Split.dev, n=50, seed=3)
    fit_from_run(OneHotTask(), tmp_path / "dev")
    with pytest.raises(ProtocolError, match="the run it was fitted on"):
        load_fit_for(tmp_path / "dev", tmp_path / "dev")
    with pytest.raises(ProtocolError, match="which is a dev run"):
        load_fit_for(tmp_path / "dev", tmp_path / "dev2")


def test_applying_across_a_different_phrasing_or_kernel_is_refused(tmp_path):
    make_run(tmp_path / "dev", Split.dev, n=50)
    make_run(tmp_path / "test", Split.test, n=50, seed=4, phrasing="p3")
    fit_from_run(OneHotTask(), tmp_path / "dev")
    with pytest.raises(ProtocolError, match="phrasing: fitted on p1 but applying to p3"):
        load_fit_for(tmp_path / "dev", tmp_path / "test")

    make_run(tmp_path / "other", Split.test, n=50, seed=5)
    RunConfig.read(tmp_path / "other").model_copy(update={"kernel": {"backend": "jev", "model_id": "jev-1.13.0"}}).write(tmp_path / "other")
    with pytest.raises(ProtocolError, match="kernel: fitted on"):
        load_fit_for(tmp_path / "dev", tmp_path / "other")


def test_a_missing_temperature_says_what_to_run(tmp_path):
    make_run(tmp_path / "dev", Split.dev, n=10)
    with pytest.raises(ProtocolError, match="recalibrate-fit <dev run>"):
        read_fit(tmp_path / "dev")
