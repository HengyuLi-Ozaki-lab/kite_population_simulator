"""The dev/test discipline, exercised through the command line exactly as an operator would break it.

Every test here is a refusal that used to be a silent success: a split that resolved to the test
data under a development label, a temperature fitted and applied to the same run, a resume that
changed the phrasing halfway through a predictions file.
"""

import json
import shutil

import yaml
from typer.testing import CliRunner

from kite.cli import REFUSED, app
from tests.conftest import only_run

runner = CliRunner()


def flat(result) -> str:
    """The rich-formatted error box, unwrapped, so an assertion is not at the mercy of the column width."""
    return " ".join(result.output.split())


def test_split_is_a_closed_set_so_a_typo_cannot_select_the_test_data(workspace):
    """`--split Dev`, `'dev '`, `dve` and `''` all used to resolve to the test set."""
    for spelling in ["Dev", "dev ", "dve", "", "TEST"]:
        result = runner.invoke(app, ["eval", "opinionqa", "--split", spelling])
        assert result.exit_code == 2, (spelling, result.output)
        assert "Invalid value for '--split'" in flat(result)
        assert "'dev', 'test'" in flat(result)


def test_prepare_refuses_a_misspelled_split_instead_of_writing_test_studies_under_it(workspace):
    result = runner.invoke(app, ["prepare-socsci210", "--split", "Dev"])
    assert result.exit_code == 2, result.output
    assert "Invalid value for '--split'" in flat(result)
    assert not (workspace / "data" / "socsci210" / "prepared" / "Dev").exists()


def test_every_choice_option_is_closed(workspace):
    """A typo in any of these used to crash late or silently take the default."""
    for args, where in [
        (["eval", "nosuchtask"], "'task'"),
        (["eval", "opinionqa", "--kernel", "Mock"], "'--kernel'"),
        (["eval", "opinionqa", "--phrasing", "P1"], "'--phrasing'"),
        (["eval", "opinionqa", "--primitive", "choise"], "'--primitive'"),
        (["eval", "opinionqa", "--mode", "Steered"], "'--mode'"),
        (["eval", "opinionqa", "--answer-mode", "probability"], "'--answer-mode'"),
        (["eval", "opinionqa", "--provider", "OpenAI"], "'--provider'"),
        (["baseline", "opinionqa", "poooled"], "'kind'"),
        (["probe", "--kernel", "jev2"], "'--kernel'"),
        (["parity-socsci210", "--split", "Test"], "'--split'"),
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 2, (args, result.output)
        assert f"Invalid value for {where}" in flat(result), (args, flat(result))


def test_a_run_records_the_waves_it_evaluated_not_only_the_splits_name(workspace):
    assert runner.invoke(app, ["eval", "opinionqa", "--split", "dev", "--mode", "steered"]).exit_code == 0
    config = yaml.safe_load((only_run(workspace, "opinionqa-dev-mock") / "config.yaml").read_text())
    assert config["task"]["split"] == "dev"
    assert config["task"]["members"] == ["W26"]  # the resolved membership, read off the data that loaded


def test_a_socsci210_run_records_the_studies_it_evaluated(workspace):
    assert runner.invoke(app, ["prepare-socsci210", "--split", "test"]).exit_code == 0
    summary = json.loads((workspace / "data" / "socsci210" / "prepared" / "test" / "summary.json").read_text())
    assert summary["studies"] == ["aaaaa", "bbbbb"]  # the prepared directory says what is inside it


def test_the_baseline_run_is_self_describing_too(workspace):
    assert runner.invoke(app, ["baseline", "opinionqa", "pooled", "--split", "dev"]).exit_code == 0
    config = yaml.safe_load((only_run(workspace, "baseline-pooled") / "config.yaml").read_text())
    assert config["task"]["members"] == ["W26"]
    assert config["git_commit"] is None or isinstance(config["git_commit"], str)


def run_eval(workspace, split, *extra):
    result = runner.invoke(app, ["eval", "opinionqa", "--kernel", "mock", "--split", split, "--mode", "steered", *extra])
    assert result.exit_code == 0, result.output
    return only_run(workspace, f"opinionqa-{split}-mock")


def test_a_temperature_cannot_be_fitted_on_the_test_split(workspace):
    """It used to return a number, exit 0, say nothing, and leave temperature.json in the test run."""
    test_run = run_eval(workspace, "test")
    result = runner.invoke(app, ["recalibrate-fit", str(test_run)])
    assert result.exit_code == REFUSED, result.output
    assert "split=test" in flat(result) and "Fit on a dev run" in flat(result)
    assert not (test_run / "temperature.json").exists()


def test_a_temperature_cannot_be_applied_to_the_run_it_came_from(workspace):
    dev = run_eval(workspace, "dev")
    assert runner.invoke(app, ["recalibrate-fit", str(dev)]).exit_code == 0
    result = runner.invoke(app, ["recalibrate-apply", str(dev), str(dev)])
    assert result.exit_code == REFUSED, result.output
    assert "refusing to apply a temperature to the run it was fitted on" in flat(result)


def test_a_dev_fitted_temperature_is_refused_on_a_second_dev_run(workspace):
    """Not the same directory, but still the split the temperature was fitted on."""
    dev = run_eval(workspace, "dev")
    assert runner.invoke(app, ["recalibrate-fit", str(dev)]).exit_code == 0
    other = workspace / "results" / "copy-of-the-dev-run"
    shutil.copytree(dev, other)
    result = runner.invoke(app, ["recalibrate-apply", str(other), str(dev)])
    assert result.exit_code == REFUSED, result.output
    assert "which is a dev run" in flat(result)


def test_a_temperature_is_refused_when_the_two_runs_disagree(workspace):
    dev = run_eval(workspace, "dev")
    assert runner.invoke(app, ["recalibrate-fit", str(dev)]).exit_code == 0
    # same task and split pair, but the test run was asked a different way (which the freeze only
    # allows under --override-frozen, and records)
    test_run = run_eval(workspace, "test", "--phrasing", "p2", "--override-frozen")
    result = runner.invoke(app, ["recalibrate-apply", str(test_run), str(dev)])
    assert result.exit_code == REFUSED, result.output
    assert "phrasing: fitted on p1 but applying to p2" in flat(result)


def test_applying_without_a_fit_says_what_to_run(workspace):
    dev, test_run = run_eval(workspace, "dev"), run_eval(workspace, "test")
    result = runner.invoke(app, ["recalibrate-apply", str(test_run), str(dev)])
    assert result.exit_code == REFUSED, result.output
    assert "recalibrate-fit <dev run>" in flat(result)


def price_mock_at(workspace, usd_per_mtok):
    (workspace / "prices.yaml").write_text(
        f"mock:\n  mock-1: {{input_per_mtok: {usd_per_mtok}, output_per_mtok: 0.0}}\n  uniform: {{input_per_mtok: 0.0, output_per_mtok: 0.0}}\n",
        encoding="utf-8",
    )


def test_resuming_keeps_the_runs_own_configuration(workspace):
    """Without this, resuming without `--phrasing p3` appended default-phrasing predictions to a p3 file."""
    first = runner.invoke(app, ["eval", "opinionqa", "--split", "dev", "--phrasing", "p3", "--limit", "2"])
    assert first.exit_code == 0, first.output
    out = only_run(workspace, "opinionqa-dev-mock")

    again = runner.invoke(app, ["eval", "opinionqa", "--resume-dir", str(out)])
    assert again.exit_code == 0, again.output
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["task"]["phrasing"] == "p3" and config["task"]["limit"] == 2  # not the defaults of the second invocation
    assert len(config["resumed"]) == 1 and "at" in config["resumed"][0]


def test_resuming_with_a_contradicting_flag_is_refused(workspace):
    first = runner.invoke(app, ["eval", "opinionqa", "--split", "dev", "--phrasing", "p3", "--limit", "2"])
    assert first.exit_code == 0, first.output
    out = only_run(workspace, "opinionqa-dev-mock")
    for flag, value, expected in [("--phrasing", "p1", "the run used 'p3', you passed 'p1'"), ("--kernel", "uniform", "'mock'")]:
        clash = runner.invoke(app, ["eval", "opinionqa", "--resume-dir", str(out), flag, value])
        assert clash.exit_code == REFUSED, clash.output
        assert "refusing to resume" in flat(clash) and expected in flat(clash)
    assert runner.invoke(app, ["eval", "socsci210", "--resume-dir", str(out)]).exit_code == REFUSED


def test_resuming_a_directory_that_is_not_a_run_is_refused(workspace):
    bare = workspace / "results" / "not-a-run"
    bare.mkdir(parents=True)
    result = runner.invoke(app, ["eval", "opinionqa", "--resume-dir", str(bare)])
    assert result.exit_code == REFUSED, result.output
    assert "does not say what it is" in flat(result)

    assert runner.invoke(app, ["baseline", "opinionqa", "pooled", "--split", "dev"]).exit_code == 0
    baseline = runner.invoke(app, ["eval", "opinionqa", "--resume-dir", str(only_run(workspace, "baseline-pooled"))])
    assert baseline.exit_code == REFUSED, baseline.output
    assert "does not record a kernel this command can continue" in flat(baseline)


def test_the_budget_stop_prints_the_command_that_resumes_the_run(workspace):
    price_mock_at(workspace, 1_000_000.0)  # every call costs about a dollar per token
    stopped = runner.invoke(app, ["eval", "opinionqa", "--split", "dev", "--phrasing", "p3", "--max-usd", "0.001"])
    assert stopped.exit_code == 2, stopped.output
    out = only_run(workspace, "opinionqa-dev-mock")

    command = stopped.output.strip().splitlines()[-1].split()
    assert command[:3] == ["uv", "run", "kite"]
    assert command[3:] == ["eval", "opinionqa", "--resume-dir", str(out), "--max-usd", "0.001"]

    price_mock_at(workspace, 0.0)
    resumed = runner.invoke(app, command[3:])  # exactly what the message printed, and nothing else
    assert resumed.exit_code == 0, resumed.output
    assert json.loads((out / "metrics.json").read_text())["n_predictions"] == 4
    assert yaml.safe_load((out / "config.yaml").read_text())["task"]["phrasing"] == "p3"


def test_a_test_run_is_blocked_when_nothing_is_frozen(workspace):
    """A missing freeze file blocks; it never falls back to the default phrasing."""
    (workspace / "frozen.yaml").unlink()
    result = runner.invoke(app, ["eval", "opinionqa", "--split", "test"])
    assert result.exit_code == REFUSED, result.output
    assert "does not exist, so nothing holds this test run" in flat(result)
    assert "tasks: socsci210: {phrasing:" in flat(result)  # the message carries the file to write
    assert not (workspace / "results").exists()


def test_the_dev_split_needs_no_freeze(workspace):
    (workspace / "frozen.yaml").unlink()
    assert runner.invoke(app, ["eval", "opinionqa", "--split", "dev"]).exit_code == 0


def test_a_test_run_is_blocked_when_the_task_was_never_frozen(workspace):
    (workspace / "frozen.yaml").write_text("commit: fixture0\ntasks: {socsci210: {phrasing: p1, primitive: choice}}\n", encoding="utf-8")
    result = runner.invoke(app, ["eval", "opinionqa", "--split", "test"])
    assert result.exit_code == REFUSED, result.output
    assert "has no entry for opinionqa" in flat(result)


def test_a_test_run_is_refused_with_a_phrasing_the_freeze_does_not_hold(workspace):
    result = runner.invoke(app, ["eval", "opinionqa", "--split", "test", "--phrasing", "p3"])
    assert result.exit_code == REFUSED, result.output
    assert "phrasing=p3" in flat(result) and "says phrasing=p1" in flat(result)
    assert "--override-frozen" in flat(result)


def test_a_test_run_records_the_freeze_it_was_held_to(workspace):
    assert runner.invoke(app, ["eval", "opinionqa", "--split", "test", "--mode", "steered"]).exit_code == 0
    frozen = yaml.safe_load((only_run(workspace, "opinionqa-test-mock") / "config.yaml").read_text())["frozen"]
    assert (frozen["commit"], frozen["phrasing"], frozen["primitive"], frozen["override"]) == ("fixture0", "p1", "choice", None)


def test_an_override_runs_and_is_recorded(workspace):
    result = runner.invoke(app, ["eval", "opinionqa", "--split", "test", "--phrasing", "p3", "--override-frozen"])
    assert result.exit_code == 0, result.output
    config = yaml.safe_load((only_run(workspace, "opinionqa-test-mock") / "config.yaml").read_text())
    assert config["task"]["phrasing"] == "p3"
    assert config["frozen"]["phrasing"] == "p1" and config["frozen"]["override"] == {"phrasing": "p3"}


def test_a_directory_without_a_config_is_not_a_run(workspace):
    """A missing config.yaml blocks; it never falls back to a default task."""
    bare = workspace / "results" / "bare"
    bare.mkdir(parents=True)
    result = runner.invoke(app, ["recalibrate-fit", str(bare)])
    assert result.exit_code == REFUSED, result.output
    assert "does not say what it is" in flat(result)
