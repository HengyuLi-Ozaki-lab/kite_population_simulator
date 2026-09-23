"""End to end through the command line, offline: fixture data, mock kernel, real files on disk."""

import json
import shutil

from typer.testing import CliRunner

from kite.cli import app
from kite.eval.opinionqa import wave_dir
from tests.conftest import only_run

runner = CliRunner()


def test_opinionqa_eval_baseline_recalibrate_report(workspace):
    """The sanctioned path: fit on the dev run, apply to the separate test run, report all three."""
    result = runner.invoke(app, ["eval", "opinionqa", "--kernel", "mock", "--split", "dev", "--mode", "steered"])
    assert result.exit_code == 0, result.output
    dev = only_run(workspace, "opinionqa-dev-mock")
    metrics = json.loads((dev / "metrics.json").read_text())
    assert metrics["n_predictions"] == 4 and metrics["failure_rate"] == 0.0
    assert json.loads((dev / "ledger.json").read_text())["calls"] == 4

    test = runner.invoke(app, ["eval", "opinionqa", "--kernel", "mock", "--split", "test", "--mode", "steered"])
    assert test.exit_code == 0, test.output
    test_run = only_run(workspace, "opinionqa-test-mock")

    assert runner.invoke(app, ["baseline", "opinionqa", "pooled", "--split", "dev"]).exit_code == 0
    pooled = only_run(workspace, "baseline-pooled")

    fit = runner.invoke(app, ["recalibrate-fit", str(dev)])
    assert fit.exit_code == 0 and "temperature = " in fit.output
    applied = runner.invoke(app, ["recalibrate-apply", str(test_run), str(dev)])
    assert applied.exit_code == 0, applied.output
    recalibrated = only_run(workspace, "recalibrated")
    # the recalibrated run says where its temperature came from, not only which predictions it scaled
    provenance = json.loads((recalibrated / "metrics.json").read_text())
    assert provenance["recalibrated_from"] == str(test_run)
    assert provenance["temperature_fitted_on"] == str(dev) and provenance["temperature_fit_split"] == "dev"

    report = runner.invoke(app, ["report", str(dev), str(pooled), str(recalibrated), "--out", str(workspace / "report")])
    assert report.exit_code == 0, report.output
    text = (workspace / "report" / "report.md").read_text()
    assert "opinionqa-dev-mock" in text and "baseline-pooled" in text
    assert (workspace / "report" / "figures" / "pareto.png").stat().st_size > 0


def test_socsci210_prepare_parity_eval_and_cache(workspace):
    prepared = runner.invoke(app, ["prepare-socsci210", "--split", "test"])
    assert prepared.exit_code == 0, prepared.output
    assert json.loads(prepared.output)["n_rows"] == 12

    parity = runner.invoke(app, ["parity-socsci210", "--split", "test"])
    assert parity.exit_code == 0 and json.loads(parity.output)["paper"]["distribution"] == 0.203

    first = runner.invoke(app, ["eval", "socsci210", "--kernel", "mock", "--split", "test", "--primitive", "score"])
    assert first.exit_code == 0, first.output
    run = only_run(workspace, "socsci210-test-mock")
    assert json.loads((run / "metrics.json").read_text())["n_predictions"] == 12

    # 12 predictions but 3 cache entries: within a cell the fixture's participants render to the same
    # persona and read the same text, so their (state, question) bytes - and cache keys - are identical
    stats = runner.invoke(app, ["cache", "stats"])
    assert json.loads(stats.output) == {"mock:mock-1": 3}
    exported = runner.invoke(app, ["cache", "export", str(workspace / "cache.jsonl.gz")])
    assert exported.exit_code == 0 and "3 entries" in exported.output


def test_check_opinionqa_reports_missing_waves(workspace):
    shutil.rmtree(wave_dir(workspace / "data" / "opinionqa", 27))
    result = runner.invoke(app, ["check-opinionqa"])
    assert result.exit_code == 1 and "W27" in result.output


def test_probe_command_with_the_mock_kernel(workspace):
    result = runner.invoke(app, ["probe", "--task", "opinionqa", "--kernel", "mock", "--n-cases", "4"])
    assert result.exit_code == 0, result.output
    probes = json.loads((only_run(workspace, "probes-opinionqa-mock") / "probes.json").read_text())
    assert probes["repeat_noise"]["noise"]["max"] == 0.0  # the mock kernel is deterministic
    assert probes["option_order"]["excess_median"] is not None
