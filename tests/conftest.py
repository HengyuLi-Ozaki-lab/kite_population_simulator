"""Shared fixtures for the command line tests: fixture data, free mock prices, an isolated workspace."""

import pytest

from kite.eval.opinionqa import WAVES, wave_dir
from tests.eval.test_opinionqa import make_wave
from tests.eval.test_socsci210 import make_raw

PRICES = "mock:\n  mock-1: {input_per_mtok: 0.0, output_per_mtok: 0.0}\n  uniform: {input_per_mtok: 0.0, output_per_mtok: 0.0}\n"
# stands in for the real freeze, which is written only after the development comparison has run
FROZEN = """commit: fixture0
dev_report: results/dev-grid/report.md
tasks:
  socsci210: {phrasing: p1, primitive: score}
  opinionqa: {phrasing: p1, primitive: choice}
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "prices.yaml").write_text(PRICES, encoding="utf-8")
    (tmp_path / "frozen.yaml").write_text(FROZEN, encoding="utf-8")
    monkeypatch.setenv("KITE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KITE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("KITE_CACHE_PATH", str(tmp_path / "cache" / "responses.sqlite"))
    monkeypatch.setenv("KITE_PRICES_PATH", str(tmp_path / "prices.yaml"))
    monkeypatch.setenv("KITE_FREEZE_PATH", str(tmp_path / "frozen.yaml"))
    for wave in WAVES:  # every wave, so a `--split test` run can load the waves that split claims
        make_wave(tmp_path / "data" / "opinionqa", wave)
    assert wave_dir(tmp_path / "data" / "opinionqa", WAVES[0]).exists()
    make_raw(tmp_path / "data" / "socsci210" / "raw")
    return tmp_path


def only_run(workspace, fragment):
    matches = [path for path in (workspace / "results").iterdir() if fragment in path.name]
    assert len(matches) == 1, [path.name for path in (workspace / "results").iterdir()]
    return matches[0]
