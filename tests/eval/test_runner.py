import asyncio
import json

import yaml

from kite.eval.phrasing import build_request, decode
from kite.eval.runner import read_predictions, run_task
from kite.eval.task import Item
from kite.kernel.ledger import BudgetedKernel, Ledger, Price, PriceTable
from kite.kernel.mock import MockKernel

OPTIONS = ["yes", "no", "unsure"]


class ToyTask:
    name = "toy"

    def __init__(self, n=10):
        self.n = n

    def items(self):
        for index in range(self.n):
            request = build_request(phrasing="p1", primitive="choice", persona=None, question=f"Question {index}?", options=OPTIONS)
            yield Item(item_id=f"item-{index}", request=request, meta={"index": index})

    def decode(self, item, response):
        return decode(response, phrasing="p1", options=OPTIONS)

    def score(self, predictions):
        return {"n_scored": len(predictions)}


def test_run_writes_predictions_metrics_ledger_and_config(tmp_path):
    ledger = Ledger()
    summary = asyncio.run(run_task(ToyTask(), MockKernel(), tmp_path, concurrency=2, ledger=ledger, config={"kernel": "mock"}))
    assert (summary.n_predicted, summary.n_failed, summary.stopped) == (10, 0, "completed")
    predictions = read_predictions(tmp_path)
    assert [p.item_id for p in predictions] == [f"item-{i}" for i in range(10)]
    assert predictions[3].meta == {"index": 3}
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_scored"] == 10 and metrics["task"] == "toy" and metrics["failure_rate"] == 0.0
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == {"kernel": "mock"}
    assert (tmp_path / "ledger.json").exists()


def test_resume_skips_finished_items(tmp_path):
    kernel = MockKernel()
    asyncio.run(run_task(ToyTask(), kernel, tmp_path, limit=4))
    summary = asyncio.run(run_task(ToyTask(), kernel, tmp_path))
    assert (summary.n_resumed, summary.n_predicted) == (4, 6)
    assert len(kernel.seen) == 10
    assert len(read_predictions(tmp_path)) == 10


def test_no_resume_starts_over(tmp_path):
    asyncio.run(run_task(ToyTask(), MockKernel(), tmp_path, limit=4))
    summary = asyncio.run(run_task(ToyTask(), MockKernel(), tmp_path, resume=False, limit=2))
    assert summary.n_resumed == 0
    assert len(read_predictions(tmp_path)) == 2


def test_failures_are_recorded_and_the_run_continues(tmp_path):
    class Flaky(MockKernel):
        async def evaluate(self, request):
            if "Question 3" in request.state["survey"]["question"]:
                raise RuntimeError("boom")
            return await super().evaluate(request)

    summary = asyncio.run(run_task(ToyTask(), Flaky(), tmp_path))
    assert (summary.n_predicted, summary.n_failed) == (9, 1)
    assert summary.failure_rate == 0.1
    failure = json.loads((tmp_path / "failures.jsonl").read_text().strip())
    assert failure == {"item_id": "item-3", "error": "RuntimeError: boom"}


def test_answers_that_do_not_sum_to_one_are_counted_in_the_run(tmp_path):
    """A truncated backend response scores like a good one, because every metric renormalizes."""

    def truncated(request, key, question):
        outcomes = question.outcomes()
        return {outcome: 0.5 / len(outcomes) for outcome in outcomes}  # sums to 0.5

    ledger = Ledger()
    asyncio.run(run_task(ToyTask(3), MockKernel(fn=truncated), tmp_path, ledger=ledger))
    assert (ledger.answers, ledger.malformed_distributions) == (3, 3)
    assert json.loads((tmp_path / "metrics.json").read_text())["malformed_distributions"] == 3
    assert json.loads((tmp_path / "ledger.json").read_text())["malformed_distributions"] == 3


def test_well_formed_answers_are_counted_and_not_flagged(tmp_path):
    ledger = Ledger()
    asyncio.run(run_task(ToyTask(3), MockKernel(), tmp_path, ledger=ledger))
    assert (ledger.answers, ledger.malformed_distributions) == (3, 0)
    assert json.loads((tmp_path / "metrics.json").read_text())["malformed_distributions"] == 0


def test_budget_stop_is_graceful_and_resumable(tmp_path):
    prices = PriceTable({"mock": {"mock-1": Price(input_per_mtok=1_000_000.0, output_per_mtok=0.0)}})
    ledger = Ledger()
    kernel = BudgetedKernel(MockKernel(), ledger, prices, max_usd=250.0)  # each call costs roughly $60-70
    summary = asyncio.run(run_task(ToyTask(), kernel, tmp_path, concurrency=1, ledger=ledger))
    assert summary.stopped == "budget"
    assert 0 < summary.n_predicted < 10
    assert summary.n_failed == 0
    finished = asyncio.run(run_task(ToyTask(), MockKernel(), tmp_path))
    assert finished.n_resumed == summary.n_predicted
    assert len(read_predictions(tmp_path)) == 10


class ClashingTask(ToyTask):
    """Two different questions under one id - what SocSci210 did across studies before ids carried the study."""

    def items(self):
        for index in range(self.n):
            request = build_request(phrasing="p1", primitive="choice", persona=None, question=f"Question {index}?", options=OPTIONS)
            yield Item(item_id=f"item-{index % 2}", request=request, meta={"index": index})


def test_a_task_that_repeats_an_item_id_is_refused(tmp_path):
    """Resume keys on the item id alone, so a repeat would make a resumed run skip work it never did."""
    import pytest

    with pytest.raises(ValueError, match="appears twice"):
        asyncio.run(run_task(ClashingTask(4), MockKernel(), tmp_path, concurrency=2))
