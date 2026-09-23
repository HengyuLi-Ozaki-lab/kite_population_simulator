"""Run a task against a kernel: resumable, append-only, failure-tolerant, budget-aware."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from kite.eval.task import FidelityTask, Item, Prediction
from kite.kernel.base import Kernel, evaluate_many
from kite.kernel.ledger import BudgetExceeded, Ledger

PREDICTIONS = "predictions.jsonl"
FAILURES = "failures.jsonl"


class RunSummary(BaseModel):
    n_predicted: int
    n_resumed: int
    n_failed: int
    stopped: Literal["completed", "budget"]

    @property
    def failure_rate(self) -> float:
        attempted = self.n_predicted + self.n_failed
        return self.n_failed / attempted if attempted else 0.0


def read_predictions(run_dir: str | Path) -> list[Prediction]:
    path = Path(run_dir) / PREDICTIONS
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [Prediction.model_validate_json(line) for line in handle if line.strip()]


def write_predictions(run_dir: str | Path, predictions: list[Prediction]) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / PREDICTIONS).open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(prediction.model_dump_json() + "\n")


def _batches(items, size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _unique(items: Iterable[Item]) -> Iterator[Item]:
    """Refuse a task that yields the same item id twice.

    Resume decides what is already done by item id alone, so a repeated id makes a resumed run skip
    items it never predicted, and nothing downstream would notice. SocSci210's sample ids restart in
    every study, which is how this was found.
    """
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"item id {item.item_id!r} appears twice in one task; ids must be unique for resume to be safe")
        seen.add(item.item_id)
        yield item


async def run_task(
    task: FidelityTask,
    kernel: Kernel,
    out_dir: str | Path,
    *,
    concurrency: int = 16,
    limit: int | None = None,
    resume: bool = True,
    ledger: Ledger | None = None,
    config: dict[str, Any] | None = None,
) -> RunSummary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {prediction.item_id for prediction in read_predictions(out)} if resume else set()
    if not resume:
        (out / PREDICTIONS).unlink(missing_ok=True)
    (out / FAILURES).unlink(missing_ok=True)  # failed items are retried on every run

    todo = (item for item in _unique(islice(task.items(), limit)) if item.item_id not in done)
    n_predicted = n_failed = 0
    stopped: Literal["completed", "budget"] = "completed"

    with (out / PREDICTIONS).open("a", encoding="utf-8") as good, (out / FAILURES).open("a", encoding="utf-8") as bad:
        for batch in _batches(todo, concurrency * 4):
            results = await evaluate_many(kernel, [item.request for item in batch], concurrency=concurrency)
            for item, result in zip(batch, results, strict=True):
                if isinstance(result, BudgetExceeded):
                    stopped = "budget"
                    continue
                if ledger is not None and not isinstance(result, BaseException):
                    ledger.record_answers(result.answers.values())
                try:
                    if isinstance(result, BaseException):
                        raise result
                    prediction = Prediction(item_id=item.item_id, probs=task.decode(item, result), meta=item.meta)
                except Exception as error:
                    bad.write(json.dumps({"item_id": item.item_id, "error": f"{type(error).__name__}: {error}"}) + "\n")
                    n_failed += 1
                    continue
                good.write(prediction.model_dump_json() + "\n")
                n_predicted += 1
            good.flush()
            bad.flush()
            if stopped == "budget":
                break

    summary = RunSummary(n_predicted=n_predicted, n_resumed=len(done), n_failed=n_failed, stopped=stopped)
    predictions = read_predictions(out)
    metrics: dict[str, Any] = {"task": task.name, "run": summary.model_dump(), "failure_rate": summary.failure_rate}
    if ledger is not None:
        # answers this invocation received whose probabilities did not sum to 1; resumed items are
        # not re-counted, which is why the ledger carries the same number
        metrics["malformed_distributions"] = ledger.malformed_distributions
    if predictions:
        metrics.update(task.score(predictions))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if ledger is not None:
        ledger.write(out / "ledger.json")
    if config is not None:
        (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return summary
