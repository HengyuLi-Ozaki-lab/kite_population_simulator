"""Temperature scaling: one number that sharpens (T < 1) or flattens (T > 1) every distribution.

Fit on the development split only; apply to a test run exactly once. Both halves of that sentence
are enforced here rather than left to the operator: `fit_from_run` reads the run's own config and
refuses anything but a dev run, and `load_fit_for` refuses to apply a temperature to the run it
came from, to a second dev run, or to a run whose task, phrasing, primitive, mode or kernel differs
from the one it was fitted on. Everything applied carries the fit's directory, split and commit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize_scalar

from kite.eval.protocol import CONFIG, ProtocolError, RunConfig, Split, TaskConfig
from kite.eval.runner import read_predictions, write_predictions
from kite.eval.task import FidelityTask, Prediction

_FLOOR = 1e-9
TEMPERATURE = "temperature.json"
# what the fit and the target have to agree on: everything that changes the shape of a prediction
AGREE_ON = ("name", "phrasing", "primitive", "mode")


class Fit(BaseModel):
    """A fitted temperature and where it came from. Written to temperature.json; copied into whatever it is applied to."""

    temperature: float
    n_fit: int
    fitted_on: str
    git_commit: str | None = None
    task: TaskConfig
    kernel: dict[str, Any] = {}

    def provenance(self) -> dict[str, Any]:
        """The scalar summary that goes into the recalibrated run's metrics.json."""
        return {
            "temperature": self.temperature,
            "temperature_fitted_on": self.fitted_on,
            "temperature_fit_split": self.task.split.value,
            "temperature_fit_commit": self.git_commit,
            "temperature_n_fit": self.n_fit,
        }


def apply_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    logits = np.log(np.clip(np.asarray(probs, dtype=float), _FLOOR, None)) / temperature
    scaled = np.exp(logits - logits.max())
    return (scaled / scaled.sum()).tolist()


def cross_entropy(predicted: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], temperature: float) -> float:
    total = 0.0
    for probs, target in zip(predicted, targets, strict=True):
        scaled = np.clip(apply_temperature(probs, temperature), _FLOOR, None)
        total -= float(np.dot(np.asarray(target, dtype=float), np.log(scaled)))
    return total / len(predicted)


def fit_temperature(predicted: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    """Minimize the mean cross-entropy against the targets (one-hot answers or human distributions)."""
    result = minimize_scalar(lambda log_t: cross_entropy(predicted, targets, float(np.exp(log_t))), bounds=(-2.5, 2.5), method="bounded")
    return float(np.exp(result.x))


def fit_from_run(task: FidelityTask, run_dir: str | Path) -> float:
    """Fit on a dev run and write temperature.json into it. Refuses on any other split."""
    config = RunConfig.read(run_dir)
    if config.task.split is not Split.dev:
        raise ProtocolError(
            f"refusing to fit a temperature on {run_dir}: its {CONFIG} says split={config.task.split.value}, and a temperature fitted on the "
            "split it will be scored on is fitted on the answers it is meant to predict. Fit on a dev run and apply that to the test run."
        )
    predictions = read_predictions(run_dir)
    temperature = fit_temperature([p.probs for p in predictions], [task.target(p) for p in predictions])
    fit = Fit(
        temperature=temperature,
        n_fit=len(predictions),
        fitted_on=str(run_dir),
        git_commit=config.git_commit,
        task=config.task,
        kernel=config.kernel,
    )
    Path(run_dir, TEMPERATURE).write_text(json.dumps(fit.model_dump(mode="json"), indent=2), encoding="utf-8")
    return temperature


def read_fit(fit_dir: str | Path) -> Fit:
    path = Path(fit_dir) / TEMPERATURE
    if not path.exists():
        raise ProtocolError(f"{path} does not exist; run `kite recalibrate-fit <dev run>` first")
    try:
        return Fit.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as error:
        raise ProtocolError(f"{path} is not a temperature this version can trace back to a run: {error}") from None


def _kernel_id(kernel: dict[str, Any]) -> tuple[Any, Any]:
    return kernel.get("backend"), kernel.get("model_id")


def load_fit_for(fit_dir: str | Path, target_dir: str | Path) -> tuple[Fit, RunConfig]:
    """The temperature fitted in `fit_dir`, checked against the run it is about to be applied to."""
    fit_path, target_path = Path(fit_dir).resolve(), Path(target_dir).resolve()
    if fit_path == target_path:
        raise ProtocolError(
            f"refusing to apply a temperature to the run it was fitted on ({target_dir}): that scores the fit on its own data. "
            "Pass the test run first and the dev run it was fitted on second."
        )
    fit = read_fit(fit_dir)
    target = RunConfig.read(target_dir)
    if fit.task.split is not Split.dev:
        raise ProtocolError(f"{fit_dir} holds a temperature fitted on the {fit.task.split.value} split; only a dev fit may be applied")
    if target.task.split is not Split.test:
        raise ProtocolError(
            f"refusing to apply a dev-fitted temperature to {target_dir}, which is a {target.task.split.value} run: "
            "the temperature was fitted on that split, so applying it there reports a number tuned on its own answers."
        )
    differences = [
        f"{field}: fitted on {getattr(fit.task, field).value} but applying to {getattr(target.task, field).value}"
        for field in AGREE_ON
        if getattr(fit.task, field) != getattr(target.task, field)
    ]
    if _kernel_id(fit.kernel) != _kernel_id(target.kernel):
        differences.append(f"kernel: fitted on {_kernel_id(fit.kernel)} but applying to {_kernel_id(target.kernel)}")
    if differences:
        raise ProtocolError(
            f"refusing to apply the temperature from {fit_dir} to {target_dir}: a temperature describes one model answering one way.\n  "
            + "\n  ".join(differences)
        )
    return fit, target


def apply_to_run(task: FidelityTask, source_dir: str | Path, out_dir: str | Path, fit: Fit) -> dict:
    scaled = [Prediction(item_id=p.item_id, probs=apply_temperature(p.probs, fit.temperature), meta=p.meta) for p in read_predictions(source_dir)]
    write_predictions(out_dir, scaled)
    metrics = {"task": task.name, "recalibrated_from": str(source_dir), **fit.provenance(), **task.score(scaled)}
    Path(out_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
