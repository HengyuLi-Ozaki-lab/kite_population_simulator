"""E3: does a simulated person's answers hang together the way a real person's do?

Jev evaluates every question in parallel and in isolation, so for one persona the answers to
different questions are conditionally independent given that persona. Real people's answers
correlate beyond what demographics explain. This measures the gap.

For each (study, condition) we take people who answered several of the study's outcome questions and
build three inter-item correlation matrices over the same people and the same questions:

  real       their actual answers
  expected   the mean of Jev's predicted distribution per (person, question) - the correlation that
             persona differences alone can carry, with no sampling noise; an upper bound
  sampled    one answer drawn per (person, question), averaged over repeats - what a simulation
             built on this kernel would actually produce

Two headline numbers per condition: whether the *structure* is recovered (the correlation between
the real pairwise correlations and the simulated ones) and whether the *magnitude* is (the ratio of
their mean absolute correlations).

Usage:
    uv run python scripts/e3_item_correlation.py --studies mp3dr nj5dx sffyb 9263n r5hwx \\
        --max-participants 150 --max-usd 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kite.config import Settings
from kite.eval import metrics
from kite.eval.runner import read_predictions, run_task
from kite.eval.socsci210 import SocSci210Task, prepare, split_studies
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger

MIN_TASKS = 3  # a person needs at least three answers before "do they hang together" means anything
MIN_PEOPLE = 30  # and a condition needs enough people for a correlation to be worth reading


def matrices(frame: pd.DataFrame, value: str, tasks: list[int]) -> np.ndarray | None:
    """Spearman correlation between every pair of tasks, over people present on all of them."""
    wide = frame.pivot_table(index="participant", columns="task_num", values=value).reindex(columns=tasks).dropna()
    if len(wide) < MIN_PEOPLE or wide.nunique().min() < 2:
        return None
    rho = spearmanr(wide.to_numpy()).statistic
    return np.atleast_2d(rho)


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


async def predict(studies: list[str], max_participants: int, max_usd: float, out: Path, phrasing: str, primitive: str):
    settings = Settings()
    prepared = out / "prepared"
    summary = prepare(settings.data_dir / "socsci210" / "raw", prepared, studies)
    print(json.dumps({k: v for k, v in summary.items() if "unsupported" not in k and k != "studies"}))

    task = SocSci210Task(prepared, phrasing=phrasing, primitive=primitive, max_participants=max_participants)
    n_items = sum(1 for _ in task.items())
    print(f"items: {n_items:,}")

    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=max_usd), settings, ledger)
    try:
        result = await run_task(task, built.kernel, out / "run", concurrency=settings.max_concurrency, ledger=ledger)
    finally:
        await built.aclose()
    print(f"{result.model_dump()}\n{json.dumps(ledger.summary())}")
    return task, read_predictions(out / "run")


def analyse(task: SocSci210Task, predictions) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    records = []
    for prediction in predictions:
        meta = prediction.meta
        probs = metrics.as_dist(prediction.probs)
        records.append(
            {
                "study_id": meta["study_id"],
                "condition_num": meta["condition_num"],
                "task_num": meta["task_num"],
                "participant": meta["participant"],
                "real": meta["true_index"] / (meta["n_levels"] - 1),
                "expected": metrics.mean_position(probs),
                **{f"draw{i}": int(rng.choice(len(probs), p=probs)) / (meta["n_levels"] - 1) for i in range(5)},
            }
        )
    frame = pd.DataFrame.from_records(records)

    for (study, condition), group in frame.groupby(["study_id", "condition_num"]):
        counts = group.groupby("participant")["task_num"].nunique()
        people = counts[counts >= MIN_TASKS].index
        group = group[group["participant"].isin(people)]
        tasks = sorted(group["task_num"].unique())
        if len(tasks) < MIN_TASKS:
            continue
        real = matrices(group, "real", tasks)
        expected = matrices(group, "expected", tasks)
        drawn = [matrices(group, f"draw{i}", tasks) for i in range(5)]
        if real is None or expected is None or any(d is None for d in drawn):
            continue
        sampled = np.mean(drawn, axis=0)

        r_off, e_off, s_off = off_diagonal(real), off_diagonal(expected), off_diagonal(sampled)
        rows.append(
            {
                "study": study,
                "condition": int(condition),
                "tasks": len(tasks),
                "people": int(len(people)),
                "pairs": len(r_off),
                "real_mean_abs": float(np.abs(r_off).mean()),
                "expected_mean_abs": float(np.abs(e_off).mean()),
                "sampled_mean_abs": float(np.abs(s_off).mean()),
                "structure_expected": float(np.corrcoef(r_off, e_off)[0, 1]) if np.std(e_off) > 1e-9 else np.nan,
                "structure_sampled": float(np.corrcoef(r_off, s_off)[0, 1]) if np.std(s_off) > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--max-participants", type=int, default=150)
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--phrasing", default="p3")
    parser.add_argument("--primitive", default="choice")
    parser.add_argument("--out", type=Path, default=Path("results/e3"))
    args = parser.parse_args()

    seen = set(split_studies(Settings().data_dir / "socsci210" / "raw")["unseen"])
    leaked = sorted(set(args.studies) & seen)
    if leaked:
        raise SystemExit(f"refusing to explore on test studies: {leaked}")

    args.out.mkdir(parents=True, exist_ok=True)
    task, predictions = asyncio.run(predict(args.studies, args.max_participants, args.max_usd, args.out, args.phrasing, args.primitive))
    table = analyse(task, predictions)
    table.to_csv(args.out / "correlations.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    weights = table["pairs"]
    mean = {column: float(np.average(table[column], weights=weights)) for column in ("real_mean_abs", "expected_mean_abs", "sampled_mean_abs")}

    print("\nweighted by the number of question pairs:")
    for label, column in (("real", "real_mean_abs"), ("expected (persona only)", "expected_mean_abs"), ("sampled", "sampled_mean_abs")):
        print(f"  mean |correlation|, {label:24} {mean[column]:.3f}")
    for label, column in (("expected", "expected_mean_abs"), ("sampled", "sampled_mean_abs")):
        print(f"  magnitude recovered, {label:9} {mean[column] / mean['real_mean_abs']:.1%} of real")
    for label, column in (("expected", "structure_expected"), ("sampled", "structure_sampled")):
        valid = table[column].notna()
        recovered = float(np.average(table.loc[valid, column], weights=weights[valid]))
        print(f"  structure recovered, {label:9} r = {recovered:.3f}  ({valid.sum()} of {len(table)} conditions)")


if __name__ == "__main__":
    main()
