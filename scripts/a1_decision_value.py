"""A1: the decision value of a simulated experiment, on one finished run.

The measures, their floor and their human reference are defined in `kite.eval.decision_value`.
This script applies them to the predictions of one run directory and adds the question a
practitioner would ask next: how many simulated respondents per condition does it take before the
answer stops changing, and what does that cost?

The development split is for calibration. The test split is refused until the pass criteria have
been written to `configs/eval/decision_value.yaml` and committed - they are echoed into the output
so that a result can always be read next to the criteria that were in force.

Usage:
    uv run python scripts/a1_decision_value.py --run results/<run> --split dev
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kite.eval import decision_value as dv
from kite.eval.runner import read_predictions
from kite.eval.scales import Scale

FROZEN = Path("configs/eval/decision_value.yaml")
SIZES = (5, 10, 20, 50, 100, 200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=200)
    args = parser.parse_args()

    frozen = None
    if args.split == "test":
        if not FROZEN.exists():
            raise SystemExit(f"refusing the test split: {FROZEN} does not exist. Calibrate on dev, write the criteria, commit, then run this once.")
        frozen = yaml.safe_load(FROZEN.read_text())
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    recorded = (config.get("task") or {}).get("split")
    if str(recorded) != args.split:
        raise SystemExit(f"{args.run} was run on split {recorded!r}, not {args.split!r}")

    prepared = Path("data/socsci210/prepared") / args.split
    scales = {key: Scale(**raw) for key, raw in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    human = dv.human_cells(rows, scales)
    comparable = dv.comparable_tasks(scales)
    predictions = read_predictions(args.run)
    blocks = dv.build_blocks(comparable, human, dv.predicted_means(predictions))

    rng = np.random.default_rng(0)
    parts = dv.evaluate(blocks)
    observed = parts.summary()
    null = dv.permutation_null(blocks, n_perm=args.n_perm, rng=rng)
    interval = dv.bootstrap_by_study(parts, n_boot=args.n_boot, rng=rng)
    comparison = dv.half_sample_comparison(blocks, human, n_splits=args.n_splits, rng=rng)
    curve = dv.effect_curve(blocks, human, n_splits=args.n_splits, rng=rng)

    all_scaled = {(k.split("|")[0], int(k.split("|")[2])) for k in scales}
    print(f"run {args.run.name}  ({args.split}, {len(predictions):,} predictions)")
    print(f"tasks with a scale: {len(all_scaled)}; with one scale across >=2 conditions: {len(comparable)}; scored: {len(blocks)}")
    print(f"studies: {len(set(parts.study))}; reliable contrasts (|z| >= {dv.RELIABLE_Z}): {int(parts.sign_n.sum())}")
    print(f"tasks with >=3 conditions (the ones a choice can be made in): {int((parts.available > 0).sum())}\n")

    result = {"run": str(args.run), "split": args.split, "n_predictions": len(predictions), "n_tasks": len(blocks), "frozen_criteria": frozen}
    print(f"{'':18}{'observed':>10}{'95% CI (studies)':>20}{'null mean':>11}{'null p95':>10}{'perm p':>9}{'pilot':>9}{'sim vs same judge':>19}")
    for name in ("sign_accuracy", "captured_gain"):
        p = float((np.sum(null[name] >= observed[name]) + 1) / (len(null[name]) + 1))
        result[name] = {
            "observed": observed[name],
            "ci": interval[name],
            "null_mean": float(np.nanmean(null[name])),
            "null_p95": float(np.nanpercentile(null[name], 95)),
            "permutation_p": p,
            "pilot": comparison[f"pilot_{name}"],
            "simulation_same_judge": comparison[f"simulation_{name}"],
        }
        r = result[name]
        print(
            f"{name:18}{r['observed']:>10.3f}{f'[{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]':>20}{r['null_mean']:>11.3f}{r['null_p95']:>10.3f}"
            f"{p:>9.4f}{r['pilot']:>9.3f}{r['simulation_same_judge']:>19.3f}"
        )
    print("\n'pilot': a human pilot with half the respondents, judged by the other half.")
    print("'sim vs same judge': the simulation judged by that same other half, so the two are on one footing.")

    print("\nagreement with the judging half, by the size of the effect in that half (all contrasts):")
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    result["effect_curve"] = curve.to_dict("records")

    by_reliability = dv.reliability_curve(blocks)
    print("\nthe simulation against the full-sample sign, by how reliable that sign is (ceiling = a simulation that knew the truth):")
    print(by_reliability.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    result["reliability_curve"] = by_reliability.replace({np.inf: None}).to_dict("records")

    ledger = json.loads((args.run / "ledger.json").read_text()) if (args.run / "ledger.json").exists() else {}
    unit = ledger["usd"] / ledger["calls"] if ledger.get("calls") and ledger.get("usd") else None  # a run made from cache has no price
    cells_per_task = float(np.mean([len(b.cells) for b in blocks]))
    print(f"\nstability against the number of simulated respondents per cell (a task here has {cells_per_task:.1f} conditions on average):")
    per_cell = pd.Series([f"{p.meta['study_id']}|{p.meta['condition_num']}|{p.meta['task_num']}" for p in predictions]).value_counts()
    stability = []
    for n in [s for s in SIZES if s <= per_cell.max()]:
        sized = dv.evaluate(dv.build_blocks(comparable, human, dv.predicted_means(predictions, n_per_cell=n))).summary()
        cost = unit * n * cells_per_task if unit else None
        stability.append({"n_per_cell": n, **sized, "usd_per_experiment": cost})
    table = pd.DataFrame(stability)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    result["stability"] = stability
    result["unit_cost_usd"] = unit

    result["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (args.run / "decision_value.json").write_text(json.dumps(result, indent=1, default=str))  # the criteria file carries a date
    print(f"\nwritten to {args.run / 'decision_value.json'}")


if __name__ == "__main__":
    main()
