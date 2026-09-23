"""Derive the run that `--max-per-cell n` would have produced from a run made with a larger cap.

Subsampling is nested - the first n respondents of a cell under the task's own ordering are the same
whatever the cap - so a smaller run does not have to be paid for again, and a baseline scored on the
small subsample can be compared with the kernel on exactly the same items.

Usage:
    uv run python scripts/nested_subsample.py --run results/<run> --split test --max-per-cell 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from kite.eval.runner import read_predictions, write_predictions
from kite.eval.socsci210 import SocSci210Task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--max-per-cell", type=int, required=True)
    args = parser.parse_args()

    task = SocSci210Task(Path("data/socsci210/prepared") / args.split, max_per_cell=args.max_per_cell)
    wanted = {f"{row.study_id}:{row.sample_id}" for row in task._frame.itertuples(index=False)}
    kept = []
    for prediction in read_predictions(args.run):
        item_id = prediction.item_id if ":" in prediction.item_id else f"{prediction.meta['study_id']}:{prediction.item_id}"
        if item_id in wanted:
            kept.append(prediction.model_copy(update={"item_id": item_id}))
    missing = len(wanted) - len(kept)

    out = args.run.with_name(f"{args.run.name}-first{args.max_per_cell}")
    out.mkdir(exist_ok=True)
    write_predictions(out, kept)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    config["task"]["max_per_cell"] = args.max_per_cell
    config["derived_from"] = str(args.run)
    (out / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    metrics = {"task": "socsci210", "failure_rate": missing / len(wanted), **task.score(kept)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{out}: {len(kept):,} of {len(wanted):,} items ({missing} missing from the source run)")


if __name__ == "__main__":
    main()
