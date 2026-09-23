"""A3c: the flagship and the kernel on the seen-pool studies that the publication-status test needs.

Everything is fixed in configs/eval/publication_moderator.yaml, committed before the flagship's first
prediction on these studies: the 91 seen studies with at least one reliable human contrast, every cell of a
multi-condition task that holds one, and the same 5 simulated respondents per cell for both models. The
flagship gets exactly what A3b gave it on test (prompt, batching and effort from a3_codex_baseline.py);
the kernel gets the frozen p3 phrasing. Refuses to run while the criteria file or this script has
uncommitted changes.

Usage:
    uv run python scripts/a3c_seen_runs.py --model jev      # needs the Jev key in the environment
    uv run python scripts/a3c_seen_runs.py --model astra    # local Codex; resumable with --out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from a3_codex_baseline import BATCH, PROMPT, make_batches, run

from kite.config import Settings
from kite.eval import decision_value as dv
from kite.eval.runner import run_task, write_predictions
from kite.eval.scales import Scale
from kite.eval.socsci210 import SocSci210Task, prepare, split_studies
from kite.eval.task import Prediction
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger

CRITERIA = Path("configs/eval/publication_moderator.yaml")
RAW = Path("data/socsci210/raw")
SEEN = Path("data/socsci210/prepared/a3c-seen")
MAX_PER_CELL = 5
ASTRA = ("gpt-6-astra", "high")


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def scope(prepared: Path) -> tuple[set[str], dict[str, int]]:
    """Cells of every multi-condition task that holds a reliable human contrast, and each study's count of them."""
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    human = dv.human_cells(rows, scales)
    means = {k: float(m) for k, m in zip(human.keys, human.mean, strict=True) if np.isfinite(m)}
    cells, reliable = set(), {}
    for block in dv.build_blocks(dv.comparable_tasks(scales), human, means):
        _, _, _, z = dv._pairs(block.human, block.se)
        n = int((np.abs(z) >= dv.RELIABLE_Z).sum())
        if n:
            cells.update(human.keys[i] for i in block.cells)
            reliable[block.study] = reliable.get(block.study, 0) + n
    return cells, reliable


def seen_task(phrasing: str) -> SocSci210Task:
    """The seen studies in scope, prepared once, with rows outside the scope's cells dropped."""
    if not (SEEN / "rows.parquet").exists():
        full = SEEN.with_name("a3c-seen-all")
        if not (full / "rows.parquet").exists():
            prepare(RAW, full, split_studies(RAW)["seen"])
        _, reliable = scope(full)
        prepare(RAW, SEEN, sorted(reliable))
    cells, _ = scope(SEEN)
    task = SocSci210Task(SEEN, phrasing=phrasing, primitive="choice", max_per_cell=MAX_PER_CELL)
    keys = task._frame["study_id"].astype(str) + "|" + task._frame["condition_num"].astype(str) + "|" + task._frame["task_num"].astype(str)
    task._frame = task._frame[keys.isin(cells)].reset_index(drop=True)
    return task


def run_jev(out: Path, max_usd: float) -> None:
    task = seen_task("p3")
    settings, ledger = Settings(), Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=max_usd), settings, ledger)
    config = {"task": {"name": "socsci210", "split": "a3c-seen", "max_per_cell": MAX_PER_CELL, "members": task.members()}, "kernel": "jev p3"}

    async def go():
        try:
            return await run_task(task, built.kernel, out, ledger=ledger, config=config)
        finally:
            await built.aclose()

    print(asyncio.run(go()))


def run_astra(out: Path) -> None:
    task = seen_task("p1")  # phrasing is irrelevant here: only the persona, question and options are used
    entries, meta = [], {}
    for row in task._frame.itertuples(index=False):
        case, item_id = task._case(row), f"{row.study_id}:{row.sample_id}"
        meta[item_id] = (task._meta(row), task._scale(row.study_id, int(row.condition_num), int(row.task_num)))
        task_key = (row.study_id, int(row.task_num))
        entries.append({"item_id": item_id, "task": task_key, "persona": case.persona, "question": case.question, "options": case.options})
    batches = make_batches(entries)
    cache = out / "raw.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    ledger = {"calls": 0, "failed_calls": 0, "tokens": 0, "call_seconds": []}
    print(f"{len(entries):,} items in {len(batches)} batches of up to {BATCH}; {len(done)} already done; writing to {out}")
    started = time.monotonic()
    asyncio.run(run(batches, done, cache, ledger, *ASTRA))
    predictions = [Prediction(item_id=i, probs=meta[i][1].expand(done[i]), meta=meta[i][0]) for i in meta if i in done]
    write_predictions(out, predictions)
    seconds = ledger.pop("call_seconds")
    ledger |= {"wall_seconds": time.monotonic() - started, "median_call_seconds": float(np.median(seconds)) if seconds else None}
    ledger |= {"items": len(predictions), "missing": len(meta) - len(predictions)}
    previous = json.loads((out / "ledger.json").read_text()) if (out / "ledger.json").exists() else []
    (out / "ledger.json").write_text(json.dumps([*previous, ledger] if isinstance(previous, list) else [previous, ledger], indent=1))
    config = {"task": {"split": "a3c-seen", "max_per_cell": MAX_PER_CELL, "members": task.members()}}
    config |= {"model": ASTRA[0], "effort": ASTRA[1], "prompt": PROMPT}
    (out / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print(json.dumps(ledger, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["jev", "astra"], required=True)
    parser.add_argument("--out", type=Path, default=None, help="an earlier run directory to resume")
    parser.add_argument("--max-usd", type=float, default=2.0)
    args = parser.parse_args()
    if not committed(str(CRITERIA), "scripts/a3c_seen_runs.py", "scripts/a3_codex_baseline.py"):
        raise SystemExit("refusing to run: commit the criteria file and the run scripts first")
    out = args.out or Path("results/a3c") / f"{datetime.now():%Y%m%d-%H%M%S}-seen-{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    run_jev(out, args.max_usd) if args.model == "jev" else run_astra(out)


if __name__ == "__main__":
    main()
