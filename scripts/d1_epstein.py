"""D1 run stage: the kernel on every Epstein item, then the flagship on the sparse anchors and the audit pool.

Everything that defines the run is in configs/eval/epstein_criteria.yaml, committed before the first prediction;
this script refuses to run while that file, the screens file, the task module or itself is uncommitted. No human
rating is read at this stage (the loader would refuse anyway).

    PYTHONPATH=scripts uv run python scripts/d1_epstein.py jev                       # 73,600 kernel predictions, resumable
    PYTHONPATH=scripts uv run python scripts/d1_epstein.py astra --jev-run <dir>     # anchors (12 per cell) + audit pool, via Codex

Anchors are (persona, headline) states of the application pool, drawn per wave x arm x cell with a fixed seed,
where a cell is veracity x the tertile of the kernel's control-arm P(yes) for that state; the same anchor states
are predicted under the control arm and under the treatment, so the flagship's effect is paired. The audit pool
is every state of the 30 audit personas per wave under every arm. One arm per Codex batch, no persona or
headline twice in a batch (the flagship must not see a state with and without its treatment).
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
from c3_llm_arechar import PROMPT, make_batches, run

from kite.config import Settings
from kite.eval import epstein
from kite.eval.runner import read_predictions, run_task, write_predictions
from kite.eval.task import Prediction
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger

CRITERIA = Path("configs/eval/epstein_criteria.yaml")
GUARDED = (str(CRITERIA), str(epstein.SCREENS), "src/kite/eval/epstein.py", "scripts/d1_epstein.py", "scripts/c3_llm_arechar.py")
OUT = Path("results/d1")


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def p_yes(probs) -> float:
    p = np.asarray(probs, dtype=float)
    return float(p[1] / max(p.sum(), 1e-12))


def cells_from_control(jev: pd.DataFrame, edges_by: str = "wave") -> tuple[pd.DataFrame, dict]:
    """Every (wave, persona, item) state's cell: veracity x tertile of the kernel's control P(yes), edges per wave."""
    control = jev[(jev["arm"] == "control") & (jev["pool"] == "application")][["wave", "id", "item_num", "true", "p_yes"]]
    edges = {}
    cells = []
    for (wave, truth), group in control.groupby(["wave", "true"]):
        lo, hi = np.quantile(group["p_yes"], [1 / 3, 2 / 3])
        edges[f"{wave}:{'true' if truth else 'false'}"] = [float(lo), float(hi)]
    for row in control.itertuples(index=False):
        lo, hi = edges[f"{row.wave}:{'true' if row.true else 'false'}"]
        tertile = 0 if row.p_yes <= lo else 1 if row.p_yes <= hi else 2
        cells.append({"wave": row.wave, "id": row.id, "item_num": row.item_num, "cell": f"{'true' if row.true else 'false'}:{tertile}"})
    return pd.DataFrame(cells), edges


def stage_jev(args) -> None:
    cfg = yaml.safe_load(CRITERIA.read_text())["run"]
    task = epstein.EpsteinTask(personas_per_wave=cfg["personas_per_wave"], audit_per_wave=cfg["audit_per_wave"], seed=cfg["seed"])
    out = args.out or OUT / f"{datetime.now():%Y%m%d-%H%M%S}-jev"
    settings, ledger = Settings(), Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=args.max_usd), settings, ledger)
    config = {"task": {"name": "epstein", "arms": task.arms, "members": task.members(), **cfg}, "kernel": "jev p3", "criteria": str(CRITERIA)}

    async def go():
        try:
            return await run_task(task, built.kernel, out, ledger=ledger, config=config)
        finally:
            await built.aclose()

    print(asyncio.run(go()))
    print(json.dumps({k: v for k, v in json.loads((out / "ledger.json").read_text()).items() if k in ("usd", "calls", "answers")}))


def stage_astra(args) -> None:
    cfg = yaml.safe_load(CRITERIA.read_text())
    run_cfg, model_cfg = cfg["run"], cfg["flagship"]
    task = epstein.EpsteinTask(personas_per_wave=run_cfg["personas_per_wave"], audit_per_wave=run_cfg["audit_per_wave"], seed=run_cfg["seed"])
    jev = pd.DataFrame([{**p.meta, "p_yes": p_yes(p.probs)} for p in read_predictions(args.jev_run)])
    cells, edges = cells_from_control(jev[jev["kind"] == "share"])
    rng = np.random.default_rng(run_cfg["seed"])
    wanted: set[tuple[int, int, str, int]] = set()  # (wave, id, arm, item)
    for wave, arms in task.arms.items():
        share_arms = [a for a in arms if a != "accuracy_only"]
        pool = cells[cells["wave"] == wave]
        for _cell, group in pool.groupby("cell"):
            states = group.sample(min(run_cfg["anchors_per_cell"], len(group)), random_state=int(rng.integers(2**31)))
            for arm in share_arms:  # the same anchor states under the control and under every treatment
                wanted.update((wave, int(s.id), arm, int(s.item_num)) for s in states.itertuples(index=False))
    audit = task.people[task.people["pool"] == "audit"]
    for person in audit.itertuples(index=False):
        for arm in task.arms[int(person.wave)]:
            if arm != "accuracy_only":
                wanted.update((int(person.wave), int(person.id), arm, k) for k in task.headlines)
    entries, meta = [], {}
    for item in task.items():
        key = (item.meta["wave"], item.meta["id"], item.meta["arm"], item.meta["item_num"])
        if key not in wanted:
            continue
        survey = item.request.state["survey"]
        meta[item.item_id] = {**item.meta, "anchor": item.meta["pool"] == "application"}
        entry = {
            "item_id": item.item_id,
            "condition": item.meta["arm"],
            "item": item.meta["item_num"],
            "person": (item.meta["wave"], item.meta["id"]),
        }
        entry |= {"respondent": item.request.state["respondent"], "context": survey["context"], "question": survey["question"]}
        entries.append({**entry, "options": epstein.question_for(item.meta["arm"], task.screens)[1]})
    batches = make_batches(entries)
    out = args.out or OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{model_cfg['id']}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "raw.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    ledger = {"calls": 0, "failed_calls": 0, "tokens": 0, "call_seconds": []}
    n_anchor = sum(m["anchor"] for m in meta.values())
    print(
        f"{len(entries):,} flagship items ({n_anchor:,} anchor, {len(meta) - n_anchor:,} audit) in {len(batches)} batches; {len(done)} done; -> {out}"
    )
    started = time.monotonic()
    asyncio.run(run(batches, done, cache, ledger, model_cfg["id"], model_cfg["reasoning_effort"]))
    predictions = [Prediction(item_id=i, probs=done[i], meta=meta[i]) for i in meta if i in done]
    write_predictions(out, predictions)
    seconds = ledger.pop("call_seconds")
    ledger |= {
        "wall_seconds": time.monotonic() - started,
        "median_call_seconds": float(np.median(seconds)) if seconds else None,
        "items": len(predictions),
        "missing": len(meta) - len(predictions),
    }
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {"jev_run": str(args.jev_run), "cell_edges": edges, "model": model_cfg, "prompt": PROMPT, **run_cfg}, allow_unicode=True, sort_keys=False
        )
    )
    print(json.dumps(ledger, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["jev", "astra"])
    parser.add_argument("--jev-run", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="resume an earlier run directory")
    parser.add_argument("--max-usd", type=float, default=3.5)
    args = parser.parse_args()
    if not committed(*GUARDED):
        raise SystemExit("refusing to run: commit the criteria, the screens, the task module and the scripts first")
    if args.stage == "jev":
        stage_jev(args)
    else:
        if args.jev_run is None:
            raise SystemExit("--jev-run is required for the flagship stage (the cells come from the kernel's control predictions)")
        stage_astra(args)


if __name__ == "__main__":
    main()
