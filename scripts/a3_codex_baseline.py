"""A3: a flagship-LLM baseline on the same items, through the local Codex CLI.

G1 compared the kernel with *published* LLM numbers, on a different task set, and needed a
normalisation argument to do it. This puts an LLM on exactly the items the kernel was scored on -
the first `--max-per-cell` simulated respondents of every cell, the kernel's own nested subsample -
so that the marginal metrics, the condition sensitivity and the decision value can be compared with
nothing in between. The kernel's decision value is flat from 5 respondents per cell upward, which is
why 5 is enough here.

The LLM gets what the kernel gets: the rendered persona, the question with its stimulus, and the
option descriptions, under the same third-person framing. It is asked for a probability per option.
The prompt adds one sentence the kernel never sees - a reminder that real answers vary - because
verbalised probabilities without it are known to be overconfident, and a baseline that was set up to
lose would prove nothing.

Several items share one Codex call, which a single-item kernel call cannot do, so a batch never
holds two items of the same task: seeing two conditions of one question side by side would let the
model contrast them, an advantage the kernel does not have.

Everything that defines the baseline - prompt, model, effort, batch size - is a constant in this
file, and the test split is refused while the file has uncommitted changes, so the configuration
that touches test is always one that is on record.

Usage:
    uv run python scripts/a3_codex_baseline.py --split dev --limit-batches 8      # pipeline check
    uv run python scripts/a3_codex_baseline.py --split test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from kite.eval.runner import write_predictions
from kite.eval.socsci210 import SocSci210Task
from kite.eval.task import Prediction

MODEL = "gpt-5.6-luna"  # the default; --model picks another from MODELS
EFFORT = "low"
MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra")
EFFORTS = ("low", "medium", "high")
BATCH = 20
PARALLEL = 4
ATTEMPTS = 3

PROMPT = """You are predicting how survey respondents answered. Each case below gives a respondent's
profile, the survey question they were asked (including anything they were shown first), and the
answer options in order.

For each case, give the probability that this respondent chose each option. Think about how people
with this profile actually answer such questions, and remember that real answers vary: do not put
nearly all of the probability on one option unless real respondents would be nearly unanimous.

Give one probability per option, in the order listed; they must be non-negative and sum to 1.
The cases are unrelated to each other; treat each one on its own. Return one entry per case id.

Cases:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "probabilities": {"type": "array", "items": {"type": "number"}}},
                "required": ["id", "probabilities"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["predictions"],
    "additionalProperties": False,
}


class UsageLimit(RuntimeError):
    """Codex says the account is out of quota: stop, do not retry."""


def make_batches(entries: list[dict]) -> list[list[dict]]:
    """Deal items out task by task, so that no batch holds two items of the same task."""
    by_task = defaultdict(list)
    for entry in entries:
        by_task[entry["task"]].append(entry)
    batches = []
    while any(by_task.values()):
        live = [task for task, items in by_task.items() if items]
        for start in range(0, len(live), BATCH):
            batches.append([by_task[task].pop() for task in live[start : start + BATCH]])
    return batches


def codex(batch: list[dict], model: str = MODEL, effort: str = EFFORT) -> tuple[dict[str, list[float]], int]:
    cases = [{"id": f"c{i:02d}", "respondent": e["persona"], "question": e["question"], "options": e["options"]} for i, e in enumerate(batch)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "schema.json").write_text(json.dumps(SCHEMA))
        command = ["codex", "exec", "-m", model, "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral"]
        command += ["--output-schema", str(tmp / "schema.json"), "-o", str(tmp / "out.json"), "-c", f'model_reasoning_effort="{effort}"', "-"]
        done = subprocess.run(
            command, input=PROMPT + json.dumps(cases, ensure_ascii=False, indent=1), text=True, capture_output=True, cwd=tmp, timeout=900
        )
        log = done.stdout + done.stderr
        if re.search(r"usage limit|rate limit reached|quota", log, re.IGNORECASE) and not (tmp / "out.json").exists():
            raise UsageLimit(log[-400:])
        if done.returncode != 0 or not (tmp / "out.json").exists():
            raise RuntimeError(f"codex exited {done.returncode}: {log[-400:]}")
        parsed = json.loads((tmp / "out.json").read_text())
    used = re.search(r"tokens used\s*\n?\s*([\d,]+)", log)
    out = {}
    for case, entry in zip(cases, batch, strict=True):
        found = next((p["probabilities"] for p in parsed["predictions"] if p["id"] == case["id"]), None)
        if found is None or len(found) != len(entry["options"]):
            continue
        probs = np.asarray(found, dtype=float)
        if np.all(np.isfinite(probs)) and np.all(probs >= 0) and probs.sum() > 0:
            out[entry["item_id"]] = (probs / probs.sum()).tolist()
    return out, int(used.group(1).replace(",", "")) if used else 0


async def run(batches: list[list[dict]], done: dict, cache: Path, ledger: dict, model: str = MODEL, effort: str = EFFORT) -> None:
    semaphore, stop = asyncio.Semaphore(PARALLEL), asyncio.Event()

    async def one(batch: list[dict]) -> None:
        todo = [e for e in batch if e["item_id"] not in done]
        for _ in range(ATTEMPTS):
            if not todo or stop.is_set():
                return
            async with semaphore:
                if stop.is_set():
                    return
                started = time.monotonic()
                try:
                    got, tokens = await asyncio.to_thread(codex, todo, model, effort)
                except UsageLimit as error:
                    print(f"  usage limit reached, stopping: {error}")
                    stop.set()
                    return
                except Exception as error:  # noqa: BLE001 - a failed call is retried, then its items are reported as missing
                    print(f"  call failed: {str(error)[:160]}")
                    ledger["failed_calls"] += 1
                    continue
                ledger["calls"] += 1
                ledger["tokens"] += tokens
                ledger["call_seconds"].append(time.monotonic() - started)
            done.update(got)
            cache.write_text(json.dumps(done))
            todo = [e for e in todo if e["item_id"] not in done]
            if ledger["calls"] % 10 == 0:
                print(f"  {ledger['calls']} calls, {len(done)} items, {ledger['tokens']:,} tokens")

    await asyncio.gather(*(one(batch) for batch in batches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--max-per-cell", type=int, default=5)
    parser.add_argument("--limit-batches", type=int, default=None, help="pipeline check: only the first N batches")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", choices=MODELS, default=MODEL)
    parser.add_argument("--effort", choices=EFFORTS, default=EFFORT)
    args = parser.parse_args()

    me = "scripts/a3_codex_baseline.py"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", me], capture_output=True, text=True).stdout.strip()
    if args.split == "test" and (dirty or args.limit_batches):
        raise SystemExit("refusing the test split: commit this file first, and run it whole - the configuration that touches test must be on record")

    prepared = Path("data/socsci210/prepared") / args.split
    task = SocSci210Task(prepared, phrasing="p1", primitive="choice", max_per_cell=args.max_per_cell)
    entries, meta = [], {}
    for row in task._frame.itertuples(index=False):
        case, item_id = task._case(row), f"{row.study_id}:{row.sample_id}"
        meta[item_id] = (task._meta(row), task._scale(row.study_id, int(row.condition_num), int(row.task_num)))
        entries.append(
            {
                "item_id": item_id,
                "task": (row.study_id, int(row.task_num)),
                "persona": case.persona,
                "question": case.question,
                "options": case.options,
            }
        )
    batches = make_batches(entries)[: args.limit_batches]

    name = f"socsci210-{args.split}-codex-{args.model}-{args.effort}" + ("-pilot" if args.limit_batches else "")
    out = args.out or Path("results") / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "raw.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    ledger = {"calls": 0, "failed_calls": 0, "tokens": 0, "call_seconds": []}
    print(f"{len(entries):,} items in {len(batches)} batches of up to {BATCH}; {len(done)} already done; writing to {out}")

    started = time.monotonic()
    asyncio.run(run(batches, done, cache, ledger, args.model, args.effort))

    wanted = [e["item_id"] for batch in batches for e in batch]
    predictions = [Prediction(item_id=i, probs=meta[i][1].expand(done[i]), meta=meta[i][0]) for i in wanted if i in done]
    write_predictions(out, predictions)
    missing = len(wanted) - len(predictions)
    seconds = ledger.pop("call_seconds")
    ledger |= {"usd": None, "wall_seconds": time.monotonic() - started, "median_call_seconds": float(np.median(seconds)) if seconds else None}
    ledger |= {"items": len(predictions), "missing": missing, "tokens_per_item": ledger["tokens"] / max(len(predictions), 1)}
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    config = {
        "task": {"name": "socsci210", "split": args.split, "max_per_cell": args.max_per_cell, "members": task.members()},
        "kernel": {"backend": "codex-cli", "model_id": args.model, "effort": args.effort, "batch": BATCH, "prompt": PROMPT},
        "git_commit": commit,
        "dirty": bool(dirty),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    if predictions:
        metrics = {"task": "socsci210", "failure_rate": missing / len(wanted), **task.score(predictions)}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        keys = ("distribution", "distribution_published_convention", "accuracy_macro", "ece", "entropy_ratio_median", "condition_sensitivity_r")
        print(json.dumps({k: metrics.get(k) for k in keys}, indent=1))
    print(json.dumps(ledger, indent=1))


if __name__ == "__main__":
    main()
