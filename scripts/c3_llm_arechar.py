"""C3 with a flagship LLM: can it see the accuracy prompt that the kernel cannot?

Everything is fixed in configs/eval/flagship_baseline.yaml, committed before this script's first held-out
prediction, and the criteria are configs/eval/arechar_criteria.yaml, frozen for the kernel and applied here
unchanged. This script refuses to run while either file or the script itself has uncommitted changes.

The LLM gets what the kernel got: the same persona, the same context in the same order, the same headline,
question and options, built by kite.eval.arechar. Items share a call, so a batch only ever holds one
condition, and never two items of the same headline or the same respondent - otherwise the model could
compare a headline with and without the pretest, an advantage the kernel does not have.

The kernel is re-scored on exactly the respondents simulated here (the 30 per country and condition are
nested inside its 100), so the two are compared on the same people.

Usage:
    uv run python scripts/c3_llm_arechar.py                      # primary conditions: share_only, prompt
    uv run python scripts/c3_llm_arechar.py --secondary          # adds tips, reusing everything already done
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kite.eval import arechar
from kite.eval.runner import read_predictions

CONFIG = Path("configs/eval/flagship_baseline.yaml")
CSV = Path("data/arechar/CR.csv")
QSF = Path("data/arechar/questionnaires/CRUS.qsf")
KERNEL_RUN = Path("results/c2/20260922-011615-held_out")
BATCH, PARALLEL, ATTEMPTS = 20, 4, 3

PROMPT = """You are predicting how survey respondents answered. Each case below gives a respondent's
profile, what the survey had shown them before the question ("context"), the question itself, and the
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


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def make_batches(entries: list[dict], seed: int = 0) -> list[list[dict]]:
    """One condition per batch; within a batch every headline and every respondent differs.

    Items are dealt headline by headline, taking first from the headlines with the most items left, so the
    batches stay full instead of trailing off into many small ones at the end.
    """
    rng = random.Random(seed)
    batches = []
    for condition in sorted({e["condition"] for e in entries}):
        by_head: dict[int, list[dict]] = {}
        for e in entries:
            if e["condition"] == condition:
                by_head.setdefault(e["item"], []).append(e)
        for pool in by_head.values():
            rng.shuffle(pool)
        while any(by_head.values()):
            batch, people = [], set()
            for head in sorted((h for h, pool in by_head.items() if pool), key=lambda h: -len(by_head[h])):
                if len(batch) == BATCH:
                    break
                pool = by_head[head]
                pick = next((i for i, e in enumerate(pool) if e["person"] not in people), None)
                if pick is not None:
                    e = pool.pop(pick)
                    batch.append(e)
                    people.add(e["person"])
            batches.append(batch)
    return batches


def codex(batch: list[dict], model: str, effort: str) -> tuple[dict[str, list[float]], int]:
    cases = [
        {"id": f"c{i:02d}", "respondent": e["respondent"], "context": e["context"], "question": e["question"], "options": e["options"]}
        for i, e in enumerate(batch)
    ]
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


async def run(batches, done: dict, cache: Path, ledger: dict, model: str, effort: str) -> None:
    semaphore, stop = asyncio.Semaphore(PARALLEL), asyncio.Event()

    async def one(batch):
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
                except Exception as error:  # noqa: BLE001 - retried, then reported as missing
                    print(f"  call failed: {str(error)[:160]}")
                    ledger["failed_calls"] += 1
                    continue
                ledger["calls"] += 1
                ledger["tokens"] += tokens
                ledger["call_seconds"].append(time.monotonic() - started)
            done.update(got)
            cache.write_text(json.dumps(done))
            todo = [e for e in todo if e["item_id"] not in done]
            if ledger["calls"] % 20 == 0:
                print(f"  {ledger['calls']} calls, {len(done)} items, {ledger['tokens']:,} tokens")

    await asyncio.gather(*(one(b) for b in batches))


def frame(meta: dict, probs_by_id: dict) -> pd.DataFrame:
    rows = []
    for item_id, probs in probs_by_id.items():
        p = np.asarray(probs, dtype=float)
        rows.append({**meta[item_id], "expected": float(np.dot(p / p.sum(), np.arange(1, 7)))})
    return pd.DataFrame(rows)


def score(model: pd.DataFrame, human: pd.DataFrame, rng) -> dict:
    out = {"n_predictions": len(model), "effects": {}, "headline_sharing": {}}
    for treatment in ("prompt", "tips"):
        if treatment in set(model["condition"]):
            e = arechar.treatment_effect(model, human, treatment, rng)
            positive = e["human_ci"][0] > 0
            e["verdict"] = "PASS" if positive and e["model_pooled"] > e["model_null_p95"] else "FAIL" if positive else "NOT APPLICABLE"
            out["effects"][treatment] = e
    share = model[model["condition"] == "share_only"]
    for country in sorted(share["country"].unique()):
        m, h = share[share["country"] == country], human[(human["country"] == country) & (human["condition"] == "share_only")]
        out["headline_sharing"][country] = arechar.headline_level(m, h, rng, n_perm=1000)
    passed = sum(v["r"] > v["null_p95"] and v["p"] < 0.01 for v in out["headline_sharing"].values())
    out["headline_sharing_passed"] = f"{passed} of {len(out['headline_sharing'])}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secondary", action="store_true", help="also run the secondary conditions")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not committed(str(CONFIG), str(arechar.CRITERIA), "scripts/c3_llm_arechar.py"):
        raise SystemExit("refusing to run: commit the flagship config, the Arechar criteria and this script first")
    config = yaml.safe_load(CONFIG.read_text())
    cfg, model_cfg = config["arechar"], config["model"]
    conditions = cfg["conditions"]["primary"] + (cfg["conditions"]["secondary"] if args.secondary else [])

    task = arechar.ArecharTask(CSV, QSF, "held_out", max_respondents=cfg["respondents_per_country_condition"], parts=[cfg["part"]])
    entries, meta = [], {}
    for item in task.items():
        if item.meta["condition"] not in conditions:
            continue
        survey = item.request.state["survey"]
        meta[item.item_id] = item.meta
        entries.append(
            {
                "item_id": item.item_id,
                "condition": item.meta["condition"],
                "item": item.meta["item"],
                "person": (item.meta["country"], item.meta["id"]),
                "respondent": item.request.state["respondent"],
                "context": survey["context"],
                "question": survey["question"],
                "options": arechar.QUESTIONS[item.meta["kind"]][1],
            }
        )
    batches = make_batches(entries)
    # a new run directory, or an earlier one passed with --out to resume it or to add the secondary conditions
    out = args.out or Path("results/c3-flagship") / f"{datetime.now():%Y%m%d-%H%M%S}-{model_cfg['id']}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "raw.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    ledger = {"calls": 0, "failed_calls": 0, "tokens": 0, "call_seconds": []}
    print(f"{len(entries):,} items ({', '.join(conditions)}) in {len(batches)} batches; {len(done)} already done; writing to {out}")
    started = time.monotonic()
    asyncio.run(run(batches, done, cache, ledger, model_cfg["id"], model_cfg["reasoning_effort"]))

    human = arechar.load_ratings(CSV, "held_out")
    human = human[human["part"] == cfg["part"]].assign(true=lambda d: d["item"].map(arechar.is_true))
    ids = [e["item_id"] for e in entries if e["item_id"] in done]
    llm = frame(meta, {i: done[i] for i in ids})
    kernel_preds = {p.item_id: p.probs for p in read_predictions(KERNEL_RUN) if p.item_id in set(ids)}
    kernel = frame(meta, kernel_preds)
    rng = np.random.default_rng(0)
    report = {
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "model": model_cfg, "conditions": conditions, "items_wanted": len(entries), "items_done": len(ids),
        "flagship": score(llm, human, rng), "kernel_same_respondents": score(kernel, human, rng),
    }  # fmt: skip
    seconds = ledger.pop("call_seconds")
    report["ledger"] = ledger | {"wall_seconds": time.monotonic() - started, "median_call_seconds": float(np.median(seconds)) if seconds else None}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    for name in ("flagship", "kernel_same_respondents"):
        s = report[name]
        print(f"\n{name}: {s['n_predictions']:,} predictions; headline-level sharing passes in {s['headline_sharing_passed']} countries")
        for t, e in s["effects"].items():
            ci = [round(x, 3) for x in e["human_ci"]]
            model_part = f"model {e['model_pooled']:+.3f} (null p95 {e['model_null_p95']:+.3f})"
            print(f"  {t:6} {model_part}   human {e['human_pooled']:+.3f} {ci}  -> {e['verdict']}")
    print(json.dumps(report["ledger"]))


if __name__ == "__main__":
    main()
