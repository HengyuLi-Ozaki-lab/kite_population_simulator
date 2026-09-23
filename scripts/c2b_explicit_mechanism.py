"""C2b: the diagnostic declared in configs/eval/arechar_criteria.yaml, run after the criteria were frozen.

On the calibration part the kernel showed no accuracy-prompt effect (-0.03, inside its null band;
people +0.15). Two explanations predict different things here:

  can represent, cannot infer   told the mechanism outright, the model's prompt effect appears
  cannot represent              even then, its sharing predictions do not move toward discernment

The prompt condition is re-run for 100 US respondents per condition (half A, calibration data only)
with one sentence added to the context, exactly as declared: "This made the respondent think about
whether headlines are accurate." The share-only and the faithful prompt predictions for the same people
are the calibration run's, from cache. Exploratory; not a gate; never used to change the frozen design.

Usage:
    KITE_RPM=120 uv run python scripts/c2b_explicit_mechanism.py
"""

from __future__ import annotations

import asyncio
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from kite.config import Settings
from kite.eval import arechar
from kite.eval.phrasing import build_request, decode
from kite.kernel.base import evaluate_many
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger

SENTENCE = "This made the respondent think about whether headlines are accurate."
CSV = Path("data/arechar/CR.csv")
QSF = Path("data/arechar/questionnaires/CRUS.qsf")


def requests(task: arechar.ArecharTask, explicit: bool) -> list[tuple[dict, object]]:
    out = []
    for item in task.items():
        if item.meta["condition"] not in ("prompt", "share_only"):
            continue
        request = item.request
        if explicit and item.meta["condition"] == "prompt":
            row = item.meta
            person = task.people.loc[(row["country"], row["id"])].to_dict()
            question, options = arechar.QUESTIONS["share"]
            context = arechar.context_for("prompt", task.materials, zlib.crc32(f"neutral:{row['country']}:{row['id']}".encode()))
            request = build_request(
                phrasing="p3",
                primitive="choice",
                persona=arechar.render_persona(person, row["country"]),
                question=f"{task.headlines[row['item']]}\n\n{question}",
                options=options,
                context=f"{context} {SENTENCE}",
            )
        out.append((item.meta, request))
    return out


async def predict(pairs, settings: Settings) -> pd.DataFrame:
    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=0.3), settings, ledger)
    try:
        responses = await evaluate_many(built.kernel, [r for _, r in pairs], concurrency=4)
    finally:
        await built.aclose()
    print(f"  {json.dumps({k: ledger.summary()[k] for k in ('calls', 'cache_hits', 'usd')})}")
    rows = []
    for (meta, _), response in zip(pairs, responses, strict=True):
        if isinstance(response, BaseException):
            raise response
        probs = np.asarray(decode(response, phrasing="p3", options=arechar.QUESTIONS["share"][1]), dtype=float)
        rows.append({**meta, "expected": float(np.dot(probs / probs.sum(), np.arange(1, 7)))})
    return pd.DataFrame(rows)


def effect(frame: pd.DataFrame) -> dict:
    def discern(d: pd.DataFrame, column: str) -> float:
        by = d.groupby("true")[column].mean()
        return float(by[True] - by[False])

    prompt, share = frame[frame["condition"] == "prompt"], frame[frame["condition"] == "share_only"]
    return {
        "model": discern(prompt, "expected") - discern(share, "expected"),
        "human": discern(prompt, "rating") - discern(share, "rating"),
        "model_false_shift": prompt.loc[~prompt["true"], "expected"].mean() - share.loc[~share["true"], "expected"].mean(),
        "model_true_shift": prompt.loc[prompt["true"], "expected"].mean() - share.loc[share["true"], "expected"].mean(),
    }


def main() -> None:
    settings = Settings()
    task = arechar.ArecharTask(CSV, QSF, "calibration", max_respondents=100)
    print(f"{task.ratings['id'].nunique()} US respondents, half A")
    results = {}
    for label, explicit in (("faithful", False), ("explicit", True)):
        print(f"{label}:")
        results[label] = effect(asyncio.run(predict(requests(task, explicit), settings)))
    print(json.dumps(results, indent=1))
    Path("results/c2/explicit_mechanism.json").write_text(json.dumps({"sentence": SENTENCE, **results}, indent=1))


if __name__ == "__main__":
    main()
