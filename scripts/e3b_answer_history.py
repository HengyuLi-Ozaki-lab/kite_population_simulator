"""E3b: is the individual channel open at all?

E3 found that a persona built from demographics carries almost no information about how one
particular person answers - cross-validated, a demographic model predicts an individual's answer at
r = 0.013, and the kernel reaches 0.169 only because it brings prior knowledge. The bottleneck is
the input, so the question is whether the kernel can use individual information when it is actually
given some.

This is an **oracle probe**: it puts the person's own real answers to other questions into the state
and asks for a held-out one. Handing over real answers is cheating, and the result is not a
simulation anybody could run - it is an upper bound. If accuracy does not move even with real
answers in hand, no amount of state design will open this channel and the individual layer has to
live outside the kernel.

Usage:
    uv run python scripts/e3b_answer_history.py --studies nj5dx sffyb --history 0 3 6 --max-usd 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from kite.config import Settings
from kite.eval import metrics
from kite.eval.phrasing import build_request, decode
from kite.eval.scales import Scale, strip_response_instruction
from kite.eval.socsci210 import SocSci210Task, prepare, split_studies
from kite.kernel.base import evaluate_many
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger
from kite.population.persona import Persona

QUESTION_CHARS = 320  # the tail of each earlier question, enough to identify it without flooding the state
MIN_PEOPLE = 30


def bin_of(scale: Scale, response: int) -> int:
    """Which described option a raw answer falls in.

    Not the same as `index_of`: a scale longer than ten levels is described in ten bins, so the
    level index runs past the end of `descriptions()`.
    """
    for position, (first, last) in enumerate(scale.bins()):
        if first <= response <= last:
            return position
    raise ValueError(f"response {response} is outside {scale.lo}..{scale.hi}")


def shorten(stimuli: str) -> str:
    text = strip_response_instruction(stimuli)
    return text if len(text) <= QUESTION_CHARS else "..." + text[-QUESTION_CHARS:]


def build(frame: pd.DataFrame, scales: dict[str, Scale], n_history: int, rng: np.random.Generator):
    """One request per (person, target question), with `n_history` of that person's real answers."""
    items = []
    for (study, condition, participant), person in frame.groupby(["study_id", "condition_num", "participant"]):
        person = person.sort_values("task_num")
        if len(person) < n_history + 1:
            continue
        for _, target in person.iterrows():
            others = person[person["task_num"] != target["task_num"]]
            if len(others) < n_history:
                continue
            history = others.sample(n_history, random_state=int(rng.integers(1 << 31))) if n_history else others.iloc[:0]
            scale = scales[f"{study}|{condition}|{target['task_num']}"]
            answered = [
                {
                    "question": shorten(row["stimuli"]),
                    "answer": (lambda sc: sc.descriptions()[bin_of(sc, int(row["response"]))])(scales[f"{study}|{condition}|{row['task_num']}"]),
                }
                for _, row in history.iterrows()
            ]
            persona = Persona.from_socsci210(json.loads(target["demographic"])).render()
            if answered:
                persona = {**persona, "answers_given_earlier_in_this_survey": answered}
            request = build_request(
                phrasing="p3",
                primitive="choice",
                persona=persona,
                question=strip_response_instruction(target["stimuli"]),
                options=scale.descriptions(),
            )
            items.append(
                {
                    "request": request,
                    "options": scale.descriptions(),
                    "cell": f"{study}|{condition}|{target['task_num']}",
                    "participant": int(participant),
                    "real": scale.index_of(int(target["response"])) / (scale.n_levels - 1),
                    "n_history": n_history,
                }
            )
    return items


async def run_arm(items, max_usd: float, settings: Settings) -> list[float]:
    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=max_usd), settings, ledger)
    try:
        responses = await evaluate_many(built.kernel, [item["request"] for item in items], concurrency=settings.max_concurrency)
    finally:
        await built.aclose()
    predictions = []
    for item, response in zip(items, responses, strict=True):
        if isinstance(response, BaseException):
            predictions.append(np.nan)
            continue
        probs = decode(response, phrasing="p3", options=item["options"])
        predictions.append(metrics.mean_position(probs))
    print(f"    {json.dumps(ledger.summary())}")
    return predictions


def score(items, predictions) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cell": [i["cell"] for i in items],
            "participant": [i["participant"] for i in items],
            "real": [i["real"] for i in items],
            "predicted": predictions,
        }
    ).dropna()
    rows = []
    for cell, g in frame.groupby("cell"):
        if len(g) < MIN_PEOPLE or g["predicted"].nunique() < 2 or g["real"].nunique() < 2:
            continue
        r = pearsonr(g["predicted"], g["real"]).statistic
        rows.append(
            {
                "cell": cell,
                "n": len(g),
                "r_individual": spearmanr(g["predicted"], g["real"]).statistic,
                "sd_predicted": g["predicted"].std(),
                "sd_warranted": abs(r) * g["real"].std(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--history", nargs="+", type=int, default=[0, 3, 6])
    parser.add_argument("--max-participants", type=int, default=150)
    parser.add_argument("--max-usd", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("results/e3b"))
    args = parser.parse_args()

    settings = Settings()
    unseen = set(split_studies(settings.data_dir / "socsci210" / "raw")["unseen"])
    if leaked := sorted(set(args.studies) & unseen):
        raise SystemExit(f"refusing to explore on test studies: {leaked}")

    args.out.mkdir(parents=True, exist_ok=True)
    prepared = args.out / "prepared"
    print(
        json.dumps(
            {
                k: v
                for k, v in prepare(settings.data_dir / "socsci210" / "raw", prepared, args.studies).items()
                if "unsupported" not in k and k != "studies"
            }
        )
    )
    scales = {key: Scale(**raw) for key, raw in json.loads((prepared / "scales.json").read_text()).items()}
    frame = SocSci210Task(prepared, max_participants=args.max_participants)._frame

    summary = []
    for n_history in args.history:
        items = build(frame, scales, n_history, np.random.default_rng(n_history))
        print(f"\n  history={n_history}: {len(items):,} requests")
        table = score(items, asyncio.run(run_arm(items, args.max_usd, settings)))
        table.to_csv(args.out / f"history_{n_history}.csv", index=False)
        summary.append(
            {
                "history": n_history,
                "cells": len(table),
                "r_individual": float(table["r_individual"].mean()),
                "r_median": float(table["r_individual"].median()),
                "sd_predicted": float(table["sd_predicted"].mean()),
                "shrinkage": float(table["sd_predicted"].mean() / table["sd_warranted"].mean()),
            }
        )

    result = pd.DataFrame(summary)
    result.to_csv(args.out / "summary.csv", index=False)
    print("\n" + result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    base = result.iloc[0]["r_individual"]
    print(f"\nbaseline (no history) r = {base:.3f}")
    for _, row in result.iloc[1:].iterrows():
        print(f"  with {int(row['history'])} of the person's real answers: r = {row['r_individual']:.3f} ({row['r_individual'] / base:.1f}x)")
    print("""
Real answers in the state is cheating and sets an upper bound. A large jump means the kernel can use
individual information when it has some, so the fix is to supply it. No movement means the channel
is closed and the individual layer has to live outside the kernel.""")


if __name__ == "__main__":
    main()
