"""E3d: can an LLM compress a person's answer history into something the kernel absorbs better?

E3b showed the kernel takes in individual information from about three earlier answers and then
stops: with six, a cross-validated regression on the same answers reaches r = 0.427 while the kernel
stays at 0.355. The information is there; the kernel cannot integrate it at length.

So each person's six history answers are compressed by an LLM (GPT-5.6 Luna through the local Codex
CLI) into one or two sentences, and three arms are compared on the same people and the same held-out
questions:

  none      no history
  raw       the six answers, verbatim
  summary   the LLM's summary of those same six answers

The content is identical in raw and summary; only the representation differs. The summary prompt
forbids inferring anything the answers do not show - an LLM that invents traits would make the
simulation look more coherent while adding nothing true about the person.

This is still an oracle probe: the history holds real answers. It asks whether a better
representation of real information helps, not whether a simulation can run this way.

Usage:
    uv run python scripts/e3d_compressed_history.py --studies nj5dx sffyb --max-usd 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from e3b_answer_history import bin_of, shorten
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

N_HISTORY = 6
BATCH = 25
CODEX_MODEL = "gpt-5.6-luna"
CODEX_EFFORT = "low"
CODEX_PARALLEL = 4
MIN_PEOPLE = 30

PROMPT = """You are compressing survey answers. For each respondent below you are given questions they
answered and the answer each one chose.

For each respondent, write one or two short sentences, at most 40 words, summarising the attitudes
and response tendencies that these specific answers show.

Rules:
- Use only what the listed answers show. Do not infer anything the answers do not support.
- Do not mention or guess demographics, personality, or anything about the person beyond these answers.
- Keep each attitude's direction and strength faithful to the answer given ("strongly agrees",
  "mildly opposes", "chose the midpoint").
- If the answers are mixed or neutral, say so. Do not force them into a coherent story.
- Describe the content of the questions; do not refer to question numbers.

Return one entry per respondent id.

Respondents:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["id", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["summaries"],
    "additionalProperties": False,
}


def codex_batch(batch: list[dict]) -> dict[str, str]:
    """One Codex call summarising up to BATCH respondents; returns {id: summary}."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "schema.json").write_text(json.dumps(SCHEMA))
        prompt = PROMPT + json.dumps(batch, ensure_ascii=False, indent=1)
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "-m",
                CODEX_MODEL,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--output-schema",
                str(tmp / "schema.json"),
                "-o",
                str(tmp / "out.json"),
                "-c",
                f'model_reasoning_effort="{CODEX_EFFORT}"',
                "-",
            ],
            input=prompt,
            text=True,
            capture_output=True,
            cwd=tmp,
            timeout=600,
        )
        if completed.returncode != 0 or not (tmp / "out.json").exists():
            raise RuntimeError(f"codex failed ({completed.returncode}): {completed.stderr[-500:]}")
        parsed = json.loads((tmp / "out.json").read_text())
    return {entry["id"]: entry["summary"] for entry in parsed["summaries"]}


async def summarise(histories: dict[str, list[dict]], cache: Path) -> dict[str, str]:
    """Summaries for every person, cached on disk so a rerun does not repeat the LLM work."""
    done: dict[str, str] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [pid for pid in histories if pid not in done]
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"  summaries: {len(done)} cached, {len(todo)} to write in {len(batches)} Codex calls")
    semaphore = asyncio.Semaphore(CODEX_PARALLEL)

    async def one(ids: list[str]) -> None:
        async with semaphore:
            for attempt in range(3):
                payload = [{"id": pid, "answers": histories[pid]} for pid in ids]
                try:
                    got = await asyncio.to_thread(codex_batch, payload)
                except Exception as error:  # noqa: BLE001 - retried, then reported
                    print(f"    batch of {len(ids)} failed (attempt {attempt + 1}): {error}")
                    continue
                missing = [pid for pid in ids if not got.get(pid, "").strip()]
                done.update({pid: got[pid].strip() for pid in ids if pid not in missing})
                cache.write_text(json.dumps(done, ensure_ascii=False, indent=1))
                if not missing:
                    return
                ids = missing  # retry only what came back empty
            print(f"    giving up on {len(ids)} respondents after three attempts")

    await asyncio.gather(*(one(batch) for batch in batches))
    return done


async def predict(items: list[dict], max_usd: float, settings: Settings) -> list[float]:
    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=max_usd), settings, ledger)
    try:
        responses = await evaluate_many(built.kernel, [i["request"] for i in items], concurrency=settings.max_concurrency)
    finally:
        await built.aclose()
    print(f"    {json.dumps(ledger.summary())}")
    out = []
    for item, response in zip(items, responses, strict=True):
        if isinstance(response, BaseException):
            out.append(np.nan)
            continue
        out.append(metrics.mean_position(decode(response, phrasing="p3", options=item["options"])))
    return out


def score(items: list[dict], predictions: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"cell": [i["cell"] for i in items], "real": [i["real"] for i in items], "predicted": predictions}).dropna()
    rows = []
    for cell, g in frame.groupby("cell"):
        if len(g) < MIN_PEOPLE or g["predicted"].nunique() < 2 or g["real"].nunique() < 2:
            continue
        r = pearsonr(g["predicted"], g["real"]).statistic
        rows.append(
            {
                "cell": cell,
                "r_individual": spearmanr(g["predicted"], g["real"]).statistic,
                "sd_predicted": g["predicted"].std(),
                "sd_warranted": abs(r) * g["real"].std(),
            }
        )
    return pd.DataFrame(rows)


def regression_ceiling(frame: pd.DataFrame, history_tasks: dict[str, list[int]]) -> float:
    """Cross-validated ridge on the same six history answers, for the same targets."""
    values = []
    for (study, _), g in frame.groupby(["study_id", "condition_num"]):
        wide = g.pivot_table(index="participant", columns="task_num", values="response").dropna()
        hist = [t for t in history_tasks[study] if t in wide.columns]
        if len(wide) < 60 or len(hist) < N_HISTORY:
            continue
        X = wide[hist].to_numpy(float)
        for target in (t for t in wide.columns if t not in hist):
            y = wide[target].to_numpy(float)
            if np.std(y) < 1e-9:
                continue
            idx = np.random.default_rng(0).permutation(len(y))
            pred = np.zeros(len(y))
            for fold in np.array_split(idx, 5):
                train = np.setdiff1d(idx, fold)
                mx, my = X[train].mean(0), y[train].mean()
                beta = np.linalg.solve((X[train] - mx).T @ (X[train] - mx) + np.eye(X.shape[1]), (X[train] - mx).T @ (y[train] - my))
                pred[fold] = (X[fold] - mx) @ beta + my
            values.append(spearmanr(pred, y).statistic)
    return float(np.nanmean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--max-participants", type=int, default=150)
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("results/e3d"))
    args = parser.parse_args()

    settings = Settings()
    unseen = set(split_studies(settings.data_dir / "socsci210" / "raw")["unseen"])
    if leaked := sorted(set(args.studies) & unseen):
        raise SystemExit(f"refusing to explore on test studies: {leaked}")

    args.out.mkdir(parents=True, exist_ok=True)
    prepared = args.out / "prepared"
    prepare(settings.data_dir / "socsci210" / "raw", prepared, args.studies)
    scales = {key: Scale(**raw) for key, raw in json.loads((prepared / "scales.json").read_text()).items()}
    frame = SocSci210Task(prepared, max_participants=args.max_participants)._frame

    # the same six history questions for everyone in a study, so raw and summary carry identical content
    history_tasks = {s: sorted(g["task_num"].unique())[:N_HISTORY] for s, g in frame.groupby("study_id")}
    histories, people = {}, {}
    for (study, condition, participant), person in frame.groupby(["study_id", "condition_num", "participant"]):
        answers = {int(r.task_num): r for r in person.itertuples(index=False)}
        if not all(t in answers for t in history_tasks[study]):
            continue
        pid = f"{study}|{int(condition)}|{int(participant)}"
        histories[pid] = [
            {
                "question": shorten(answers[t].stimuli),
                "answer": (sc := scales[f"{study}|{int(condition)}|{t}"]).descriptions()[bin_of(sc, int(answers[t].response))],
            }
            for t in history_tasks[study]
        ]
        people[pid] = (study, int(condition), answers)
    print(f"people with a complete history: {len(people)}")

    summaries = asyncio.run(summarise(histories, args.out / "summaries.json"))
    (args.out / "codex.json").write_text(json.dumps({"model": CODEX_MODEL, "effort": CODEX_EFFORT, "prompt": PROMPT}, indent=1))

    arms = {"none": None, "raw": "raw", "summary": "summary"}
    summary_rows = []
    for arm, kind in arms.items():
        items = []
        for pid, (study, condition, answers) in people.items():
            if kind == "summary" and pid not in summaries:
                continue
            persona = Persona.from_socsci210(json.loads(next(iter(answers.values())).demographic)).render()
            if kind == "raw":
                persona["answers_given_earlier_in_this_survey"] = histories[pid]
            elif kind == "summary":
                persona["what_their_earlier_answers_in_this_survey_show"] = summaries[pid]
            for task, row in answers.items():
                if task in history_tasks[study]:
                    continue
                scale = scales[f"{study}|{condition}|{task}"]
                items.append(
                    {
                        "request": build_request(
                            phrasing="p3",
                            primitive="choice",
                            persona=persona,
                            question=strip_response_instruction(row.stimuli),
                            options=scale.descriptions(),
                        ),
                        "options": scale.descriptions(),
                        "cell": f"{study}|{condition}|{task}",
                        "real": scale.index_of(int(row.response)) / (scale.n_levels - 1),
                    }
                )
        print(f"\n  arm={arm}: {len(items):,} requests")
        table = score(items, asyncio.run(predict(items, args.max_usd, settings)))
        table.to_csv(args.out / f"arm_{arm}.csv", index=False)
        summary_rows.append(
            {
                "arm": arm,
                "cells": len(table),
                "r_individual": float(table["r_individual"].mean()),
                "r_median": float(table["r_individual"].median()),
                "shrinkage": float(table["sd_predicted"].mean() / table["sd_warranted"].mean()),
            }
        )

    ceiling = regression_ceiling(frame, history_tasks)
    result = pd.DataFrame(summary_rows)
    result["share_of_ceiling"] = result["r_individual"] / ceiling
    result.to_csv(args.out / "summary.csv", index=False)
    print("\n" + result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nregression ceiling on the same six answers: r = {ceiling:.3f}")
    print(f"summaries written: {len(summaries)} of {len(histories)}")


if __name__ == "__main__":
    main()
