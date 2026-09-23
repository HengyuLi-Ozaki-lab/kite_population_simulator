"""E3c: does a simulated person who remembers their own answers hang together like a real one?

E3 found that answering each question independently gives a simulated population whose answers
barely correlate (0.040 against a real 0.313). E3b found that the kernel *can* use a person's own
earlier answers, up to about three of them, when those answers are real.

This is the version a simulation could actually run, with nothing borrowed from the real people:
each simulated person answers the study's questions in order, one answer is **drawn** from each
predicted distribution, and the last `window` of their own drawn answers go into the state for the
next question. Their inter-item correlation matrix is then compared with the real one, over the
same personas and the same questions.

window 0 reproduces E3's independent answering, as the baseline. Three outcomes are possible and
each means something different: correlation near the real level (the individual layer works in
this form), far above it (the model copies its own previous answers - self-reinforcing
stereotyping), or near zero (conditioning on one's own history does not propagate).

Usage:
    uv run python scripts/e3c_autoregressive.py --studies nj5dx sffyb --windows 0 1 3 --max-usd 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from e3b_answer_history import bin_of, shorten
from scipy.stats import spearmanr

from kite.config import Settings
from kite.eval.phrasing import build_request, decode
from kite.eval.scales import Scale, strip_response_instruction
from kite.eval.socsci210 import SocSci210Task, prepare, split_studies
from kite.kernel.base import evaluate_many
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger
from kite.population.persona import Persona

MIN_PEOPLE = 30


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


async def simulate(people: dict, scales: dict[str, Scale], window: int, max_usd: float, settings: Settings, seed: int, trace: list | None = None):
    """Answer every person's questions in order; return {person: {task: drawn position in [0, 1]}}.

    `trace`, when given, receives one record per answer - the person, the round, the task, the
    predicted distribution and the option drawn - for analyses that need more than the correlations.
    """
    rng = np.random.default_rng(seed)
    history: dict = {person: [] for person in people}
    drawn: dict = {person: {} for person in people}
    n_rounds = max(len(info["tasks"]) for info in people.values())

    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=max_usd), settings, ledger)
    try:
        for round_index in range(n_rounds):
            batch = []
            for person, info in people.items():
                if round_index >= len(info["tasks"]):
                    continue
                task = info["tasks"][round_index]
                scale = scales[f"{info['study']}|{info['condition']}|{task['task_num']}"]
                persona = dict(info["persona"])
                if window and history[person]:
                    persona["answers_given_earlier_in_this_survey"] = history[person][-window:]
                request = build_request(
                    phrasing="p3",
                    primitive="choice",
                    persona=persona,
                    question=strip_response_instruction(task["stimuli"]),
                    options=scale.descriptions(),
                )
                batch.append((person, task, scale, request))

            responses = await evaluate_many(built.kernel, [b[3] for b in batch], concurrency=settings.max_concurrency)
            for (person, task, scale, _), response in zip(batch, responses, strict=True):
                if isinstance(response, BaseException):
                    raise response
                probs = np.asarray(decode(response, phrasing="p3", options=scale.descriptions()), dtype=float)
                probs = probs / probs.sum()
                choice = int(rng.choice(len(probs), p=probs))
                drawn[person][task["task_num"]] = choice / (len(probs) - 1)
                if trace is not None:
                    trace.append({"person": person, "round": round_index, "task_num": task["task_num"], "choice": choice, "probs": probs.tolist()})
                history[person].append({"question": shorten(task["stimuli"]), "answer": scale.descriptions()[choice]})
            print(f"    window={window} round {round_index + 1}/{n_rounds}: {len(batch)} answers")
    finally:
        await built.aclose()
    print(f"    {json.dumps(ledger.summary())}")
    return drawn


def load_people(frame: pd.DataFrame) -> dict:
    """One entry per (study, condition, participant), with their questions in task order."""
    people = {}
    for (study, condition, participant), person in frame.groupby(["study_id", "condition_num", "participant"]):
        person = person.sort_values("task_num")
        people[(study, int(condition), int(participant))] = {
            "study": study,
            "condition": int(condition),
            "persona": Persona.from_socsci210(json.loads(person.iloc[0]["demographic"])).render(),
            "tasks": person[["task_num", "stimuli", "response"]].to_dict("records"),
        }
    return people


def compare(people: dict, drawn: dict, scales: dict[str, Scale]) -> pd.DataFrame:
    rows = []
    groups: dict = {}
    for person, info in people.items():
        groups.setdefault((info["study"], info["condition"]), []).append(person)
    for (study, condition), members in groups.items():
        tasks = sorted({t["task_num"] for p in members for t in people[p]["tasks"]})
        real = pd.DataFrame(
            {
                p: {
                    t["task_num"]: bin_of(scales[f"{study}|{condition}|{t['task_num']}"], int(t["response"]))
                    / (len(scales[f"{study}|{condition}|{t['task_num']}"].bins()) - 1)
                    for t in people[p]["tasks"]
                }
                for p in members
            }
        ).T.reindex(columns=tasks)
        simulated = pd.DataFrame({p: drawn[p] for p in members}).T.reindex(columns=tasks)
        both = real.notna().all(axis=1) & simulated.notna().all(axis=1)
        real, simulated = real[both], simulated[both]
        if len(real) < MIN_PEOPLE or real.nunique().min() < 2 or simulated.nunique().min() < 2:
            continue
        r_real = spearmanr(real.to_numpy()).statistic
        r_sim = spearmanr(simulated.to_numpy()).statistic
        pairs_real, pairs_sim = off_diagonal(r_real), off_diagonal(r_sim)
        idx = np.triu_indices(len(tasks), k=1)
        adjacent = np.abs(idx[0] - idx[1]) == 1
        rows.append(
            {
                "study": study,
                "condition": condition,
                "people": len(real),
                "pairs": len(pairs_real),
                "real_mean_abs": float(np.abs(pairs_real).mean()),
                "sim_mean_abs": float(np.abs(pairs_sim).mean()),
                "sim_adjacent": float(np.abs(pairs_sim[adjacent]).mean()),
                "sim_distant": float(np.abs(pairs_sim[~adjacent]).mean()),
                "real_adjacent": float(np.abs(pairs_real[adjacent]).mean()),
                "real_distant": float(np.abs(pairs_real[~adjacent]).mean()),
                "structure": float(np.corrcoef(pairs_real, pairs_sim)[0, 1]) if np.std(pairs_sim) > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", type=int, default=[0, 1, 3])
    parser.add_argument("--max-participants", type=int, default=150)
    parser.add_argument("--max-usd", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/e3c"))
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

    people = load_people(frame)
    print(f"people: {len(people)}, answers per window: {sum(len(p['tasks']) for p in people.values()):,}")

    summary = []
    for window in args.windows:
        print(f"\n  window={window}")
        drawn = asyncio.run(simulate(people, scales, window, args.max_usd, settings, args.seed))
        table = compare(people, drawn, scales)
        table.to_csv(args.out / f"window_{window}.csv", index=False)
        w = table["pairs"]
        summary.append(
            {
                "window": window,
                "conditions": len(table),
                "real_mean_abs": float(np.average(table["real_mean_abs"], weights=w)),
                "sim_mean_abs": float(np.average(table["sim_mean_abs"], weights=w)),
                "sim_adjacent": float(np.average(table["sim_adjacent"], weights=w)),
                "sim_distant": float(np.average(table["sim_distant"], weights=w)),
                "real_adjacent": float(np.average(table["real_adjacent"], weights=w)),
                "real_distant": float(np.average(table["real_distant"], weights=w)),
                "structure": float(np.average(table["structure"].fillna(0), weights=w)),
            }
        )

    result = pd.DataFrame(summary)
    result.to_csv(args.out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    real = result["real_mean_abs"].iloc[0]
    print(f"\nreal inter-item |correlation|: {real:.3f}")
    for _, row in result.iterrows():
        print(
            f"  window {int(row['window'])}: simulated {row['sim_mean_abs']:.3f} ({row['sim_mean_abs'] / real:.0%} of real), "
            f"structure r = {row['structure']:.3f}; adjacent {row['sim_adjacent']:.3f} vs distant {row['sim_distant']:.3f}"
        )
    print("""
adjacent vs distant: the window only shows recent answers, so if simulated correlation is much
higher for neighbouring questions than distant ones while real correlation is not, the coherence is
an artefact of copying the last answer rather than a person's stable disposition.""")


if __name__ == "__main__":
    main()
