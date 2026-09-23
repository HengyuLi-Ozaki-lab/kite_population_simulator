"""B1: does remembering your own answers bend the population's marginals?

E3c showed that a simulated person who sees their own last few drawn answers hangs together much
more like a real person. The risk of that mechanism is self-reinforcement: if each answer pulls the
next one further the same way, the population drifts - toward the extremes, or simply away from the
distribution people actually gave - and a long simulation built on it would be unusable however
coherent its individuals look.

This re-runs E3c's simulation with the same seed, so every request is already cached and nothing is
spent, records every drawn answer, and compares each cell's distribution of drawn answers with the
real one, for independent answering (window 0) and for self-conditioned answering (windows 1, 3).

Criterion, written before the first run (2026-09-21), judged on the clean study sffyb only - nj5dx
repeats task numbers within a person and is reported separately:

  Sampling alone moves the distance: redrawing the window-0 answers from the very same predicted
  distributions with a different seed gives a different distance to the human cells. That spread is
  the noise floor. **No drift detected** means the window-3 distance does not exceed the 95th
  percentile of the window-0 distances over 200 redraws. If it does, the size is reported as the
  share of the kernel's advantage over a uniform guess that the self-conditioning gives back.

Also reported, without a threshold because none has been calibrated: the same comparison on the
averaged predicted distributions (no sampling noise, the sharper diagnostic of a systematic shift),
the share of answers at either end of the scale, and early against late questions.

Usage:
    uv run python scripts/b1_marginal_drift.py --studies nj5dx sffyb --windows 0 1 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from e3b_answer_history import bin_of
from e3c_autoregressive import load_people, simulate

from kite.config import Settings
from kite.eval import metrics
from kite.eval.scales import Scale
from kite.eval.socsci210 import SocSci210Task, prepare, split_studies

N_REDRAWS = 200
MIN_DRAWS = 30
CLEAN = {"sffyb"}  # nj5dx repeats task numbers within a person; see docs/data/socsci210-schema.md


def human_by_option(cells: dict, scales: dict[str, Scale]) -> dict[str, np.ndarray]:
    """Human counts per *described option*: a scale longer than ten levels is described in ten bins."""
    out = {}
    for key, counts in cells.items():
        scale = scales[key]
        binned = np.zeros(len(scale.bins()))
        for index, count in enumerate(counts):
            binned[bin_of(scale, scale.lo + index)] += count
        out[key] = binned
    return out


def cell_key(record: dict) -> str:
    study, condition, _ = record["person"]
    return f"{study}|{condition}|{record['task_num']}"


def distances(choices_by_cell: dict[str, list[int]], probs_by_cell: dict[str, list], human: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for key, choices in choices_by_cell.items():
        if len(choices) < MIN_DRAWS or key not in human or human[key].sum() == 0:
            continue
        n_options = len(human[key])
        drawn = np.bincount(choices, minlength=n_options).astype(float)
        mean_probs = np.mean(probs_by_cell[key], axis=0)
        out[key] = {
            "drawn": metrics.wasserstein_unit(drawn, human[key]),
            "expected": metrics.wasserstein_unit(mean_probs, human[key]),
            "uniform": metrics.wasserstein_unit(np.ones(n_options), human[key]),
            "ends_drawn": float((drawn[0] + drawn[-1]) / drawn.sum()),
            "ends_human": float((human[key][0] + human[key][-1]) / human[key].sum()),
            "mean_drawn": metrics.mean_position(drawn),
            "mean_human": metrics.mean_position(human[key]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", type=int, default=[0, 1, 3])
    parser.add_argument("--max-participants", type=int, default=150)
    parser.add_argument("--max-usd", type=float, default=0.2, help="every request should be cached; a small cap makes a divergence loud")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/b1"))
    args = parser.parse_args()

    settings = Settings()
    unseen = set(split_studies(settings.data_dir / "socsci210" / "raw")["unseen"])
    if leaked := sorted(set(args.studies) & unseen):
        raise SystemExit(f"refusing to explore on test studies: {leaked}")
    if 0 not in args.windows:
        raise SystemExit("window 0 is the baseline and the source of the noise floor; include it")

    args.out.mkdir(parents=True, exist_ok=True)
    prepared = args.out / "prepared"
    prepare(settings.data_dir / "socsci210" / "raw", prepared, args.studies)
    scales = {key: Scale(**raw) for key, raw in json.loads((prepared / "scales.json").read_text()).items()}
    human = human_by_option(json.loads((prepared / "cells.json").read_text()), scales)
    people = load_people(SocSci210Task(prepared, max_participants=args.max_participants)._frame)

    traces = {}
    for window in args.windows:
        print(f"\n  window={window}")
        trace: list = []
        asyncio.run(simulate(people, scales, window, args.max_usd, settings, args.seed, trace=trace))
        traces[window] = trace
        pd.DataFrame(
            [
                {
                    "study": r["person"][0],
                    "condition": r["person"][1],
                    "participant": r["person"][2],
                    **{k: r[k] for k in ("round", "task_num", "choice")},
                }
                for r in trace
            ]
        ).to_csv(args.out / f"draws_window_{window}.csv", index=False)

    # which questions come early and which late, per study, by the order people met them
    last_round = defaultdict(int)
    for record in traces[0]:
        last_round[record["person"][0]] = max(last_round[record["person"][0]], record["round"])

    rows = []
    per_window = {}
    for window, trace in traces.items():
        choices, probs, rounds = defaultdict(list), defaultdict(list), defaultdict(list)
        for record in trace:
            key = cell_key(record)
            choices[key].append(record["choice"])
            probs[key].append(record["probs"])
            rounds[key].append(record["round"])
        table = distances(choices, probs, human)
        per_window[window] = table
        for key, values in table.items():
            study = key.split("|")[0]
            rows.append({"window": window, "cell": key, "study": study, "late": np.mean(rounds[key]) > last_round[study] / 2, **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "cells.csv", index=False)

    # the noise floor: redraw window-0 answers from the same predicted distributions
    rng = np.random.default_rng(12345)
    base_probs = defaultdict(list)
    for record in traces[0]:
        base_probs[cell_key(record)].append(record["probs"])
    floor = defaultdict(list)
    for _ in range(N_REDRAWS):
        by_study = defaultdict(list)
        for key, plist in base_probs.items():
            if key not in per_window[0]:
                continue
            redrawn = [int(rng.choice(len(p), p=np.asarray(p) / np.sum(p))) for p in plist]
            hist = np.bincount(redrawn, minlength=len(human[key])).astype(float)
            by_study[key.split("|")[0]].append(metrics.wasserstein_unit(hist, human[key]))
        for study, values in by_study.items():
            floor[study].append(float(np.mean(values)))

    print()
    summary = []
    for study, group in frame.groupby("study"):
        floor_values = np.asarray(floor[study])
        p95 = float(np.percentile(floor_values, 95))
        base = group[group["window"] == 0]
        print(f"{study} ({'clean' if study in CLEAN else 'task numbers repeat within a person'}), {base['cell'].nunique()} cells")
        print(f"  noise floor of the window-0 distance over {N_REDRAWS} redraws: mean {floor_values.mean():.4f}, p95 {p95:.4f}")
        for window, g in group.groupby("window"):
            uniform_gap = g["uniform"].mean() - base["expected"].mean()
            given_back = (g["expected"].mean() - base["expected"].mean()) / uniform_gap if uniform_gap > 0 else float("nan")
            exceeds = float((floor_values >= g["drawn"].mean()).mean())
            summary.append(
                {
                    "study": study,
                    "window": window,
                    "distance_drawn": g["drawn"].mean(),
                    "floor_p95": p95,
                    "share_of_redraws_at_least_this_far": exceeds,
                    "distance_expected": g["expected"].mean(),
                    "distance_uniform": g["uniform"].mean(),
                    "advantage_given_back": given_back,
                    "ends_drawn": g["ends_drawn"].mean(),
                    "ends_human": g["ends_human"].mean(),
                    "mean_shift": (g["mean_drawn"] - g["mean_human"]).mean(),
                    "distance_early": g.loc[~g["late"], "drawn"].mean(),
                    "distance_late": g.loc[g["late"], "drawn"].mean(),
                }
            )
            verdict = "within the floor" if g["drawn"].mean() <= p95 else "BEYOND the floor"
            print(
                f"  window {window}: drawn {g['drawn'].mean():.4f} ({verdict}; {exceeds:.0%} of redraws are at least this far)  "
                f"expected {g['expected'].mean():.4f}  uniform {g['uniform'].mean():.4f}  advantage given back {given_back:+.1%}"
            )
            print(
                f"            answers at either end: simulated {g['ends_drawn'].mean():.3f} vs human {g['ends_human'].mean():.3f};  "
                f"early {g.loc[~g['late'], 'drawn'].mean():.4f} / late {g.loc[g['late'], 'drawn'].mean():.4f}"
            )
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
