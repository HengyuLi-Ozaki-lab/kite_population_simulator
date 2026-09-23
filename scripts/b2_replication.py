"""B2: replicate the individual layer on six clean development studies.

The criteria, the studies and the arms are fixed in configs/eval/b2_criteria.yaml, which was written
before any model output for these studies existed. This script refuses to run while that file or this
script has uncommitted changes, so the configuration that produced a result is always on record.

Arms (see the criteria file): independent (window 0), raw_memory (window 3), and corrected - the
raw_memory draws mapped per cell, rank-preserving, onto the independent arm's mean predicted
distribution. Nulls: the independent arm re-simulated with 40 further seeds (its requests do not
depend on any draw, so they replay from cache for free), and its answers redrawn 200 times from the
same predicted distributions for the marginal floor.

Usage:
    PYTHONPATH=scripts uv run python scripts/b2_replication.py --check   # free: the whole pipeline on B1's cached studies
    PYTHONPATH=scripts uv run python scripts/b2_replication.py           # the run the criteria file describes

`--check` replays nj5dx and sffyb, which B1 simulated with the same seeds and the same people, so every
request is already cached; its sffyb numbers should agree with docs/decisions/B1.md to within the
randomness of the tie-breaking. It spends at most five cents and prints no verdict.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from b1_marginal_drift import cell_key, human_by_option
from b1b_drift_diagnostics import quantile_map
from e3b_answer_history import bin_of
from e3c_autoregressive import compare, load_people, simulate
from scipy.stats import spearmanr

from kite.config import Settings
from kite.eval import metrics
from kite.eval.scales import Scale
from kite.eval.socsci210 import SocSci210Task, prepare, split_studies
from kite.kernel.ledger import BudgetExceeded

CRITERIA = Path("configs/eval/b2_criteria.yaml")
OUT = Path("results/b2")
NULL_SEEDS = range(1, 41)
REDRAWS = 200
SPLITS = 50
MIN_DRAWS = 30
ARM_USD = 3.0  # two paid arms; the criteria file's total is 6


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def run(people: dict, scales: dict, window: int, seed: int, max_usd: float, settings: Settings) -> list[dict]:
    """Simulate, retrying on a transient failure: finished calls replay from cache, and the seeded draws repeat exactly."""
    for attempt in range(3):
        trace: list[dict] = []
        try:
            asyncio.run(simulate(people, scales, window, max_usd, settings, seed, trace=trace))
            return trace
        except BudgetExceeded:
            raise
        except Exception as error:  # noqa: BLE001 - replayed from cache on the next attempt
            if "402" in str(error) or "billing" in str(error).lower():
                # out of credits is not transient; everything finished so far is cached and replays for free
                raise SystemExit(
                    f"window {window} seed {seed}: the account is out of credits after {len(trace):,} answers; top up and rerun"
                ) from error
            print(f"    window {window} seed {seed}: attempt {attempt + 1} failed ({str(error)[:120]}); replaying")
    raise RuntimeError(f"window {window} seed {seed} failed three times")


def drawn_from(trace: list[dict], choices: list[int]) -> dict:
    drawn: dict = defaultdict(dict)
    for record, choice in zip(trace, choices, strict=True):
        drawn[record["person"]][record["task_num"]] = choice / (len(record["probs"]) - 1)
    return drawn


def coherence(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for study, g in table.groupby("study"):
        w = g["pairs"]
        avg = lambda column, g=g, w=w: float(np.average(g[column], weights=w))  # noqa: E731
        out[study] = {
            "magnitude": avg("sim_mean_abs") / avg("real_mean_abs"),
            "structure": float(np.average(g["structure"].fillna(0), weights=w)),
            "local_sim": avg("sim_adjacent") / avg("sim_distant"),
            "local_real": avg("real_adjacent") / avg("real_distant"),
            "distant_vs_real": avg("sim_distant") / avg("real_distant"),
            "real_mean_abs": avg("real_mean_abs"),
        }
    return out


def distances(trace: list[dict], choices: list[int], human: dict) -> dict[str, float]:
    by_cell = defaultdict(list)
    for record, choice in zip(trace, choices, strict=True):
        by_cell[cell_key(record)].append(choice)
    per_study = defaultdict(list)
    for key, values in by_cell.items():
        if len(values) >= MIN_DRAWS and key in human:
            per_study[key.split("|")[0]].append(metrics.wasserstein_unit(np.bincount(values, minlength=len(human[key])).astype(float), human[key]))
    return {study: float(np.mean(v)) for study, v in per_study.items()}


def expected_distances(trace: list[dict], human: dict) -> tuple[dict, dict]:
    """Distance of the mean predicted distribution per cell, and of a uniform guess, averaged per study."""
    by_cell = defaultdict(list)
    for record in trace:
        by_cell[cell_key(record)].append(record["probs"])
    model, uniform = defaultdict(list), defaultdict(list)
    for key, probs in by_cell.items():
        if len(probs) >= MIN_DRAWS and key in human:
            study = key.split("|")[0]
            model[study].append(metrics.wasserstein_unit(np.mean(probs, axis=0), human[key]))
            uniform[study].append(metrics.wasserstein_unit(np.ones(len(human[key])), human[key]))
    return {s: float(np.mean(v)) for s, v in model.items()}, {s: float(np.mean(v)) for s, v in uniform.items()}


def human_split_half(people: dict, scales: dict, rng: np.random.Generator) -> dict[str, float]:
    """Structure between two random halves of the real respondents: the reliability reference for structure."""
    groups = defaultdict(list)
    for person, info in people.items():
        groups[(info["study"], info["condition"])].append(person)
    per_study = defaultdict(list)
    for (study, condition), members in groups.items():
        tasks = sorted({t["task_num"] for p in members for t in people[p]["tasks"]})
        wide = pd.DataFrame(
            {
                p: {
                    t["task_num"]: bin_of(scales[f"{study}|{condition}|{t['task_num']}"], int(t["response"]))
                    / (len(scales[f"{study}|{condition}|{t['task_num']}"].bins()) - 1)
                    for t in people[p]["tasks"]
                }
                for p in members
            }
        ).T.reindex(columns=tasks)
        wide = wide.dropna().to_numpy()
        if len(wide) < 60 or len(tasks) < 3:
            continue
        upper = np.triu_indices(len(tasks), k=1)
        values = []
        for _ in range(SPLITS):
            half = rng.permutation(len(wide)) < len(wide) // 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a question everyone in one half answered alike has no correlation
                ra, rb = spearmanr(wide[half]).statistic[upper], spearmanr(wide[~half]).statistic[upper]
            usable = np.isfinite(ra) & np.isfinite(rb)
            if usable.sum() >= 3:
                values.append(np.corrcoef(ra[usable], rb[usable])[0, 1])
        if values:
            per_study[study].append((np.nanmean(values), len(upper[0])))
    return {s: float(np.average([v for v, _ in e], weights=[n for _, n in e])) for s, e in per_study.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="free end-to-end check on B1's cached studies; no verdict")
    args = parser.parse_args()
    if not committed(str(CRITERIA), "scripts/b2_replication.py"):
        raise SystemExit("refusing to run: commit configs/eval/b2_criteria.yaml and this script first")
    criteria = yaml.safe_load(CRITERIA.read_text())
    global OUT, ARM_USD
    if args.check:
        studies, OUT, ARM_USD = ["nj5dx", "sffyb"], Path("results/b2-check"), 0.05
    else:
        studies = criteria["studies"]
    settings = Settings()
    unseen = set(split_studies(settings.data_dir / "socsci210" / "raw")["unseen"])
    if leaked := sorted(set(studies) & unseen):
        raise SystemExit(f"refusing to explore on test studies: {leaked}")

    OUT.mkdir(parents=True, exist_ok=True)
    prepared = OUT / "prepared"
    prepare(settings.data_dir / "socsci210" / "raw", prepared, studies)
    scales = {key: Scale(**raw) for key, raw in json.loads((prepared / "scales.json").read_text()).items()}
    human = human_by_option(json.loads((prepared / "cells.json").read_text()), scales)
    people = load_people(SocSci210Task(prepared, max_participants=150)._frame)
    print(f"{len(people):,} people, {sum(len(p['tasks']) for p in people.values()):,} answers per arm, studies {studies}")

    print("\nindependent arm (window 0, seed 0)")
    independent = run(people, scales, 0, 0, ARM_USD, settings)
    print("\nraw_memory arm (window 3, seed 0)")
    raw_memory = run(people, scales, 3, 0, ARM_USD, settings)

    # corrected: raw_memory draws mapped per cell onto the independent arm's mean predicted distribution
    target = defaultdict(list)
    for record in independent:
        target[cell_key(record)].append(record["probs"])
    target = {key: np.mean(v, axis=0) for key, v in target.items()}
    positions = defaultdict(list)
    for index, record in enumerate(raw_memory):
        positions[cell_key(record)].append(index)
    corrected = [0] * len(raw_memory)
    rng = np.random.default_rng(0)
    for key, indices in positions.items():
        mapped = quantile_map(np.array([raw_memory[i]["choice"] for i in indices]), target[key], rng)
        for i, value in zip(indices, mapped, strict=True):
            corrected[i] = int(value)

    arms = {
        "independent": (independent, [r["choice"] for r in independent]),
        "raw_memory": (raw_memory, [r["choice"] for r in raw_memory]),
        "corrected": (raw_memory, corrected),
    }
    measures = {name: coherence(compare(people, drawn_from(t, c), scales)) for name, (t, c) in arms.items()}
    marginal = {name: distances(t, c, human) for name, (t, c) in arms.items()}
    for name, (t, c) in arms.items():
        pd.DataFrame(
            [
                {"study": r["person"][0], "condition": r["person"][1], "participant": r["person"][2], "task_num": r["task_num"], "choice": ch}
                for r, ch in zip(t, c, strict=True)
            ]
        ).to_csv(OUT / f"draws_{name}.csv", index=False)

    print(f"\nnull: independent arm with seeds {NULL_SEEDS.start}-{NULL_SEEDS.stop - 1} (from cache)")
    null = defaultdict(lambda: defaultdict(list))
    for seed in NULL_SEEDS:
        trace = run(people, scales, 0, seed, 0.05, settings)
        for study, m in coherence(compare(people, drawn_from(trace, [r["choice"] for r in trace]), scales)).items():
            null[study]["magnitude"].append(m["magnitude"])
            null[study]["structure"].append(m["structure"])

    floor = defaultdict(list)
    redraw = np.random.default_rng(12345)
    for _ in range(REDRAWS):
        choices = [int(redraw.choice(len(r["probs"]), p=np.asarray(r["probs"]) / np.sum(r["probs"]))) for r in independent]
        for study, value in distances(independent, choices, human).items():
            floor[study].append(value)

    expected = {name: expected_distances(t, human)[0] for name, (t, _) in arms.items() if name != "corrected"}
    uniform = expected_distances(independent, human)[1]
    split_half = human_split_half(people, scales, np.random.default_rng(7))

    rows = []
    for study in [s for s in studies if s in null]:
        m_null, s_null = np.percentile(null[study]["magnitude"], 95), np.percentile(null[study]["structure"], 95)
        corr, raw, ind = measures["corrected"][study], measures["raw_memory"][study], measures["independent"][study]
        floor_p95 = float(np.percentile(floor[study], 95))
        gap = uniform[study] - expected["independent"][study]
        rows.append(
            {
                "study": study,
                "coherence_gain": bool(corr["magnitude"] > m_null and corr["structure"] > s_null),
                "marginals_kept": bool(marginal["corrected"][study] <= floor_p95),
                "magnitude_independent": ind["magnitude"],
                "magnitude_null_p95": float(m_null),
                "magnitude_raw": raw["magnitude"],
                "magnitude_corrected": corr["magnitude"],
                "structure_independent": ind["structure"],
                "structure_null_p95": float(s_null),
                "structure_raw": raw["structure"],
                "structure_corrected": corr["structure"],
                "human_split_half_structure": split_half.get(study, float("nan")),
                "distance_independent": marginal["independent"][study],
                "distance_raw": marginal["raw_memory"][study],
                "distance_corrected": marginal["corrected"][study],
                "floor_p95": floor_p95,
                "raw_drift_advantage_given_back": (expected["raw_memory"][study] - expected["independent"][study]) / gap if gap > 0 else float("nan"),
                "kernel_advantage_over_uniform": gap,
                "coherence_kept_by_correction": corr["magnitude"] / raw["magnitude"],
                "local_corrected": corr["local_sim"],
                "local_real": corr["local_real"],
                "distant_vs_real_corrected": corr["distant_vs_real"],
                "real_mean_abs": corr["real_mean_abs"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "per_study.csv", index=False)

    n_gain, n_kept = int(table["coherence_gain"].sum()), int(table["marginals_kept"].sum())
    verdict = "replicates" if n_gain >= 5 and n_kept >= 5 else "partial" if n_gain in (3, 4) and n_kept >= 5 else "fails"
    if args.check:
        verdict = "check only - no verdict"
    pd.set_option("display.width", 250)
    show = [
        "study",
        "coherence_gain",
        "marginals_kept",
        "magnitude_independent",
        "magnitude_null_p95",
        "magnitude_raw",
        "magnitude_corrected",
        "structure_independent",
        "structure_null_p95",
        "structure_raw",
        "structure_corrected",
        "human_split_half_structure",
    ]
    print("\n" + table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    show = [
        "study",
        "distance_independent",
        "distance_raw",
        "distance_corrected",
        "floor_p95",
        "raw_drift_advantage_given_back",
        "coherence_kept_by_correction",
        "local_corrected",
        "local_real",
        "distant_vs_real_corrected",
    ]
    print("\n" + table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\ncoherence_gain in {n_gain} of {len(table)}, marginals_kept in {n_kept} of {len(table)}  ->  {verdict.upper()}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (OUT / "summary.json").write_text(
        json.dumps(
            {"commit": commit, "criteria": criteria, "verdict": verdict, "coherence_gain": n_gain, "marginals_kept": n_kept, "per_study": rows},
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
