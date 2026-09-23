"""B1b: why do self-conditioned marginals drift, and does a rank-preserving correction repair them?

Exploratory, development studies only, and free: every request is already cached.

B1 found that letting simulated people see their own drawn answers bends the population's marginals
(window 3 gives back 28% of the kernel's advantage over a uniform guess on the clean study; window 1
gives back 62%). Three questions follow.

1. Is the noise floor itself sound? Window-0 requests do not depend on the draws, so simulating with
   other seeds is free; those distances should sit inside the floor that B1 built by redrawing.

2. Is it conditioning as such, or feeding back the model's own imperfect draws? E3b conditioned on
   the person's *real* answers. If the averaged predicted distributions stay as close to the human
   cells there, the damage comes from compounding the simulation's own errors.

3. Can dependence and marginals be separated? Self-conditioning supplies the dependence between a
   person's answers; independent answering supplies good marginals. A randomized quantile map, per
   cell, from the self-conditioned draws onto the independent marginal keeps each person's rank
   within the question and restores the marginal by construction. What matters is how much of the
   recovered inter-item coherence survives the map - ties on a discrete scale have to be broken at
   random, which can only attenuate it.

Usage:
    PYTHONPATH=scripts uv run python scripts/b1b_drift_diagnostics.py
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from b1_marginal_drift import cell_key, human_by_option
from e3b_answer_history import build
from e3c_autoregressive import compare, load_people, simulate

from kite.config import Settings
from kite.eval import metrics
from kite.eval.phrasing import decode
from kite.eval.scales import Scale
from kite.eval.socsci210 import SocSci210Task
from kite.kernel.base import evaluate_many
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger

PREPARED = Path("results/b1/prepared")  # written by b1_marginal_drift.py
CLEAN = "sffyb"
MIN_DRAWS = 30


def mean_distance(choices: dict[str, list[int]], human: dict[str, np.ndarray], study: str) -> float:
    values = [
        metrics.wasserstein_unit(np.bincount(c, minlength=len(human[k])).astype(float), human[k])
        for k, c in choices.items()
        if k.startswith(study + "|") and len(c) >= MIN_DRAWS and k in human
    ]
    return float(np.mean(values))


def summarise(table: pd.DataFrame, study: str) -> dict:
    g = table[table["study"] == study]
    w = g["pairs"]
    f = lambda c: float(np.average(g[c], weights=w))  # noqa: E731
    return {
        "magnitude": f("sim_mean_abs") / f("real_mean_abs"),
        "structure": f("structure"),
        "adjacent/distant": f("sim_adjacent") / f("sim_distant"),
        "real adjacent/distant": f("real_adjacent") / f("real_distant"),
        "distant vs real": f("sim_distant") / f("real_distant"),
    }


def quantile_map(choices: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT of `choices` under their own empirical distribution, inverted through `target`."""
    k = len(target)
    source = np.bincount(choices, minlength=k).astype(float)
    cdf_source = np.concatenate([[0.0], np.cumsum(source / source.sum())])
    u = rng.uniform(cdf_source[choices], cdf_source[choices + 1])
    cdf_target = np.cumsum(target / target.sum())
    return np.minimum(np.searchsorted(cdf_target, u, side="left"), k - 1)


async def oracle_arm(items: list[dict], settings: Settings) -> dict[str, list]:
    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=0.2), settings, ledger)
    try:
        responses = await evaluate_many(built.kernel, [i["request"] for i in items], concurrency=settings.max_concurrency)
    finally:
        await built.aclose()
    print(f"    calls {ledger.summary()['calls']}, cache hits {ledger.summary()['cache_hits']}")
    by_cell = defaultdict(list)
    for item, response in zip(items, responses, strict=True):
        if not isinstance(response, BaseException):
            by_cell[item["cell"]].append(decode(response, phrasing="p3", options=item["options"]))
    return by_cell


def main() -> None:
    settings = Settings()
    scales = {key: Scale(**raw) for key, raw in json.loads((PREPARED / "scales.json").read_text()).items()}
    human = human_by_option(json.loads((PREPARED / "cells.json").read_text()), scales)
    frame = SocSci210Task(PREPARED, max_participants=150)._frame
    people = load_people(frame)

    print("1. window-0 distances under other simulation seeds (each should sit inside B1's floor)")
    for seed in (1, 2, 3, 4, 5):
        trace: list = []
        asyncio.run(simulate(people, scales, 0, 0.2, settings, seed, trace=trace))
        choices = defaultdict(list)
        for r in trace:
            choices[cell_key(r)].append(r["choice"])
        print(f"   seed {seed}: sffyb {mean_distance(choices, human, 'sffyb'):.4f}   nj5dx {mean_distance(choices, human, 'nj5dx'):.4f}")

    print("\n2. conditioning on REAL answers (E3b, cached): distance of the averaged predicted distribution to the human cell")
    for n_history in (0, 3, 6):
        items = build(frame, scales, n_history, np.random.default_rng(n_history))
        by_cell = asyncio.run(oracle_arm(items, settings))
        for study in ("sffyb", "nj5dx"):
            values = [
                metrics.wasserstein_unit(np.mean(p, axis=0), human[k])
                for k, p in by_cell.items()
                if k.startswith(study + "|") and len(p) >= MIN_DRAWS and k in human
            ]
            print(f"   real history = {n_history}: {study} {np.mean(values):.4f} over {len(values)} cells")

    print(f"\n3. rank-preserving marginal correction on {CLEAN}")
    traces = {}
    for window in (0, 3):
        trace = []
        asyncio.run(simulate(people, scales, window, 0.2, settings, 0, trace=trace))
        traces[window] = [r for r in trace if r["person"][0] == CLEAN]
    target = defaultdict(list)
    for r in traces[0]:
        target[cell_key(r)].append(r["probs"])
    target = {k: np.mean(v, axis=0) for k, v in target.items()}

    by_cell = defaultdict(list)
    for index, r in enumerate(traces[3]):
        by_cell[cell_key(r)].append(index)
    rng = np.random.default_rng(0)
    corrected = [None] * len(traces[3])
    for key, indices in by_cell.items():
        mapped = quantile_map(np.array([traces[3][i]["choice"] for i in indices]), target[key], rng)
        for i, value in zip(indices, mapped, strict=True):
            corrected[i] = int(value)

    clean_people = {p: info for p, info in people.items() if p[0] == CLEAN}
    variants = {
        "window 0 (independent)": {i: r["choice"] for i, r in enumerate(traces[0])},
        "window 3 (raw memory)": {i: r["choice"] for i, r in enumerate(traces[3])},
        "window 3 + correction": dict(enumerate(corrected)),
    }
    rows = []
    for label, chosen in variants.items():
        source = traces[0] if label.startswith("window 0") else traces[3]
        drawn, choices = defaultdict(dict), defaultdict(list)
        for i, r in enumerate(source):
            n_options = len(r["probs"])
            drawn[r["person"]][r["task_num"]] = chosen[i] / (n_options - 1)
            choices[cell_key(r)].append(chosen[i])
        rows.append(
            {"variant": label, "distance to human": mean_distance(choices, human, CLEAN), **summarise(compare(clean_people, drawn, scales), CLEAN)}
        )
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n(B1's noise floor on sffyb: mean 0.2400, p95 0.2507)")


if __name__ == "__main__":
    main()
