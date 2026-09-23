"""A6: how many human respondents is one simulated cell distribution worth?

Two readings of the question, both at the level of a cell (study x condition x task), both on the
stated-scale Wasserstein distance to the full human cell distribution.

  equivalent sample   the n at which the empirical distribution of n randomly drawn respondents is,
                      on average, as close to the full cell as the simulation is

  value as a prior    treat the simulated distribution as a Dirichlet prior of strength kappa and
                      combine it with n respondents: (kappa * p_sim + counts) / (kappa + n). How much
                      closer does that get than the n respondents alone, and how many respondents
                      would it take to match it without the prior?

The second reading needs a control, because *any* smoothing helps a small sample: the same is done
with a uniform prior, and what is reported for the simulation is its value over and above that.

Protocol: kappa is chosen per n on the development run and applied unchanged to the test run. There
is no pass criterion; this is a measurement. It is a secondary analysis of the test predictions that
were made for G1.

Usage:
    uv run python scripts/a6_kappa.py --dev results/<dev run> --test results/<test run>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from kite.eval import metrics
from kite.eval.runner import read_predictions

SIZES = (1, 2, 3, 5, 8, 12, 20, 30, 50, 100)
PRIOR_SIZES = (5, 10, 20, 50)
KAPPAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0)
DRAWS = 200


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wasserstein on the unit-spaced grid, vectorised over leading axes; equals metrics.wasserstein_unit."""
    return np.abs(np.cumsum(a - b, axis=-1))[..., :-1].sum(-1) / (a.shape[-1] - 1)


def load(run: Path, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(simulated distribution, human distribution) per cell that has both."""
    cells = json.loads((Path("data/socsci210/prepared") / split / "cells.json").read_text())
    by_cell = defaultdict(list)
    for p in read_predictions(run):
        by_cell[f"{p.meta['study_id']}|{p.meta['condition_num']}|{p.meta['task_num']}"].append(p.probs)
    out = []
    for key, probs in by_cell.items():
        human = np.asarray(cells[key], dtype=float)
        if human.sum() > 0 and len(human) >= 2:
            out.append((metrics.as_dist(np.mean(probs, axis=0)), human / human.sum()))
    return out


def sample_curves(pairs, rng) -> dict:
    """Per cell: the simulation's distance, and the mean distance of n drawn respondents for each n."""
    simulated = np.array([float(distance(sim, human)) for sim, human in pairs])
    empirical = np.zeros((len(pairs), len(SIZES)))
    for i, (_, human) in enumerate(pairs):
        for j, n in enumerate(SIZES):
            empirical[i, j] = distance(rng.multinomial(n, human, size=DRAWS) / n, human).mean()
    return {"simulated": simulated, "empirical": empirical}


def equivalent_n(simulated: float, curve: np.ndarray) -> float:
    """The n at which drawn respondents match the simulation, interpolated on log n; clipped to the grid's ends."""
    if simulated >= curve[0]:
        return float(SIZES[0]) * 0.5  # worse than a single respondent
    if simulated <= curve[-1]:
        return float(SIZES[-1])
    return float(np.exp(np.interp(-simulated, -curve, np.log(SIZES))))


def prior_table(pairs, rng) -> pd.DataFrame:
    """Mean distance of (kappa * prior + counts) / (kappa + n), for the simulation and for a uniform prior."""
    rows = []
    for n in PRIOR_SIZES:
        totals = {("simulation", k): [] for k in KAPPAS} | {("uniform", k): [] for k in KAPPAS}
        for sim, human in pairs:
            counts = rng.multinomial(n, human, size=DRAWS)
            for name, prior in (("simulation", sim), ("uniform", np.full(len(human), 1 / len(human)))):
                for kappa in KAPPAS:
                    totals[(name, kappa)].append(distance((kappa * prior + counts) / (kappa + n), human).mean())
        for (name, kappa), values in totals.items():
            rows.append({"n": n, "prior": name, "kappa": kappa, "distance": float(np.mean(values))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/a6"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    check = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])
    assert abs(float(distance(*check)) - metrics.wasserstein_unit(*check)) < 1e-12

    rng = np.random.default_rng(0)
    dev, test = load(args.dev, "dev"), load(args.test, "test")

    print("1. equivalent sample size")
    summary = {}
    for name, pairs in (("dev", dev), ("test", test)):
        curves = sample_curves(pairs, rng)
        per_cell = np.array([equivalent_n(s, c) for s, c in zip(curves["simulated"], curves["empirical"], strict=True)])
        pooled = equivalent_n(float(curves["simulated"].mean()), curves["empirical"].mean(axis=0))
        summary[name] = {
            "cells": len(pairs),
            "simulated_distance": float(curves["simulated"].mean()),
            "pooled_equivalent_n": pooled,
            "median_equivalent_n": float(np.median(per_cell)),
            "p25": float(np.percentile(per_cell, 25)),
            "p75": float(np.percentile(per_cell, 75)),
            "share_worse_than_one_respondent": float((per_cell < 1).mean()),
            "share_at_least_20": float((per_cell >= 20).mean()),
        }
        print(f"   {name}: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summary[name].items()})}")
        curve = ", ".join(f"n={n}: {d:.3f}" for n, d in zip(SIZES, curves["empirical"].mean(axis=0), strict=True))
        print(f"        mean distance of n drawn respondents -> {curve}")

    print("\n2. value as a prior (kappa chosen on dev, applied to test)")
    dev_table, test_table = prior_table(dev, rng), prior_table(test, rng)
    rows = []
    for n in PRIOR_SIZES:
        row = {"n": n, "respondents_alone": float(test_table.query("n == @n and prior == 'uniform' and kappa == 0")["distance"].iloc[0])}
        for prior in ("uniform", "simulation"):
            chosen = dev_table.query("n == @n and prior == @prior").sort_values("distance").iloc[0]["kappa"]
            row[f"kappa_{prior}"] = chosen
            row[f"with_{prior}_prior"] = float(test_table.query("n == @n and prior == @prior and kappa == @chosen")["distance"].iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    # how many respondents alone would match the simulation-augmented estimate, from the test split's own curve
    alone = sample_curves(test, rng)["empirical"].mean(axis=0)
    table["respondents_needed_without_it"] = [float(np.exp(np.interp(-d, -alone, np.log(SIZES)))) for d in table["with_simulation_prior"]]
    table["needed_without_uniform"] = [float(np.exp(np.interp(-d, -alone, np.log(SIZES)))) for d in table["with_uniform_prior"]]
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    (args.out / "kappa.json").write_text(json.dumps({"equivalent": summary, "prior": table.to_dict("records")}, indent=1))
    print(f"\nwritten to {args.out / 'kappa.json'}")


if __name__ == "__main__":
    main()
