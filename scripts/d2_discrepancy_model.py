"""D2: a three-parameter discrepancy model between a kernel's predicted condition effects and people's.

Research plan v3 §4.5, v3.1 (b). For every task with a shared scale, each condition's effect is measured
against the task's first condition, on the unit scale, for people (respondent-clustered standard errors,
shared control) and for a model (mean of its predicted distributions). The model

    human effect = beta * predicted effect + b_study + u_arm,   b ~ N(0, tau^2),  u ~ N(0, sigma^2)

is fitted by maximum likelihood with the human sampling covariance in the likelihood. tau^2 is the discrepancy
a study's effects share (the part that more simulated respondents cannot remove and that must be drawn once
per study in a population experiment); sigma^2 is arm-specific. Nothing here queries a model: every prediction
comes from cached runs.

Development data: the 91 seen studies of A3c (kernel and flagship, five simulated respondents per cell;
tasks with at least one reliable contrast, so a selected set - the 20-study development run at fifty per
cell is the unselected check). Calibration: split-half over studies. Retrospective evaluation: interval
coverage on the test split. The test outcomes informed the design of this model (they are on record), so
this is a retrospective check, not a prospective one; Epstein (D1) is the prospective one.

Usage:
    uv run python scripts/d2_discrepancy_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kite.eval import decision_value as dv
from kite.eval.runner import read_predictions
from kite.eval.scales import Scale

PREPARED = {
    "seen": Path("data/socsci210/prepared/a3c-seen"),
    "dev": Path("data/socsci210/prepared/dev"),
    "test": Path("data/socsci210/prepared/test"),
}
RUNS = {
    ("seen", "jev"): "results/a3c/20260922-155916-seen-jev",
    ("seen", "astra"): "results/a3c/20260922-155915-seen-astra",
    ("dev", "jev"): "results/20260921-221629-socsci210-dev-jev-p3-choice",
    ("test", "jev"): "results/20260921-012513-socsci210-test-jev-p3-choice-first5",
    ("test", "astra"): "results/20260922-130953-socsci210-test-codex-gpt-6-astra-high",
    ("test", "jev_full"): "results/20260921-012513-socsci210-test-jev-p3-choice",
}
OUT = Path("results/d2")
LEVELS = (0.5, 0.8, 0.9)
N_BOOT, N_SPLITS, N_MIX = 80, 20, 4000


def records(split: str, model: str, n_per_cell: int | None = None, seed: int = 0) -> pd.DataFrame:
    """One row per (study, task, arm != control): the model's and people's effect against the task's first condition."""
    prepared = PREPARED[split]
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    human = dv.human_cells(rows, scales)
    predicted = dv.predicted_means(read_predictions(RUNS[(split, model)]), n_per_cell=n_per_cell, seed=seed)
    out = []
    for b in dv.build_blocks(dv.comparable_tasks(scales), human, predicted):
        for a in range(1, len(b.human)):
            out.append({"study": b.study, "task": b.task, "arm": a, "x": b.predicted[a] - b.predicted[0], "y": b.human[a] - b.human[0],
                        "var_y": b.se[a] ** 2 + b.se[0] ** 2, "var_control": b.se[0] ** 2})  # fmt: skip
    return pd.DataFrame(out)


def study_blocks(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per study: (x, y, C) with C the human sampling covariance - shared controls within a task, nothing across tasks."""
    blocks = []
    for _, s in df.groupby("study", sort=True):
        n = len(s)
        C = np.diag(s["var_y"].to_numpy())
        task = s["task"].to_numpy()
        same = task[:, None] == task[None, :]
        C = C + np.where(same & ~np.eye(n, dtype=bool), s["var_control"].to_numpy()[:, None], 0.0)
        blocks.append((s["x"].to_numpy(), s["y"].to_numpy(), C))
    return blocks


def negative_loglik(params: np.ndarray, blocks) -> float:
    beta, tau2, sigma2 = params[0], np.exp(params[1]), np.exp(params[2])
    total = 0.0
    for x, y, C in blocks:
        n = len(y)
        cov = C + tau2 * np.ones((n, n)) + sigma2 * np.eye(n)
        r = y - beta * x
        sign, logdet = np.linalg.slogdet(cov)
        total += 0.5 * (logdet + r @ np.linalg.solve(cov, r) + n * np.log(2 * np.pi))
    return float(total)


STARTS = ((0.5, np.log(1e-4), np.log(1e-4)), (1.0, np.log(1e-3), np.log(1e-3)), (0.3, np.log(1e-2), np.log(1e-5)))


def fit(df: pd.DataFrame, starts=STARTS) -> dict:
    blocks = study_blocks(df)
    best = None
    for start in starts:
        res = minimize(
            negative_loglik, np.array(start), args=(blocks,), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000}
        )
        if best is None or res.fun < best.fun:
            best = res
    return {
        "beta": float(best.x[0]),
        "tau": float(np.sqrt(np.exp(best.x[1]))),
        "sigma": float(np.sqrt(np.exp(best.x[2]))),
        "nll": float(best.fun),
        "studies": df["study"].nunique(),
        "effects": len(df),
    }


def bootstrap_fits(df: pd.DataFrame, rng, point: dict, n_boot: int = N_BOOT) -> list[dict]:
    """Study-resampled refits, each started from the point estimate (one start: the surface is smooth in three parameters)."""
    studies = sorted(df["study"].unique())
    warm = ((point["beta"], np.log(point["tau"] ** 2 + 1e-12), np.log(point["sigma"] ** 2 + 1e-12)),)
    draws = []
    for i in range(n_boot):
        chosen = rng.choice(studies, len(studies), replace=True)
        parts = [df[df["study"] == s].assign(study=f"{s}#{k}") for k, s in enumerate(chosen)]  # a study drawn twice counts twice
        draws.append(fit(pd.concat(parts, ignore_index=True), starts=warm))
        if (i + 1) % 20 == 0:
            print(f"  bootstrap {i + 1}/{n_boot}", flush=True)
    return draws


def predictive_samples(df: pd.DataFrame, draws: list[dict], rng, include_sampling: bool = True) -> np.ndarray:
    """Samples from the mixture-over-parameter-draws predictive distribution of each effect; rows = effects."""
    idx = rng.integers(0, len(draws), size=(len(df), N_MIX))
    beta = np.array([d["beta"] for d in draws])[idx]
    var = np.array([d["tau"] ** 2 + d["sigma"] ** 2 for d in draws])[idx]
    if include_sampling:
        var = var + df["var_y"].to_numpy()[:, None]
    return beta * df["x"].to_numpy()[:, None] + rng.standard_normal((len(df), N_MIX)) * np.sqrt(var)


def coverage(df: pd.DataFrame, samples: np.ndarray) -> dict:
    out = {}
    y = df["y"].to_numpy()
    for level in LEVELS:
        lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2], axis=1)
        inside = (y >= lo) & (y <= hi)
        alpha = 1 - level
        score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
        per_study = pd.DataFrame({"study": df["study"], "inside": inside, "width": hi - lo, "score": score}).groupby("study").mean()
        out[f"{level:.0%}"] = {
            "coverage": float(per_study["inside"].mean()),
            "width": float(per_study["width"].mean()),
            "interval_score": float(per_study["score"].mean()),
        }
    return out


def split_half(df: pd.DataFrame, rng, n_splits: int = N_SPLITS) -> dict:
    studies = np.array(sorted(df["study"].unique()))
    rows = []
    for _ in range(n_splits):
        half = set(rng.choice(studies, len(studies) // 2, replace=False))
        train, held = df[df["study"].isin(half)], df[~df["study"].isin(half)]
        params = fit(train, starts=STARTS[:1])
        samples = predictive_samples(held, [params], rng)
        rows.append({k: v["coverage"] for k, v in coverage(held, samples).items()})
    return pd.DataFrame(rows).mean().to_dict()


def monte_carlo_versus_discrepancy(params: dict, rng) -> list[dict]:
    """On the test split: how the model's effects move with the number of simulated respondents, against tau and sigma."""
    out = []
    for n in (5, 10, 20, 50, 100, 200):
        reps = [records("test", "jev_full", n_per_cell=n, seed=s).set_index(["study", "task", "arm"])["x"] for s in range(6)]
        x = pd.concat(reps, axis=1)
        out.append({"n_per_cell": n, "monte_carlo_sd": float(x.std(axis=1, ddof=1).mean()), "effects": len(x)})
    shared = float(np.sqrt(params["tau"] ** 2 + params["sigma"] ** 2))
    return [{**row, "discrepancy_sd": shared} for row in out]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {}
    for model in ("jev", "astra"):
        seen = records("seen", model)
        params = fit(seen)
        print(f"{model}: point fit done ({params['studies']} studies)", flush=True)
        draws = bootstrap_fits(seen, rng, params)
        ci = {
            k: [float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5))] for k in ("beta", "tau", "sigma")
        }
        test = records("test", model)
        entry = {
            "fit_seen": params,
            "ci": ci,
            "split_half_coverage": split_half(seen, rng),
            "test": {"effects": len(test), "studies": test["study"].nunique()},
        }
        entry["test"]["coverage_with_parameter_uncertainty"] = coverage(test, predictive_samples(test, draws, rng))
        entry["test"]["coverage_point_parameters"] = coverage(test, predictive_samples(test, [params], rng))
        entry["test"]["coverage_no_discrepancy"] = coverage(test, predictive_samples(test, [{**params, "tau": 0.0, "sigma": 0.0}], rng))
        if model == "jev":
            dev = records("dev", "jev")
            entry["fit_dev_unselected"] = fit(dev)
            entry["test"]["coverage_dev_fit"] = coverage(test, predictive_samples(test, [entry["fit_dev_unselected"]], rng))
            entry["monte_carlo_versus_discrepancy"] = monte_carlo_versus_discrepancy(params, rng)
        report[model] = entry
    (OUT / "discrepancy_model.json").write_text(json.dumps(report, indent=1, default=float))

    for model, e in report.items():
        p, ci = e["fit_seen"], e["ci"]
        print(f"\n{model}: seen fit on {p['studies']} studies / {p['effects']} effects: beta {p['beta']:.3f} {[round(v, 3) for v in ci['beta']]}, "
              f"tau {p['tau']:.4f} {[round(v, 4) for v in ci['tau']]}, sigma {p['sigma']:.4f} {[round(v, 4) for v in ci['sigma']]}")  # fmt: skip
        if "fit_dev_unselected" in e:
            d = e["fit_dev_unselected"]
            print(f"  unselected 20-study dev fit: beta {d['beta']:.3f}, tau {d['tau']:.4f}, sigma {d['sigma']:.4f}")
        print(f"  split-half coverage within seen: {e['split_half_coverage']}")
        t = e["test"]
        print(f"  test ({t['studies']} studies / {t['effects']} effects), coverage / mean width / interval score:")
        for name in ("coverage_with_parameter_uncertainty", "coverage_point_parameters", "coverage_no_discrepancy", "coverage_dev_fit"):
            if name in t:
                print(
                    f"    {name:36} "
                    + "  ".join(f"{lvl}: {v['coverage']:.2f} / {v['width']:.3f} / {v['interval_score']:.3f}" for lvl, v in t[name].items())
                )
        if "monte_carlo_versus_discrepancy" in e:
            print("  Monte Carlo sd of the predicted effect vs simulated respondents per cell (test split):")
            for row in e["monte_carlo_versus_discrepancy"]:
                print(f"    n={row['n_per_cell']:>3}: MC sd {row['monte_carlo_sd']:.4f}   shared discrepancy sd {row['discrepancy_sd']:.4f}")
    print(f"\nwritten to {OUT / 'discrepancy_model.json'}")


if __name__ == "__main__":
    main()
