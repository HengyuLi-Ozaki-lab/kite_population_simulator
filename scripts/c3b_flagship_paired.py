"""C3b (exploratory): the flagship's intervention effect against the kernel's, on the same respondents.

Written after the gate verdicts of scripts/c3_llm_arechar.py were on record; nothing here is a gate or changes
one. It asks three things the gate does not:
- how far above its null each model's pooled effect lies (one-sided permutation p from 5,000 shuffles);
- whether the flagship's effect is larger than the kernel's. Both predicted the same respondents and headlines,
  so the difference of their effects is the effect computed on the per-respondent differences of their
  predictions: it gets the gate's own within-country condition shuffle as its null, and an interval from
  resampling respondents within country and condition;
- which channel the effect runs through: the change in sharing of true and of false headlines, next to people's.

Usage:
    uv run python scripts/c3b_flagship_paired.py --run results/c3-flagship/<run>
"""

from __future__ import annotations

import argparse
import json
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
N_PERM, N_BOOT = 5000, 2000


def expected(meta: dict, probs_by_id: dict) -> pd.DataFrame:
    rows = []
    for item_id, probs in probs_by_id.items():
        p = np.asarray(probs, dtype=float)
        rows.append({**meta[item_id], "expected": float(np.dot(p / p.sum(), np.arange(1, 7)))})
    return pd.DataFrame(rows)


def person_sums(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per respondent: the sum and count of `column` over true and over false headlines."""
    sums = frame.groupby(["country", "id", "condition", "true"])[column].agg(["sum", "count"]).unstack("true", fill_value=0)
    sums.columns = [f"{stat}_{'T' if truth else 'F'}" for stat, truth in sums.columns]
    return sums.reset_index()[["country", "id", "condition", "sum_T", "count_T", "sum_F", "count_F"]]


def channels(a: np.ndarray, treated: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Treatment minus share-only in the mean rating of true and of false headlines (rows pooled, as in the gate)."""
    t, c = (w * treated) @ a, (w * ~treated) @ a
    return t[0] / t[1] - c[0] / c[1], t[2] / t[3] - c[2] / c[3]


def effect(a: np.ndarray, treated: np.ndarray, w: np.ndarray) -> float:
    true, false = channels(a, treated, w)
    return true - false


def pooled(sums: pd.DataFrame, treatment: str, rng) -> dict:
    """Equal weight per country, as in the gate: point, per country, permutation null and p, respondent bootstrap."""
    countries = sorted(sums["country"].unique())
    per_country, parts, null, boot = {}, [], np.zeros((N_PERM, len(countries))), np.zeros((N_BOOT, len(countries)))
    for j, country in enumerate(countries):
        s = sums[(sums["country"] == country) & sums["condition"].isin([treatment, "share_only"])]
        a = s[["sum_T", "count_T", "sum_F", "count_F"]].to_numpy(dtype=float)
        treated, ones = (s["condition"] == treatment).to_numpy(), np.ones(len(s))
        per_country[country] = effect(a, treated, ones)
        parts.append(channels(a, treated, ones))
        for i in range(N_PERM):
            null[i, j] = effect(a, rng.permutation(treated), ones)
        groups = [np.flatnonzero(treated), np.flatnonzero(~treated)]
        for i in range(N_BOOT):
            w = np.zeros(len(s))
            for members in groups:
                w[members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)))
            boot[i, j] = effect(a, treated, w)
    point, pooled_null, pooled_boot = float(np.mean(list(per_country.values()))), null.mean(axis=1), boot.mean(axis=1)
    return {
        "pooled": point,
        "p_one_sided": float((1 + (pooled_null >= point).sum()) / (1 + N_PERM)),
        "null_p95": float(np.percentile(pooled_null, 95)),
        "ci": [float(np.percentile(pooled_boot, 2.5)), float(np.percentile(pooled_boot, 97.5))],
        "true_headlines": float(np.mean([p[0] for p in parts])),
        "false_headlines": float(np.mean([p[1] for p in parts])),
        "per_country": per_country,
    }


def levels(frame: pd.DataFrame, column: str) -> dict:
    """Mean rating by condition and headline truth, rows pooled within a country and countries weighted equally."""
    means = frame.groupby(["condition", "true", "country"])[column].mean().groupby(["condition", "true"]).mean()
    out: dict = {}
    for (condition, truth), value in means.items():
        out.setdefault(condition, {})["true" if truth else "false"] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="a results/c3-flagship run directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())["arechar"]

    task = arechar.ArecharTask(CSV, QSF, "held_out", max_respondents=cfg["respondents_per_country_condition"], parts=[cfg["part"]])
    meta = {item.item_id: item.meta for item in task.items()}
    raw = json.loads((args.run / "raw.json").read_text())
    ids = [i for i in raw if i in meta]
    flagship = expected(meta, {i: raw[i] for i in ids})
    kernel = expected(meta, {p.item_id: p.probs for p in read_predictions(KERNEL_RUN) if p.item_id in set(ids)})
    assert len(kernel) == len(flagship), "the kernel must cover every item the flagship predicted"
    human = arechar.load_ratings(CSV, "held_out")
    human = human[human["part"] == cfg["part"]].assign(true=lambda d: d["item"].map(arechar.is_true))

    sums = {"flagship": person_sums(flagship, "expected"), "kernel": person_sums(kernel, "expected"), "people": person_sums(human, "rating")}
    keys = ["country", "id", "condition"]
    both = sums["flagship"].merge(sums["kernel"], on=keys, suffixes=("", "_k"), validate="one_to_one")
    assert (both["count_T"] == both["count_T_k"]).all() and (both["count_F"] == both["count_F_k"]).all()
    sums["flagship_minus_kernel"] = both.assign(sum_T=both["sum_T"] - both["sum_T_k"], sum_F=both["sum_F"] - both["sum_F_k"])[sums["kernel"].columns]

    rng = np.random.default_rng(0)
    report = {}
    for treatment in [t for t in ("prompt", "tips") if t in set(flagship["condition"])]:
        report[treatment] = {name: pooled(s, treatment, rng) for name, s in sums.items()}
        diff = report[treatment]["flagship_minus_kernel"]["per_country"]
        report[treatment]["flagship_minus_kernel"]["countries_positive"] = f"{sum(v > 0 for v in diff.values())} of {len(diff)}"
    sharing = human[human["condition"].isin(set(flagship["condition"]))]
    report["levels"] = {"flagship": levels(flagship, "expected"), "kernel": levels(kernel, "expected"), "people": levels(sharing, "rating")}
    (args.run / "paired.json").write_text(json.dumps(report, indent=1))

    for treatment, r in ((t, r) for t, r in report.items() if t != "levels"):
        print(f"\n{treatment} (sharing-discernment effect, equal weight per country; people: all respondents of the part)")
        print(f"  {'':22}{'effect':>8}{'95% CI':>18}{'p':>8}{'d true':>9}{'d false':>9}")
        for name, e in r.items():
            ci = f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
            p = "-" if name == "people" else f"{e['p_one_sided']:.4f}"
            print(f"  {name:22}{e['pooled']:>+8.3f}{ci:>18}{p:>8}{e['true_headlines']:>+9.3f}{e['false_headlines']:>+9.3f}")
        print(f"  flagship above kernel in {r['flagship_minus_kernel']['countries_positive']} countries")


if __name__ == "__main__":
    main()
