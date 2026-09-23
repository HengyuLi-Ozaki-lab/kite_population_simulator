"""A3b: the kernel, the cheapest GPT-5.6 tier and a flagship, on the same SocSci210 test items.

Implements the comparison declared in configs/eval/flagship_baseline.yaml: the frozen decision-value measures
for each model, paired study-bootstrap differences, and the common-blind-spot statistic - on the reliable
contrasts, the share all three models get wrong against the product of their error rates, which is what
that share would be if their errors were independent.

Usage:
    uv run python scripts/a3b_three_models.py --flagship results/<flagship run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kite.eval import decision_value as dv
from kite.eval.runner import read_predictions
from kite.eval.scales import Scale

RUNS = {
    "jev": "results/20260921-012513-socsci210-test-jev-p3-choice-first5",
    "luna": "results/20260921-224518-socsci210-test-codex-gpt-5.6-luna",
}
N_BOOT = 4000
MARGINAL = (
    "distribution",
    "distribution_published_convention",
    "accuracy_macro",
    "ece",
    "entropy_ratio_median",
    "condition_sensitivity_r",
    "condition_sensitivity_slope",
    "condition_spread_ratio",
)


def centred_r(blocks, weights) -> float:
    p = np.concatenate([b.predicted - b.predicted.mean() for b in blocks])
    h = np.concatenate([b.human - b.human.mean() for b in blocks])
    w = np.concatenate([np.full(len(b.predicted), weights[i]) for i, b in enumerate(blocks)])
    return float((w * p * h).sum() / np.sqrt((w * p * p).sum() * (w * h * h).sum()))


def reliable_agreement(blocks) -> tuple[np.ndarray, np.ndarray]:
    """Per reliable contrast: 1 if the model orders it like the full human sample; and the contrast's study."""
    hits, studies = [], []
    for b in blocks:
        a, c, _, z = dv._pairs(b.human, b.se)
        keep = np.abs(z) >= dv.RELIABLE_Z
        agreement = dv._agreement(b.predicted, b.human, a, c)[keep]
        hits.append(agreement)
        studies.extend([b.study] * int(keep.sum()))
    return np.concatenate(hits), np.array(studies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagship", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/a3b/three_models.json"))
    args = parser.parse_args()
    runs = {**RUNS, "astra": str(args.flagship)}

    prepared = Path("data/socsci210/prepared/test")
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    human = dv.human_cells(rows, scales)
    comparable = dv.comparable_tasks(scales)
    blocks = {k: dv.build_blocks(comparable, human, dv.predicted_means(read_predictions(v))) for k, v in runs.items()}
    keys = {k: [(b.study, b.task) for b in v] for k, v in blocks.items()}
    assert keys["jev"] == keys["luna"] == keys["astra"], "the three runs must cover the same tasks"
    parts = {k: dv.evaluate(v) for k, v in blocks.items()}
    studies, member = np.unique(parts["jev"].study, return_inverse=True)
    ones = np.ones(len(member))

    marginal = {}
    for k, v in runs.items():
        m = json.loads((Path(v) / "metrics.json").read_text())
        marginal[k] = {key: m.get(key) for key in MARGINAL}
    decision = {k: {**parts[k].summary(), "r_within_task": centred_r(blocks[k], ones)} for k in runs}

    rng = np.random.default_rng(0)
    diffs = {pair: {"sign_accuracy": [], "captured_gain": [], "r_within_task": []} for pair in (("astra", "jev"), ("astra", "luna"), ("luna", "jev"))}
    agree = {k: reliable_agreement(v) for k, v in blocks.items()}
    contrast_study = agree["jev"][1]
    wrong = {k: agree[k][0] == 0 for k in runs}
    valid = np.all([np.isfinite(agree[k][0]) for k in runs], axis=0)
    all_wrong_boot, independent_boot = [], []
    for _ in range(N_BOOT):
        draw = np.bincount(rng.integers(0, len(studies), len(studies)), minlength=len(studies))
        w = draw[member].astype(float)
        summaries = {k: parts[k].summary(w) for k in runs}
        for a, b in diffs:
            for measure in ("sign_accuracy", "captured_gain"):
                diffs[(a, b)][measure].append(summaries[a][measure] - summaries[b][measure])
            diffs[(a, b)]["r_within_task"].append(centred_r(blocks[a], w) - centred_r(blocks[b], w))
        cw = draw[np.searchsorted(studies, contrast_study)].astype(float) * valid
        if cw.sum():
            rates = {k: float((cw * wrong[k]).sum() / cw.sum()) for k in runs}
            all_wrong_boot.append(float((cw * wrong["jev"] * wrong["luna"] * wrong["astra"]).sum() / cw.sum()))
            independent_boot.append(rates["jev"] * rates["luna"] * rates["astra"])

    n = int(valid.sum())
    error_rate = {k: float(wrong[k][valid].mean()) for k in runs}
    all_wrong = float((wrong["jev"] & wrong["luna"] & wrong["astra"])[valid].mean())
    none_wrong = float((~wrong["jev"] & ~wrong["luna"] & ~wrong["astra"])[valid].mean())
    report = {
        "marginal": marginal,
        "decision": decision,
        "paired_differences": {
            f"{a}-{b}": {
                m: {
                    "point": decision[a][m] - decision[b][m],
                    "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                    "share_above_0": float(np.nanmean(np.asarray(v) > 0)),
                }
                for m, v in d.items()
            }
            for (a, b), d in diffs.items()
        },  # fmt: skip
        "common_blind_spot": {
            "reliable_contrasts": n,
            "error_rate": error_rate,
            "all_three_wrong": all_wrong,
            "if_independent": error_rate["jev"] * error_rate["luna"] * error_rate["astra"],
            "all_three_right": none_wrong,
            "all_wrong_ci": [float(np.percentile(all_wrong_boot, 2.5)), float(np.percentile(all_wrong_boot, 97.5))],
            "ratio_to_independent_ci": [
                float(np.percentile(np.divide(all_wrong_boot, independent_boot), 2.5)),
                float(np.percentile(np.divide(all_wrong_boot, independent_boot), 97.5)),
            ],
        },  # fmt: skip
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    print(f"{'':8}{'dist':>8}{'dist_pub':>10}{'acc':>8}{'ECE':>8}{'H med':>8}{'r_cond':>8}{'spread':>8}{'sign':>8}{'gain':>8}")
    for k in runs:
        m, d = marginal[k], decision[k]
        cells = [f(m["distribution"], 4), f(m["distribution_published_convention"], 4), f(m["accuracy_macro"]), f(m["ece"]),
                 f(m["entropy_ratio_median"]), f(m["condition_sensitivity_r"]), f(m["condition_spread_ratio"], 2),
                 f(d["sign_accuracy"]), f(d["captured_gain"])]  # fmt: skip
        widths = (8, 10, 8, 8, 8, 8, 8, 8, 8)
        print(f"{k:8}" + "".join(f"{c:>{w}}" for c, w in zip(cells, widths, strict=True)))
    print("\npaired differences (study bootstrap, 95% CI):")
    for pair, d in report["paired_differences"].items():
        print("  " + pair + ": " + "; ".join(f"{m} {v['point']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for m, v in d.items()))
    c = report["common_blind_spot"]
    print(f"\ncommon blind spot on {n} reliable contrasts: error rates {', '.join(f'{k} {v:.3f}' for k, v in c['error_rate'].items())}")
    print(f"  all three wrong {c['all_three_wrong']:.3f} {[round(x, 3) for x in c['all_wrong_ci']]} vs {c['if_independent']:.3f} if independent "
          f"(ratio CI {[round(x, 2) for x in c['ratio_to_independent_ci']]}); all three right {c['all_three_right']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
