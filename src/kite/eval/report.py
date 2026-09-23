"""Turn run directories into one markdown report with figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# task -> (metric shown as the headline, whether higher is better)
PRIMARY = {"socsci210": ("distribution", False), "opinionqa": ("alignment", True)}
ZERO_COST_FLOOR = 1e-4  # USD per 1k predictions, so free runs still fit on a log axis
ENTROPY_RATIO_CLIP = 4.0  # the ratio is unbounded above; a few near-unanimous cells would flatten the axis


def load_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    run = {"name": root.name, "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")), "ledger": {}, "config": {}}
    if (root / "ledger.json").exists():
        run["ledger"] = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    if (root / "config.yaml").exists():
        run["config"] = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    return run


def _row(run: dict[str, Any]) -> dict[str, Any]:
    metrics, ledger, kernel = run["metrics"], run["ledger"], run["config"].get("kernel", {})
    metric, _ = PRIMARY[metrics["task"]]
    n = metrics.get("n_predictions", 0)
    usd = ledger.get("usd", 0.0)
    return {
        "run": run["name"],
        "task": metrics["task"],
        "kernel": kernel.get("backend", "-"),
        "model": kernel.get("model_id") or "-",
        "n": n,
        metric: metrics.get(metric),
        # two conventions that both used to be called "uniform accuracy"; they differ by about 0.09 on a
        # 7-level scale, which is enough to flip "did the model beat the floor". The columns say which.
        "accuracy_macro_median_rule": metrics.get("accuracy_macro"),
        "accuracy_macro_random_draw": metrics.get("accuracy_macro_random_draw"),
        "condition_r": metrics.get("condition_sensitivity_r"),
        "ece": metrics.get("ece"),
        # the median, not the mean: the ratio is unbounded above and near-unanimous cells dominate a mean
        "entropy_ratio_median": metrics.get("entropy_ratio_median"),
        "usd": usd,
        "usd_per_1k": usd / n * 1000 if n else 0.0,
        "p50_ms": ledger.get("latency_ms_p50"),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _table(rows: list[dict[str, Any]]) -> str:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join(lines)


def _pareto(runs: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        row = _row(run)
        metric, _ = PRIMARY[row["task"]]
        if row[metric] is None:
            continue
        axes.scatter(max(row["usd_per_1k"], ZERO_COST_FLOOR), row[metric])
        axes.annotate(row["run"], (max(row["usd_per_1k"], ZERO_COST_FLOOR), row[metric]), fontsize=7)
    axes.set_xscale("log")
    axes.set_xlabel("USD per 1,000 predictions (log; free runs drawn at the floor)")
    axes.set_ylabel("headline metric (distribution: lower is better; alignment: higher is better)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _reliability(runs: list[dict[str, Any]], path: Path) -> bool:
    with_bins = [run for run in runs if run["metrics"].get("reliability_bins")]
    if not with_bins:
        return False
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for run in with_bins:
        bins = run["metrics"]["reliability_bins"]
        axes.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", label=run["name"])
    axes.set_xlabel("confidence (top probability)")
    axes.set_ylabel("share of top answers that were right")
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _entropy(runs: list[dict[str, Any]], path: Path) -> bool:
    with_ratios = [run for run in runs if run["metrics"].get("entropy_ratios")]
    if not with_ratios:
        return False
    figure, axes = plt.subplots(figsize=(7, 4))
    for run in with_ratios:
        ratios = run["metrics"]["entropy_ratios"]
        axes.hist(np.clip(ratios, None, ENTROPY_RATIO_CLIP), bins=30, alpha=0.5, label=run["name"])
        axes.axvline(float(np.median(ratios)), linestyle=":", linewidth=1)
    axes.axvline(1.0, linestyle="--", color="gray")
    axes.set_xlabel(f"H(model) / H(human); below 1 means sharper than people (clipped at {ENTROPY_RATIO_CLIP}, dotted = median)")
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _groups(runs: list[dict[str, Any]], path: Path) -> bool:
    default = [run for run in runs if run["metrics"].get("mode") == "default" and run["metrics"].get("alignment_by_group")]
    if not default:
        return False
    groups = default[0]["metrics"]["alignment_by_group"]
    names = sorted(groups, key=groups.get)
    figure, axes = plt.subplots(figsize=(7, max(4, 0.18 * len(names))))
    axes.barh(names, [groups[name] for name in names])
    axes.set_xlabel(f"alignment of the un-steered model with each group ({default[0]['name']})")
    axes.tick_params(axis="y", labelsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def build_report(run_dirs: list[str | Path], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = [load_run(run_dir) for run_dir in run_dirs]
    sections = ["# Fidelity report", "", _table([_row(run) for run in runs]), ""]
    _pareto(runs, out / "figures" / "pareto.png")
    sections += ["## Fidelity versus cost", "", "![pareto](figures/pareto.png)", ""]
    for title, name, draw in [
        ("Reliability", "reliability", _reliability),
        ("Entropy ratio", "entropy_ratio", _entropy),
        ("Whose opinions", "group_alignment", _groups),
    ]:
        if draw(runs, out / "figures" / f"{name}.png"):
            sections += [f"## {title}", "", f"![{name}](figures/{name}.png)", ""]
    report = out / "report.md"
    report.write_text("\n".join(sections), encoding="utf-8")
    return report
