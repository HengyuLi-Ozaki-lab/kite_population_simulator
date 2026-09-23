"""D1 addendum A2: headline-level treatment effects on Epstein, under configs/eval/epstein_headline_criteria.yaml.

    PYTHONPATH=scripts uv run python scripts/d1_headlines.py

Refuses to run while the criteria file or this script is uncommitted. Rebuilds the hybrid from the locked inputs and
checks its wave x arm table against results/d1/policies.json before any headline-level number is computed. Writes
results/d1/headlines.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
import d1_analysis as d1  # noqa: E402
from d1_epstein import cells_from_control  # noqa: E402

from kite.eval import epstein  # noqa: E402

CRITERIA = Path("configs/eval/epstein_headline_criteria.yaml")
POLICIES = Path("results/d1/policies.json")
OUT = Path("results/d1/headlines.json")
SYSTEMS = ("kernel", "hybrid", "flagship_audit")
PREDICTORS = (*SYSTEMS, "baseline")
ITEMS = list(range(1, 21))


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def per_headline(df: pd.DataFrame, column: str = "p_yes") -> pd.Series:
    return df.groupby(["wave", "arm", "item_num"])[column].mean()


def effects_from(means: pd.Series, arms: list[tuple[int, str]]) -> np.ndarray:
    return np.array([means[(w, a, i)] - means[(w, "control", i)] for w, a in arms for i in ITEMS])


def centre(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        out[m] = values[m] - values[m].mean()
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def rating_matrices(ratings: pd.DataFrame) -> dict[tuple[int, str], np.ndarray]:
    """(wave, arm) -> participants x 20 headlines matrix of binary sharing answers."""
    share = ratings[ratings["arm"] != "accuracy_only"]
    out = {}
    for (w, a), g in share.groupby(["wave", "arm"]):
        out[(int(w), str(a))] = (
            g.pivot_table(index="id", columns="item_num", values="rating", aggfunc="mean").reindex(columns=ITEMS).to_numpy(dtype=float)
        )
    return out


def human_effects(mats: dict, arms: list[tuple[int, str]], rng=None, half: dict | None = None) -> np.ndarray:
    """Per (arm, headline) share-rate effects; people resampled within wave x arm when rng is given, or restricted to a half."""
    means = {}
    for key, m in mats.items():
        rows = m
        if half is not None:
            rows = m[half[key]]
        if rng is not None:
            rows = rows[rng.integers(0, len(rows), len(rows))]
        means[key] = np.nanmean(rows, axis=0)
    return np.concatenate([means[(w, a)] - means[(w, "control")] for w, a in arms])


def main() -> None:
    if not committed(str(CRITERIA), "scripts/d1_headlines.py"):
        raise SystemExit("refusing to run: commit the headline criteria and this script first")
    cfg = yaml.safe_load(CRITERIA.read_text())
    locked = json.loads(POLICIES.read_text())
    jev = d1.frame(Path(locked["inputs"]["jev"]))
    jev = jev[jev["kind"] == "share"]
    astra = d1.frame(Path(locked["inputs"]["astra"]))
    cells, _ = cells_from_control(jev)
    corrected, _ = d1.hybrid(jev, astra, cells, 12)
    table = d1.discernment(corrected)
    for key, row in locked["tables"]["hybrid"].items():
        w, a = key.split(":")
        assert abs(table.loc[(int(w), a), "discernment"] - row["discernment"]) < 1e-9, "the rebuilt hybrid differs from the locked table"
    arms = [(int(k.split(":")[0]), k.split(":")[1]) for k in locked["tables"]["hybrid"] if not k.endswith(":control")]
    arms = sorted(arms)
    app = jev[jev["pool"] == "application"]
    model = {
        "kernel": effects_from(per_headline(app), arms),
        "hybrid": effects_from(per_headline(corrected), arms),
        "flagship_audit": effects_from(per_headline(astra[~astra["anchor"]]), arms),
    }
    control = app[app["arm"] == "control"].groupby(["wave", "item_num"])["p_yes"].mean()
    model["baseline"] = np.array([control[(w, i)] for w, _ in arms for i in ITEMS])
    arm_group = np.array([f"{w}:{a}" for w, a in arms for _ in ITEMS])
    veracity = np.array([i > 10 for _ in arms for i in ITEMS])
    arm_ver_group = np.array([f"{g}:{v}" for g, v in zip(arm_group, veracity, strict=True)])

    ratings = epstein.load_ratings()
    mats = rating_matrices(ratings)
    human = human_effects(mats, arms)
    c_h, cv_h = centre(human, arm_group), centre(human, arm_ver_group)
    c_m = {p: centre(v, arm_group) for p, v in model.items()}
    cv_m = {p: centre(v, arm_ver_group) for p, v in model.items()}

    stats = {
        p: {
            "within_arm_pearson": pearson(c_m[p], c_h),
            "within_arm_spearman": float(spearmanr(c_m[p], c_h)[0]),
            "within_arm_sign_agreement": float(np.mean(np.sign(c_m[p]) == np.sign(c_h))),
            "within_arm_veracity_pearson": pearson(cv_m[p], cv_h),
            "false_only_within_arm_pearson": pearson(cv_m[p][~veracity], cv_h[~veracity]),
            "true_only_within_arm_pearson": pearson(cv_m[p][veracity], cv_h[veracity]),
        }
        for p in PREDICTORS
    }
    rng = np.random.default_rng(cfg["analysis"]["seed"])
    boot = {p: {"arm": [], "arm_ver": []} for p in PREDICTORS}
    for _ in range(cfg["analysis"]["n_boot"]):
        h = human_effects(mats, arms, rng)
        ch, cvh = centre(h, arm_group), centre(h, arm_ver_group)
        for p in PREDICTORS:
            boot[p]["arm"].append(pearson(c_m[p], ch))
            boot[p]["arm_ver"].append(pearson(cv_m[p], cvh))
    boot = {p: {k: np.array(v) for k, v in d.items()} for p, d in boot.items()}

    def ci(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    intervals = {p: {"within_arm": ci(boot[p]["arm"]), "within_arm_veracity": ci(boot[p]["arm_ver"])} for p in PREDICTORS}
    differences = {
        f"hybrid-{q}": {
            "within_arm": {
                "point": stats["hybrid"]["within_arm_pearson"] - stats[q]["within_arm_pearson"],
                "ci": ci(boot["hybrid"]["arm"] - boot[q]["arm"]),
            },
            "within_arm_veracity": {
                "point": stats["hybrid"]["within_arm_veracity_pearson"] - stats[q]["within_arm_veracity_pearson"],
                "ci": ci(boot["hybrid"]["arm_ver"] - boot[q]["arm_ver"]),
            },
        }
        for q in ("kernel", "flagship_audit", "baseline")
    }
    halves = []
    for _ in range(cfg["analysis"]["n_splits"]):
        split = {key: rng.integers(0, 2, len(m)).astype(bool) for key, m in mats.items()}
        a = centre(human_effects(mats, arms, half=split), arm_group)
        b = centre(human_effects(mats, arms, half={k: ~v for k, v in split.items()}), arm_group)
        r = pearson(a, b)
        halves.append(2 * r / (1 + r))
    reliability = float(np.mean(halves))
    primary = stats["hybrid"]["within_arm_pearson"]
    lo, hi = intervals["hybrid"]["within_arm"]
    report = {
        "criteria": str(CRITERIA),
        "units": len(human),
        "arms": [f"{w}:{a}" for w, a in arms],
        "stats": stats,
        "ci95": intervals,
        "differences": differences,
        "human_split_half_reliability_within_arm": reliability,
        "attenuation_corrected_within_arm": {p: stats[p]["within_arm_pearson"] / np.sqrt(reliability) for p in PREDICTORS},
        "primary": {"statistic": primary, "ci95": [lo, hi], "verdict": "POSITIVE" if primary > 0 and lo > 0 else "NOT SHOWN"},
    }
    OUT.write_text(json.dumps(report, indent=1))
    print(f"{len(human)} units; human split-half reliability (within arm, Spearman-Brown) {reliability:.3f}")
    print(pd.DataFrame(stats).T.round(3).to_string())
    r3 = lambda values: [round(v, 3) for v in values]  # noqa: E731
    for p in PREDICTORS:
        s, i = stats[p], intervals[p]
        line = f"  {p:15} within-arm r {s['within_arm_pearson']:+.3f} {r3(i['within_arm'])};"
        print(line + f"  within arm x veracity {s['within_arm_veracity_pearson']:+.3f} {r3(i['within_arm_veracity'])}")
    for k, d in differences.items():
        line = f"  {k:24} within-arm {d['within_arm']['point']:+.3f} {r3(d['within_arm']['ci'])};"
        print(line + f"  arm x veracity {d['within_arm_veracity']['point']:+.3f} {r3(d['within_arm_veracity']['ci'])}")
    print("PRIMARY:", json.dumps(report["primary"]))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
