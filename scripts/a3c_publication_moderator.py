"""A3c: does GPT-6 Astra's edge over the kernel grow where a study's results were already public?

Implements configs/eval/publication_moderator.yaml exactly: per reliable human contrast (|z| >= 3), each
model's sign agreement with the full human sample; within a group of studies, the flagship's agreement
minus the kernel's, pooled over contrasts with each study weighted min(n, 20) / n; the interaction is that
advantage among results_public studies minus the advantage among the rest, tested against 10,000
permutations of the labels across studies, with a 95% interval from resampling studies within each group.

Refuses to run until the criteria file, the labels file and this script are committed, so the labels that
enter the test are the ones on record before any per-study result was computed.

Usage:
    uv run python scripts/a3c_publication_moderator.py --astra results/a3c/<seen astra run> --jev results/a3c/<seen jev run>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kite.eval import decision_value as dv
from kite.eval.runner import read_predictions
from kite.eval.scales import Scale

CRITERIA = Path("configs/eval/publication_moderator.yaml")
LABELS = Path("docs/data/tess_publication_status.csv")
PREPARED = {"seen": Path("data/socsci210/prepared/a3c-seen"), "unseen": Path("data/socsci210/prepared/test")}
UNSEEN_RUNS = {
    "astra": "results/20260922-130953-socsci210-test-codex-gpt-6-astra-high",
    "jev": "results/20260921-012513-socsci210-test-jev-p3-choice-first5",
}
CAP, N_PERM, N_BOOT, MIN_GROUP = 20, 10_000, 4_000, 15
SIZE_BINS = (0.0, 0.02, 0.05, 0.10, np.inf)


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def contrasts(prepared: Path, runs: dict[str, str]) -> pd.DataFrame:
    """One row per reliable contrast: study, task, each model's sign agreement, and the human difference's size."""
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    human = dv.human_cells(rows, scales)
    comparable = dv.comparable_tasks(scales)
    means = {m: dv.predicted_means(read_predictions(run)) for m, run in runs.items()}
    shared = set.intersection(*(set(v) for v in means.values()))  # both models on exactly the same cells
    blocks = {m: {(b.study, b.task): b for b in dv.build_blocks(comparable, human, {k: v[k] for k in shared})} for m, v in means.items()}
    records = []
    for key, b in sorted(blocks["jev"].items()):
        a, c, difference, z = dv._pairs(b.human, b.se)
        keep = np.abs(z) >= dv.RELIABLE_Z
        if not keep.any():
            continue
        agree = {m: dv._agreement(blocks[m][key].predicted, b.human, a, c)[keep] for m in runs}
        for i in range(int(keep.sum())):
            records.append((b.study, b.task, agree["astra"][i], agree["jev"][i], abs(difference[keep][i])))
    return pd.DataFrame(records, columns=["study", "task", "astra", "jev", "size"])


def per_study(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(diff=df["astra"] - df["jev"]).groupby("study")
    out = pd.DataFrame({"n": g.size(), "sum_diff": g["diff"].sum(), "sum_astra": g["astra"].sum(), "sum_jev": g["jev"].sum()})
    out["weight"] = np.minimum(out["n"], CAP) / out["n"]  # per contrast; a study's total weight is min(n, CAP)
    return out


def pooled(s: pd.DataFrame, column: str, member: np.ndarray, draws: np.ndarray | None = None) -> float:
    """Capped, contrast-pooled mean of `column` over the studies flagged in `member` (optionally with bootstrap counts)."""
    k = member.astype(float) * (1.0 if draws is None else draws)
    return float((k * s["weight"] * s[column]).sum() / (k * s["weight"] * s["n"]).sum())


def interaction(s: pd.DataFrame, label: np.ndarray, rng: np.random.Generator) -> dict:
    label = label.astype(bool)
    if min(label.sum(), (~label).sum()) < 2:  # nothing to compare
        return {"interaction": None, "studies": [int(label.sum()), int((~label).sum())], "underpowered": True}
    point = pooled(s, "sum_diff", label) - pooled(s, "sum_diff", ~label)
    null = np.array([(lambda p: pooled(s, "sum_diff", p) - pooled(s, "sum_diff", ~p))(rng.permutation(label)) for _ in range(N_PERM)])
    boot = []
    idx_pub, idx_priv = np.flatnonzero(label), np.flatnonzero(~label)
    for _ in range(N_BOOT):
        draws = np.zeros(len(s))
        np.add.at(draws, rng.choice(idx_pub, len(idx_pub)), 1)
        np.add.at(draws, rng.choice(idx_priv, len(idx_priv)), 1)
        boot.append(pooled(s, "sum_diff", label, draws) - pooled(s, "sum_diff", ~label, draws))
    groups = {}
    for name, member in (("public", label), ("not_public", ~label)):
        groups[name] = {
            "studies": int(member.sum()),
            "contrasts": int(s.loc[member, "n"].sum()),
            "astra_agreement": pooled(s, "sum_astra", member),
            "jev_agreement": pooled(s, "sum_jev", member),
            "advantage": pooled(s, "sum_diff", member),
        }
    return {
        "interaction": point,
        "p_one_sided": float((1 + (null >= point).sum()) / (1 + N_PERM)),
        "null_p95": float(np.percentile(null, 95)),
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "groups": groups,
        "underpowered": bool(min(label.sum(), (~label).sum()) < MIN_GROUP),
    }


def verdict(result: dict) -> str:
    if result["underpowered"] or result["interaction"] is None:
        return "DESCRIPTIVE ONLY (a group has fewer than 15 studies)"
    if result["interaction"] > 0 and result["p_one_sided"] < 0.05:
        return "RECALL SUPPORTED"
    return f"NOT DETECTED (the data allow a recall effect of at most {result['ci'][1]:+.3f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra", required=True, help="the seen-pool flagship run")
    parser.add_argument("--jev", required=True, help="the seen-pool kernel run")
    parser.add_argument("--out", type=Path, default=Path("results/a3c/analysis.json"))
    args = parser.parse_args()
    if not committed(str(CRITERIA), str(LABELS), "scripts/a3c_publication_moderator.py"):
        raise SystemExit("refusing to run: commit the criteria, the labels and this script first")

    labels = pd.read_csv(LABELS).set_index("study_id")
    rng = np.random.default_rng(0)
    report = {"labels_commit": subprocess.run(["git", "log", "-1", "--format=%h", "--", str(LABELS)], capture_output=True, text=True).stdout.strip()}
    parts = {"seen": contrasts(PREPARED["seen"], {"astra": args.astra, "jev": args.jev}), "unseen": contrasts(PREPARED["unseen"], UNSEEN_RUNS)}

    for part, df in parts.items():
        s = per_study(df).join(labels, how="inner")
        missing = sorted(set(df["study"]) - set(s.index))
        entry = {"studies": len(s), "contrasts": int(s["n"].sum()), "unlabelled_studies": missing}
        entry["primary"] = interaction(s, s["results_public"].to_numpy(), rng)
        entry["primary"]["verdict"] = verdict(entry["primary"])
        entry["published"] = interaction(s, s["published"].to_numpy(), rng)
        clear = s[~s["unclear"].astype(bool)]
        entry["excluding_unclear"] = interaction(clear, clear["results_public"].to_numpy(), rng)
        entry["by_size"] = {}
        for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:], strict=True):
            sub = per_study(df[(df["size"] >= lo) & (df["size"] < hi)]).join(labels, how="inner")
            if sub["results_public"].nunique() == 2:
                entry["by_size"][f"{lo:.2f}-{hi:.2f}"] = interaction(sub, sub["results_public"].to_numpy(), rng)
        era, public = s["field_start_year"].fillna(0).to_numpy() >= 2015, s["results_public"].to_numpy().astype(bool)
        entry["by_era"] = {}
        for name, newer in (("before_2015", False), ("2015_on", True)):
            cells = {group: (era == newer) & (public == flag) for group, flag in (("public", True), ("not_public", False))}
            entry["by_era"][name] = {group: pooled(s, "sum_diff", member) for group, member in cells.items() if member.any()}
        pub = s[s["published"].astype(bool) & s["citations"].notna()]
        if len(pub) >= 5:
            advantage = (pub["sum_diff"] / pub["n"]).to_numpy()
            logc = np.log1p(pub["citations"].to_numpy(dtype=float))
            rho = float(spearmanr(advantage, logc).statistic)
            null = np.array([spearmanr(advantage, rng.permutation(logc)).statistic for _ in range(N_PERM)])
            entry["citations"] = {"studies": len(pub), "spearman": rho, "p_one_sided": float((1 + (null >= rho).sum()) / (1 + N_PERM))}
        report[part] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    for part in ("seen", "unseen"):
        e, p = report[part], report[part]["primary"]
        if p["interaction"] is None:
            print(f"\n{part}: groups too small to compare {p['studies']}")
            continue
        role = "primary" if part == "seen" else "secondary"
        print(f"\n{part} ({role}): {e['studies']} studies, {e['contrasts']} reliable contrasts; unlabelled {e['unlabelled_studies']}")
        for name, g in p["groups"].items():
            shares = f"Astra {g['astra_agreement']:.3f}  Jev {g['jev_agreement']:.3f}  advantage {g['advantage']:+.3f}"
            print(f"  {name:10} {g['studies']:3} studies {g['contrasts']:5} contrasts  {shares}")
        print(f"  interaction {p['interaction']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  p {p['p_one_sided']:.4f}  -> {p['verdict']}")


if __name__ == "__main__":
    main()
