"""D1 analysis: lock the model-side policies first, open the human outcomes second.

    PYTHONPATH=scripts uv run python scripts/d1_analysis.py predict --jev <dir> --astra <dir>   # -> results/d1/policies.json, no rating read
    PYTHONPATH=scripts uv run python scripts/d1_analysis.py score --policies results/d1/policies.json  # opens the ratings (gated), one pass

predict: per wave and arm, the predicted sharing discernment (mean P(yes) on true cards minus false cards over the
application pool) for three systems - the kernel alone, the hybrid (kernel tilted so that each cell's treated
mean equals the kernel's control mean plus the flagship's paired anchor effect), and the flagship directly on the
audit pool - and each system's chosen arm per wave (highest predicted discernment, control included). The file
carries the SHA-256 of both prediction files, so the locked policies can be checked against the runs later.

score: people's discernment per wave and arm from the ratings; the primary endpoint is the mean within-wave policy
value of the hybrid minus the kernel, with participant resampling within wave x arm and a permutation null that
shuffles the model's arm scores within waves; the secondary endpoints are the effect MAE, the two channels, the
flagship's policy value, the approximation error, the split-half human pilot, and the coverage of the D2
discrepancy intervals. Everything reported was declared in configs/eval/epstein_criteria.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from d1_epstein import cells_from_control, p_yes  # noqa: F401 - p_yes is re-exported for the tests

from kite.eval import epstein
from kite.eval.runner import read_predictions
from kite.operators import tilt_cell

CRITERIA = Path("configs/eval/epstein_criteria.yaml")
D2 = Path("results/d2/discrepancy_model.json")
VALUES = np.array([0.0, 1.0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(run: Path) -> pd.DataFrame:
    rows = [{**p.meta, "probs": np.asarray(p.probs, float), "p_yes": p_yes(p.probs)} for p in read_predictions(run)]
    return pd.DataFrame(rows)


def discernment(df: pd.DataFrame, column: str = "p_yes") -> pd.DataFrame:
    """Per wave and arm: mean on true cards minus mean on false cards, and the two channels."""
    by = df.groupby(["wave", "arm", "true"])[column].mean().unstack("true")
    out = pd.DataFrame({"true": by[True], "false": by[False]})
    out["discernment"] = out["true"] - out["false"]
    return out


def hybrid(jev: pd.DataFrame, astra: pd.DataFrame, cells: pd.DataFrame, anchors_per_cell: int) -> tuple[pd.DataFrame, list[dict]]:
    """The kernel's application-pool predictions with every treated cell tilted to the flagship's paired anchor effect."""
    app = jev[jev["pool"] == "application"].merge(cells, on=["wave", "id", "item_num"], how="left")
    anchors = astra[astra["anchor"]].merge(cells, on=["wave", "id", "item_num"], how="left")
    out, records = app.copy(), []
    for (wave, cell), group in app.groupby(["wave", "cell"]):
        control = group[group["arm"] == "control"]
        a_control = anchors[(anchors["wave"] == wave) & (anchors["cell"] == cell) & (anchors["arm"] == "control")].set_index(["id", "item_num"])[
            "p_yes"
        ]
        for arm in sorted(set(group["arm"]) - {"control"}):
            treated = group[group["arm"] == arm]
            a_treated = anchors[(anchors["wave"] == wave) & (anchors["cell"] == cell) & (anchors["arm"] == arm)].set_index(["id", "item_num"])[
                "p_yes"
            ]
            paired = a_control.index.intersection(a_treated.index)
            record = {
                "wave": int(wave),
                "cell": cell,
                "arm": arm,
                "anchors": int(len(paired)),
                "weight": float(len(treated) / (app["wave"] == wave).sum()),
            }
            if len(paired) < anchors_per_cell // 2 or len(control) == 0:
                records.append({**record, "status": "skipped"})
                continue
            shift = float(a_treated.loc[paired].mean() - a_control.loc[paired].mean())
            target = float(control["p_yes"].mean() + shift)
            probs = np.stack([np.array([1 - p, p]) for p in treated["p_yes"]])
            tilted, result = tilt_cell(probs, VALUES, target)
            if not result.feasible:
                records.append({**record, "status": "infeasible", "requested": target, "achieved": result.achieved})
                continue
            out.loc[treated.index, "p_yes"] = tilted[:, 1]
            records.append({**record, "status": "ok", "shift": shift, "requested": target, "achieved": result.achieved, "alpha": result.alpha})
    return out, records


def choose(table: pd.DataFrame) -> dict[int, str]:
    return {int(w): str(g["discernment"].idxmax()[1]) for w, g in table.groupby(level="wave")}


def stage_predict(args) -> None:
    cfg = yaml.safe_load(CRITERIA.read_text())
    jev = frame(args.jev)
    jev = jev[jev["kind"] == "share"]
    astra = frame(args.astra)
    cells, edges = cells_from_control(jev)
    stored = yaml.safe_load((args.astra / "config.yaml").read_text())["cell_edges"]
    if {k: [round(x, 9) for x in v] for k, v in stored.items()} != {k: [round(x, 9) for x in v] for k, v in edges.items()}:
        raise SystemExit("the cell edges recomputed from the kernel run differ from those the flagship stage used")
    corrected, records = hybrid(jev, astra, cells, cfg["run"]["anchors_per_cell"])
    tables = {
        "kernel": discernment(jev[jev["pool"] == "application"]),
        "hybrid": discernment(corrected),
        "flagship_audit": discernment(astra[~astra["anchor"]]),
        "kernel_audit": discernment(jev[jev["pool"] == "audit"]),
    }
    policies = {name: choose(t) for name, t in tables.items()}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    inputs = {"jev": str(args.jev), "astra": str(args.astra)}
    inputs |= {"jev_sha256": sha256(args.jev / "predictions.jsonl"), "astra_sha256": sha256(args.astra / "predictions.jsonl")}
    waves = sorted({r["wave"] for r in records})
    out = {
        "criteria": str(CRITERIA),
        "commit": commit,
        "inputs": inputs,
        "tables": {name: {f"{w}:{a}": {k: float(v) for k, v in row.items()} for (w, a), row in t.iterrows()} for name, t in tables.items()},
        "policies": policies,
        "corrections": records,
        "uncorrected_weight": {int(w): float(sum(r["weight"] for r in records if r["wave"] == w and r["status"] != "ok")) for w in waves},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for name, pol in policies.items():
        print(f"{name:15} chooses {pol}")
    counts = {status: sum(r["status"] == status for r in records) for status in ("ok", "infeasible", "skipped")}
    print(f"corrections: {counts}")
    print(f"locked to {args.out}")


def human_tables(ratings: pd.DataFrame) -> pd.DataFrame:
    share = ratings[ratings["arm"] != "accuracy_only"].copy()
    share["yes"] = share["rating"].astype(float)
    return discernment(share, "yes")


def person_discernment(ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: mean yes on true minus false cards; the unit of resampling."""
    share = ratings[ratings["arm"] != "accuracy_only"]
    by = share.groupby(["wave", "arm", "id", "true"])["rating"].mean().unstack("true")
    out = (by[True] - by[False]).rename("d").reset_index()
    return out


def policy_value(person: pd.DataFrame, policy: dict[int, str], rng=None) -> float:
    """Mean over waves of people's discernment in the arm the policy chose (resampled per wave x arm if rng is given)."""
    values = []
    for wave, arm in policy.items():
        d = person[(person["wave"] == wave) & (person["arm"] == arm)]["d"].to_numpy()
        if rng is not None:
            d = d[rng.integers(0, len(d), len(d))]
        values.append(d.mean())
    return float(np.mean(values))


def stage_score(args) -> None:
    cfg = yaml.safe_load(CRITERIA.read_text())
    locked = json.loads(Path(args.policies).read_text())
    for key in ("jev", "astra"):
        if sha256(Path(locked["inputs"][key]) / "predictions.jsonl") != locked["inputs"][f"{key}_sha256"]:
            raise SystemExit(f"the {key} predictions changed since the policies were locked")
    ratings = epstein.load_ratings()  # refuses unless the criteria file is committed
    human = human_tables(ratings)
    person = person_discernment(ratings)
    rng = np.random.default_rng(cfg["analysis"]["seed"])
    n_boot, n_perm = cfg["analysis"]["n_boot"], cfg["analysis"]["n_perm"]
    policies = {k: {int(w): a for w, a in v.items()} for k, v in locked["policies"].items()}
    report = {
        "locked": str(args.policies),
        "human": {f"{w}:{a}": {k: float(v) for k, v in row.items()} for (w, a), row in human.iterrows()},
        "endpoints": {},
    }

    # primary: hybrid minus kernel policy value; secondary: flagship minus kernel, hybrid minus flagship
    for name, (a, b) in {
        "hybrid_minus_kernel": ("hybrid", "kernel"),
        "flagship_minus_kernel": ("flagship_audit", "kernel"),
        "hybrid_minus_flagship": ("hybrid", "flagship_audit"),
    }.items():
        point = policy_value(person, policies[a]) - policy_value(person, policies[b])
        boot = np.array([policy_value(person, policies[a], rng) - policy_value(person, policies[b], rng) for _ in range(n_boot)])
        disagree = [w for w in policies[a] if policies[a][w] != policies[b][w]]
        report["endpoints"][name] = {
            "point": point,
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "waves_that_differ": disagree,
        }
    # chance baseline for each system: permute its arm scores within waves
    for system in ("kernel", "hybrid", "flagship_audit"):
        table = locked["tables"][system]
        waves = sorted({int(k.split(":")[0]) for k in table})
        null = []
        for _ in range(n_perm):
            policy = {}
            for w in waves:
                arms = [k.split(":")[1] for k in table if int(k.split(":")[0]) == w]
                policy[w] = arms[rng.integers(0, len(arms))]
            null.append(policy_value(person, policy))
        observed = policy_value(person, policies[system])
        report["endpoints"][f"{system}_value"] = {
            "observed": observed,
            "chance_mean": float(np.mean(null)),
            "chance_p95": float(np.percentile(null, 95)),
            "p_one_sided": float((1 + sum(v >= observed for v in null)) / (1 + n_perm)),
        }
    # human reference: half chooses, the other half judges, all systems judged on the same half
    refs = {"pilot": [], **{s: [] for s in ("kernel", "hybrid", "flagship_audit")}}
    for _ in range(cfg["analysis"]["n_splits"]):
        person["half"] = rng.integers(0, 2, len(person))
        choosing, judging = person[person["half"] == 0], person[person["half"] == 1]
        pilot = {int(w): str(g.groupby("arm")["d"].mean().idxmax()) for w, g in choosing.groupby("wave")}
        refs["pilot"].append(policy_value(judging, pilot))
        for s in ("kernel", "hybrid", "flagship_audit"):
            refs[s].append(policy_value(judging, policies[s]))
    report["endpoints"]["same_judge"] = {k: float(np.mean(v)) for k, v in refs.items()}
    # effects against each wave's control: MAE and channels per system
    effects = {}
    for system in ("kernel", "hybrid", "flagship_audit"):
        table, errs, ch = locked["tables"][system], [], {"true": [], "false": []}
        for key, row in table.items():
            w, a = key.split(":")
            if a == "control":
                continue
            ctrl_m, ctrl_h = table[f"{w}:control"], report["human"][f"{w}:control"]
            h = report["human"][key]
            errs.append(abs((row["discernment"] - ctrl_m["discernment"]) - (h["discernment"] - ctrl_h["discernment"])))
            for c in ("true", "false"):
                ch[c].append(abs((row[c] - ctrl_m[c]) - (h[c] - ctrl_h[c])))
        effects[system] = {
            "effect_mae": float(np.mean(errs)),
            "true_channel_mae": float(np.mean(ch["true"])),
            "false_channel_mae": float(np.mean(ch["false"])),
            "n_effects": len(errs),
        }
    report["effects"] = effects
    # coverage of the D2 discrepancy intervals (kernel parameters) on the eight wave-relative effects of the hybrid and the kernel
    if D2.exists():
        d2 = json.loads(D2.read_text())["jev"]["fit_seen"]
        cov = {}
        for system in ("kernel", "hybrid"):
            inside = []
            for key, row in locked["tables"][system].items():
                w, a = key.split(":")
                if a == "control":
                    continue
                x = row["discernment"] - locked["tables"][system][f"{w}:control"]["discernment"]
                y = report["human"][key]["discernment"] - report["human"][f"{w}:control"]["discernment"]
                sd = np.sqrt(d2["tau"] ** 2 + d2["sigma"] ** 2)
                inside.append(abs(y - d2["beta"] * x) <= 1.645 * sd)
            cov[system] = {"coverage_90": float(np.mean(inside)), "n": len(inside)}
        report["d2_coverage"] = cov
    Path(args.out).write_text(json.dumps(report, indent=1))
    e = report["endpoints"]
    hk = e["hybrid_minus_kernel"]
    print(f"primary  hybrid - kernel policy value: {hk['point']:+.4f} {[round(v, 4) for v in hk['ci']]}  waves that differ {hk['waves_that_differ']}")
    for s in ("kernel", "hybrid", "flagship_audit"):
        v = e[f"{s}_value"]
        line = f"  {s:15} value {v['observed']:+.4f}  chance mean {v['chance_mean']:+.4f} p95 {v['chance_p95']:+.4f}  p {v['p_one_sided']:.3f}"
        print(line + f"  same-judge {e['same_judge'][s]:+.4f}")
    print(f"  {'pilot':15} same-judge {e['same_judge']['pilot']:+.4f}")
    for s, v in effects.items():
        print(f"  {s:15} effect MAE {v['effect_mae']:.4f}  true-channel {v['true_channel_mae']:.4f}  false-channel {v['false_channel_mae']:.4f}")
    if "d2_coverage" in report:
        print("  D2 90% coverage:", report["d2_coverage"])
    print(f"written to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--jev", type=Path, required=True)
    p.add_argument("--astra", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/d1/policies.json"))
    s = sub.add_parser("score")
    s.add_argument("--policies", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("results/d1/report.json"))
    args = parser.parse_args()
    if subprocess.run(["git", "status", "--porcelain", str(CRITERIA), "scripts/d1_analysis.py"], capture_output=True, text=True).stdout.strip():
        raise SystemExit("refusing to run: commit the criteria file and this script first")
    stage_predict(args) if args.stage == "predict" else stage_score(args)


if __name__ == "__main__":
    main()
