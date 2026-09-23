"""C2 / C3: the kernel on Arechar et al. 2023 - calibration on US x half A, then the held-out parts once.

calibration   every US respondent's ratings of odd-numbered headlines (the only part the loader releases
              until configs/eval/arechar_criteria.yaml is committed). Besides the model's own measures it
              records what the criteria need: a null for each measure, the human reference for each, and
              how the measures move with the number of simulated respondents per condition.
held_out      refused unless the criteria file is committed; runs the parts it names, once.

Usage:
    uv run python scripts/c2_arechar.py --phase calibration
    uv run python scripts/c2_arechar.py --phase held_out        # only after the criteria are committed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kite.config import Settings
from kite.eval import arechar
from kite.eval.runner import read_predictions, run_task
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger

CSV = Path("data/arechar/CR.csv")
QSF = Path("data/arechar/questionnaires/CRUS.qsf")
OUT = Path("results/c2")
SIZES = (25, 50, 100, 150, None)


def frame_of(predictions) -> pd.DataFrame:
    rows = []
    for p in predictions:
        probs = np.asarray(p.probs, dtype=float)
        rows.append({**p.meta, "expected": float(np.dot(probs / probs.sum(), np.arange(1, 7)))})
    return pd.DataFrame(rows)


def item_r(d: pd.DataFrame, condition: str, column: str = "expected") -> float:
    g = d[d["condition"] == condition].groupby("item")
    return float(np.corrcoef(g[column].mean(), g["rating"].mean())[0, 1])


def discernment(d: pd.DataFrame, condition: str, column: str) -> float:
    g = d[d["condition"] == condition].groupby("true")[column].mean()
    return float(g[True] - g[False])


def nulls(d: pd.DataFrame, rng: np.random.Generator, n: int = 2000) -> dict:
    """What a model with no information about the relevant thing would score, by permutation."""
    out = {}
    for condition in ("share_only", "accuracy"):
        g = d[d["condition"] == condition].groupby("item")
        model, human = g["expected"].mean().to_numpy(), g["rating"].mean().to_numpy()
        values = [np.corrcoef(rng.permutation(model), human)[0, 1] for _ in range(n)]
        out[f"item_r_{condition}"] = {"null_p95": float(np.percentile(values, 95)), "null_mean": float(np.mean(values))}
    # discernment: shuffle which items are true, keeping the number of true items
    items = d["item"].unique()
    truth = {i: arechar.is_true(int(i)) for i in items}
    values = {c: [] for c in ("share_only", "accuracy", "prompt", "tips")}
    for _ in range(n // 4):
        shuffled = dict(zip(items, rng.permutation([truth[i] for i in items]), strict=True))
        e = d.assign(true=d["item"].map(shuffled))
        for c in values:
            values[c].append(discernment(e, c, "expected"))
    out["discernment"] = {c: {"null_p95": float(np.percentile(v, 95)), "null_p05": float(np.percentile(v, 5))} for c, v in values.items()}
    # treatment effects: shuffle condition labels between the treated and the share-only respondents
    for treatment in ("prompt", "tips"):
        pair = d[d["condition"].isin([treatment, "share_only"])]
        people = pair[["id", "condition"]].drop_duplicates()
        effects = []
        for _ in range(n // 4):
            relabel = dict(zip(people["id"], rng.permutation(people["condition"].to_numpy()), strict=True))
            e = pair.assign(condition=pair["id"].map(relabel))
            effects.append(discernment(e, treatment, "expected") - discernment(e, "share_only", "expected"))
        out[f"{treatment}_effect"] = {"null_p95": float(np.percentile(effects, 95)), "null_p05": float(np.percentile(effects, 5))}
    return out


def stability(d: pd.DataFrame, seed: int = 0) -> list[dict]:
    """The model's measures when only the first n respondents per condition are simulated (nested by hash)."""
    import zlib

    people = d[["country", "id", "condition"]].drop_duplicates()
    people = people.assign(order=[zlib.crc32(f"{seed}:{c}:{i}".encode()) for c, i in zip(people["country"], people["id"], strict=True)])
    out = []
    for size in SIZES:
        chosen = people.sort_values("order").groupby("condition").head(size) if size else people
        sub = d.merge(chosen[["id"]], on="id")
        human = d  # the human side always uses everybody
        row = {"respondents_per_condition": size or "all", "predictions": len(sub)}
        for condition in ("share_only", "accuracy"):
            m = sub[sub["condition"] == condition].groupby("item")["expected"].mean()
            h = human[human["condition"] == condition].groupby("item")["rating"].mean().reindex(m.index)
            row[f"item_r_{condition}"] = float(np.corrcoef(m, h)[0, 1])
        for treatment in ("prompt", "tips"):
            row[f"{treatment}_effect_model"] = discernment(sub, treatment, "expected") - discernment(sub, "share_only", "expected")
        out.append(row)
    return out


LANGUAGE = {c: "english" for c in ("US", "UK", "AU", "ZA", "NG")} | {"IN": "mixed", "PN": "mixed"}


def analyse_part(model: pd.DataFrame, human: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Every measure the criteria name, for one part of the split."""
    per_country = {}
    for country in sorted(model["country"].unique()):
        mc, hc = model[model["country"] == country], human[human["country"] == country]
        entry = {}
        for condition in ("share_only", "accuracy"):
            m, h = mc[mc["condition"] == condition], hc[hc["condition"] == condition]
            entry[f"headline_{condition}"] = arechar.headline_level(m, h, rng)
            entry[f"discernment_{condition}"] = arechar.discernment_test(m, h, rng)
        per_country[country] = entry
    return {
        "per_country": per_country,
        "individual": arechar.individual_level(model, rng, n_perm=200),
        "effects": {t: arechar.treatment_effect(model, human, t, rng) for t in ("prompt", "tips")},
    }


def judge(part: str, result: dict) -> dict:
    """Apply configs/eval/arechar_criteria.yaml to one part."""
    need = 1 if part == "new_headlines" else 12
    countries = result["per_country"]

    def significant(entry: dict) -> bool:
        return entry["r"] > entry["null_p95"] and entry["p"] < 0.01

    counts = {
        "headline_sharing": sum(significant(e["headline_share_only"]) for e in countries.values()),
        "headline_accuracy": sum(significant(e["headline_accuracy"]) for e in countries.values()),
        "discernment": sum(e["discernment_accuracy"]["model"] > e["discernment_accuracy"]["null_p95"] for e in countries.values()),
    }
    individual_ok = result["individual"]["r"] > result["individual"]["null_p95"]
    intervention, agreement = {}, {}
    for treatment, e in result["effects"].items():
        human_positive = e["human_ci"][0] > 0
        model_positive = e["model_pooled"] > e["model_null_p95"]
        intervention[treatment] = "PASS" if human_positive and model_positive else "FAIL" if human_positive else "NOT APPLICABLE"
        reliable = {c: v for c, v in e["per_country"].items() if v["human_ci"][0] > 0 or v["human_ci"][1] < 0}
        same = sum(bool(np.sign(v["model"]) == np.sign(v["human"])) for v in reliable.values())
        agreement[treatment] = {"reliable_countries": len(reliable), "same_sign": same}
    return {
        "countries": len(countries),
        "needed": need,
        "counts": counts,
        "individual_ok": bool(individual_ok),
        "item_and_person": bool(all(v >= need for v in counts.values()) and individual_ok),
        "intervention": intervention,
        "sign_agreement_on_reliable_countries": agreement,
    }


def by_language(result: dict) -> dict:
    groups: dict[str, dict[str, list[float]]] = {}
    for country, e in result["per_country"].items():
        g = groups.setdefault(LANGUAGE.get(country, "translated"), {"sharing": [], "accuracy": [], "discernment_gap": []})
        g["sharing"].append(e["headline_share_only"]["r"])
        g["accuracy"].append(e["headline_accuracy"]["r"])
        g["discernment_gap"].append(e["discernment_accuracy"]["model"] - e["discernment_accuracy"]["human"])
    return {name: {k: float(np.median(v)) for k, v in g.items()} | {"countries": len(g["sharing"])} for name, g in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["calibration", "held_out"], required=True)
    parser.add_argument("--max-usd", type=float, default=2.0)
    args = parser.parse_args()

    settings = Settings()
    criteria = None
    if args.phase == "held_out":
        criteria = yaml.safe_load(arechar.CRITERIA.read_text()) if arechar.CRITERIA.exists() else None
        if criteria is None or subprocess.run(["git", "status", "--porcelain", str(arechar.CRITERIA)], capture_output=True, text=True).stdout.strip():
            raise SystemExit(f"refusing the held-out phase: commit {arechar.CRITERIA} first")
    task = arechar.ArecharTask(
        CSV, QSF, args.phase, max_respondents=(criteria or {}).get("respondents_per_country_condition"), parts=(criteria or {}).get("parts")
    )
    n_items = len(task.ratings)
    print(f"{args.phase}: {n_items:,} ratings from {task.ratings[['country', 'id']].drop_duplicates().shape[0]:,} respondents in {task.members()}")

    out = OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{args.phase}"
    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=args.max_usd), settings, ledger)

    async def go():
        try:
            return await run_task(
                task, built.kernel, out, concurrency=settings.max_concurrency, ledger=ledger, config={"phase": args.phase, "criteria": criteria}
            )
        finally:
            await built.aclose()

    summary = asyncio.run(go())
    print(json.dumps(summary.model_dump()), json.dumps(ledger.summary()))
    predictions = read_predictions(out)
    scores = arechar.score_arechar(predictions)
    print(json.dumps({k: v for k, v in scores.items() if k != "by_part"}, indent=1))

    if args.phase == "held_out":
        model = frame_of(predictions)
        human = arechar.load_ratings(CSV, "held_out")
        human["true"] = human["item"].map(arechar.is_true)
        rng = np.random.default_rng(0)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        report = {"commit": commit, "criteria": criteria, "parts": {}}
        for part in criteria["parts"]:
            result = analyse_part(model[model["part"] == part], human[human["part"] == part], rng)
            report["parts"][part] = {"result": result, "verdict": judge(part, result), "by_language": by_language(result)}
            v = report["parts"][part]["verdict"]
            print(f"\n{part}: {v['countries']} countries, need {v['needed']}; counts {v['counts']}; individual ok {v['individual_ok']}")
            print(f"  item_and_person: {'PASS' if v['item_and_person'] else 'FAIL'};  intervention: {v['intervention']}")
            for t, e in result["effects"].items():
                ci = [round(x, 3) for x in e["human_ci"]]
                print(f"  {t:6} model {e['model_pooled']:+.3f} (null p95 {e['model_null_p95']:+.3f})   human {e['human_pooled']:+.3f} {ci}")
            print(f"  by language: {json.dumps(report['parts'][part]['by_language'])}")
        (out / "held_out.json").write_text(json.dumps(report, indent=1, default=str))

    if args.phase == "calibration":
        d = frame_of(predictions)
        human = json.loads((OUT / "human_references.json").read_text()) if (OUT / "human_references.json").exists() else None
        report = {"scores": scores, "nulls": nulls(d, np.random.default_rng(0)), "stability": stability(d), "human_references": human}
        (out / "calibration.json").write_text(json.dumps(report, indent=1, default=str))
        print("\nnulls:", json.dumps(report["nulls"], indent=1))
        print("\nstability against simulated respondents per condition:")
        print(pd.DataFrame(report["stability"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
