"""Export the aggregate numbers behind the showcase page into one JSON file.

Only aggregates leave the machine: cell and headline means, correlation matrices, curves. No respondent-level
row, no study stimulus text and no headline text is exported - the two datasets state no licence, so
headlines appear on the page as short topic tags written for it (TAGS below), not as quotations.

Usage:
    PYTHONPATH=scripts uv run python scripts/export_showcase_data.py --out <path>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from e3b_answer_history import bin_of
from e3c_autoregressive import load_people
from scipy.stats import spearmanr

from kite.eval import arechar
from kite.eval import decision_value as dv
from kite.eval.runner import read_predictions
from kite.eval.scales import Scale
from kite.eval.socsci210 import SocSci210Task

RUNS = {
    "uniform": "results/20260921-034251-socsci210-test-baseline-uniform",
    "pooled": "results/20260921-034257-socsci210-test-baseline-pooled",
    "jev": "results/20260921-012513-socsci210-test-jev-p3-choice",
    "jev_first5": "results/20260921-012513-socsci210-test-jev-p3-choice-first5",
    "luna": "results/20260921-224518-socsci210-test-codex-gpt-5.6-luna",
    "astra": "results/20260922-130953-socsci210-test-codex-gpt-6-astra-high",
}
MODELS = ("jev", "luna", "astra")
CALIBRATION = "results/c2/20260922-004807-calibration"
HELD_OUT = "results/c2/20260922-011615-held_out"
FLAGSHIP = "results/c3-flagship/20260922-133803-gpt-6-astra"
LANGUAGE = {"US": "英", "UK": "英", "AU": "英", "ZA": "英", "NG": "英", "IN": "混", "PN": "混"}  # everyone else read a translation
PUBLISHED = {"GPT-4o": 0.1740, "GPT-4o few-shot": 0.1610, "Socrates-LLaMA3-8B": 0.1530, "Socrates-Qwen2.5-14B": 0.1510, "empirical bound": 0.1250}
PUBLISHED_UNIFORM = 0.2030

TAGS = {  # short paraphrased topics for the 45 Arechar headlines, written for this page
    1: "口罩与疫苗削弱免疫", 2: "热水陈皮薄荷膏除病毒", 3: "疫苗内置追踪芯片", 4: "疫苗改造人类基因", 5: "mRNA 疫苗改变 DNA",
    6: "抗生素可治新冠", 7: "1918 大流感死于疫苗", 8: "新冠是细菌、5G 放大", 9: "戴口罩致缺氧", 10: "热蒸汽与茶治愈新冠",
    11: "社交距离无效", 12: "疫苗致女性不育", 13: "疫苗含纳米追踪芯片", 14: "日晒高温防新冠", 15: "超额死亡等同流感季",
    16: "可乐也能测出阳性", 17: "2020 年死亡反而更少", 18: "联合国承认疫苗有毒", 19: "柠檬小苏打茶杀病毒", 20: "某族裔血统抗新冠",
    21: "护士接种后身亡", 22: "紫外灯或干手机杀病毒", 23: "O 型血不受影响", 24: "口罩会致人死亡", 25: "新冠不如流感致命",
    26: "饮酒可防治新冠", 27: "蚊蝇传播新冠", 28: "大量儿童接种后出事", 29: "疫苗含胎儿组织", 30: "胡椒可防新冠",
    31: "鞋子传播风险很低", 32: "测温无法检出新冠", 33: "变异或引发新一波", 34: "勤洗手降低风险", 35: "新冠或严重伤心脏",
    36: "少数族裔感染风险更高", 37: "新冠远比疫苗危险", 38: "多数感染症状轻中度", 39: "症状消失仍带病毒", 40: "疫苗罕见过敏疑云",
    41: "封城改善空气质量", 42: "二次感染增多", 43: "电子烟增加感染风险", 44: "平台提醒点赞谣言者", 45: "地塞米松降低重症死亡",
}  # fmt: skip


def metrics(run: str) -> dict:
    return json.loads((Path(run) / "metrics.json").read_text())


def gaming() -> list[dict]:
    uniform = metrics(RUNS["uniform"])["distribution_published_convention"]
    rows = []
    labels = {
        "uniform": "均匀猜测",
        "pooled": "合并真人（条件盲）",
        "jev": "Jev · 每格 200 人",
        "jev_first5": "Jev · 每格 5 人",
        "luna": "GPT-5.6 Luna · 每格 5 人",
        "astra": "GPT-6 Astra · 每格 5 人",
    }
    for key, label in labels.items():
        m = metrics(RUNS[key])
        rows.append(
            {
                "key": key,
                "label": label,
                "x": m["distribution_published_convention"] / uniform,
                "r": m.get("condition_sensitivity_r"),
                "source": "ours",
            }
        )
    for label, value in PUBLISHED.items():
        rows.append({"key": label, "label": label, "x": value / PUBLISHED_UNIFORM, "r": None, "source": "published"})
    return rows


def decision() -> dict:
    results = {name: json.loads((Path(RUNS[name]) / "decision_value.json").read_text()) for name in MODELS}
    prepared = Path("data/socsci210/prepared/test")
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    human = dv.human_cells(rows, scales)
    comparable = dv.comparable_tasks(scales)
    blocks = {name: dv.build_blocks(comparable, human, dv.predicted_means(read_predictions(RUNS[name]))) for name in MODELS}
    tasks = []
    for row in zip(*(blocks[name] for name in MODELS), strict=True):
        j = row[0]
        assert all((b.study, b.task) == (j.study, j.task) for b in row)
        if len(j.human) < 3:
            continue
        tasks.append({"study": j.study, "task": j.task, "human": j.human.round(4).tolist(), "se": j.se.round(4).tolist(),
                      **{name: b.predicted.round(4).tolist() for name, b in zip(MODELS, row, strict=True)}})  # fmt: skip
    keep = ("sign_accuracy", "captured_gain")
    three = json.loads(Path("results/a3b/three_models.json").read_text())
    return {
        "summary": {name: {k: d[k] for k in keep} for name, d in results.items()},
        "effect_curve": {name: d["effect_curve"] for name, d in results.items()},
        "stability": results["jev"]["stability"],
        "blind_spot": three["common_blind_spot"],
        "astra_minus_jev": three["paired_differences"]["astra-jev"],
        "three_models": {name: {k: d[k] for k in keep} for name, d in three["decision"].items()},  # same 3,615 items, 5 per cell
        "tasks": tasks,
    }


def arechar_calibration() -> dict:
    predictions = read_predictions(CALIBRATION)
    rows = []
    for p in predictions:
        probs = np.asarray(p.probs, dtype=float)
        rows.append({**p.meta, "expected": float(np.dot(probs / probs.sum(), np.arange(1, 7)))})
    frame = pd.DataFrame(rows)
    items = []
    for item, g in frame.groupby("item"):
        human = g.groupby("condition")["rating"].mean().round(3).to_dict()
        model = g.groupby("condition")["expected"].mean().round(3).to_dict()
        items.append({"k": int(item), "true": arechar.is_true(int(item)), "tag": TAGS[int(item)], "human": human, "model": model})
    cells = {}
    for (condition, truth), g in frame.groupby(["condition", "true"]):
        cells.setdefault(condition, {})["true" if truth else "false"] = {
            "human": round(g["rating"].mean(), 3),
            "model": round(g["expected"].mean(), 3),
        }
    calibration = json.loads((Path(CALIBRATION) / "calibration.json").read_text())
    explicit = json.loads(Path("results/c2/explicit_mechanism.json").read_text())
    return {"items": items, "cells": cells, "scores": calibration["scores"], "nulls": calibration["nulls"], "explicit": explicit}


METRIC_KEYS = (
    "magnitude_independent",
    "magnitude_raw",
    "magnitude_corrected",
    "structure_independent",
    "structure_raw",
    "structure_corrected",
    "human_split_half_structure",
    "distant_vs_real_corrected",
    "raw_drift_advantage_given_back",
    "coherence_kept_by_correction",
)


def arechar_held_out() -> dict:
    report = json.loads((Path(HELD_OUT) / "held_out.json").read_text())
    parts = {}
    for part, v in report["parts"].items():
        result = v["result"]
        countries = []
        for country, e in sorted(result["per_country"].items()):
            prompt, tips = result["effects"]["prompt"]["per_country"][country], result["effects"]["tips"]["per_country"][country]
            countries.append(
                {
                    "c": country,
                    "lang": LANGUAGE.get(country, "译"),
                    "share_r": e["headline_share_only"]["r"],
                    "share_null": e["headline_share_only"]["null_p95"],
                    "share_rel": e["headline_share_only"]["human_reliability"],
                    "acc_r": e["headline_accuracy"]["r"],
                    "acc_null": e["headline_accuracy"]["null_p95"],
                    "acc_rel": e["headline_accuracy"]["human_reliability"],
                    "disc_model": e["discernment_accuracy"]["model"],
                    "disc_human": e["discernment_accuracy"]["human"],
                    "prompt_model": prompt["model"],
                    "prompt_human": prompt["human"],
                    "prompt_ci": prompt["human_ci"],
                    "tips_model": tips["model"],
                    "tips_human": tips["human"],
                    "tips_ci": tips["human_ci"],
                }
            )
        keep = ("model_pooled", "model_null_p95", "human_pooled", "human_ci")
        effects = {t: {k: e[k] for k in keep} for t, e in result["effects"].items()}
        parts[part] = {"countries": countries, "verdict": v["verdict"], "effects": effects, "individual": result["individual"]}
    return parts


def flagship() -> dict:
    """C3b: GPT-6 Astra and the kernel on the same respondents of the both-new part (aggregates only)."""
    run = Path(FLAGSHIP)
    report = json.loads((run / "report.json").read_text())
    primary = json.loads((run / "report-primary.json").read_text())
    paired = json.loads((run / "paired.json").read_text())
    names = {"flagship": "astra", "kernel_same_respondents": "jev"}
    out = {"verdicts": {}, "headline_sharing": {}, "treatments": {}, "levels": paired.pop("levels")}
    for name, short in names.items():
        keep = ("model_pooled", "model_null_p95", "human_pooled", "human_ci", "verdict")
        out["verdicts"][short] = {t: {k: e[k] for k in keep} for t, e in report[name]["effects"].items()}
        rs = [v["r"] for v in report[name]["headline_sharing"].values()]
        out["headline_sharing"][short] = {"passed": report[name]["headline_sharing_passed"], "median_r": float(np.median(rs))}
    for treatment, r in paired.items():
        human_ci = {c: v["human_ci"] for c, v in report["flagship"]["effects"][treatment]["per_country"].items()}
        keep = ("pooled", "ci", "p_one_sided", "null_p95", "true_headlines", "false_headlines", "per_country")
        entry = {short: {k: r[name][k] for k in keep} for name, short in (("flagship", "astra"), ("kernel", "jev"), ("people", "people"))}
        entry["diff"] = {k: r["flagship_minus_kernel"][k] for k in (*keep, "countries_positive")}
        entry["human_ci"] = human_ci
        out["treatments"][treatment] = entry
    tokens = primary["ledger"]["tokens"] + (report["ledger"]["tokens"] if report["conditions"] != primary["conditions"] else 0)
    out["tokens_per_prediction"] = tokens / report["flagship"]["n_predictions"]
    out["predictions"] = report["flagship"]["n_predictions"]
    # the kernel's own held-out bill per prediction, against the flagship's tokens at list prices ($10 / $50 per MTok)
    kernel_usd = json.loads((Path(HELD_OUT) / "ledger.json").read_text())["usd"] / len(read_predictions(HELD_OUT))
    out["cost_ratio"] = [out["tokens_per_prediction"] * price / 1e6 / kernel_usd for price in (10, 50)]
    return out


def b2() -> dict:
    prepared = Path("results/b2/prepared")
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    people = load_people(SocSci210Task(prepared, max_participants=150)._frame)
    real = {}
    for (study, condition, participant), info in people.items():
        for t in info["tasks"]:
            real[(study, condition, participant, int(t["task_num"]))] = bin_of(scales[f"{study}|{condition}|{t['task_num']}"], int(t["response"]))
    arms = {arm: pd.read_csv(f"results/b2/draws_{arm}.csv") for arm in ("independent", "raw_memory", "corrected")}
    per_study = pd.read_csv("results/b2/per_study.csv").set_index("study")
    out = []
    for study in per_study.index:
        mats = defaultdict(list)
        weights = []
        tasks = None
        for condition in sorted({c for (s, c, _, _) in real if s == study}):
            sim = {
                arm: d[(d["study"] == study) & (d["condition"] == condition)].pivot_table(index="participant", columns="task_num", values="choice")
                for arm, d in arms.items()
            }
            keys = [(p, t) for (s, c, p, t) in real if s == study and c == condition]
            wide = pd.Series({k: real[(study, condition) + k] for k in keys}).unstack()
            common = wide.dropna().index
            for arm in arms:
                common = common.intersection(sim[arm].dropna().index)
            tasks = sorted(wide.columns)
            if len(common) < 30:
                continue
            weights.append(len(common))
            mats["real"].append(spearmanr(wide.loc[common, tasks].to_numpy()).statistic)
            for arm in arms:
                mats[arm].append(spearmanr(sim[arm].loc[common, tasks].to_numpy()).statistic)
        matrices = {name: np.average(np.stack(ms), axis=0, weights=weights) for name, ms in mats.items()}
        n = len(tasks)
        lag = {name: [float(np.mean([abs(m[i, i + d]) for i in range(n - d)])) for d in range(1, n)] for name, m in matrices.items()}
        row = per_study.loc[study]
        out.append({
            "study": study, "n_tasks": n, "people": int(sum(weights)),
            "matrices": {name: np.round(m, 3).tolist() for name, m in matrices.items()}, "lag": lag,
            "metrics": {k: float(row[k]) for k in METRIC_KEYS},
        })  # fmt: skip
    return {"studies": out}


D1_DIR = Path("results/d1")
D1_SYSTEMS = ("kernel", "hybrid", "flagship_audit")
D1_COST = {"kernel_usd": 1.60, "kernel_predictions": 73600, "anchor_items": 1080, "audit_items": 9000, "flagship_tokens": 4098094}
D3_COST = {
    "kernel_usd": 4.53,
    "kernel_predictions": 136901,
    "anchor_items": 2082,
    "anchor_tokens": 1004653,
    "flagship5_items": 3615,
    "flagship5_tokens": 2570752,
}
CUBE_ARMS = ("control", "evaluation", "generic_norms", "tips")  # wave 3 of Epstein: the arms the simulator can switch between


def d1() -> dict:
    """D1 + appendix A: the ten wave-relative effects, per-wave arm tables and policies, robustness, headline heterogeneity."""
    report = json.loads((D1_DIR / "report.json").read_text())
    policies = json.loads((D1_DIR / "policies.json").read_text())
    robust = json.loads((D1_DIR / "robustness.json").read_text())
    heads = json.loads((D1_DIR / "headlines.json").read_text())
    human, tables = report["human"], policies["tables"]
    effects, waves = [], {}
    for key, v in human.items():
        w, a = key.split(":")
        waves.setdefault(int(w), {})[a] = {"human": v["discernment"], **{s: tables[s][key]["discernment"] for s in D1_SYSTEMS}}
        if a != "control":
            row = {"wave": int(w), "arm": a, "human": v["discernment"] - human[f"{w}:control"]["discernment"]}
            row |= {s: tables[s][key]["discernment"] - tables[s][f"{w}:control"]["discernment"] for s in D1_SYSTEMS}
            effects.append(row)
    keep = ("kernel_minus_hybrid_abs_error", "leave_one_wave_out", "calibrated_parents", "sampling_floor", "persona_count_curve", "raw_mae")
    return {
        "effects": effects,
        "waves": waves,
        "policies": policies["policies"],
        "endpoints": report["endpoints"],
        "effect_mae": report["effects"],
        "robustness": {k: robust[k] for k in keep},
        "headlines": {
            "stats": heads["stats"],
            "ci95": heads["ci95"],
            "reliability": heads["human_split_half_reliability_within_arm"],
            "primary": heads["primary"],
        },
        "corrections": {"ok": sum(r["status"] == "ok" for r in policies["corrections"]), "n": len(policies["corrections"])},
        "cost": D1_COST,
    }


def d3() -> dict:
    """D3: the frozen test-split report and the dev-split table."""
    test = json.loads(Path("results/b/report.json").read_text())
    dev = json.loads(Path("results/b/dev-20260923-022128.json").read_text())
    keep = ("blocks", "studies", "effects", "table", "ci95", "differences", "agreement", "primary")
    return {
        "test": {k: test[k] for k in keep},
        "dev": {k: dev[k] for k in ("blocks", "studies", "effects", "table", "differences", "agreement")},
        "cost": D3_COST,
    }


def d2() -> dict:
    d = json.loads(Path("results/d2/discrepancy_model.json").read_text())
    out = {}
    for model in ("jev", "astra"):
        m = d[model]
        out[model] = {
            "fit": {k: m["fit_seen"][k] for k in ("beta", "tau", "sigma", "studies", "effects")},
            "ci": m["ci"],
            "split_half": m["split_half_coverage"],
            "test": {k: m["test"][k] for k in ("effects", "studies", "coverage_with_parameter_uncertainty", "coverage_no_discrepancy")},
        }
    out["jev"]["mc_vs_discrepancy"] = d["jev"]["monte_carlo_versus_discrepancy"]
    return out


def s1() -> dict:
    execute = json.loads(Path("results/c/execute.json").read_text())
    live = json.loads(Path("results/c/20260923-113433-live/throughput.json").read_text())
    return {
        "live": live,
        "throughput": execute["throughput"],
        "crn": execute["crn"],
        "exact": execute["exact"],
        "wind_tunnel": execute["wind_tunnel"],
        "peak_rss_mb": execute["peak_rss_mb"],
    }


def cube() -> dict:
    """Wave-3 application pool of D1: P(share) per persona x card x arm, kernel and D1-corrected, for the in-page simulator."""
    import d1_analysis as d1a
    from d1_epstein import cells_from_control

    jev = d1a.frame(D1_DIR / "20260922-213832-jev")
    jev = jev[jev["kind"] == "share"]
    astra = d1a.frame(D1_DIR / "20260922-225014-gpt-6-astra")
    cells, _ = cells_from_control(jev)
    corrected, _ = d1a.hybrid(jev, astra, cells, 12)
    app = jev[(jev["pool"] == "application") & (jev["wave"] == 3)]
    hyb = corrected[corrected["wave"] == 3]
    personas = sorted(app["id"].unique())
    index = {pid: i for i, pid in enumerate(personas)}

    def grid(frame: pd.DataFrame, arm: str) -> list[list[float]]:
        g = np.full((len(personas), 20), np.nan)
        for r in frame[frame["arm"] == arm].itertuples(index=False):
            g[index[r.id], r.item_num - 1] = r.p_yes
        assert not np.isnan(g).any(), arm
        return np.round(g, 3).tolist()

    return {
        "wave": 3,
        "personas": len(personas),
        "arms": list(CUBE_ARMS),
        "true_cards": [k > 10 for k in range(1, 21)],
        "kernel": {arm: grid(app, arm) for arm in CUBE_ARMS},
        "hybrid": {arm: grid(hyb, arm) for arm in CUBE_ARMS if arm != "control"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    data = {
        "meta": {"commit": commit, "model": "jev-1.13.0"},
        "gaming": gaming(),
        "decision": decision(),
        "arechar": {**arechar_calibration(), "held_out": arechar_held_out(), "flagship": flagship()},
        "b2": b2(),
        "d1": d1(),
        "d3": d3(),
        "d2": d2(),
        "s1": s1(),
        "cube": cube(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB): {len(data['decision']['tasks'])} tasks, "
          f"{len(data['arechar']['items'])} headlines, {len(data['b2']['studies'])} B2 studies")  # fmt: skip


if __name__ == "__main__":
    main()
