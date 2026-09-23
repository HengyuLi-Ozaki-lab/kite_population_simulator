"""B: paired flagship anchors on SocSci210 - k personas per task, each predicted under every condition of the task.

The multi-fidelity claim in one experiment: the kernel predicts every state of the population; the flagship is
called only on a few anchor personas per task, under every condition of that task (paired, as in D1), and the
kernel's condition effects are replaced by the flagship's paired effects. Development on the 91 seen studies
(kernel and flagship 5-per-cell runs exist), then one frozen evaluation on the 37 test studies.

    PYTHONPATH=scripts uv run python scripts/b_paired_anchors.py count   --split dev|test
    PYTHONPATH=scripts uv run python scripts/b_paired_anchors.py astra   --split dev|test [--out <dir>]   # Codex, resumable
    PYTHONPATH=scripts uv run python scripts/b_paired_anchors.py analyse --split dev  --astra <dir>          # exploration
    PYTHONPATH=scripts uv run python scripts/b_paired_anchors.py analyse --split test --astra <dir>          # frozen; needs the criteria

Anchor personas are the first K respondents of a (study, task) block in hash order, nested, so k = 1, 2 are
subsets of k = 3. Each anchor keeps its own demographics and is shown the stimulus of every condition of the
block; the stimulus of a condition is the one shown to the block's first respondent (hash order) in that cell.
Batches never hold two items of one task (a3_codex_baseline.make_batches), so no persona meets two conditions
in one call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from a3_codex_baseline import PROMPT, make_batches, run
from a3c_seen_runs import seen_task

from kite.eval import decision_value as dv
from kite.eval import metrics
from kite.eval.runner import read_predictions, write_predictions
from kite.eval.scales import Scale, strip_response_instruction
from kite.eval.socsci210 import SocSci210Task
from kite.eval.task import Prediction
from kite.population.persona import Persona

CRITERIA = Path("configs/eval/socsci210_paired_anchors.yaml")
GUARDED = (str(CRITERIA), "scripts/b_paired_anchors.py", "scripts/a3_codex_baseline.py")
OUT = Path("results/b")
K = 3
SEED = 0
ASTRA = ("gpt-6-astra", "high")
PREPARED = {"dev": Path("data/socsci210/prepared/a3c-seen"), "test": Path("data/socsci210/prepared/test")}
KERNEL = {"dev": "results/a3c/20260922-155916-seen-jev", "test": "results/20260921-012513-socsci210-test-jev-p3-choice"}
FLAGSHIP5 = {"dev": "results/a3c/20260922-155915-seen-astra", "test": "results/20260922-130953-socsci210-test-codex-gpt-6-astra-high"}


def committed(*paths: str) -> bool:
    return not subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()


def task_for(split: str) -> SocSci210Task:
    return seen_task("p1") if split == "dev" else SocSci210Task(PREPARED[split], phrasing="p1", primitive="choice")


def anchor_entries(task: SocSci210Task, k: int = K, seed: int = SEED) -> tuple[list[dict], dict[str, tuple[dict, Scale]], dict]:
    """Codex entries for every (block, anchor persona, condition), their meta and a few counts for the config."""
    frame = task._frame
    frame = frame.assign(_row_order=[zlib.crc32(f"{seed}:{s}".encode()) for s in frame["sample_id"]])
    comparable = dv.comparable_tasks(task._scales)
    entries, meta = [], {}
    stimuli_variants, blocks_used = [], 0
    for (study, task_num), conditions in sorted(comparable.items()):
        block = frame[(frame["study_id"] == study) & (frame["task_num"] == task_num)]
        present = [c for c in conditions if (block["condition_num"] == c).any()]
        if len(present) < 2:
            continue
        blocks_used += 1
        people = block.drop_duplicates("participant")[["participant", "demographic"]]
        people = people.assign(_order=[zlib.crc32(f"{seed}:{study}:{p}".encode()) for p in people["participant"]]).sort_values("_order").head(k)
        stimulus = {}
        for c in present:
            cell = block[block["condition_num"] == c].sort_values("_row_order")
            stimulus[c] = cell["stimuli"].iloc[0]
            stimuli_variants.append(int(cell["stimuli"].nunique()))
        for rank, person in enumerate(people.itertuples(index=False)):
            persona = Persona.from_socsci210(json.loads(person.demographic)).render()
            for c in present:
                scale = task._scale(study, int(c), int(task_num))
                item_id = f"{study}:{task_num}:{c}:{person.participant}"
                info = {"study_id": study, "condition_num": int(c), "task_num": int(task_num), "participant": int(person.participant)}
                meta[item_id] = ({**info, "anchor_rank": rank, "n_levels": scale.n_levels}, scale)
                entry = {"item_id": item_id, "task": (study, int(task_num)), "persona": persona}
                entries.append({**entry, "question": strip_response_instruction(stimulus[c]), "options": scale.descriptions()})
    several = int(sum(v > 1 for v in stimuli_variants))
    counts = {"blocks": blocks_used, "items": len(entries), "cells_with_several_stimuli": several, "cells": len(stimuli_variants)}
    return entries, meta, counts


def stage_astra(args) -> None:
    task = task_for(args.split)
    entries, meta, counts = anchor_entries(task)
    batches = make_batches(entries)
    out = args.out or OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{args.split}-paired-{ASTRA[0]}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "raw.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    ledger = {"calls": 0, "failed_calls": 0, "tokens": 0, "call_seconds": []}
    print(f"{counts} in {len(batches)} batches; {len(done)} done; -> {out}")
    started = time.monotonic()
    asyncio.run(run(batches, done, cache, ledger, *ASTRA))
    predictions = [Prediction(item_id=i, probs=meta[i][1].expand(done[i]), meta=meta[i][0]) for i in meta if i in done]
    write_predictions(out, predictions)
    seconds = ledger.pop("call_seconds")
    ledger |= {"wall_seconds": time.monotonic() - started, "median_call_seconds": float(np.median(seconds)) if seconds else None}
    ledger |= {"items": len(predictions), "missing": len(meta) - len(predictions)}
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    config = {"split": args.split, "k": K, "seed": SEED, **counts, "kernel": KERNEL[args.split], "flagship5": FLAGSHIP5[args.split]}
    config |= {"model": ASTRA[0], "effort": ASTRA[1], "prompt": PROMPT}
    (out / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print(json.dumps(ledger, indent=1))


# ------------------------------------------------ analysis (importable) ------------------------------------------------


def paired_shifts(predictions, k: int) -> dict[tuple[str, int], dict]:
    """Per block: the flagship's paired effect of every condition against the block's first condition, over the first k anchors."""
    by_block: dict[tuple[str, int], dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))  # block -> person -> cond -> pos
    for p in predictions:
        m = p.meta
        if m["anchor_rank"] < k:
            by_block[(m["study_id"], m["task_num"])][m["participant"]][m["condition_num"]] = metrics.mean_position(p.probs)
    shifts = {}
    for block, people in by_block.items():
        conditions = sorted({c for pos in people.values() for c in pos})
        ref = conditions[0]
        out = {}
        for c in conditions[1:]:
            diffs = [pos[c] - pos[ref] for pos in people.values() if c in pos and ref in pos]
            if diffs:
                out[c] = float(np.mean(diffs))
        shifts[block] = {"ref": ref, "shift": out, "anchors": len(people)}
    return shifts


def hybrid_means(kernel: dict[str, float], shifts: dict, s: float) -> dict[str, float]:
    """Cell means of the hybrid: the kernel's reference-condition mean plus s times the flagship's paired shift."""
    out = {}
    for (study, task), block in shifts.items():
        ref_key = f"{study}|{block['ref']}|{task}"
        if ref_key not in kernel:
            continue
        out[ref_key] = kernel[ref_key]
        for c, shift in block["shift"].items():
            out[f"{study}|{c}|{task}"] = kernel[ref_key] + s * shift
    return out


def load_split(split: str):
    prepared = PREPARED[split]
    scales = {k: Scale(**v) for k, v in json.loads((prepared / "scales.json").read_text()).items()}
    rows = pd.read_parquet(prepared / "rows.parquet", columns=["study_id", "participant", "condition_num", "task_num", "response"])
    return dv.human_cells(rows, scales), dv.comparable_tasks(scales)


def effects(blocks):
    """Effects against each block's first condition: (predicted, human, study) arrays."""
    P, H, S = [], [], []
    for b in blocks:
        P.extend(b.predicted[1:] - b.predicted[0])
        H.extend(b.human[1:] - b.human[0])
        S.extend([b.study] * (len(b.human) - 1))
    return np.array(P), np.array(H), np.array(S)


def centred_r(blocks, weights) -> float:
    p = np.concatenate([b.predicted - b.predicted.mean() for b in blocks])
    h = np.concatenate([b.human - b.human.mean() for b in blocks])
    w = np.concatenate([np.full(len(b.predicted), weights[i]) for i, b in enumerate(blocks)])
    return float((w * p * h).sum() / np.sqrt((w * p * p).sum() * (w * h * h).sum()))


def weighted_r(x, y, w) -> float:
    x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
    return float((w * x * y).sum() / np.sqrt((w * x * x).sum() * (w * y * y).sum()))


def reliable_hits(blocks) -> np.ndarray:
    hits = []
    for b in blocks:
        a, c, _, z = dv._pairs(b.human, b.se)
        hits.append(dv._agreement(b.predicted, b.human, a, c)[np.abs(z) >= dv.RELIABLE_Z])
    return np.concatenate(hits)


class Systems:
    """Cell means of every system on one split, restricted to the blocks all of them cover."""

    def __init__(self, split: str, astra_run: Path, ks=(1, 2, 3), s: float = 1.0):
        self.human, comparable = load_split(split)
        kernel = dv.predicted_means(read_predictions(KERNEL[split]))
        flagship = dv.predicted_means(read_predictions(FLAGSHIP5[split]))
        anchors = list(read_predictions(astra_run))
        flagship3 = dv.predicted_means(read_predictions(FLAGSHIP5[split]), n_per_cell=3)  # unpaired, equal budget to k = 3
        means = {"kernel": kernel, "flagship5": flagship, "flagship3": flagship3}
        for k in ks:
            means[f"hybrid_k{k}"] = hybrid_means(kernel, paired_shifts(anchors, k), s)
        # every system is scored on exactly the same cells: a block keeps the conditions all systems predict, and needs two
        shared = {}
        for (study, task), conditions in comparable.items():
            kept = [c for c in conditions if all(f"{study}|{c}|{task}" in m for m in means.values())]
            if len(kept) >= 2:
                shared[(study, task)] = kept
        blocks = {name: dv.build_blocks(shared, self.human, m) for name, m in means.items()}
        common = set.intersection(*[{(b.study, b.task) for b in v} for v in blocks.values()])
        self.blocks = {name: [b for b in v if (b.study, b.task) in common] for name, v in blocks.items()}
        for name, v in self.blocks.items():
            assert [(b.study, b.task, len(b.cells)) for b in v] == [(b.study, b.task, len(b.cells)) for b in self.blocks["kernel"]], name
        self.studies, self.member = np.unique([b.study for b in self.blocks["kernel"]], return_inverse=True)
        self.s = s

    def scale(self, name: str, s: float):
        """Blocks of a system with its effects rescaled by s (the reference condition unchanged)."""
        out = []
        for b in self.blocks[name]:
            out.append(dv.TaskBlock(b.study, b.task, b.cells, b.predicted[0] + s * (b.predicted - b.predicted[0]), b.human, b.se))
        return out

    def summary(self, name: str, w=None, s: float | None = None) -> dict[str, float]:
        blocks = self.blocks[name] if s is None else self.scale(name, s)
        w = np.ones(len(blocks)) if w is None else w
        P, H, _ = effects(blocks)
        wp = np.concatenate([np.full(len(b.human) - 1, w[i]) for i, b in enumerate(blocks)])
        parts = dv.evaluate(blocks).summary(w)
        return {
            "r_within_task": centred_r(blocks, w),
            "sign_accuracy": parts["sign_accuracy"],
            "captured_gain": parts["captured_gain"],
            "effect_mae": float(np.average(np.abs(P - H), weights=wp)),
            "magnitude_ratio": float((wp * np.abs(P)).sum() / (wp * np.abs(H)).sum()),
        }

    def approximation(self, name: str, w=None) -> float:
        """Pearson r between a system's effects and the 5-per-cell flagship run's effects (how well it approximates the flagship)."""
        blocks, flag = self.blocks[name], self.blocks["flagship5"]
        w = np.ones(len(blocks)) if w is None else w
        P, _, _ = effects(blocks)
        F, _, _ = effects(flag)
        wp = np.concatenate([np.full(len(b.human) - 1, w[i]) for i, b in enumerate(blocks)])
        return weighted_r(P, F, wp)

    def agreement(self, a: str, b: str) -> dict[str, float]:
        """On reliable contrasts: how often a and b order them alike, and each one's accuracy when they agree and when they disagree."""
        ha, hb = reliable_hits(self.blocks[a]), reliable_hits(self.blocks[b])
        ok = np.isfinite(ha) & np.isfinite(hb)
        ha, hb = ha[ok], hb[ok]
        agree = (ha == hb) & (ha != 0.5)
        return {
            "contrasts": int(ok.sum()),
            "agree_share": float(agree.mean()),
            "accuracy_when_agree": float(ha[agree].mean()) if agree.any() else float("nan"),
            f"{a}_accuracy_when_disagree": float(ha[~agree].mean()) if (~agree).any() else float("nan"),
            f"{b}_accuracy_when_disagree": float(hb[~agree].mean()) if (~agree).any() else float("nan"),
        }

    def bootstrap_weights(self, rng):
        return np.bincount(rng.integers(0, len(self.studies), len(self.studies)), minlength=len(self.studies))[self.member].astype(float)


def dev_slope(systems: Systems, name: str) -> dict[str, float]:
    """OLS slope of human effects on a system's effects, on all blocks and under leave-one-study-out."""
    blocks = systems.blocks[name]
    P, H, S = effects(blocks)
    full = float(np.polyfit(P, H, 1)[0])
    loo = []
    for study in np.unique(S):
        keep = S != study
        loo.append(float(np.polyfit(P[keep], H[keep], 1)[0]))
    return {"slope": full, "loo_mean": float(np.mean(loo)), "loo_sd": float(np.std(loo))}


MEASURES = ("r_within_task", "sign_accuracy", "captured_gain", "effect_mae", "approx_r_vs_flagship5")
CALIBRATED = ("kernel", "flagship5", "hybrid_k3")


def stage_analyse(args) -> None:
    frozen = yaml.safe_load(CRITERIA.read_text()) if args.split == "test" else None
    systems = Systems(args.split, args.astra)  # raw effects; the dev slopes enter only the calibrated-MAE rows
    n_boot = frozen["analysis"]["n_boot"] if frozen else 2000
    rng = np.random.default_rng(0)
    names = ["kernel", "flagship5", "flagship3", "hybrid_k1", "hybrid_k2", "hybrid_k3"]
    table = {n: {**systems.summary(n), "approx_r_vs_flagship5": systems.approximation(n)} for n in names}
    slopes = {n: dev_slope(systems, n) for n in CALIBRATED}  # on the test split these are descriptive; the frozen ones are used
    used = {n: (frozen["hybrid"]["dev_slopes"][n] if frozen else slopes[n]["slope"]) for n in CALIBRATED}
    for n in CALIBRATED:
        table[f"{n} x dev slope"] = systems.summary(n, s=used[n])
    boot = {n: {m: [] for m in MEASURES} for n in names}
    for _ in range(n_boot):
        w = systems.bootstrap_weights(rng)
        for n in names:
            sm = systems.summary(n, w)
            for m in MEASURES[:4]:
                boot[n][m].append(sm[m])
            boot[n]["approx_r_vs_flagship5"].append(systems.approximation(n, w))
    boot = {n: {m: np.array(v) for m, v in d.items()} for n, d in boot.items()}

    def diff(a, b, m, level=95):
        d = boot[a][m] - boot[b][m]
        lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
        return {"point": table[a][m] - table[b][m], "ci": [float(np.percentile(d, lo)), float(np.percentile(d, hi))], "level": level}

    def ci(a, m, level=95):
        lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
        return [float(np.percentile(boot[a][m], lo)), float(np.percentile(boot[a][m], hi))]

    differences = {}
    for m in MEASURES:
        for n in ("hybrid_k1", "hybrid_k2", "hybrid_k3"):
            differences[f"{n}-kernel:{m}"] = diff(n, "kernel", m)
            differences[f"{n}-flagship5:{m}"] = diff(n, "flagship5", m, 90)
        differences[f"flagship5-kernel:{m}"] = diff("flagship5", "kernel", m)
        differences[f"hybrid_k3-flagship3:{m}"] = diff("hybrid_k3", "flagship3", m)
    report = {
        "split": args.split,
        "astra_run": str(args.astra),
        "kernel_run": KERNEL[args.split],
        "flagship5_run": FLAGSHIP5[args.split],
        "blocks": len(systems.blocks["kernel"]),
        "studies": len(systems.studies),
        "effects": int(sum(len(b.human) - 1 for b in systems.blocks["kernel"])),
        "slopes_fitted_here": slopes,
        "slopes_used_for_calibrated_mae": used,
        "table": table,
        "ci95": {n: {m: ci(n, m) for m in MEASURES} for n in names},
        "differences": differences,
        "agreement": {f"kernel~{n}": systems.agreement("kernel", n) for n in ("flagship5", "hybrid_k3", "hybrid_k1")},
    }
    if frozen:
        name = f"hybrid_k{frozen['hybrid']['k']}"
        p = frozen["primary_endpoint"]
        gain = differences[f"{name}-kernel:approx_r_vs_flagship5"]
        approx = table[name]["approx_r_vs_flagship5"]
        passed = gain["point"] > 0 and gain["ci"][0] > 0 and approx >= p["approx_r_at_least"]
        report["primary"] = {
            "system": name,
            "approx_r": approx,
            "approx_r_ci95": report["ci95"][name]["approx_r_vs_flagship5"],
            "approx_gain_over_kernel": gain,
            "approx_r_at_least": p["approx_r_at_least"],
            "verdict": "PASS" if passed else "FAIL",
        }
    out = args.out or (OUT / (f"dev-{datetime.now():%Y%m%d-%H%M%S}.json" if args.split == "dev" else "report.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"{args.split}: {report['blocks']} blocks, {report['studies']} studies, {report['effects']} effects")
    print(pd.DataFrame(table).T.round(3).to_string())
    for key, d in report["differences"].items():
        print(f"  {key:40} {d['point']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] ({d['level']}%)")
    for key, d in report["agreement"].items():
        print(f"  {key}: {json.dumps({k: round(v, 3) for k, v in d.items()})}")
    print("  slopes fitted here:", json.dumps({n: {k: round(v, 3) for k, v in d.items()} for n, d in slopes.items()}), "used:", used)
    if frozen:
        print("  PRIMARY:", json.dumps(report["primary"]))
    print(f"written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["count", "astra", "analyse"])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--astra", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.split == "test" and args.stage != "count" and not committed(*GUARDED):
        raise SystemExit("refusing to run on the test split: commit the criteria file and the scripts first")
    if args.stage == "count":
        entries, _, counts = anchor_entries(task_for(args.split))
        print(json.dumps(counts), f"batches {len(make_batches(entries))}")
    elif args.stage == "astra":
        stage_astra(args)
    else:
        if args.astra is None:
            raise SystemExit("--astra <anchor run dir> is required")
        stage_analyse(args)


if __name__ == "__main__":
    main()
