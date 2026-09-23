"""D0: a sparse LLM correction surface, prototyped on Arechar from predictions that already exist.

Research plan v3 §4.4 and v3.1 (a). Nothing new is queried: the flagship's predictions on the both-new part
(30 respondents per country and condition, C3b) supply the anchors, the kernel's predictions on the same part
(100 per country and condition, C3) are what gets corrected, and the human ratings are the reference.

For each country, intervention and correction cell, the flagship's effect on the sampled anchor states
(treated anchors minus control anchors) becomes the target shift for the kernel: the kernel's treated
distributions in that cell are tilted so that their mean equals the kernel's control mean plus that shift.
Three cell representations are compared - one cell per country, veracity only, veracity x headline baseline
tertile - at 6, 12 and 24 anchors per condition and cell.

Two things are measured. Approximation fidelity: how close the corrected kernel's effect is to the
flagship's own effect on the anchors it did not see. Human fidelity: how close it is to people's effect.
The second is development evidence: the flagship's predictions on this part were already used to reach the
C3b verdict, so nothing here is a frozen test. The frozen test is D1 (Epstein), deferred.

Usage:
    uv run python scripts/d0_correction_prototype.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kite.eval import arechar
from kite.eval.runner import read_predictions
from kite.operators import tilt_cell

CSV = Path("data/arechar/CR.csv")
KERNEL_RUN = Path("results/c2/20260922-011615-held_out")
FLAGSHIP_RAW = Path("results/c3-flagship/20260922-133803-gpt-6-astra/raw.json")
OUT = Path("results/d0/prototype.json")
PART = "both_new"
INTERVENTIONS = ("prompt", "tips")
ANCHORS = (6, 12, 24)
REPRESENTATIONS = ("global", "veracity", "veracity_x_tertile")
VALUES = np.arange(1, 7, dtype=float)
SEEDS = range(5)


def expected(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    return (p / p.sum(axis=-1, keepdims=True)) @ VALUES


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kernel_rows = []
    for p in read_predictions(KERNEL_RUN):
        if p.meta["part"] == PART and p.meta["condition"] in ("share_only", *INTERVENTIONS):
            meta = {k: p.meta[k] for k in ("country", "id", "condition", "item", "true")}
            kernel_rows.append({**meta, "item_id": p.item_id, "probs": np.asarray(p.probs, float)})
    kernel = pd.DataFrame(kernel_rows)
    kernel["expected"] = expected(np.stack(kernel["probs"].to_numpy()))
    raw = json.loads(FLAGSHIP_RAW.read_text())
    flagship = kernel[kernel["item_id"].isin(raw)].drop(columns=["probs", "expected"]).copy()
    flagship["expected"] = expected(np.stack([np.asarray(raw[i], float) for i in flagship["item_id"]]))
    human = arechar.load_ratings(CSV, "held_out")
    human = human[human["part"] == PART].assign(true=lambda d: d["item"].map(arechar.is_true))
    return kernel, flagship, human


def assign_cells(kernel: pd.DataFrame, representation: str) -> pd.Series:
    """The correction cell of every (persona, headline) state; treatment-independent by construction."""
    if representation == "global":
        return pd.Series("all", index=kernel.index)
    veracity = kernel["true"].map({True: "true", False: "false"})
    if representation == "veracity":
        return veracity
    control = kernel[kernel["condition"] == "share_only"]
    baseline = control.groupby("item")["expected"].mean()  # headline-level kernel baseline, pooled over countries
    tertile = {}
    for truth, group in baseline.groupby(baseline.index.map(arechar.is_true)):
        edges = np.quantile(group.to_numpy(), [1 / 3, 2 / 3])
        for item, value in group.items():
            tertile[item] = f"{'true' if truth else 'false'}:{int(np.searchsorted(edges, value, side='right'))}"
    return kernel["item"].map(tertile)


def correct(kernel: pd.DataFrame, flagship: pd.DataFrame, cells: pd.Series, intervention: str, n_anchor: int, rng) -> tuple[pd.DataFrame, dict]:
    """The kernel frame with the intervention's rows tilted, plus the anchor ids and feasibility record."""
    out = kernel.copy()
    fl = flagship.assign(cell=cells.reindex(flagship.index))
    ke = kernel.assign(cell=cells)
    anchors, record = set(), {"cells": 0, "infeasible": 0, "skipped": 0, "weight_uncorrected": 0.0}
    treated_total = int((ke["condition"] == intervention).sum())
    for (country, cell), _ in ke.groupby(["country", "cell"]):
        f = fl[(fl["country"] == country) & (fl["cell"] == cell)]
        control_pool, treated_pool = f[f["condition"] == "share_only"], f[f["condition"] == intervention]
        k_treated = ke[(ke["country"] == country) & (ke["cell"] == cell) & (ke["condition"] == intervention)]
        k_control = ke[(ke["country"] == country) & (ke["cell"] == cell) & (ke["condition"] == "share_only")]
        if len(k_treated) == 0:
            continue
        record["cells"] += 1
        if len(control_pool) < n_anchor or len(treated_pool) < n_anchor or len(k_control) == 0:
            record["skipped"] += 1
            record["weight_uncorrected"] += len(k_treated) / treated_total
            continue
        c_anchor = control_pool.sample(n_anchor, random_state=int(rng.integers(2**31)))
        t_anchor = treated_pool.sample(n_anchor, random_state=int(rng.integers(2**31)))
        anchors |= set(c_anchor["item_id"]) | set(t_anchor["item_id"])
        shift = t_anchor["expected"].mean() - c_anchor["expected"].mean()
        target = k_control["expected"].mean() + shift
        tilted, result = tilt_cell(np.stack(k_treated["probs"].to_numpy()), VALUES, target)
        if not result.feasible:
            record["infeasible"] += 1
            record["weight_uncorrected"] += len(k_treated) / treated_total
            continue
        out.loc[k_treated.index, "expected"] = expected(tilted)
    return out, {**record, "anchors": sorted(anchors)}


def effects(model: pd.DataFrame, human: pd.DataFrame, intervention: str, rng) -> dict:
    e = arechar.treatment_effect(model, human, intervention, rng, n_perm=200, n_boot=200)
    channels = {}
    for truth, name in ((True, "true"), (False, "false")):
        m = model[model["true"] == truth]
        per_country = m.groupby(["country", "condition"])["expected"].mean().unstack("condition")
        channels[name] = float((per_country[intervention] - per_country["share_only"]).mean())
    per_country = {c: v["model"] for c, v in e["per_country"].items()}
    out = {"pooled": e["model_pooled"], "null_p95": e["model_null_p95"], "human": e["human_pooled"], "human_ci": e["human_ci"]}
    return {**out, "per_country": per_country, "channels": channels}


def main() -> None:
    kernel, flagship, human = load_frames()
    rng = np.random.default_rng(0)
    report = {"part": PART, "kernel_rows": len(kernel), "flagship_rows": len(flagship), "reference": {}, "results": {}}
    for intervention in INTERVENTIONS:
        reference = {"kernel": effects(kernel, human, intervention, rng), "flagship_all": effects(flagship, human, intervention, rng)}
        report["reference"][intervention] = reference
    for representation in REPRESENTATIONS:
        cells = assign_cells(kernel, representation)
        for n_anchor in ANCHORS:
            for intervention in INTERVENTIONS:
                runs = []
                for seed in SEEDS:
                    corrected, record = correct(kernel, flagship, cells, intervention, n_anchor, np.random.default_rng(seed))
                    audit = flagship[~flagship["item_id"].isin(record["anchors"])]
                    e_corrected = effects(corrected, human, intervention, rng)
                    e_audit = effects(audit, human, intervention, rng)
                    summary = {k: v for k, v in record.items() if k != "anchors"}
                    run = {"seed": seed, "corrected": e_corrected, "flagship_audit": e_audit, "record": summary}
                    runs.append({**run, "n_anchor_states": len(record["anchors"])})
                key = f"{representation}/{n_anchor}/{intervention}"
                pooled = np.array([r["corrected"]["pooled"] for r in runs])
                audit_pooled = np.array([r["flagship_audit"]["pooled"] for r in runs])
                report["results"][key] = {
                    "runs": runs,
                    "corrected_pooled_mean": float(pooled.mean()),
                    "corrected_pooled_sd": float(pooled.std(ddof=1)),
                    "approximation_error": float(np.mean(np.abs(pooled - audit_pooled))),
                    "human_error": float(np.mean(np.abs(pooled - runs[0]["corrected"]["human"]))),
                    "anchor_states": float(np.mean([r["n_anchor_states"] for r in runs])),
                    "uncorrected_weight": float(np.mean([r["record"]["weight_uncorrected"] for r in runs])),
                }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=float))

    for intervention in INTERVENTIONS:
        ref = report["reference"][intervention]
        ci = [round(x, 3) for x in ref["kernel"]["human_ci"]]
        kernel_line = f"kernel {ref['kernel']['pooled']:+.3f} (null p95 {ref['kernel']['null_p95']:+.3f})"
        flagship_line = f"flagship on all 30/cell {ref['flagship_all']['pooled']:+.3f}"
        print(f"\n{intervention}: people {ref['kernel']['human']:+.3f} {ci}; {kernel_line}; {flagship_line}")
        print(f"  {'representation':20}{'anchors':>8}{'states':>8}{'corrected':>11}{'sd':>7}{'|Δ flagship|':>14}{'|Δ people|':>12}{'uncorr.w':>10}")
        for representation in REPRESENTATIONS:
            for n_anchor in ANCHORS:
                r = report["results"][f"{representation}/{n_anchor}/{intervention}"]
                head = f"  {representation:20}{n_anchor:>8}{r['anchor_states']:>8.0f}"
                head += f"{r['corrected_pooled_mean']:>+11.3f}{r['corrected_pooled_sd']:>7.3f}"
                print(head + f"{r['approximation_error']:>14.3f}{r['human_error']:>12.3f}{r['uncorrected_weight']:>10.2f}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
