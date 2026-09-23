"""D1 robustness of the declared secondary endpoint (effect error): no new endpoint, no new look at outcomes.

    PYTHONPATH=scripts uv run python scripts/d1_robustness.py

Reads the locked tables (results/d1/policies.json), the human tables (results/d1/report.json), the ratings (for the
participant bootstrap) and the prediction files (for the persona-count curve). Writes results/d1/robustness.json.

  1. kernel minus hybrid absolute error, per effect: participant bootstrap (people resampled within wave x arm) and
     effect bootstrap intervals; leave-one-wave-out.
  2. calibrated parents: each system's effects rescaled by a slope fitted on the other four waves (so the comparison is
     against the best single-scale version of the kernel and of the flagship, out of sample).
  3. the human sampling floor: the error a perfect model would show because people's effects are themselves estimates.
  4. the persona-count curve: kernel effect error against the number of simulated respondents, and the hybrid at the
     same counts with the same 1,080 anchors - does compute spent on more respondents or on the correction reduce error?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import d1_analysis as d1  # noqa: E402
from d1_epstein import cells_from_control  # noqa: E402

from kite.eval import epstein  # noqa: E402

POLICIES = Path("results/d1/policies.json")
REPORT = Path("results/d1/report.json")
OUT = Path("results/d1/robustness.json")
SYSTEMS = ("kernel", "hybrid", "flagship_audit")
COUNTS = (10, 20, 30, 50, 100, 200)
DRAWS = 20


def effect_frame(tables: dict, human: dict) -> pd.DataFrame:
    rows = []
    for key in human:
        w, a = key.split(":")
        if a == "control":
            continue
        row = {"wave": int(w), "arm": a, "human": human[key]["discernment"] - human[f"{w}:control"]["discernment"]}
        for s in SYSTEMS:
            row[s] = tables[s][key]["discernment"] - tables[s][f"{w}:control"]["discernment"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["wave", "arm"]).reset_index(drop=True)


def mae(frame: pd.DataFrame, system: str, human: str = "human") -> float:
    return float((frame[system] - frame[human]).abs().mean())


def human_effects(person: pd.DataFrame, waves_arms: list[tuple[int, str]], rng=None) -> np.ndarray:
    """Wave-relative effects from participant discernment, optionally with people resampled within wave x arm."""
    means = {}
    for (wave, arm), g in person.groupby(["wave", "arm"]):
        d = g["d"].to_numpy()
        if rng is not None:
            d = d[rng.integers(0, len(d), len(d))]
        means[(wave, arm)] = d.mean()
    return np.array([means[(w, a)] - means[(w, "control")] for w, a in waves_arms])


def main() -> None:
    locked = json.loads(POLICIES.read_text())
    report = json.loads(REPORT.read_text())
    E = effect_frame(locked["tables"], report["human"])
    waves_arms = list(zip(E["wave"], E["arm"], strict=True))
    out = {"n_effects": len(E), "raw_mae": {s: mae(E, s) for s in SYSTEMS}}

    # 1. intervals for kernel minus hybrid absolute error
    ratings = epstein.load_ratings()
    person = d1.person_discernment(ratings)
    rng = np.random.default_rng(0)
    diffs_people, maes_people = [], {s: [] for s in SYSTEMS}
    for _ in range(2000):
        h = human_effects(person, waves_arms, rng)
        errors = {s: np.abs(E[s].to_numpy() - h) for s in SYSTEMS}
        for s in SYSTEMS:
            maes_people[s].append(errors[s].mean())
        diffs_people.append((errors["kernel"] - errors["hybrid"]).mean())
    d = (E["kernel"] - E["human"]).abs() - (E["hybrid"] - E["human"]).abs()
    diffs_effects = [d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)]
    out["kernel_minus_hybrid_abs_error"] = {
        "point": float(d.mean()),
        "participant_bootstrap_ci95": [float(np.percentile(diffs_people, 2.5)), float(np.percentile(diffs_people, 97.5))],
        "participant_bootstrap_p_le_0": float(np.mean(np.array(diffs_people) <= 0)),
        "effect_bootstrap_ci95": [float(np.percentile(diffs_effects, 2.5)), float(np.percentile(diffs_effects, 97.5))],
        "mae_participant_ci95": {s: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for s, v in maes_people.items()},
    }
    out["leave_one_wave_out"] = {
        int(w): {s: mae(E[E["wave"] != w], s) for s in SYSTEMS}
        | {"relative": mae(E[E["wave"] != w], "hybrid") / mae(E[E["wave"] != w], "kernel") - 1}
        for w in sorted(E["wave"].unique())
    }

    # 2. calibrated parents, out of sample by wave
    calibrated = {}
    for s in SYSTEMS:
        errors, slopes = [], []
        for w in sorted(E["wave"].unique()):
            train, test = E[E["wave"] != w], E[E["wave"] == w]
            slope = float(np.polyfit(train[s], train["human"], 1)[0])
            slopes.append(slope)
            errors.extend((slope * test[s] - test["human"]).abs().tolist())
        calibrated[s] = {"mae_loo_slope": float(np.mean(errors)), "slopes": slopes, "slope_all": float(np.polyfit(E[s], E["human"], 1)[0])}
    out["calibrated_parents"] = calibrated

    # 3. the human sampling floor
    se = []
    for w, a in waves_arms:
        t, c = person[(person["wave"] == w) & (person["arm"] == a)]["d"], person[(person["wave"] == w) & (person["arm"] == "control")]["d"]
        se.append(float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))))
    se = np.array(se)
    floor = float(np.mean(np.sqrt(2 / np.pi) * se))
    rmse = {s: float(np.sqrt(((E[s] - E["human"]) ** 2).mean())) for s in SYSTEMS}
    out["sampling_floor"] = {
        "effect_se_mean": float(se.mean()),
        "mae_floor_if_model_perfect": floor,
        "rmse_observed": rmse,
        "rmse_model_only": {s: float(np.sqrt(max(v**2 - float((se**2).mean()), 0.0))) for s, v in rmse.items()},
    }

    # 4. persona-count curve
    jev = d1.frame(Path(locked["inputs"]["jev"]))
    jev = jev[jev["kind"] == "share"]
    astra = d1.frame(Path(locked["inputs"]["astra"]))
    cells, _ = cells_from_control(jev)
    app = jev[jev["pool"] == "application"]
    people = app[["wave", "id"]].drop_duplicates()
    human = E["human"].to_numpy()

    def effects_of(table: pd.DataFrame) -> np.ndarray:
        return np.array([table.loc[(w, a), "discernment"] - table.loc[(w, "control"), "discernment"] for w, a in waves_arms])

    curve = []
    rng = np.random.default_rng(0)
    for n in COUNTS:
        k_err, h_err, skipped = [], [], []
        for _ in range(1 if n == 200 else DRAWS):
            chosen = pd.concat([g.sample(min(n, len(g)), random_state=int(rng.integers(2**31))) for _, g in people.groupby("wave")])
            subset = jev.merge(chosen, on=["wave", "id"])
            k_err.append(float(np.abs(effects_of(d1.discernment(subset)) - human).mean()))
            corrected, records = d1.hybrid(subset, astra, cells, 12)
            h_err.append(float(np.abs(effects_of(d1.discernment(corrected)) - human).mean()))
            skipped.append(sum(r["status"] != "ok" for r in records))
        curve.append(
            {
                "personas_per_wave": n,
                "kernel_mae": float(np.mean(k_err)),
                "kernel_sd": float(np.std(k_err)),
                "hybrid_mae": float(np.mean(h_err)),
                "hybrid_sd": float(np.std(h_err)),
                "corrections_not_ok": float(np.mean(skipped)),
            }
        )
    out["persona_count_curve"] = curve
    out["flagship_audit_reference"] = {"personas_per_wave": 30, "predictions": 9000, "mae": out["raw_mae"]["flagship_audit"]}

    OUT.write_text(json.dumps(out, indent=1))
    k = out["kernel_minus_hybrid_abs_error"]
    r4 = lambda values: [round(v, 4) for v in values]  # noqa: E731
    line = f"kernel - hybrid abs error {k['point']:+.4f}: people-bootstrap {r4(k['participant_bootstrap_ci95'])}"
    print(line + f" (P<=0 {k['participant_bootstrap_p_le_0']:.3f}); effect-bootstrap {r4(k['effect_bootstrap_ci95'])}")
    print("MAE people-bootstrap CIs:", {s: [round(v, 4) for v in ci] for s, ci in k["mae_participant_ci95"].items()})
    print("leave-one-wave-out:", {w: round(v["relative"], 3) for w, v in out["leave_one_wave_out"].items()})
    print(
        "calibrated parents (LOO slope):",
        {s: round(v["mae_loo_slope"], 4) for s, v in calibrated.items()},
        {s: [round(x, 2) for x in v["slopes"]] for s, v in calibrated.items()},
    )
    f = out["sampling_floor"]
    line = f"sampling floor: mean SE {f['effect_se_mean']:.4f}, MAE floor {f['mae_floor_if_model_perfect']:.4f};"
    observed = {s: round(v, 4) for s, v in rmse.items()}
    model_only = {s: round(v, 4) for s, v in f["rmse_model_only"].items()}
    print(line + f" RMSE observed {observed}; model-only {model_only}")
    print(pd.DataFrame(curve).round(4).to_string(index=False))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
