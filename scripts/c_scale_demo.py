"""C (research plan v4 §3): the scale demonstration - live kernel throughput, then execution from the table.

    set -a && . ./.env && set +a && PYTHONPATH=scripts uv run python scripts/c_scale_demo.py live
    PYTHONPATH=scripts uv run python scripts/c_scale_demo.py execute --live results/c/<live run>

live: wave 3 of Epstein, the 500 completed participants ranked 231-730 in the task's hash order (disjoint from D1's 230),
under the control and the tips arm, 20 cards -> 20,000 states that no earlier run predicted. Records wall time, achieved
requests per minute, latency p50/p95, dollars and dollars per 1,000 predictions (ledger.json in the run directory).

execute: those 20,000 states as a table; a population of N agents, agent i carrying persona i mod 500, deciding on each
of the 20 cards (20 steps) under both arms with one event-keyed uniform per (agent, step) shared across arms (common
random numbers). Measures agent-steps per second and peak memory at N = 10^4, 10^5, 10^6 (vectorised tier) and 10^4
(scalar tier); the Rao-Blackwell expectation; the variance of the treatment effect with and without common random
numbers; and a wind-tunnel report for the fresh population: discernment by arm, the kernel's tips effect, the same
effect after the D1 correction (the 1,080 flagship anchors already paid for, applied through the frozen cells - zero new
flagship calls), Monte Carlo spread over seeds, and the D2 discrepancy interval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kite.config import Settings
from kite.engine import tabulated, vectorized
from kite.eval import epstein
from kite.eval.runner import read_predictions, run_task
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger
from kite.operators.tilt import tilt_cell

OUT = Path("results/c")
WAVE, ARMS, SKIP, FRESH = 3, ["control", "tips"], 230, 500
D1_POLICIES = Path("results/d1/policies.json")
D1_ASTRA_CONFIG = Path("results/d1/20260922-225014-gpt-6-astra/config.yaml")
D2 = {"beta": 0.925, "tau": 0.037, "sigma": 0.080}  # docs/decisions/D2.md, kernel, fit on the seen studies
SIZES = (10_000, 100_000, 1_000_000)
VALUES = np.array([0.0, 1.0])


def fresh_task() -> epstein.EpsteinTask:
    task = epstein.EpsteinTask(personas_per_wave=SKIP + FRESH, audit_per_wave=0, arms={WAVE: ARMS})
    people = task.people[task.people["wave"] == WAVE].iloc[SKIP : SKIP + FRESH]
    task.people = people.reset_index(drop=True)
    return task


def stage_live(args) -> None:
    task = fresh_task()
    out = args.out or OUT / f"{datetime.now():%Y%m%d-%H%M%S}-live"
    settings, ledger = Settings(), Ledger()
    built = build_kernel(KernelSpec(backend="jev", max_usd=args.max_usd), settings, ledger)
    config = {
        "task": {"name": "epstein-fresh", "wave": WAVE, "arms": ARMS, "personas": FRESH, "ranks": f"{SKIP + 1}-{SKIP + FRESH}"},
        "kernel": "jev p3",
    }
    config |= {"settings": {"rpm": settings.rpm, "max_concurrency": settings.max_concurrency}}
    started = time.monotonic()

    async def go():
        try:
            return await run_task(task, built.kernel, out, ledger=ledger, config=config, concurrency=settings.max_concurrency)
        finally:
            await built.aclose()

    summary = asyncio.run(go())
    wall = time.monotonic() - started
    led = json.loads((out / "ledger.json").read_text())
    n = summary.n_predicted
    throughput = {
        "predictions": n,
        "wall_seconds": wall,
        "predictions_per_minute": 60 * n / wall if wall else None,
        "calls_per_minute": 60 * led.get("calls", 0) / wall if wall else None,
        "cache_hits": led.get("cache_hits"),
        "latency_ms_p50": led.get("latency_ms_p50"),
        "latency_ms_p95": led.get("latency_ms_p95"),
        "usd": led.get("usd"),
        "usd_per_1k": 1000 * led.get("usd", 0) / n if n else None,
        "failed": summary.n_failed,
        "stopped": summary.stopped,
    }
    (out / "throughput.json").write_text(json.dumps(throughput, indent=1))
    print(json.dumps(throughput, indent=1))


def load_states(live: Path) -> tuple[list[int], np.ndarray, dict]:
    """Personas, the (persona, card, arm) -> P(yes) cube and the veracity of each card."""
    rows = [{**p.meta, "p_yes": float(p.probs[1] / max(sum(p.probs), 1e-12))} for p in read_predictions(live)]
    frame = pd.DataFrame(rows)
    personas = sorted(frame["id"].unique())
    cube = np.full((len(personas), 20, len(ARMS)), np.nan)
    index = {pid: i for i, pid in enumerate(personas)}
    for r in frame.itertuples(index=False):
        cube[index[r.id], r.item_num - 1, ARMS.index(r.arm)] = r.p_yes
    if np.isnan(cube).any():
        raise SystemExit(f"{int(np.isnan(cube).sum())} states missing from {live}")
    truth = {int(k): bool(v) for k, v in frame.drop_duplicates("item_num").set_index("item_num")["true"].items()}
    return personas, cube, truth


def corrected_cube(cube: np.ndarray, truth: dict) -> tuple[np.ndarray, list[dict]]:
    """The D1 correction applied to the fresh population: the frozen cell edges and the recorded per-cell flagship shifts."""
    edges = yaml.safe_load(D1_ASTRA_CONFIG.read_text())["cell_edges"]
    shifts = {
        r["cell"]: r["shift"]
        for r in json.loads(D1_POLICIES.read_text())["corrections"]
        if r["wave"] == WAVE and r["arm"] == "tips" and r["status"] == "ok"
    }
    out, records = cube.copy(), []
    control, tips = cube[:, :, 0], cube[:, :, 1]
    for veracity in (False, True):
        cards = np.array([truth[k] == veracity for k in range(1, 21)])
        lo, hi = edges[f"{WAVE}:{'true' if veracity else 'false'}"]
        tertile = np.where(control <= lo, 0, np.where(control <= hi, 1, 2))
        for t in range(3):
            mask = cards[None, :] & (tertile == t)
            cell = f"{'true' if veracity else 'false'}:{t}"
            target = float(control[mask].mean() + shifts[cell])
            probs = np.stack([1 - tips[mask], tips[mask]], axis=1)
            tilted, result = tilt_cell(probs, VALUES, target)
            records.append(
                {
                    "cell": cell,
                    "states": int(mask.sum()),
                    "shift": shifts[cell],
                    "requested": target,
                    "achieved": result.achieved,
                    "feasible": result.feasible,
                }
            )
            if result.feasible:
                out[:, :, 1][mask] = tilted[:, 1]
    return out, records


def discernment(yes: dict[str, np.ndarray], truth: dict) -> dict[str, float]:
    """yes[arm] = per-card share of yes answers (20,) -> discernment = mean on true cards minus mean on false cards."""
    true_cards = np.array([truth[k] for k in range(1, 21)])
    return {arm: float(v[true_cards].mean() - v[~true_cards].mean()) for arm, v in yes.items()}


def execute_vectorised(cube: np.ndarray, n_agents: int, seed: int, *, crn: bool = True) -> tuple[dict[str, np.ndarray], float]:
    """N agents x 20 steps x both arms; returns per-arm per-card yes shares and the wall seconds."""
    n_personas = cube.shape[0]
    cumulative = vectorized.cumulative_table(np.stack([1 - cube.reshape(-1), cube.reshape(-1)], axis=1))  # row = persona*40 + card*2 + arm
    base = (np.arange(n_agents) % n_personas) * 40
    yes = {arm: np.zeros(20) for arm in ARMS}
    started = time.perf_counter()
    for step in range(20):
        states = {arm: base + step * 2 + a for a, arm in enumerate(ARMS)}
        if crn:
            counts = vectorized.run(cumulative, states, seed=seed, step=step)
        else:
            counts = {
                arm: vectorized.run(cumulative, {arm: rows}, seed=seed, step=step, event=a)[arm] for a, (arm, rows) in enumerate(states.items())
            }
        for arm in ARMS:
            yes[arm][step] = counts[arm][1] / n_agents
    return yes, time.perf_counter() - started


def execute_scalar(cube: np.ndarray, n_agents: int, seed: int) -> tuple[dict[str, np.ndarray], float]:
    n_personas = cube.shape[0]
    table = tabulated.Table(["No", "Yes"])
    for p in range(n_personas):
        for card in range(20):
            for a, arm in enumerate(ARMS):
                table.put(f"{p}:{card}:{arm}", [1 - cube[p, card, a], cube[p, card, a]])
    yes = {arm: np.zeros(20) for arm in ARMS}
    started = time.perf_counter()
    for step in range(20):
        agents = ((i, 1.0, {arm: f"{i % n_personas}:{step}:{arm}" for arm in ARMS}) for i in range(n_agents))
        counts = tabulated.run(table, agents, ARMS, seed=seed, step=step)
        for arm in ARMS:
            yes[arm][step] = counts[arm][1] / n_agents
    return yes, time.perf_counter() - started


def stage_execute(args) -> None:
    personas, cube, truth = load_states(args.live)
    hybrid, records = corrected_cube(cube, truth)
    report = {"live": str(args.live), "personas": len(personas), "states": int(cube.size), "correction": records}

    # exact (Rao-Blackwell) discernment per arm, kernel and corrected
    exact = {}
    for name, c in (("kernel", cube), ("hybrid", hybrid)):
        yes = {arm: c[:, :, a].mean(axis=0) for a, arm in enumerate(ARMS)}
        d = discernment(yes, truth)
        exact[name] = {**d, "effect": d["tips"] - d["control"]}
    report["exact"] = exact

    # throughput
    rows = []
    for n in SIZES:
        yes, seconds = execute_vectorised(cube, n, seed=0)
        d = discernment(yes, truth)
        rows.append(
            {
                "tier": "vectorised",
                "agents": n,
                "agent_steps": n * 20,
                "seconds": seconds,
                "agent_steps_per_second": n * 20 / seconds,
                "effect": d["tips"] - d["control"],
            }
        )
    yes, seconds = execute_scalar(cube, SIZES[0], seed=0)
    d = discernment(yes, truth)
    rows.append(
        {
            "tier": "scalar",
            "agents": SIZES[0],
            "agent_steps": SIZES[0] * 20,
            "seconds": seconds,
            "agent_steps_per_second": SIZES[0] * 20 / seconds,
            "effect": d["tips"] - d["control"],
        }
    )
    report["throughput"] = rows
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    # common random numbers: the treatment effect's spread over seeds with shared versus independent uniforms
    effects = {"crn": [], "independent": []}
    for seed in range(100):
        for mode, crn in (("crn", True), ("independent", False)):
            yes, _ = execute_vectorised(cube, SIZES[0], seed=seed, crn=crn)
            d = discernment(yes, truth)
            effects[mode].append(d["tips"] - d["control"])
    report["crn"] = {
        "agents": SIZES[0],
        "seeds": 100,
        "effect_sd_crn": float(np.std(effects["crn"])),
        "effect_sd_independent": float(np.std(effects["independent"])),
    }

    # the wind-tunnel report for the fresh population
    mc = []
    for seed in range(10):
        yes, _ = execute_vectorised(hybrid, SIZES[-1], seed=seed)
        d = discernment(yes, truth)
        mc.append(d["tips"] - d["control"])
    half = 1.645 * np.sqrt(D2["tau"] ** 2 + D2["sigma"] ** 2)
    report["wind_tunnel"] = {
        "population": SIZES[-1],
        "kernel_effect_exact": exact["kernel"]["effect"],
        "hybrid_effect_exact": exact["hybrid"]["effect"],
        "hybrid_effect_mc_sd_1e6": float(np.std(mc)),
        "d2_interval_90_for_people": {
            name: [D2["beta"] * exact[name]["effect"] - half, D2["beta"] * exact[name]["effect"] + half] for name in ("kernel", "hybrid")
        },
        "note": "kernel-only and D1-corrected effects of the tips screen on wave-3 personas never seen by the flagship; "
        "the interval is the D2 discrepancy model, not a claim about these people",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "execute.json").write_text(json.dumps(report, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))
    print("exact:", json.dumps({k: {m: round(v, 4) for m, v in d.items()} for k, d in exact.items()}))
    print("crn:", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in report["crn"].items()}))
    print("peak RSS MB:", round(report["peak_rss_mb"], 1))
    print("wind tunnel:", json.dumps(report["wind_tunnel"], indent=1))
    print(f"correction: {sum(r['feasible'] for r in records)}/{len(records)} cells feasible")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["live", "execute"])
    parser.add_argument("--live", type=Path, default=None, help="the live run directory (execute stage)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-usd", type=float, default=0.7)
    args = parser.parse_args()
    if args.stage == "live":
        stage_live(args)
    else:
        if args.live is None:
            raise SystemExit("--live <run dir> is required")
        stage_execute(args)


if __name__ == "__main__":
    main()
