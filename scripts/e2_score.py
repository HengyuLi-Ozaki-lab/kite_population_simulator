"""E2: score a revealed forecast against a study's results, with the rules fixed in forecasts/README.md.

    uv run python scripts/e2_score.py <forecast_id>

Reads forecasts/revealed/<id>.json (checked against its LEDGER.jsonl digest) and forecasts/results/<id>.json (the
study's observed effects with their sources), and writes forecasts/scores/<id>.json. Rules: sign accuracy on reliable
contrasts (|z| >= 3 in the study's own data), Spearman correlation of the within-study effects, coverage of the 80% and
90% intervals, mean absolute error on the unit scale, and whether the largest predicted effect is the largest observed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from kite.ledger import verify

ROOT = Path("forecasts")


def score(predicted: dict, observed: dict[str, float], reliable: set[str], ranked: list[str]) -> dict:
    ids = [i for i in observed if i in predicted]
    err = [abs(predicted[i]["predicted_effect"] - observed[i]) for i in ids]
    in80 = [predicted[i]["interval80"][0] <= observed[i] <= predicted[i]["interval80"][1] for i in ids]
    in90 = [predicted[i]["interval90"][0] <= observed[i] <= predicted[i]["interval90"][1] for i in ids]
    signs = [np.sign(predicted[i]["predicted_effect"]) == np.sign(observed[i]) for i in ids if i in reliable]
    rho = spearmanr([predicted[i]["predicted_effect"] for i in ranked], [observed[i] for i in ranked]).statistic if len(ranked) > 2 else None
    top_pred = max(ranked, key=lambda i: predicted[i]["predicted_effect"])
    top_obs = max(ranked, key=lambda i: observed[i])
    return {
        "effects": len(ids),
        "sign_accuracy_reliable": f"{sum(signs)}/{len(signs)}",
        "spearman_within_study": None if rho is None else round(float(rho), 3),
        "coverage80": f"{sum(in80)}/{len(ids)}",
        "coverage90": f"{sum(in90)}/{len(ids)}",
        "mae": round(float(np.mean(err)), 4),
        "largest_effect_hit": top_pred == top_obs,
        "largest_predicted": top_pred,
        "largest_observed": top_obs,
        "per_effect": {
            i: {"predicted": predicted[i]["predicted_effect"], "observed": observed[i], "in80": a, "in90": b}
            for i, a, b in zip(ids, in80, in90, strict=True)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast_id")
    fid = parser.parse_args().forecast_id
    revealed = json.loads((ROOT / "revealed" / f"{fid}.json").read_text(encoding="utf-8"))
    line = next(json.loads(x) for x in (ROOT / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(x)["forecast_id"] == fid)
    if not verify(revealed["record"], revealed["salt"], line["digest"]):
        raise SystemExit(f"{fid}: the revealed record does not match its ledger digest")
    results = json.loads((ROOT / "results" / f"{fid}.json").read_text(encoding="utf-8"))
    predicted = {e["id"]: e for e in revealed["record"]["effects"]}
    observed = {k: v["value"] for k, v in results["observed"].items()}
    reliable = {k for k, v in results["observed"].items() if v.get("reliable")}
    ranked = [k for k in observed if k in predicted and k in reliable]
    out = {
        "forecast_id": fid,
        "group": revealed["record"]["group"],
        "sealed_at": line["sealed_at"],
        "scored": datetime.now(UTC).isoformat(timespec="seconds"),
        "primary": score(predicted, observed, reliable, ranked),
    }
    sens = results.get("sensitivity_symmetric_per_level")
    if sens:
        alt = {**observed, **{k: v for k, v in sens.items() if k in predicted}}
        out["sensitivity_symmetric_per_level"] = score(predicted, alt, reliable, ranked)
    (ROOT / "scores").mkdir(exist_ok=True)
    (ROOT / "scores" / f"{fid}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["primary"].items() if k != "per_effect"}, indent=1))
    if sens:
        print("sensitivity:", json.dumps({k: v for k, v in out["sensitivity_symmetric_per_level"].items() if k != "per_effect"}))


if __name__ == "__main__":
    main()
