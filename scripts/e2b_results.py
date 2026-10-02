"""E2b: the observed human effects of the post-cutoff studies, by the rules fixed before the data were read.

Rules: docs/decisions/E2.md, "E2b 的真人效应怎么算" (commit 5f7611f, before any data file was opened). Writes
forecasts/results/<forecast_id>.json in the format scripts/e2_score.py reads. The data files stay in the private
data/forecasts/e2b/ (their SHA-256 in data/forecasts/e2b/data_SHA256SUMS); only these aggregates are published.

    uv run python scripts/e2b_results.py c1
    uv run python scripts/e2b_results.py c4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("forecasts/results")
C1_DATA = Path("data/forecasts/e2b/c1/data/data_S3.csv")
C4_DATA = Path("data/forecasts/e2b/c4/data/01_clean_distribution_2025-02-26.csv")
C1_CONDS = {  # forecast condition -> the data's `cond` value and belief column stem
    "m1": ("science", "science", 100),
    "m2": ("country", "country", 193),
    "m3": ("public_belief", "belief", 100),
    "m4": ("public_problem", "problem", 100),
    "m5": ("public_action", "action", 100),
    "m6": ("public_cbd", "cbd", 100),
    "m7": ("public_anger", "anger", 100),
    "m8": ("ig_sign", "ig_sign", 100),
    "m9": ("og_sign", "og_sign", 100),
    "m10": ("og_values", "og_values", 100),
}
PARTY = {1: "Democrat", 2: "Independent", 3: "Republican"}  # inferred from counts (57/13/29%) against the 58.5/10/31.5 design
C4_POLICY = {"EITC": "eitc", "minimumwage": "minwage", "incometax": "progtax", "PRO": "proact", "medicare": "m4a", "promise": "college"}
C4_FRAMING = {"predist": "predistribution", "redist": "redistribution", "control": "control"}


def welch(a: pd.Series, b: pd.Series, scale: float, wa: pd.Series | None = None, wb: pd.Series | None = None) -> dict:
    """Difference in means of a and b divided by `scale`, with Welch's standard error (weighted if weights are given)."""
    if wa is None:
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    else:

        def wstats(x: pd.Series, w: pd.Series) -> tuple[float, float]:
            w = w / w.sum()
            m = float((w * x).sum())
            n_eff = 1 / float((w**2).sum())
            return m, float((w * (x - m) ** 2).sum()) / n_eff

        (ma, va), (mb, vb) = wstats(a, wa), wstats(b, wb)
        diff, se = ma - mb, np.sqrt(va + vb)
    value, se = float(diff / scale), float(se / scale)
    return {
        "value": round(value, 5),
        "se": round(se, 5),
        "z": round(value / se, 2),
        "reliable": bool(abs(value / se) >= 3),
        "n": [int(len(a)), int(len(b))],
    }


def c1() -> dict:
    cols = ["cond", "pol", "support", "weights_full", "weights_party"]
    cols += [f"{p}norm_{stem}" for _, stem, _ in C1_CONDS.values() for p in ("", "post_")]
    df = pd.read_csv(C1_DATA, usecols=cols)
    df["cond"] = df["cond"].fillna("control")
    df["party"] = df["pol"].map(PARTY)
    sup = df.dropna(subset=["support"])
    control, partisan_control = sup[sup["cond"] == "control"], sup[(sup["cond"] == "control") & sup["party"].isin(["Democrat", "Republican"])]
    observed, weighted = {}, {}
    for m, (code, stem, span) in C1_CONDS.items():
        arm = sup[sup["cond"] == code]
        ref, wcol = (control, "weights_full") if m in ("m1", "m2", "m3", "m4", "m5", "m6", "m7") else (partisan_control, "weights_party")
        observed[f"support:{m}-control"] = {
            **welch(arm["support"], ref["support"], 6),
            "source": (
                f"data_S3.csv: cond == '{code}' vs cond missing (control)"
                f"{'' if ref is control else ', Democrats and Republicans (pol 1, 3)'}; support 1-7, unweighted"
            ),
        }
        weighted[f"support:{m}-control"] = welch(arm["support"], ref["support"], 6, arm[wcol], ref[wcol])["value"]
        if m in ("m8", "m9", "m10"):
            for party in ("Democrat", "Republican"):
                a, b = arm[arm["party"] == party], ref[ref["party"] == party]
                observed[f"support:{m}-control:{party}"] = {**welch(a["support"], b["support"], 6), "source": f"as above, {party}s only"}
        treated = df[df["cond"] == code].dropna(subset=[f"norm_{stem}", f"post_norm_{stem}"])
        change = (treated[f"post_norm_{stem}"] - treated[f"norm_{stem}"]) / span
        se = float(change.std(ddof=1) / np.sqrt(len(change)))
        observed[f"belief:{m}"] = {
            "value": round(float(change.mean()), 5),
            "se": round(se, 5),
            "z": round(float(change.mean()) / se, 2),
            "reliable": bool(abs(change.mean() / se) >= 3),
            "n": int(len(change)),
            "source": f"data_S3.csv: cond == '{code}', post_norm_{stem} minus norm_{stem} within respondent, divided by {span}",
        }
    return {
        "forecast_id": "pc-geiger-cbd-norms-001",
        "study": (
            "PsyArXiv 10.31234/osf.io/y42db, Study 3; data OSF wqab7 /Study3/2_data/data_S3.csv "
            "(sha256 13715c1e0dcdec376737d0349a634d0d9afcb7b9fa1ad3175f6fe96cf4c98f3b)"
        ),
        "read_on": "2026-09-30",
        "estimand": (
            "Difference in mean support (1-7, 'don't know' already absent from the file) between each message and the control, "
            "divided by 6; m8-m10 against Democrats and Republicans in the control. Unweighted (rule fixed before reading the data)."
        ),
        "data_notes": (
            f"{len(df)} respondents, all flagged complete, consented, passing the attention check, not duplicate or fraud; none excluded. "
            "pol has three codes in the file; 1 = Democrat, 2 = Independent, 3 = Republican is inferred from the group sizes against the "
            "recruitment design (no codebook read)."
        ),
        "observed": observed,
        "sensitivity_weighted": {"note": "Same contrasts with the authors' weights (weights_full for m1-m7, weights_party for m8-m10).", **weighted},
    }


def c4() -> dict:
    df = pd.read_csv(C4_DATA, usecols=["RespondentId", "Policy", "treatment", "PolicySupport"], low_memory=False)
    df = df.dropna(subset=["PolicySupport"]).assign(policy=lambda f: f["Policy"].map(C4_POLICY), framing=lambda f: f["treatment"].map(C4_FRAMING))
    person = df.groupby(["RespondentId", "framing"])["PolicySupport"].mean().unstack()
    contrasts = (("predistribution", "redistribution"), ("predistribution", "control"), ("redistribution", "control"))
    observed = {}
    for a, b in contrasts:
        diff = ((person[a] - person[b]) / 100).dropna()
        value, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
        observed[f"pooled:{a}-{b}"] = {
            "value": round(value, 5),
            "se": round(se, 5),
            "z": round(value / se, 2),
            "reliable": bool(abs(value / se) >= 3),
            "n": int(len(diff)),
            "source": "per-respondent mean PolicySupport (0-100) by framing, paired difference over respondents, divided by 100",
        }
    for policy, part in df.groupby("policy"):
        for a, b in contrasts:
            observed[f"{policy}:{a}-{b}"] = {
                **welch(part.loc[part["framing"] == a, "PolicySupport"], part.loc[part["framing"] == b, "PolicySupport"], 100),
                "source": f"PolicySupport for {policy}, framing groups compared between respondents, divided by 100",
            }
    return {
        "forecast_id": "pc-predistribution-framing-001",
        "study": (
            "SocArXiv 10.31235/osf.io/gkr5e / PNAS 10.1073/pnas.2617191123; data OSF kndq7 /Replication package/Study2/Data/"
            "01_clean_distribution_2025-02-26.csv (sha256 cfe30a1ea3cb8a5a04388cdc8215a6fd574b185b708663e55806283bf6fdaccb)"
        ),
        "read_on": "2026-09-30",
        "estimand": (
            "Differences in PolicySupport (mean of the two 0-100 items, divided by 100) between framings; pooled contrasts "
            "paired within respondent, per-policy contrasts between respondents."
        ),
        "data_notes": f"{df['RespondentId'].nunique()} respondents, {len(df)} rows (respondent x policy) in the authors' cleaned file.",
        "observed": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["c1", "c4"])
    result = {"c1": c1, "c4": c4}[parser.parse_args().study]()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{result['forecast_id']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
