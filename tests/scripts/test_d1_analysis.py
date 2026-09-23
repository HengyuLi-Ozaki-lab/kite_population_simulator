"""The D1 analysis on synthetic predictions and ratings: cells, tilt targets, policies and policy value."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import d1_analysis as d1  # noqa: E402


def synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, arows, cells = [], [], []
    arms = {2: ["control", "evaluation", "long_evaluation"], 3: ["control", "tips"]}
    for wave, wave_arms in arms.items():
        for pool, ids in (("application", range(30)), ("audit", range(100, 106))):
            for pid in ids:
                for item in range(1, 21):
                    true = item > 10
                    base = 0.55 if true else 0.45
                    for arm in wave_arms:
                        p = base + (0.0 if arm == "control" else -0.05 * (not true))  # kernel: small false-sharing drop
                        rows.append(
                            {
                                "wave": wave,
                                "id": pid,
                                "pool": pool,
                                "arm": arm,
                                "item_num": item,
                                "true": true,
                                "kind": "share",
                                "p_yes": float(np.clip(p + rng.normal(0, 0.03), 0.01, 0.99)),
                            }
                        )
                        if pool == "audit" or pid < 12:  # flagship: audit pool and 12 anchor personas
                            q = base + (0.0 if arm == "control" else -0.15 * (not true))  # flagship sees a bigger false drop
                            arows.append(
                                {
                                    "wave": wave,
                                    "id": pid,
                                    "pool": pool,
                                    "arm": arm,
                                    "item_num": item,
                                    "true": true,
                                    "anchor": pool == "application",
                                    "p_yes": float(np.clip(q + rng.normal(0, 0.03), 0.01, 0.99)),
                                }
                            )
    jev, astra = pd.DataFrame(rows), pd.DataFrame(arows)
    control = jev[(jev["arm"] == "control") & (jev["pool"] == "application")]
    for (wave, true), g in control.groupby(["wave", "true"]):
        lo, hi = np.quantile(g["p_yes"], [1 / 3, 2 / 3])
        for r in g.itertuples(index=False):
            cells.append(
                {
                    "wave": wave,
                    "id": r.id,
                    "item_num": r.item_num,
                    "cell": f"{'true' if true else 'false'}:{0 if r.p_yes <= lo else 1 if r.p_yes <= hi else 2}",
                }
            )
    return jev, astra, pd.DataFrame(cells)


def test_hybrid_moves_treated_cells_to_the_flagship_effect():
    jev, astra, cells = synthetic()
    corrected, records = d1.hybrid(jev, astra, cells, anchors_per_cell=12)
    assert all(r["status"] == "ok" for r in records)
    k, h = d1.discernment(jev[jev["pool"] == "application"]), d1.discernment(corrected)
    # the kernel's own treatment effect on discernment is ~+0.05; the flagship's ~+0.15, and the hybrid follows the flagship
    kernel_effect = k.loc[(2, "evaluation"), "discernment"] - k.loc[(2, "control"), "discernment"]
    hybrid_effect = h.loc[(2, "evaluation"), "discernment"] - h.loc[(2, "control"), "discernment"]
    assert 0.02 < kernel_effect < 0.08 and 0.11 < hybrid_effect < 0.19
    assert np.allclose(
        corrected.loc[corrected["arm"] == "control", "p_yes"], jev.loc[(jev["pool"] == "application") & (jev["arm"] == "control"), "p_yes"]
    )


def test_policies_and_policy_value():
    jev, astra, cells = synthetic()
    table = d1.discernment(jev[jev["pool"] == "application"])
    policy = d1.choose(table)
    assert set(policy) == {2, 3} and policy[3] == "tips"
    ratings = pd.DataFrame(
        [
            {"wave": w, "id": i, "arm": a, "item_num": k, "true": k > 10, "rating": int((k > 10) or (a == "control" and k % 2 == 0))}
            for w, arms in {2: ["control", "evaluation", "long_evaluation"], 3: ["control", "tips"]}.items()
            for a in arms
            for i in range(20)
            for k in range(1, 21)
        ]
    )
    person = d1.person_discernment(ratings)
    assert len(person) == 100 and abs(person[person["arm"] == "control"]["d"].mean() - 0.5) < 1e-9
    assert abs(d1.policy_value(person, {2: "evaluation", 3: "tips"}) - 1.0) < 1e-9
    assert abs(d1.policy_value(person, {2: "control", 3: "control"}) - 0.5) < 1e-9
    human = d1.human_tables(ratings)
    assert abs(human.loc[(2, "control"), "discernment"] - 0.5) < 1e-9 and abs(human.loc[(3, "tips"), "discernment"] - 1.0) < 1e-9
