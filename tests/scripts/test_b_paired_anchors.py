"""Paired anchor shifts and the hybrid's cell means, on synthetic predictions."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import b_paired_anchors as b  # noqa: E402

from kite.eval.task import Prediction  # noqa: E402


def anchor(study, task, condition, participant, rank, probs):
    meta = {"study_id": study, "task_num": task, "condition_num": condition, "participant": participant, "anchor_rank": rank, "n_levels": len(probs)}
    return Prediction(item_id=f"{study}:{task}:{condition}:{participant}", probs=probs, meta=meta)


def test_paired_shifts_average_the_first_k_anchors_against_the_first_condition():
    preds = [
        anchor("s", 1, 1, 10, 0, [1.0, 0.0, 0.0]),  # anchor 0: position 0.0 under condition 1
        anchor("s", 1, 2, 10, 0, [0.0, 0.0, 1.0]),  # ... 1.0 under condition 2 -> shift +1.0
        anchor("s", 1, 1, 11, 1, [0.0, 1.0, 0.0]),  # anchor 1: 0.5 under condition 1
        anchor("s", 1, 2, 11, 1, [0.0, 1.0, 0.0]),  # ... 0.5 under condition 2 -> shift 0.0
        anchor("s", 1, 1, 12, 2, [0.0, 1.0, 0.0]),  # anchor 2 has no condition-2 prediction: ignored for the pair
    ]
    one = b.paired_shifts(preds, k=1)
    two = b.paired_shifts(preds, k=3)
    assert one[("s", 1)]["ref"] == 1 and abs(one[("s", 1)]["shift"][2] - 1.0) < 1e-12
    assert abs(two[("s", 1)]["shift"][2] - 0.5) < 1e-12 and two[("s", 1)]["anchors"] == 3


def test_hybrid_means_keep_the_kernel_reference_and_add_the_scaled_shift():
    kernel = {"s|1|1": 0.30, "s|2|1": 0.35, "s|3|1": 0.10}
    shifts = {("s", 1): {"ref": 1, "shift": {2: 0.20, 3: -0.10}, "anchors": 2}}
    means = b.hybrid_means(kernel, shifts, s=0.5)
    assert means == {"s|1|1": 0.30, "s|2|1": 0.30 + 0.10, "s|3|1": 0.30 - 0.05}
    assert b.hybrid_means({"other|1|1": 0.5}, shifts, s=1.0) == {}


def test_weighted_r_matches_numpy_when_weights_are_equal():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert abs(b.weighted_r(x, y, np.ones(50)) - np.corrcoef(x, y)[0, 1]) < 1e-12
