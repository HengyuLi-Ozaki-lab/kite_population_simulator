"""Distances between distributions and scores for individual predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance


def as_dist(values: Sequence[float]) -> np.ndarray:
    """Weights renormalized to sum to 1. Counts are welcome here; NaN and infinity are not.

    `as_dist([nan, 1.0])` used to return `[nan, nan]`: NaN is not negative and a NaN total is not
    `<= 0`, so it passed both guards and propagated into whatever metric was being computed.
    """
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"a distribution cannot contain NaN or infinity: {list(values)}")
    total = array.sum()
    if total <= 0 or np.any(array < 0):
        raise ValueError("a distribution needs non-negative entries with a positive sum")
    return array / total


def wasserstein(p: Sequence[float], q: Sequence[float], positions: Sequence[float]) -> float:
    return float(wasserstein_distance(positions, positions, as_dist(p), as_dist(q)))


def opinionqa_alignment(p: Sequence[float], q: Sequence[float], ordinal: Sequence[float]) -> float:
    """OpinionQA's alignment: 1 - WD / (max ordinal - min ordinal). 1 is identical, 0 is opposite poles."""
    return 1.0 - wasserstein(p, q, ordinal) / (max(ordinal) - min(ordinal))


def wasserstein_unit(p: Sequence[float], q: Sequence[float]) -> float:
    """SocSci210's distribution distance: responses rescaled to [0, 1], then Wasserstein."""
    return wasserstein(p, q, np.linspace(0.0, 1.0, len(p)))


def observed_range_positions(human: Sequence[float]) -> np.ndarray:
    """Level positions rescaled by the range the humans in this cell actually used, not the stated scale.

    Levels outside that range fall outside [0, 1], which is intended: they are further from the
    answers people gave than the stated scale suggests. A cell where everyone chose one level has no
    range, so the divisor floors at 1 and positions are whole levels away from that answer.
    """
    counts = np.asarray(human, dtype=float)
    used = np.flatnonzero(counts)
    if len(used) == 0:
        raise ValueError("a cell needs at least one human answer")
    low, high = int(used[0]), int(used[-1])
    return (np.arange(len(counts)) - low) / max(high - low, 1)


def wasserstein_observed(p: Sequence[float], human: Sequence[float]) -> float:
    """The distribution distance under the convention the published SocSci210 numbers use.

    Reproduces their model-free "Uniform Guess" row to 0.0006; see docs/data/socsci210-schema.md.
    """
    return wasserstein(p, human, observed_range_positions(human))


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    return float(0.5 * np.abs(as_dist(p) - as_dist(q)).sum())


def entropy(p: Sequence[float]) -> float:
    dist = as_dist(p)
    nonzero = dist[dist > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, between 0 and 1."""
    a, b = as_dist(p), as_dist(q)
    return float(entropy((a + b) / 2) - (entropy(a) + entropy(b)) / 2)


LOW_ENTROPY_BITS = 0.1  # below this the humans are all but unanimous and the ratio's denominator is unstable


def entropy_ratio(model: Sequence[float], human: Sequence[float]) -> float | None:
    """H(model) / H(human). Below 1 means the model is sharper than people are. None if H(human) = 0."""
    human_entropy = entropy(human)
    if human_entropy == 0:
        return None
    return entropy(model) / human_entropy


def describe_entropy_ratios(ratios: Sequence[float]) -> dict[str, float | None]:
    """Median and quartiles, because the mean of an unbounded ratio is not a summary.

    A cell where people nearly all agree has H(human) near zero and a ratio in the hundreds: one such
    cell in a thousand lifts the mean from 1.00 to 2.01 while the median does not move. Callers also
    report how many cells sit below `LOW_ENTROPY_BITS`, which is what makes that happen.
    """
    if len(ratios) == 0:
        return {"entropy_ratio_median": None, "entropy_ratio_p25": None, "entropy_ratio_p75": None, "entropy_ratio_mean": None}
    values = np.asarray(ratios, dtype=float)
    return {
        "entropy_ratio_median": float(np.median(values)),
        "entropy_ratio_p25": float(np.percentile(values, 25)),
        "entropy_ratio_p75": float(np.percentile(values, 75)),
        "entropy_ratio_mean": float(values.mean()),
    }


def mean_position(p: Sequence[float]) -> float:
    """The distribution's mean level, rescaled to [0, 1] so scales of different lengths compare."""
    dist = as_dist(p)
    if len(dist) < 2:
        raise ValueError("a scale needs at least two levels")
    return float(np.dot(dist, np.arange(len(dist)))) / (len(dist) - 1)


def regrid(weights: Sequence[float], n_levels: int) -> np.ndarray:
    """Move a histogram onto a scale of `n_levels`, keeping its shape on [0, 1].

    Each level's mass is split between the two target levels its [0, 1] position falls between,
    which leaves `mean_position` exactly unchanged. The identity when the lengths already match.
    """
    source = np.asarray(weights, dtype=float)
    if len(source) < 2 or n_levels < 2:
        raise ValueError("a scale needs at least two levels")
    exact = np.arange(len(source)) * (n_levels - 1) / (len(source) - 1)
    below = np.floor(exact).astype(int)
    fraction = exact - below
    target = np.zeros(n_levels, dtype=float)
    np.add.at(target, below, source * (1.0 - fraction))
    np.add.at(target, np.minimum(below + 1, n_levels - 1), source * fraction)
    return target


FLAT = 1e-9  # a mean position that moves less than this across conditions is flat, not faintly sensitive


def _fit_centred(pairs: np.ndarray) -> dict[str, float | None]:
    """Pearson r, the slope of human-on-predicted, and std(predicted) / std(human), on centred pairs.

    A flat side gives None rather than a number: correlating a condition-blind predictor's rounding
    noise with the human means yields a meaningless r of any size. `FLAT` is the cutoff, on a mean
    position that lives in [0, 1].
    """
    predicted = pairs[:, 0] - pairs[:, 0].mean()
    human = pairs[:, 1] - pairs[:, 1].mean()
    predicted_rms, human_rms = float(np.sqrt((predicted**2).mean())), float(np.sqrt((human**2).mean()))
    spread_ratio = predicted_rms / human_rms if human_rms > FLAT else None
    if predicted_rms <= FLAT or human_rms <= FLAT:
        return {"r": None, "slope": None, "spread_ratio": 0.0 if spread_ratio is not None else None}
    cross = float((predicted * human).mean())
    return {"r": cross / (predicted_rms * human_rms), "slope": cross / predicted_rms**2, "spread_ratio": spread_ratio}


def condition_sensitivity(
    by_task: dict[Any, Sequence[tuple[float, float]]], *, n_boot: int = 1000, n_perm: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Does the prediction move across experimental conditions the way people do?

    `by_task` maps a (study, task) key to that task's cells as (predicted, human) mean positions in
    [0, 1]. Each task is centred on its own mean, which removes the task's overall level and leaves
    only movement across conditions; the centred pairs are then pooled. A task with one condition
    says nothing about condition sensitivity and is dropped.

    The bootstrap resamples whole tasks, not cells: cells within a task share a question and a
    respondent pool, so they are not independent and resampling them would understate the interval.

    The bootstrap interval says how precisely `r` is known; it does **not** say whether `r` is
    distinguishable from chance, and the two were conflated when the gate was first written. The
    permutation null answers the second question: shuffling the predicted means within each task
    breaks the pairing with the condition while keeping every marginal, so it is the distribution of
    `r` under "the prediction carries no information about which condition this is". It matters
    because that null is wide — measured on the development split it reaches 0.28 at its 95th
    percentile, above the 0.25 the gate originally asked for.
    """
    blocks = [np.asarray(pairs, dtype=float) for _, pairs in sorted(by_task.items()) if len(pairs) >= 2]
    blocks = [block - block.mean(axis=0) for block in blocks]
    empty = {
        "r": None,
        "slope": None,
        "spread_ratio": None,
        "n_tasks": 0,
        "n_cells": 0,
        "ci_low": None,
        "ci_high": None,
        "null_p95": None,
        "permutation_p": None,
    }
    if not blocks:
        return empty
    pooled = np.concatenate(blocks)
    observed = _fit_centred(pooled)
    generator = np.random.default_rng(seed)
    resampled = [_fit_centred(np.concatenate([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])) for _ in range(n_boot)]
    values = [fit["r"] for fit in resampled if fit["r"] is not None]
    interval = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (None, None)

    null = []
    for _ in range(n_perm):
        shuffled = np.concatenate([np.column_stack([generator.permutation(block[:, 0]), block[:, 1]]) for block in blocks])
        fit = _fit_centred(shuffled)
        if fit["r"] is not None:
            null.append(fit["r"])
    if null and observed["r"] is not None:
        null_array = np.asarray(null)
        null_p95 = float(np.percentile(null_array, 95))
        # +1 in both parts so the p-value can never be reported as exactly zero
        permutation_p = float((np.sum(null_array >= observed["r"]) + 1) / (len(null_array) + 1))
    else:
        null_p95, permutation_p = None, None

    return {
        **empty,
        **observed,
        "n_tasks": len(blocks),
        "n_cells": len(pooled),
        "ci_low": interval[0],
        "ci_high": interval[1],
        "null_p95": null_p95,
        "permutation_p": permutation_p,
    }


def median_index(p: Sequence[float]) -> int:
    """The point prediction that minimizes expected absolute error."""
    cumulative = np.cumsum(as_dist(p))
    return int(np.searchsorted(cumulative, 0.5 - 1e-12))


def normalized_abs_error(predicted_index: int, true_index: int, n_levels: int) -> float:
    return abs(predicted_index - true_index) / (n_levels - 1)


def random_draw_abs_error(p: Sequence[float], true_index: int) -> float:
    """Expected normalized absolute error of **one answer drawn** from `p`.

    The other convention is `normalized_abs_error(median_index(p), ...)`: the error of the single best
    point prediction. The two differ a lot — on a 7-level scale a uniform predictor scores 0.619 drawing
    and 0.714 taking the middle level — so a table must say which one it means. The paper's "Uniform
    Guess" row (0.612 over its mixture of scale lengths) is the drawing convention.
    """
    dist = as_dist(p)
    return float(np.dot(dist, np.abs(np.arange(len(dist)) - true_index)) / (len(dist) - 1))


def uniform_expected_abs_error(true_index: int, n_levels: int) -> float:
    """Expected normalized absolute error of a uniformly random guess: `random_draw_abs_error` of a flat p."""
    return float(np.abs(np.arange(n_levels) - true_index).mean() / (n_levels - 1))


def brier(p: Sequence[float], true_index: int) -> float:
    dist = as_dist(p)
    target = np.zeros_like(dist)
    target[true_index] = 1.0
    return float(((dist - target) ** 2).sum())


def ece(confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error plus the per-bin table behind a reliability diagram."""
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(corrects, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total, bins = 0.0, []
    for index in range(n_bins):
        mask = which == index
        if not mask.any():
            continue
        mean_conf, accuracy = float(conf[mask].mean()), float(hit[mask].mean())
        total += mask.mean() * abs(mean_conf - accuracy)
        bins.append({"lo": float(edges[index]), "hi": float(edges[index + 1]), "n": int(mask.sum()), "confidence": mean_conf, "accuracy": accuracy})
    return float(total), bins
