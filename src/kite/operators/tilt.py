"""Exponential tilting: move a cell of categorical distributions to a declared mean.

An intervention operator states what the mean response of a cell should be (for example the kernel's
control-condition mean plus an effect estimated elsewhere). This module re-weights every distribution in the
cell by exp(alpha * value), with one alpha shared by the whole cell, chosen so that the weighted cell mean
equals the target. The tilt keeps each distribution's support, is monotone in alpha, and is undone by -alpha.
It matches a mean and nothing more: it does not recover the full response distribution or any individual
treatment effect, and the caller must say where the target came from.

A target that would need an implausibly large alpha is reported as infeasible rather than clipped: silently
clipping would change the estimand. Values are rescaled to [0, 1] inside, so alpha has the same meaning on
a 1-6 rating scale and on a binary outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_MAX = 10.0  # exp(10) between the two ends of the scale: beyond this the tilt replaces the distribution instead of correcting it
DEFAULT_FLOOR = 0.005  # the kernel reports probabilities to two decimals, so exact zeros are rounding, not certainty


def floored(probs: np.ndarray, floor: float = DEFAULT_FLOOR) -> np.ndarray:
    """Rows with every option at least `floor`, renormalised. Applied consistently before any tilt."""
    p = np.asarray(probs, dtype=float)
    p = np.maximum(p, floor)
    return p / p.sum(axis=-1, keepdims=True)


def _unit(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    span = v.max() - v.min()
    if span <= 0:
        raise ValueError("values must span more than one point")
    return (v - v.min()) / span


def tilt(probs: np.ndarray, values: np.ndarray, alpha: float) -> np.ndarray:
    """Every row re-weighted by exp(alpha * unit value) and renormalised. Rows are assumed already floored."""
    p = np.asarray(probs, dtype=float)
    weights = np.exp(alpha * _unit(values))
    q = p * weights
    return q / q.sum(axis=-1, keepdims=True)


def cell_mean(probs: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean response over the rows of a cell, in the original value units."""
    p = np.asarray(probs, dtype=float)
    v = np.asarray(values, dtype=float)
    w = np.ones(p.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must have positive total")
    return float(w @ (p @ v) / w.sum())


@dataclass(frozen=True)
class TiltResult:
    """What a solve produced. `feasible` False means the cell was left as it was; `achieved` is then its untilted mean."""

    requested: float
    achieved: float
    alpha: float
    feasible: bool
    lower: float  # the means reachable within |alpha| <= alpha_max
    upper: float


def solve(
    probs: np.ndarray,
    values: np.ndarray,
    target: float,
    weights: np.ndarray | None = None,
    *,
    floor: float = DEFAULT_FLOOR,
    tol: float = 1e-6,
    alpha_max: float = ALPHA_MAX,
) -> TiltResult:
    """The shared alpha that brings the floored cell's weighted mean to `target`, by bisection.

    The weighted mean is increasing in alpha (each row's tilted mean has derivative equal to its variance), so
    the root is unique when it exists within [-alpha_max, alpha_max].
    """
    p = floored(probs, floor)
    mean_at = lambda a: cell_mean(tilt(p, values, a), values, weights)  # noqa: E731
    lower, upper = mean_at(-alpha_max), mean_at(alpha_max)
    if not (lower - tol <= target <= upper + tol):
        return TiltResult(requested=target, achieved=cell_mean(p, values, weights), alpha=0.0, feasible=False, lower=lower, upper=upper)
    lo, hi = -alpha_max, alpha_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    alpha = 0.5 * (lo + hi)
    return TiltResult(requested=target, achieved=mean_at(alpha), alpha=alpha, feasible=True, lower=lower, upper=upper)


def tilt_cell(
    probs: np.ndarray,
    values: np.ndarray,
    target: float,
    weights: np.ndarray | None = None,
    *,
    floor: float = DEFAULT_FLOOR,
    tol: float = 1e-6,
    alpha_max: float = ALPHA_MAX,
) -> tuple[np.ndarray, TiltResult]:
    """The cell's distributions moved to `target` (floored and untilted if the target is infeasible), with the record."""
    result = solve(probs, values, target, weights, floor=floor, tol=tol, alpha_max=alpha_max)
    p = floored(probs, floor)
    return (tilt(p, values, result.alpha) if result.feasible else p), result
