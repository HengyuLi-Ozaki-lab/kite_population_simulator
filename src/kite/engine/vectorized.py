"""Array-sized execution of the tabulated tier, for populations of 10^5 agents and up.

Same guarantees as `tabulated.run` - one uniform per (seed, agent, step, event), shared across conditions,
independent of order and batching - but the uniforms come from an integer mixing function (SplitMix64 finaliser)
evaluated on whole arrays instead of a per-event hash of a string key, so a million agents take milliseconds.
The two streams are different functions of the key; either is a valid event-keyed source, and a run must say
which it used.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on uint64 arrays."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def event_uniforms(seed: int, agents: np.ndarray, step: int = 0, event: int = 0) -> np.ndarray:
    """One uniform in [0, 1) per agent index for this (seed, step, event); a pure function of the key."""
    a = np.asarray(agents, dtype=np.uint64)
    gamma = np.uint64(0x9E3779B97F4A7C15)  # SplitMix64's increment: keeps the all-zero key (seed 0, agent 0, step 0) off the fixed point 0
    with np.errstate(over="ignore"):
        key = _mix((a + np.uint64(1)) * gamma + np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF) + gamma)
        key = _mix(
            key
            ^ ((np.uint64(int(step)) + np.uint64(1)) * np.uint64(0xD1B54A32D192ED03))
            ^ ((np.uint64(int(event)) + np.uint64(1)) * np.uint64(0x8CB92BA72F3D8DD7))
        )
    return (key >> np.uint64(11)).astype(np.float64) / float(2**53)


def cumulative_table(probs: np.ndarray) -> np.ndarray:
    """Rows of probabilities -> rows of cumulative probabilities ending exactly at 1."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    c = np.cumsum(p, axis=1)
    c[:, -1] = 1.0
    return c


def run(
    cumulative: np.ndarray,
    states: Mapping[str, np.ndarray],
    weights: np.ndarray | None = None,
    *,
    seed: int,
    step: int = 0,
    event: int = 0,
) -> dict[str, np.ndarray]:
    """Weighted option counts per condition. `states[condition]` gives each agent's row of `cumulative` in that condition."""
    n = len(next(iter(states.values())))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    u = event_uniforms(seed, np.arange(n), step, event)
    n_options = cumulative.shape[1]
    out = {}
    for condition, rows in states.items():
        c = cumulative[np.asarray(rows)]
        choice = np.minimum((c < u[:, None]).sum(axis=1), n_options - 1)  # first option whose cumulative probability exceeds u
        out[condition] = np.bincount(choice, weights=w, minlength=n_options)
    return out


def expectation(probs: np.ndarray, states: Mapping[str, np.ndarray], weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Weighted expected counts per condition, with no sampling error."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    n = len(next(iter(states.values())))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    return {condition: w @ p[np.asarray(rows)] for condition, rows in states.items()}
