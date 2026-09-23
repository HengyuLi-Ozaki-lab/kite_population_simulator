"""The tabulated tier: a population executed from a table of unique states to response distributions.

The kernel is asked once per unique state (persona, content, situation, condition); the table holds the answer.
Executing a population is then an ordinary simulation: each agent's response in each condition is drawn from
its state's distribution with an event-keyed uniform shared across conditions. Where only aggregates are
needed, `expectation` sums the distributions directly and has no sampling error at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from kite.engine.random import event_uniform


class Table:
    """State key -> probability vector over the options, in one fixed option order."""

    def __init__(self, options: Sequence[str], rows: Mapping[str, Sequence[float]] | None = None) -> None:
        self.options = tuple(options)
        self._rows: dict[str, np.ndarray] = {}
        for key, probs in (rows or {}).items():
            self.put(key, probs)

    def put(self, key: str, probs: Sequence[float]) -> None:
        p = np.asarray(probs, dtype=float)
        if p.shape != (len(self.options),) or (p < 0).any() or p.sum() <= 0:
            raise ValueError(f"state {key!r}: need {len(self.options)} non-negative probabilities with positive total")
        self._rows[key] = p / p.sum()

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def probs(self, key: str) -> np.ndarray:
        return self._rows[key]

    def draw(self, key: str, u: float) -> int:
        """The option whose cumulative probability first exceeds u: inverse-CDF sampling, so a larger u is a later option."""
        cumulative = np.cumsum(self._rows[key])
        return int(min(np.searchsorted(cumulative, u, side="right"), len(self.options) - 1))


def run(
    table: Table,
    agents: Iterable[tuple[str, float, Mapping[str, str]]],
    conditions: Sequence[str],
    *,
    seed: int,
    step: int = 0,
    event: str = "decide",
) -> dict[str, np.ndarray]:
    """One decision per agent per condition. `agents` yields (agent id, weight, {condition: state key}).

    Returns, per condition, the weighted count of agents on each option. The same uniform serves every
    condition of an agent, so conditions differ only through their distributions.
    """
    counts = {c: np.zeros(len(table.options)) for c in conditions}
    for agent_id, weight, states in agents:
        u = event_uniform(seed, agent_id, step, event)
        for condition in conditions:
            counts[condition][table.draw(states[condition], u)] += weight
    return counts


def expectation(table: Table, agents: Iterable[tuple[str, float, Mapping[str, str]]], conditions: Sequence[str]) -> dict[str, np.ndarray]:
    """The weighted expected count per option and condition - the limit of `run` over seeds, with no sampling error."""
    totals = {c: np.zeros(len(table.options)) for c in conditions}
    for _, weight, states in agents:
        for condition in conditions:
            totals[condition] += weight * table.probs(states[condition])
    return totals
