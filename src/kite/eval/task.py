"""What an evaluation task must provide so the runner, baselines and recalibration can drive it."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from kite.kernel.types import KernelRequest, KernelResponse


class Item(BaseModel):
    item_id: str
    request: KernelRequest
    meta: dict[str, Any]  # JSON-serializable; copied onto the prediction


class Prediction(BaseModel):
    item_id: str
    probs: list[float]
    meta: dict[str, Any]


class ProbeCase(BaseModel):
    """Raw ingredients of one survey prediction, for probes that build their own requests."""

    persona: dict[str, str] | None
    question: str
    options: list[str]
    context: str | None = None


class FidelityTask(Protocol):
    name: str

    def items(self) -> Iterator[Item]: ...

    def members(self) -> list[str]:
        """The study ids or waves this task covers: the split's name resolved to actual data."""
        ...

    def decode(self, item: Item, response: KernelResponse) -> list[float]:
        """Probabilities over the item's answer levels."""
        ...

    def target(self, prediction: Prediction) -> list[float]:
        """What the prediction is scored against: a one-hot response or a human distribution."""
        ...

    def score(self, predictions: Sequence[Prediction]) -> dict[str, Any]: ...

    def baseline(self, kind: str) -> list[Prediction]:
        """Model-free predictions: 'uniform' or 'pooled' (human distribution that ignores condition/group)."""
        ...

    def probe_cases(self, limit: int) -> list[ProbeCase]: ...
