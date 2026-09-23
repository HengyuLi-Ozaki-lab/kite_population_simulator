"""Spend accounting: a price table, a per-run ledger, and a wrapper that enforces a dollar cap."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from kite.kernel.base import Kernel, estimate_output_tokens, estimate_tokens
from kite.kernel.types import KernelAnswer, KernelRequest, KernelResponse, Usage


class Price(BaseModel):
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000


class PriceTable:
    """`{backend: {model: {input_per_mtok, output_per_mtok}}}`, loaded from configs/prices.yaml."""

    def __init__(self, prices: dict[str, dict[str, Price]]) -> None:
        self._prices = prices

    @classmethod
    def load(cls, path: str | Path) -> PriceTable:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls({backend: {model: Price(**p) for model, p in models.items()} for backend, models in raw.items()})

    def lookup(self, backend: str, model: str) -> Price:
        try:
            return self._prices[backend][model]
        except KeyError:
            raise KeyError(
                f"no price for backend={backend!r} model={model!r}; add it to configs/prices.yaml from the provider's pricing page before running"
            ) from None


class Ledger:
    def __init__(self) -> None:
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0
        self.reserved_usd = 0.0
        self.answers = 0
        self.malformed_distributions = 0
        self._latencies: list[float] = []
        self._started = time.time()

    @property
    def committed_usd(self) -> float:
        """Spent plus reserved: what a cap has to be checked against while calls are in flight."""
        return self.usd + self.reserved_usd

    def reserve(self, usd: float) -> None:
        """Book a projected cost before awaiting, so concurrent callers do not all read one stale balance."""
        self.reserved_usd += usd

    def release(self, usd: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - usd)

    def record_call(self, price: Price, usage: Usage, latency_ms: float, *, reserved: float = 0.0) -> None:
        self.release(reserved)
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.usd += price.cost(usage.input_tokens, usage.output_tokens)
        self._latencies.append(latency_ms)

    def record_lookup(self, hits: int, misses: int) -> None:
        self.cache_hits += hits
        self.cache_misses += misses

    def record_answers(self, answers: Iterable[KernelAnswer]) -> None:
        """Count answers whose probabilities do not sum to 1 within `PROB_SUM_TOLERANCE`.

        Every metric renormalizes, so a truncated or garbled backend response otherwise scores like
        a good one and leaves no trace. The count belongs in the run's artifacts, not in a log line.
        """
        for answer in answers:
            self.answers += 1
            self.malformed_distributions += int(not answer.is_normalized)

    def summary(self) -> dict:
        latencies = np.array(self._latencies) if self._latencies else np.array([0.0])
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
            "reserved_usd": round(self.reserved_usd, 6),  # zero once every call has settled
            "answers": self.answers,
            "malformed_distributions": self.malformed_distributions,
            "wall_seconds": round(time.time() - self._started, 3),
            "latency_ms_p50": float(np.percentile(latencies, 50)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push the run past its dollar cap."""


class BudgetedKernel:
    """The dollar cap. Projects a call's cost, reserves it, and settles to the real cost afterwards.

    The projection has to cover what the backend will actually be billed for: output tokens (free
    on Jev, the expensive half for the LLM baselines) and `n_samples`, since the adapter makes that
    many API calls per `evaluate`. And it has to be reserved *before* awaiting: reading the ledger
    before the call and writing it after let `concurrency` calls pass on one stale balance.
    """

    def __init__(
        self, inner: Kernel, ledger: Ledger, prices: PriceTable, *, max_usd: float, n_samples: int = 1, output_tokens: int | None = None
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._price = prices.lookup(inner.name, inner.model_id)
        self._max_usd = max_usd
        self._n_samples = max(1, n_samples)
        self._output_tokens = output_tokens
        self.name = inner.name
        self.model_id = inner.model_id

    def projected(self, request: KernelRequest) -> float:
        output = self._output_tokens if self._output_tokens is not None else estimate_output_tokens(request)
        return self._n_samples * self._price.cost(estimate_tokens(request), output)

    async def evaluate(self, request: KernelRequest) -> KernelResponse:
        projected = self.projected(request)
        if self._ledger.committed_usd + projected > self._max_usd:
            raise BudgetExceeded(
                f"spent ${self._ledger.usd:.4f} with ${self._ledger.reserved_usd:.4f} in flight; "
                f"this call is projected at ${projected:.4f}, which would exceed the ${self._max_usd:.2f} cap"
            )
        self._ledger.reserve(projected)
        try:
            response = await self._inner.evaluate(request)
        except BaseException:
            self._ledger.release(projected)
            raise
        self._ledger.record_call(self._price, response.usage, response.latency_ms, reserved=projected)
        return response
