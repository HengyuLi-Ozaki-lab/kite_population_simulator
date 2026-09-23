"""Assemble a backend and its wrappers in one fixed order: Cached(Budgeted(Limited(backend)))."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from kite.config import Settings
from kite.kernel.base import Kernel
from kite.kernel.cache import CachedKernel, CacheStore, Granularity
from kite.kernel.ledger import BudgetedKernel, Ledger, PriceTable
from kite.kernel.limiter import LimitedKernel, RateLimiter
from kite.kernel.mock import MockKernel, uniform_probs


class Backend(StrEnum):
    jev = "jev"
    adapter = "adapter"
    mock = "mock"
    uniform = "uniform"


class Provider(StrEnum):
    openai = "openai"
    anthropic = "anthropic"


class AnswerMode(StrEnum):
    probabilities = "probabilities"
    discrete = "discrete"


class KernelSpec(BaseModel):
    backend: Backend = Backend.mock
    model_id: str | None = None
    provider: Provider | None = None
    answer_mode: AnswerMode = AnswerMode.probabilities
    n_samples: int = 1
    use_cache: bool = True
    cache_granularity: Granularity | None = None
    salt: str = ""
    max_usd: float = 5.0


class BuiltKernel:
    """The composed kernel plus the handles needed to shut it down cleanly."""

    def __init__(self, kernel: Kernel, backend: Kernel, store: CacheStore | None) -> None:
        self.kernel = kernel
        self._backend = backend
        self._store = store

    async def aclose(self) -> None:
        close = getattr(self._backend, "aclose", None)
        if close is not None:
            await close()
        if self._store is not None:
            self._store.close()


def build_kernel(spec: KernelSpec, settings: Settings, ledger: Ledger) -> BuiltKernel:
    options = ""
    if spec.backend == "jev":
        from kite.kernel.jev import PINNED_MODEL, JevKernel

        backend: Kernel = JevKernel(model_id=spec.model_id or PINNED_MODEL)
        default_granularity: Granularity = "question"
    elif spec.backend == "adapter":
        from kite.kernel.adapter import AdapterKernel

        if spec.provider is None or spec.model_id is None:
            raise ValueError("the adapter backend needs both provider and model_id")
        adapter = AdapterKernel(spec.provider, spec.model_id, answer_mode=spec.answer_mode, n_samples=spec.n_samples)
        backend, options = adapter, adapter.options
        default_granularity = "request"
    elif spec.backend == "uniform":
        backend = MockKernel(model_id="uniform", fn=uniform_probs)
        default_granularity = "question"
    else:
        backend = MockKernel(model_id=spec.model_id or "mock-1")
        default_granularity = "question"

    kernel: Kernel = backend
    if spec.backend in ("jev", "adapter"):
        kernel = LimitedKernel(kernel, RateLimiter(settings.rpm, settings.tps))
    # only the adapter turns one `evaluate` into several API calls; charging the others n_samples
    # would stop a run at a fraction of its cap
    calls_per_request = spec.n_samples if spec.backend is Backend.adapter else 1
    kernel = BudgetedKernel(kernel, ledger, PriceTable.load(settings.prices_path), max_usd=spec.max_usd, n_samples=calls_per_request)

    store: CacheStore | None = None
    if spec.use_cache:
        store = CacheStore(settings.cache_path)
        kernel = CachedKernel(
            kernel,
            store,
            granularity=spec.cache_granularity or default_granularity,
            options=options,
            salt=spec.salt,
            on_lookup=ledger.record_lookup,
        )
    return BuiltKernel(kernel, backend, store)
