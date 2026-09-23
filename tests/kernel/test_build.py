import asyncio

import pytest

from kite.config import Settings
from kite.kernel.build import KernelSpec, build_kernel
from kite.kernel.ledger import Ledger
from kite.kernel.types import KernelRequest, QuestionSpec

PRICES = "mock:\n  mock-1: {input_per_mtok: 0.0, output_per_mtok: 0.0}\n  uniform: {input_per_mtok: 0.0, output_per_mtok: 0.0}\n"
REQUEST = KernelRequest(state="s", questions={"q": QuestionSpec(type="choice", instructions="pick", criteria={"a": None, "b": None})})


def make_settings(tmp_path) -> Settings:
    prices = tmp_path / "prices.yaml"
    prices.write_text(PRICES, encoding="utf-8")
    return Settings(cache_path=tmp_path / "cache.sqlite", prices_path=prices)


def test_mock_kernel_is_cached_and_metered(tmp_path):
    ledger = Ledger()
    built = build_kernel(KernelSpec(backend="mock"), make_settings(tmp_path), ledger)

    async def scenario():
        first = await built.kernel.evaluate(REQUEST)
        second = await built.kernel.evaluate(REQUEST)
        await built.aclose()
        return first, second

    first, second = asyncio.run(scenario())
    assert first.cached == {"q": False} and second.cached == {"q": True}
    assert ledger.calls == 1
    assert (ledger.cache_hits, ledger.cache_misses) == (1, 1)


def test_uniform_backend(tmp_path):
    built = build_kernel(KernelSpec(backend="uniform", use_cache=False), make_settings(tmp_path), Ledger())
    response = asyncio.run(built.kernel.evaluate(REQUEST))
    assert response.answers["q"].probs == {"a": 0.5, "b": 0.5}


def test_adapter_needs_provider_and_model(tmp_path):
    with pytest.raises(ValueError, match="provider and model_id"):
        build_kernel(KernelSpec(backend="adapter"), make_settings(tmp_path), Ledger())
