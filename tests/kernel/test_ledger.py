import asyncio
import json

import pytest

from kite.kernel.base import estimate_output_tokens, evaluate_many
from kite.kernel.ledger import BudgetedKernel, BudgetExceeded, Ledger, Price, PriceTable
from kite.kernel.mock import MockKernel
from kite.kernel.types import KernelRequest, QuestionSpec, Usage

REQUEST = KernelRequest(state="x" * 3500, questions={"q": QuestionSpec(type="noul", instructions="x")})


class SlowKernel(MockKernel):
    """Suspends inside `evaluate`, like a real backend: every concurrent caller is in flight at once."""

    async def evaluate(self, request):
        await asyncio.sleep(0.01)
        return await super().evaluate(request)


def test_price_cost_is_per_million_tokens():
    assert Price(input_per_mtok=0.042, output_per_mtok=0.0).cost(1_000_000, 123) == pytest.approx(0.042)
    assert Price(input_per_mtok=1.0, output_per_mtok=5.0).cost(2_000_000, 1_000_000) == pytest.approx(7.0)


def test_price_table_load_and_missing_entry(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text("jev:\n  jev-1.13.0: {input_per_mtok: 0.042, output_per_mtok: 0.0}\n", encoding="utf-8")
    table = PriceTable.load(path)
    assert table.lookup("jev", "jev-1.13.0").input_per_mtok == 0.042
    with pytest.raises(KeyError, match="configs/prices.yaml"):
        table.lookup("adapter-openai", "some-model")


def test_ledger_accumulates_and_writes(tmp_path):
    ledger = Ledger()
    price = Price(input_per_mtok=1.0, output_per_mtok=2.0)
    ledger.record_call(price, Usage(input_tokens=1_000_000, output_tokens=500_000), latency_ms=100.0)
    ledger.record_call(price, Usage(input_tokens=0, output_tokens=0), latency_ms=300.0)
    ledger.record_lookup(hits=3, misses=1)
    summary = ledger.summary()
    assert summary["calls"] == 2
    assert summary["usd"] == pytest.approx(2.0)
    assert (summary["cache_hits"], summary["cache_misses"]) == (3, 1)
    assert summary["latency_ms_p50"] == pytest.approx(200.0)
    ledger.write(tmp_path / "ledger.json")
    assert json.loads((tmp_path / "ledger.json").read_text())["input_tokens"] == 1_000_000


def test_budgeted_kernel_records_spend_and_stops_before_the_cap():
    prices = PriceTable({"mock": {"mock-1": Price(input_per_mtok=1000.0, output_per_mtok=0.0)}})
    ledger = Ledger()
    inner = MockKernel()
    kernel = BudgetedKernel(inner, ledger, prices, max_usd=1.5)  # each call costs about $1

    asyncio.run(kernel.evaluate(REQUEST))
    assert ledger.calls == 1 and ledger.usd == pytest.approx(1.0, rel=0.05)
    with pytest.raises(BudgetExceeded):
        asyncio.run(kernel.evaluate(REQUEST))
    assert len(inner.seen) == 1


def test_budgeted_kernel_needs_a_price():
    with pytest.raises(KeyError):
        BudgetedKernel(MockKernel(), Ledger(), PriceTable({}), max_usd=1.0)


def test_the_cap_holds_when_calls_run_concurrently():
    """The check read the ledger before awaiting and `record_call` wrote it after, so up to
    `concurrency` calls passed on one stale balance - 16 by default, and every one of them billed."""
    prices = PriceTable({"mock": {"mock-1": Price(input_per_mtok=1000.0, output_per_mtok=0.0)}})
    ledger = Ledger()
    inner = SlowKernel()
    kernel = BudgetedKernel(inner, ledger, prices, max_usd=3.5)  # each call costs about $1

    results = asyncio.run(evaluate_many(kernel, [REQUEST] * 16, concurrency=16))
    refused = [result for result in results if isinstance(result, BudgetExceeded)]
    assert len(inner.seen) == 3 and len(refused) == 13
    assert ledger.usd == pytest.approx(3.0, rel=0.05) and ledger.usd <= 3.5
    assert ledger.reserved_usd == 0.0  # every reservation settled


def test_a_failing_call_releases_its_reservation():
    class Broken(MockKernel):
        async def evaluate(self, request):
            await asyncio.sleep(0)
            raise RuntimeError("boom")

    ledger = Ledger()
    kernel = BudgetedKernel(Broken(), ledger, PriceTable({"mock": {"mock-1": Price(input_per_mtok=1000.0, output_per_mtok=0.0)}}), max_usd=3.5)
    with pytest.raises(RuntimeError):
        asyncio.run(kernel.evaluate(REQUEST))
    assert (ledger.reserved_usd, ledger.committed_usd) == (0.0, 0.0)


def test_the_projection_prices_output_tokens():
    """Output was assumed free, so a model whose output is the expensive half had no cap at all."""
    prices = PriceTable({"mock": {"mock-1": Price(input_per_mtok=0.0, output_per_mtok=1000.0)}})
    ledger = Ledger()
    inner = MockKernel()
    kernel = BudgetedKernel(inner, ledger, prices, max_usd=estimate_output_tokens(REQUEST) * 1000.0 / 1e6 / 2)
    with pytest.raises(BudgetExceeded, match="projected at"):
        asyncio.run(kernel.evaluate(REQUEST))
    assert inner.seen == []


def test_the_projection_counts_every_sample_the_adapter_will_send():
    prices = PriceTable({"mock": {"mock-1": Price(input_per_mtok=1000.0, output_per_mtok=0.0)}})
    one = BudgetedKernel(MockKernel(), Ledger(), prices, max_usd=100.0, n_samples=1)
    five = BudgetedKernel(MockKernel(), Ledger(), prices, max_usd=100.0, n_samples=5)
    assert five.projected(REQUEST) == pytest.approx(5 * one.projected(REQUEST))
    # a cap between one call and five refuses before spending five times over it
    capped = BudgetedKernel(MockKernel(), Ledger(), prices, max_usd=3 * one.projected(REQUEST), n_samples=5)
    with pytest.raises(BudgetExceeded):
        asyncio.run(capped.evaluate(REQUEST))


def test_the_measured_overrun_is_gone():
    """The reviewer's case: five samples, output priced, sixteen concurrent calls, a $0.05 cap.

    Before: $0.3070 spent against the cap, 6.1x over. The mock kernel reports only input tokens, so
    this asserts on the projection the cap is enforced against as well as on what was recorded.
    """
    prices = PriceTable({"mock": {"mock-1": Price(input_per_mtok=3.0, output_per_mtok=15.0)}})
    ledger = Ledger()
    inner = SlowKernel()
    kernel = BudgetedKernel(inner, ledger, prices, max_usd=0.05, n_samples=5)
    asyncio.run(evaluate_many(kernel, [REQUEST] * 16, concurrency=16))
    assert len(inner.seen) * kernel.projected(REQUEST) <= 0.05
    assert ledger.usd <= 0.05
