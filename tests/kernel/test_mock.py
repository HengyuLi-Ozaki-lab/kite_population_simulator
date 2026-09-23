import asyncio

import pytest

from kite.kernel.base import Kernel, estimate_tokens, evaluate_many
from kite.kernel.mock import MockKernel, uniform_probs
from kite.kernel.types import KernelRequest, QuestionSpec


def make_request(state="hello", n_options=3) -> KernelRequest:
    criteria = {f"option {i}": None for i in range(n_options)}
    return KernelRequest(state=state, questions={"q": QuestionSpec(type="choice", instructions="pick", criteria=criteria)})


def test_mock_is_a_kernel_and_is_deterministic():
    kernel = MockKernel()
    assert isinstance(kernel, Kernel)
    first = asyncio.run(kernel.evaluate(make_request()))
    second = asyncio.run(MockKernel().evaluate(make_request()))
    assert first.answers["q"].probs == second.answers["q"].probs
    assert sum(first.answers["q"].probs.values()) == pytest.approx(1.0)
    assert first.backend == "mock" and first.model == "mock-1"
    assert first.cached == {"q": False}


def test_mock_depends_on_state():
    kernel = MockKernel()
    a = asyncio.run(kernel.evaluate(make_request("state a")))
    b = asyncio.run(kernel.evaluate(make_request("state b")))
    assert a.answers["q"].probs != b.answers["q"].probs


def test_uniform_function():
    response = asyncio.run(MockKernel(model_id="uniform", fn=uniform_probs).evaluate(make_request(n_options=4)))
    assert list(response.answers["q"].probs.values()) == [0.25] * 4


def test_estimate_tokens_grows_with_payload():
    assert estimate_tokens(make_request("x" * 700)) > estimate_tokens(make_request("x"))


def test_evaluate_many_keeps_order_and_returns_exceptions():
    class Flaky:
        name, model_id = "flaky", "f-1"

        async def evaluate(self, request):
            if request.state == "boom":
                raise RuntimeError("boom")
            return await MockKernel().evaluate(request)

    results = asyncio.run(evaluate_many(Flaky(), [make_request("a"), make_request("boom"), make_request("c")], concurrency=2))
    assert results[0].answers and results[2].answers
    assert isinstance(results[1], RuntimeError)
