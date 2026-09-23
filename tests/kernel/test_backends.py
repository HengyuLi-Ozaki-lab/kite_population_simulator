"""JevKernel and AdapterKernel against fake clients: no network, no API keys."""

import asyncio
from types import SimpleNamespace

import pytest

from kite.kernel.adapter import AdapterKernel
from kite.kernel.jev import JevKernel
from kite.kernel.sdk_bridge import answer_from_sdk, to_sdk_question
from kite.kernel.types import KernelRequest, QuestionSpec

REQUEST = KernelRequest(
    state={"ticket": "charged twice"},
    questions={
        "urgent": QuestionSpec(type="noul", instructions="Is it urgent?"),
        "team": QuestionSpec(type="choice", instructions="Which team?", criteria={"billing": None, "tech": None}),
        "anger": QuestionSpec(type="score", instructions="How angry?", criteria=["calm", "annoyed", "furious"]),
    },
)


def sdk_like_answers(team_probs):
    return {
        "urgent": SimpleNamespace(type="noul", noul=0.9),
        "team": SimpleNamespace(type="choice", choice="billing", probabilities=team_probs, confidence=0.8),
        "anger": SimpleNamespace(type="score", score=1.5, probabilities={0: 0.1, 1: 0.3, 2: 0.6}, confidence=0.5),
    }


def test_to_sdk_question_builds_real_sdk_objects():
    for spec in REQUEST.questions.values():
        assert to_sdk_question(spec).model_dump() == spec.to_api()


def test_answer_from_sdk_stringifies_score_levels():
    answer = answer_from_sdk(SimpleNamespace(type="score", score=1.5, probabilities={0: 0.25, 1: 0.75}, confidence=0.4))
    assert answer.probs == {"0": 0.25, "1": 0.75}


class FakeJevClient:
    def __init__(self):
        self.calls = []

    async def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))

        class Result:
            answers = sdk_like_answers({"billing": 0.85, "tech": 0.15})
            usage = SimpleNamespace(input_tokens=312, output_tokens=None)
            model = "jev-1.13.0"

            @property
            def request_id(self):
                raise RuntimeError("no request id header")

        return Result()


def test_jev_kernel_converts_answers_and_pins_the_model():
    client = FakeJevClient()
    response = asyncio.run(JevKernel(client=client).evaluate(REQUEST))
    state, questions, model = client.calls[0]
    assert model == "jev-1.13.0"
    assert state == {"ticket": "charged twice"}
    assert questions["team"].model_dump() == REQUEST.questions["team"].to_api()
    assert response.answers["urgent"].probs["yes"] == 0.9
    assert response.answers["team"].value == "billing"
    assert response.answers["anger"].probs == {"0": 0.1, "1": 0.3, "2": 0.6}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (312, 0)
    assert response.request_id is None
    assert response.backend == "jev"


class FakeAdapterClient:
    def __init__(self):
        self.n = 0

    async def system_one(self, state, questions):
        self.n += 1
        probs = {"billing": 1.0, "tech": 0.0} if self.n % 2 else {"billing": 0.0, "tech": 1.0}
        return SimpleNamespace(
            answers=sdk_like_answers(probs),
            usage=SimpleNamespace(input_tokens_total=100, output_tokens_total=20),
        )


def test_adapter_kernel_averages_samples_and_sums_usage():
    kernel = AdapterKernel("openai", "some-model", answer_mode="discrete", n_samples=4, client=FakeAdapterClient())
    response = asyncio.run(kernel.evaluate(REQUEST))
    assert kernel.name == "adapter-openai"
    assert kernel.options == "mode=discrete;n=4"
    assert response.answers["team"].probs == {"billing": pytest.approx(0.5), "tech": pytest.approx(0.5)}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (400, 80)
    assert response.model == "some-model"
