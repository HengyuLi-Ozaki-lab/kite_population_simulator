"""A network-free kernel: deterministic pseudo-distributions, or an injected function."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from kite.kernel.base import estimate_tokens
from kite.kernel.types import KernelAnswer, KernelRequest, KernelResponse, QuestionSpec, Usage, compact_json

ProbFn = Callable[[KernelRequest, str, QuestionSpec], dict[str, float]]


def hash_probs(request: KernelRequest, key: str, question: QuestionSpec) -> dict[str, float]:
    """A distribution derived from sha256(state, question): stable across runs and machines."""
    payload = compact_json(request.state) + "\x1f" + compact_json(question.to_api())
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    outcomes = question.outcomes()
    weights = [digest[index % len(digest)] + 1 for index in range(len(outcomes))]
    total = float(sum(weights))
    return {outcome: weight / total for outcome, weight in zip(outcomes, weights, strict=True)}


def uniform_probs(request: KernelRequest, key: str, question: QuestionSpec) -> dict[str, float]:
    outcomes = question.outcomes()
    return {outcome: 1.0 / len(outcomes) for outcome in outcomes}


class MockKernel:
    name = "mock"

    def __init__(self, model_id: str = "mock-1", fn: ProbFn | None = None) -> None:
        self.model_id = model_id
        self._fn = fn or hash_probs
        self.seen: list[KernelRequest] = []

    async def evaluate(self, request: KernelRequest) -> KernelResponse:
        self.seen.append(request)
        answers = {key: KernelAnswer.from_probs(question, self._fn(request, key, question)) for key, question in request.questions.items()}
        return KernelResponse(
            answers=answers,
            usage=Usage(input_tokens=estimate_tokens(request), output_tokens=0),
            model=self.model_id,
            backend=self.name,
            cached=dict.fromkeys(answers, False),
        )
