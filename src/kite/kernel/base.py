"""The Kernel protocol and helpers every backend and wrapper shares."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from kite.kernel.types import KernelRequest, KernelResponse, compact_json


@runtime_checkable
class Kernel(Protocol):
    """Anything that turns a state plus typed questions into probability distributions."""

    name: str
    model_id: str

    async def evaluate(self, request: KernelRequest) -> KernelResponse: ...


OUTPUT_TOKENS_PER_QUESTION = 48  # the JSON scaffolding, the chosen answer and a confidence
OUTPUT_TOKENS_PER_OUTCOME = 12  # one `"option label": 0.123,` line


def estimate_tokens(request: KernelRequest) -> int:
    """Conservative pre-call token estimate from the serialized payload length."""
    chars = len(compact_json(request.state))
    chars += sum(len(compact_json(question.to_api())) for question in request.questions.values())
    return max(1, math.ceil(chars / 3.5))


def estimate_output_tokens(request: KernelRequest) -> int:
    """Conservative pre-call estimate of the answer: one probability per outcome, per question.

    Jev's output is free, so this costs nothing there. For the LLM baselines output is the
    expensive half of most price tables, and a cap that assumes zero is not a cap at all.
    """
    return sum(OUTPUT_TOKENS_PER_QUESTION + OUTPUT_TOKENS_PER_OUTCOME * len(question.outcomes()) for question in request.questions.values())


async def evaluate_many(kernel: Kernel, requests: Sequence[KernelRequest], *, concurrency: int = 16) -> list[KernelResponse | BaseException]:
    """Evaluate requests concurrently, preserving order. Failures come back as exception objects."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(request: KernelRequest) -> KernelResponse:
        async with semaphore:
            return await kernel.evaluate(request)

    return await asyncio.gather(*(one(request) for request in requests), return_exceptions=True)
