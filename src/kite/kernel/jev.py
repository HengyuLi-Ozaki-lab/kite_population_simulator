"""The real thing: TypeSafe's Jev through the official async SDK."""

from __future__ import annotations

import time
from typing import Any

from kite.kernel.sdk_bridge import answer_from_sdk, to_sdk_question
from kite.kernel.types import KernelRequest, KernelResponse, Usage

PINNED_MODEL = "jev-1.13.0"


class JevKernel:
    """Pins a versioned model id; never use the `jev-latest` alias for published numbers."""

    name = "jev"

    def __init__(self, model_id: str = PINNED_MODEL, client: Any | None = None) -> None:
        self.model_id = model_id
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            client = AsyncTypeSafeClient(model=model_id)
        self._client = client

    async def evaluate(self, request: KernelRequest) -> KernelResponse:
        questions = {name: to_sdk_question(spec) for name, spec in request.questions.items()}
        started = time.perf_counter()
        result = await self._client.system_one(request.state, questions, model=self.model_id)
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            request_id = result.request_id
        except Exception:
            request_id = None
        return KernelResponse(
            answers={name: answer_from_sdk(answer) for name, answer in result.answers.items()},
            usage=Usage(input_tokens=result.usage.input_tokens or 0, output_tokens=result.usage.output_tokens or 0),
            latency_ms=latency_ms,
            model=result.model,
            backend=self.name,
            cached=dict.fromkeys(result.answers, False),
            request_id=request_id,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()
