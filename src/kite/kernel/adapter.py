"""LLM baselines through system-one-adapter: the same interface, backed by OpenAI or Anthropic."""

from __future__ import annotations

import asyncio
import time
from typing import Any

# the closed sets live next to KernelSpec, which is what a run records; `build` does not import this
# module at import time, so naming them from here cannot cycle
from kite.kernel.build import AnswerMode, Provider
from kite.kernel.sdk_bridge import answer_from_sdk, to_sdk_question
from kite.kernel.types import KernelAnswer, KernelRequest, KernelResponse, Usage


class AdapterKernel:
    """`probabilities`: the LLM states a distribution. `discrete`: it answers once per sample and
    `n_samples` answers are averaged into an empirical distribution."""

    def __init__(
        self,
        provider: Provider,
        model_id: str,
        *,
        answer_mode: AnswerMode = "probabilities",
        n_samples: int = 1,
        client: Any | None = None,
    ) -> None:
        self.name = f"adapter-{provider}"
        self.model_id = model_id
        self.options = f"mode={answer_mode};n={n_samples}"
        self._provider = provider
        self._n_samples = n_samples
        if client is None:
            from system_one_adapter import AsyncSystemOneAdapterClient

            client = AsyncSystemOneAdapterClient(
                structured_outputs=True,
                llm_answer_mode=answer_mode,
                normalize_probabilities=True,
                n_retry_malformed_structure=2,
                provider=provider,
                model=model_id,
            )
        self._client = client

    async def evaluate(self, request: KernelRequest) -> KernelResponse:
        questions = {name: to_sdk_question(spec) for name, spec in request.questions.items()}
        started = time.perf_counter()
        results = await asyncio.gather(*(self._client.system_one(request.state, questions) for _ in range(self._n_samples)))
        latency_ms = (time.perf_counter() - started) * 1000.0

        answers: dict[str, KernelAnswer] = {}
        for name, spec in request.questions.items():
            samples = [answer_from_sdk(result.answers[name]) for result in results]
            mean = {outcome: sum(sample.probs.get(outcome, 0.0) for sample in samples) / len(samples) for outcome in spec.outcomes()}
            answers[name] = KernelAnswer.from_probs(spec, mean)

        return KernelResponse(
            answers=answers,
            usage=Usage(
                input_tokens=sum(result.usage.input_tokens_total for result in results),
                output_tokens=sum(result.usage.output_tokens_total for result in results),
            ),
            latency_ms=latency_ms,
            model=self.model_id,
            backend=self.name,
            cached=dict.fromkeys(answers, False),
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()
