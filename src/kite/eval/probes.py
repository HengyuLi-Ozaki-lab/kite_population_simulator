"""E0 probes: cheap experiments on the real model that decide engineering trade-offs.

Each probe manipulates one thing and measures how far the answer moves, as a total-variation
distance. On its own that number means little: asking Jev the identical question twice already
moves the answer (measured against the live API on 2026-09-20: TV 0.03-0.04 between identical
calls, partly because probabilities come back rounded to two decimals). So every probe also
measures that noise floor on the same cases and reports the **excess** over it. A manipulation
matters only when it moves the answer further than simply re-asking does.

Pass when the excess median is <= 0.02. Pass the *uncached* kernel: probes must reach the model
on every call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from kite.eval.metrics import total_variation
from kite.eval.phrasing import GENERIC_RESPONDENT, build_request, decode
from kite.eval.task import ProbeCase
from kite.kernel.base import Kernel, evaluate_many
from kite.kernel.types import KernelRequest, KernelResponse, QuestionSpec

EXCESS_LIMIT = 0.02
NOISE_REPEATS = 2

Probs = list[float]


def describe(distances: Sequence[float]) -> dict[str, Any]:
    if len(distances) == 0:
        return {"n": 0, "median": None, "p95": None, "max": None}
    values = np.asarray(distances, dtype=float)
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def summarize(effect: Sequence[float], noise: Sequence[float]) -> dict[str, Any]:
    """Effect distances next to the noise floor measured on the same cases."""
    report: dict[str, Any] = {"effect": describe(effect), "noise": describe(noise)}
    if report["effect"]["median"] is None or report["noise"]["median"] is None:
        return {**report, "excess_median": None, "passed": None}
    excess = report["effect"]["median"] - report["noise"]["median"]
    return {**report, "excess_median": excess, "passed": bool(excess <= EXCESS_LIMIT)}


async def _run(kernel: Kernel, requests: Sequence[KernelRequest], concurrency: int) -> list[KernelResponse]:
    results = await evaluate_many(kernel, requests, concurrency=concurrency)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results  # type: ignore[return-value]


def _probs(response: KernelResponse, options: list[str], key: str = "answer") -> Probs:
    return [response.answers[key].probs[option] for option in options]


def _solo(case: ProbeCase, options: list[str] | None = None) -> KernelRequest:
    return build_request(
        phrasing="p1",
        primitive="choice",
        persona=case.persona,
        question=case.question,
        options=options or case.options,
        context=case.context,
    )


def _pairwise(rounds: Sequence[Sequence[Probs]]) -> list[float]:
    """Distances between repeats of one condition: the noise floor."""
    return [total_variation(rounds[0][index], later[index]) for later in rounds[1:] for index in range(len(rounds[0]))]


async def _repeat(
    kernel: Kernel,
    build: Callable[[], list[KernelRequest]],
    read: Callable[[Sequence[KernelResponse]], list[Probs]],
    *,
    repeats: int,
    concurrency: int,
) -> list[list[Probs]]:
    return [read(await _run(kernel, build(), concurrency)) for _ in range(repeats)]


async def _against_solo(
    kernel: Kernel,
    cases: Sequence[ProbeCase],
    variant: Callable[[int, ProbeCase], KernelRequest],
    read_variant: Callable[[KernelResponse, ProbeCase], Probs],
    *,
    concurrency: int,
    repeats: int,
) -> dict[str, Any]:
    """The shared shape: one request per case in the plain condition, one in the manipulated one."""
    plain = await _repeat(
        kernel,
        lambda: [_solo(case) for case in cases],
        lambda responses: [_probs(response, case.options) for response, case in zip(responses, cases, strict=True)],
        repeats=repeats + 1,
        concurrency=concurrency,
    )
    variants = await _run(kernel, [variant(index, case) for index, case in enumerate(cases)], concurrency)
    effect = [
        total_variation(plain[0][index], read_variant(response, case)) for index, (response, case) in enumerate(zip(variants, cases, strict=True))
    ]
    return summarize(effect, _pairwise(plain))


async def repeat_noise(kernel: Kernel, cases: Sequence[ProbeCase], *, repeats: int = 5, concurrency: int = 8) -> dict:
    """How far the same request moves on its own. The control every other probe subtracts."""
    rounds = await _repeat(
        kernel,
        lambda: [_solo(case) for case in cases],
        lambda responses: [_probs(response, case.options) for response, case in zip(responses, cases, strict=True)],
        repeats=repeats,
        concurrency=concurrency,
    )
    return {"noise": describe(_pairwise(rounds)), "repeats": repeats}


async def irrelevant_field(kernel: Kernel, cases: Sequence[ProbeCase], *, concurrency: int = 8, repeats: int = NOISE_REPEATS) -> dict:
    """Add a meaningless `uid` to the state: is the model swayed by content carrying no information?"""

    def variant(index: int, case: ProbeCase) -> KernelRequest:
        request = _solo(case)
        return KernelRequest(state={**request.state, "uid": f"probe-{index:06d}"}, questions=request.questions)

    return await _against_solo(
        kernel, cases, variant, lambda response, case: _probs(response, case.options), concurrency=concurrency, repeats=repeats
    )


async def option_order(kernel: Kernel, cases: Sequence[ProbeCase], *, concurrency: int = 8, repeats: int = NOISE_REPEATS) -> dict:
    """Reverse the option list: a position-biased model answers differently."""
    return await _against_solo(
        kernel,
        cases,
        lambda index, case: _solo(case, list(reversed(case.options))),
        lambda response, case: decode(response, phrasing="p1", options=case.options),
        concurrency=concurrency,
        repeats=repeats,
    )


def _indexed(target: str, options: list[str]) -> QuestionSpec:
    return QuestionSpec(type="choice", instructions={"question": f"Which answer did {target}?"}, criteria=dict.fromkeys(options))


def _flatten(groups: Sequence[Sequence[ProbeCase]]) -> list[tuple[int, Sequence[ProbeCase], int, ProbeCase]]:
    """Every case with the *position* of its group, never the group's value.

    Looking the group up by value (`groups.index(group)`) resolves two value-equal groups to the same
    response, so every case in the second group would be compared against the first group's answer and
    score TV = 0 — a silent pass on the probe that decides whether per-question caching is sound.
    """
    return [(at, group, index, case) for at, group in enumerate(groups) for index, case in enumerate(group)]


async def batching_invariance(
    kernel: Kernel, cases: Sequence[ProbeCase], *, group_size: int = 5, concurrency: int = 8, repeats: int = NOISE_REPEATS
) -> dict:
    """K questions in one request versus K requests over the same state.

    Per-question caching is sound only if a question's answer does not depend on which other
    questions shared its request.
    """
    cases = list({case.question: case for case in cases}.values())  # identical questions would make this trivial
    groups = [cases[start : start + group_size] for start in range(0, len(cases) - group_size + 1, group_size)]
    if not groups:
        return summarize([], [])

    def state_of(group: Sequence[ProbeCase]) -> dict[str, Any]:
        return {"respondent": GENERIC_RESPONDENT, "survey": {"questions": [case.question for case in group]}}

    def specs_of(group: Sequence[ProbeCase]) -> dict[str, QuestionSpec]:
        return {f"q{index}": _indexed(f"`respondent` give to `survey.questions[{index}]`", case.options) for index, case in enumerate(group)}

    flat = _flatten(groups)

    together = await _repeat(
        kernel,
        lambda: [KernelRequest(state=state_of(group), questions=specs_of(group)) for group in groups],
        lambda responses: [_probs(responses[at], case.options, f"q{index}") for at, _, index, case in flat],
        repeats=repeats + 1,
        concurrency=concurrency,
    )
    apart = await _run(
        kernel,
        [KernelRequest(state=state_of(group), questions={f"q{index}": specs_of(group)[f"q{index}"]}) for _, group, index, _ in flat],
        concurrency,
    )
    effect = [
        total_variation(together[0][position], _probs(apart[position], case.options, f"q{index}"))
        for position, (_, _, index, case) in enumerate(flat)
    ]
    return summarize(effect, _pairwise(together))


async def packing_perturbation(
    kernel: Kernel,
    cases: Sequence[ProbeCase],
    *,
    sizes: Sequence[int] = (2, 4, 8, 16),
    concurrency: int = 8,
    repeats: int = NOISE_REPEATS,
) -> dict:
    """Several personas in one state versus one persona per state, on the same question.

    Packing is how a simulation would evaluate a whole population in few requests; this says
    whether a persona's answer survives having neighbours.
    """
    with_persona = [case for case in cases if case.persona]
    report: dict[str, Any] = {}
    for size in sizes:
        groups = [with_persona[start : start + size] for start in range(0, len(with_persona) - size + 1, size)]
        if not groups:
            continue
        flat = [(group, case) for group in groups for case in group]

        def solo(flat: list[tuple[Sequence[ProbeCase], ProbeCase]] = flat) -> list[KernelRequest]:
            return [
                KernelRequest(
                    state={"respondents": [case.persona], "survey": {"question": group[0].question}},
                    questions={"q0": _indexed("`respondents[0]` give to `survey.question`", group[0].options)},
                )
                for group, case in flat
            ]

        def read_solo(responses: Sequence[KernelResponse], flat: list[tuple[Sequence[ProbeCase], ProbeCase]] = flat) -> list[Probs]:
            return [_probs(response, group[0].options, "q0") for response, (group, _) in zip(responses, flat, strict=True)]

        alone = await _repeat(kernel, solo, read_solo, repeats=repeats + 1, concurrency=concurrency)
        packed = await _run(
            kernel,
            [
                KernelRequest(
                    state={"respondents": [case.persona for case in group], "survey": {"question": group[0].question}},
                    questions={f"q{index}": _indexed(f"`respondents[{index}]` give to `survey.question`", group[0].options) for index in range(size)},
                )
                for group in groups
            ],
            concurrency,
        )
        effect = [
            total_variation(alone[0][group_index * size + index], _probs(packed[group_index], group[0].options, f"q{index}"))
            for group_index, group in enumerate(groups)
            for index in range(len(group))
        ]
        report[f"k={size}"] = summarize(effect, _pairwise(alone))
    return report


async def latency(kernel: Kernel, cases: Sequence[ProbeCase], *, concurrency: int = 1) -> dict:
    responses = await _run(kernel, [_solo(case) for case in cases], concurrency)
    values = np.asarray([response.latency_ms for response in responses])
    tokens = np.asarray([response.usage.input_tokens for response in responses])
    return {
        "n": len(values),
        "concurrency": concurrency,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_input_tokens": float(tokens.mean()),
    }


PROBES = {
    "repeat_noise": repeat_noise,
    "irrelevant_field": irrelevant_field,
    "batching_invariance": batching_invariance,
    "packing_perturbation": packing_perturbation,
    "option_order": option_order,
    "latency": latency,
}
