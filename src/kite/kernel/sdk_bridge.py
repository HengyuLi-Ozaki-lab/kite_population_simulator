"""Conversions between our types and typesafe_sdk objects, shared by the Jev and adapter backends."""

from __future__ import annotations

from typing import Any

from kite.kernel.types import KernelAnswer, QuestionSpec


def to_sdk_question(spec: QuestionSpec) -> Any:
    """Build a typesafe_sdk question object. The LLM adapter dispatches on these classes."""
    from typesafe_sdk import Choice, Noul, Score

    if spec.type == "choice":
        return Choice(instructions=spec.instructions, criteria=spec.criteria)
    if spec.type == "score":
        return Score(instructions=spec.instructions, criteria=spec.criteria)
    return Noul(instructions=spec.instructions, criteria=spec.criteria)


def answer_from_sdk(answer: Any) -> KernelAnswer:
    """Convert a typesafe_sdk answer (or anything shaped like one) into a KernelAnswer."""
    kind = answer.type
    if kind == "noul":
        return KernelAnswer.from_api({"type": "noul", "noul": answer.noul})
    raw = {"type": kind, "probabilities": dict(answer.probabilities), "confidence": answer.confidence}
    if kind == "choice":
        raw["choice"] = answer.choice
    else:
        raw["score"] = answer.score
    return KernelAnswer.from_api(raw)
