"""Typed request and response models shared by every kernel backend."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonValue = Any
QuestionType = Literal["choice", "score", "noul"]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
PROB_SUM_TOLERANCE = 1e-3  # the spec's tolerance for "these probabilities sum to 1"


def compact_json(value: JsonValue) -> str:
    """Serialize a value exactly as it is sent: insertion order, compact separators, raw Unicode."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class QuestionSpec(BaseModel):
    """One typed question, mirroring the TypeSafe API's raw question dictionary."""

    model_config = ConfigDict(frozen=True)

    type: QuestionType
    instructions: JsonValue
    criteria: JsonValue = None

    @model_validator(mode="after")
    def _check_criteria(self) -> QuestionSpec:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice criteria must be a dict with 2..{MAX_CHOICE_OPTIONS} options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
                raise ValueError(f"score criteria must be a list with {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels")
        elif self.criteria is not None:
            if not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}:
                raise ValueError("noul criteria must be None or a dict with keys 'true' and/or 'false'")
        return self

    def outcomes(self) -> list[str]:
        """Probability keys an answer to this question carries, in order."""
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(index) for index in range(len(self.criteria))]
        return ["yes", "no"]

    def to_api(self) -> dict[str, Any]:
        """The raw question dictionary accepted by POST /v1/systemone."""
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


class KernelRequest(BaseModel):
    """A state plus named questions. Dict order is the order sent to the model."""

    model_config = ConfigDict(frozen=True)

    state: JsonValue
    questions: dict[str, QuestionSpec] = Field(min_length=1)


class KernelAnswer(BaseModel):
    """A probability distribution over a question's outcomes."""

    model_config = ConfigDict(frozen=True)

    type: QuestionType
    probs: dict[str, float]
    value: str | float
    confidence: float | None = None

    @model_validator(mode="after")
    def _check_probs(self) -> KernelAnswer:
        """Where a backend's answer enters the system, and so where a malformed one has to stop.

        Non-finite and negative entries are rejected: NaN passes every guard further down - it is
        neither negative nor a non-positive sum, so `as_dist` divides by a NaN total and returns
        NaN - and ends up in a reported metric, indistinguishable from a number. A sum away from 1
        is counted rather than rejected; see `normalization_error` and `Ledger.record_answers`.
        """
        if not self.probs:
            raise ValueError("an answer needs at least one outcome")
        for outcome, probability in self.probs.items():
            if not math.isfinite(probability):
                raise ValueError(f"probability for {outcome!r} is not a finite number: {probability!r}")
            if probability < 0.0:
                raise ValueError(f"probability for {outcome!r} is negative: {probability!r}")
        return self

    @property
    def normalization_error(self) -> float:
        """How far the probabilities are from summing to 1."""
        return abs(math.fsum(self.probs.values()) - 1.0)

    @property
    def is_normalized(self) -> bool:
        return self.normalization_error <= PROB_SUM_TOLERANCE

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> KernelAnswer:
        """Build from one entry of the API's `answers` map."""
        kind = raw["type"]
        if kind == "noul":
            p_yes = float(raw["noul"])
            return cls(type="noul", probs={"yes": p_yes, "no": 1.0 - p_yes}, value=p_yes)
        probs = {str(key): float(value) for key, value in raw["probabilities"].items()}
        if kind == "choice":
            return cls(type="choice", probs=probs, value=str(raw["choice"]), confidence=raw.get("confidence"))
        if kind == "score":
            return cls(type="score", probs=probs, value=float(raw["score"]), confidence=raw.get("confidence"))
        raise ValueError(f"unknown answer type: {kind!r}")

    @classmethod
    def from_probs(cls, question: QuestionSpec, probs: dict[str, float]) -> KernelAnswer:
        """Build from a distribution over `question.outcomes()` (used by mock and table backends)."""
        outcomes = question.outcomes()
        if set(probs) != set(outcomes):
            raise ValueError(f"probs keys {sorted(probs)} do not match outcomes {sorted(outcomes)}")
        ordered = {key: float(probs[key]) for key in outcomes}
        if question.type == "noul":
            return cls(type="noul", probs=ordered, value=ordered["yes"])
        top = max(ordered.values())
        if question.type == "choice":
            return cls(type="choice", probs=ordered, value=max(outcomes, key=ordered.__getitem__), confidence=top)
        expectation = sum(int(key) * p for key, p in ordered.items())
        return cls(type="score", probs=ordered, value=expectation, confidence=top)


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0


class KernelResponse(BaseModel):
    """Answers keyed like the request's questions, plus accounting metadata."""

    model_config = ConfigDict(frozen=True)

    answers: dict[str, KernelAnswer]
    usage: Usage = Usage()
    latency_ms: float = 0.0
    model: str
    backend: str
    cached: dict[str, bool] = Field(default_factory=dict)
    request_id: str | None = None
