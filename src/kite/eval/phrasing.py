"""Turn "which option would this person pick?" into kernel requests, and decode the answers.

Three phrasings are compared on a dev split and one is frozen before any test run:
  p1  third-person prediction (the default candidate: it matches what calibration means)
  p2  first-person role play
  p3  one yes/no question per option, renormalized
For ordinal options the primitive can be `choice` or `score`; score allows at most ten levels.
"""

from __future__ import annotations

from enum import StrEnum

from kite.kernel.types import MAX_SCORE_LEVELS, KernelRequest, KernelResponse, QuestionSpec


class Phrasing(StrEnum):
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"


class Primitive(StrEnum):
    choice = "choice"
    score = "score"


GENERIC_RESPONDENT = "a randomly selected adult living in the United States"
ANSWER_KEY = "answer"

_INSTRUCTIONS = {
    "p1": {
        "question": "Which answer did `respondent` give to `survey.question`?",
        "focus": "Predict this particular respondent's own answer, not the most reasonable or most socially desirable one.",
    },
    "p2": {
        "question": "You are the person described in `you`. Which answer do you give to `survey.question`?",
        "focus": "Answer as this person actually would, not as an ideal respondent would.",
    },
}


def effective_primitive(primitive: Primitive, n_options: int) -> Primitive:
    return "choice" if n_options > MAX_SCORE_LEVELS else primitive


def build_request(
    *,
    phrasing: Phrasing,
    primitive: Primitive,
    persona: dict[str, str] | None,
    question: str,
    options: list[str],
    context: str | None = None,
) -> KernelRequest:
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("options must be at least two distinct strings")
    survey: dict[str, str] = {}
    if context:
        survey["context"] = context
    survey["question"] = question
    who = persona if persona else GENERIC_RESPONDENT
    state = {"you" if phrasing == "p2" else "respondent": who, "survey": survey}

    if phrasing == "p3":
        questions = {
            f"option_{index}": QuestionSpec(
                type="noul",
                instructions={"question": "Did `respondent` give this answer to `survey.question`?", "answer": option},
            )
            for index, option in enumerate(options)
        }
        return KernelRequest(state=state, questions=questions)

    if effective_primitive(primitive, len(options)) == "score":
        spec = QuestionSpec(type="score", instructions=_INSTRUCTIONS[phrasing], criteria=list(options))
    else:
        spec = QuestionSpec(type="choice", instructions=_INSTRUCTIONS[phrasing], criteria=dict.fromkeys(options))
    return KernelRequest(state=state, questions={ANSWER_KEY: spec})


def decode(response: KernelResponse, *, phrasing: Phrasing, options: list[str]) -> list[float]:
    """Probabilities aligned with `options`, whatever phrasing and primitive produced them."""
    if phrasing == "p3":
        yes = [response.answers[f"option_{index}"].probs["yes"] for index in range(len(options))]
        total = sum(yes)
        return [p / total for p in yes] if total > 0 else [1.0 / len(options)] * len(options)
    answer = response.answers[ANSWER_KEY]
    if answer.type == "score":
        return [answer.probs[str(index)] for index in range(len(options))]
    return [answer.probs[option] for option in options]
