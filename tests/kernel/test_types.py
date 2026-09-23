import pytest
from pydantic import ValidationError

from kite.kernel.types import PROB_SUM_TOLERANCE, KernelAnswer, KernelRequest, QuestionSpec, compact_json


def test_compact_json_keeps_insertion_order_and_raw_unicode():
    assert compact_json({"b": 1, "a": "café"}) == '{"b":1,"a":"café"}'


def test_choice_needs_between_2_and_255_options():
    QuestionSpec(type="choice", instructions="q", criteria={"a": None, "b": "desc"})
    with pytest.raises(ValidationError):
        QuestionSpec(type="choice", instructions="q", criteria={"only": None})
    with pytest.raises(ValidationError):
        QuestionSpec(type="choice", instructions="q", criteria={str(i): None for i in range(256)})


def test_score_needs_between_2_and_10_levels():
    QuestionSpec(type="score", instructions="q", criteria=["low", "high"])
    with pytest.raises(ValidationError):
        QuestionSpec(type="score", instructions="q", criteria=["only"])
    with pytest.raises(ValidationError):
        QuestionSpec(type="score", instructions="q", criteria=[str(i) for i in range(11)])


def test_noul_criteria_is_optional_but_keys_are_checked():
    QuestionSpec(type="noul", instructions="q")
    QuestionSpec(type="noul", instructions="q", criteria={"true": "yes means", "false": "no means"})
    with pytest.raises(ValidationError):
        QuestionSpec(type="noul", instructions="q", criteria={"maybe": "?"})


def test_outcomes_per_type():
    assert QuestionSpec(type="choice", instructions="q", criteria={"x": None, "y": None}).outcomes() == ["x", "y"]
    assert QuestionSpec(type="score", instructions="q", criteria=["a", "b", "c"]).outcomes() == ["0", "1", "2"]
    assert QuestionSpec(type="noul", instructions="q").outcomes() == ["yes", "no"]


def test_to_api_omits_missing_criteria():
    assert QuestionSpec(type="noul", instructions="q").to_api() == {"type": "noul", "instructions": "q"}
    spec = QuestionSpec(type="score", instructions={"question": "q"}, criteria=["a", "b"])
    assert spec.to_api() == {"type": "score", "instructions": {"question": "q"}, "criteria": ["a", "b"]}


def test_request_needs_at_least_one_question():
    with pytest.raises(ValidationError):
        KernelRequest(state="s", questions={})


def test_answer_from_api_noul():
    answer = KernelAnswer.from_api({"type": "noul", "noul": 0.92})
    assert answer.probs == {"yes": 0.92, "no": pytest.approx(0.08)}
    assert answer.value == 0.92
    assert answer.confidence is None


def test_answer_from_api_choice_and_score():
    choice = KernelAnswer.from_api({"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.7})
    assert (choice.value, choice.probs, choice.confidence) == ("b", {"a": 0.2, "b": 0.8}, 0.7)
    score = KernelAnswer.from_api({"type": "score", "score": 1.6, "legend": {"0": "x"}, "probabilities": {0: 0.1, 1: 0.2, 2: 0.7}, "confidence": 0.6})
    assert score.probs == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert score.value == 1.6


def test_answer_from_probs_computes_value_per_type():
    score_q = QuestionSpec(type="score", instructions="q", criteria=["a", "b", "c"])
    assert KernelAnswer.from_probs(score_q, {"0": 0.5, "1": 0.0, "2": 0.5}).value == pytest.approx(1.0)
    choice_q = QuestionSpec(type="choice", instructions="q", criteria={"x": None, "y": None})
    assert KernelAnswer.from_probs(choice_q, {"y": 0.9, "x": 0.1}).value == "y"
    noul_q = QuestionSpec(type="noul", instructions="q")
    assert KernelAnswer.from_probs(noul_q, {"yes": 0.3, "no": 0.7}).value == 0.3


def test_answer_from_probs_rejects_wrong_keys():
    choice_q = QuestionSpec(type="choice", instructions="q", criteria={"x": None, "y": None})
    with pytest.raises(ValueError):
        KernelAnswer.from_probs(choice_q, {"x": 1.0})


def test_an_answer_rejects_probabilities_that_are_not_numbers():
    """NaN used to pass every guard downstream and land in a reported metric."""
    for bad in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValidationError):
            KernelAnswer(type="choice", probs={"a": bad, "b": 1.0}, value="b")
    with pytest.raises(ValidationError, match="not a finite number"):
        KernelAnswer.from_api({"type": "choice", "choice": "a", "probabilities": {"a": float("nan"), "b": 1.0}})
    with pytest.raises(ValidationError, match="negative"):
        KernelAnswer.from_api({"type": "noul", "noul": 1.4})  # p(no) would be -0.4


def test_an_answer_knows_how_far_it_is_from_summing_to_one():
    """A sum outside tolerance is counted, not rejected: the run has to show it happened."""
    truncated = KernelAnswer(type="choice", probs={"a": 0.4, "b": 0.4}, value="a")
    assert truncated.normalization_error == pytest.approx(0.2) and truncated.is_normalized is False
    fine = KernelAnswer(type="choice", probs={"a": 0.4, "b": 0.6 - 1e-9}, value="b")
    assert fine.is_normalized is True and fine.normalization_error < PROB_SUM_TOLERANCE
