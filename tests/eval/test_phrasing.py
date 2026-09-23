import asyncio

import pytest

from kite.eval.phrasing import GENERIC_RESPONDENT, build_request, decode, effective_primitive
from kite.kernel.mock import MockKernel
from kite.kernel.types import KernelAnswer, KernelResponse

OPTIONS = ["Strongly agree", "Agree", "Disagree"]
PERSONA = {"political_party": "Democrat"}


def test_p1_choice_request_snapshot():
    request = build_request(
        phrasing="p1", primitive="choice", persona=PERSONA, question="Is the economy good?", options=OPTIONS, context="You read X."
    )
    assert request.state == {
        "respondent": {"political_party": "Democrat"},
        "survey": {"context": "You read X.", "question": "Is the economy good?"},
    }
    spec = request.questions["answer"]
    assert spec.type == "choice"
    assert spec.criteria == {"Strongly agree": None, "Agree": None, "Disagree": None}
    assert spec.instructions["question"] == "Which answer did `respondent` give to `survey.question`?"


def test_missing_persona_and_context():
    request = build_request(phrasing="p1", primitive="choice", persona=None, question="Q?", options=OPTIONS)
    assert request.state == {"respondent": GENERIC_RESPONDENT, "survey": {"question": "Q?"}}


def test_p2_uses_you_and_score_uses_a_list():
    request = build_request(phrasing="p2", primitive="score", persona=PERSONA, question="Q?", options=OPTIONS)
    assert "you" in request.state and "respondent" not in request.state
    spec = request.questions["answer"]
    assert (spec.type, spec.criteria) == ("score", OPTIONS)
    assert "`you`" in spec.instructions["question"]


def test_score_falls_back_to_choice_above_ten_options():
    many = [f"option {i}" for i in range(11)]
    assert effective_primitive("score", 11) == "choice"
    assert build_request(phrasing="p1", primitive="score", persona=None, question="Q?", options=many).questions["answer"].type == "choice"


def test_p3_builds_one_noul_per_option():
    request = build_request(phrasing="p3", primitive="choice", persona=PERSONA, question="Q?", options=OPTIONS)
    assert list(request.questions) == ["option_0", "option_1", "option_2"]
    assert request.questions["option_1"].type == "noul"
    assert request.questions["option_1"].instructions["answer"] == "Agree"


def test_options_must_be_distinct():
    with pytest.raises(ValueError):
        build_request(phrasing="p1", primitive="choice", persona=None, question="Q?", options=["a", "a"])
    with pytest.raises(ValueError):
        build_request(phrasing="p1", primitive="choice", persona=None, question="Q?", options=["a"])


@pytest.mark.parametrize(("phrasing", "primitive"), [("p1", "choice"), ("p1", "score"), ("p2", "choice"), ("p3", "choice")])
def test_decode_round_trip_through_the_mock_kernel(phrasing, primitive):
    request = build_request(phrasing=phrasing, primitive=primitive, persona=PERSONA, question="Q?", options=OPTIONS)
    response = asyncio.run(MockKernel().evaluate(request))
    probs = decode(response, phrasing=phrasing, options=OPTIONS)
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0)


def test_decode_p3_normalizes_and_handles_all_zero():
    def response(yes_values):
        answers = {f"option_{i}": KernelAnswer(type="noul", probs={"yes": p, "no": 1 - p}, value=p) for i, p in enumerate(yes_values)}
        return KernelResponse(answers=answers, model="m", backend="b")

    assert decode(response([0.2, 0.2, 0.4]), phrasing="p3", options=OPTIONS) == pytest.approx([0.25, 0.25, 0.5])
    assert decode(response([0.0, 0.0, 0.0]), phrasing="p3", options=OPTIONS) == pytest.approx([1 / 3] * 3)
