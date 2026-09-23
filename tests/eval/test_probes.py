import asyncio
import hashlib
import random

from kite.eval import probes
from kite.eval.task import ProbeCase
from kite.kernel.mock import MockKernel, uniform_probs
from kite.kernel.types import KernelAnswer, KernelResponse, Usage, compact_json

CASES = [
    ProbeCase(persona={"political_party": party, "age_group": age}, question=f"Question {index}?", options=["Yes", "No", "Unsure"])
    for index, (party, age) in enumerate([(p, a) for p in ("Democrat", "Republican") for a in ("18-29", "30-49", "50-64", "65+")] * 2)
]


def _spread(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {outcome: weight / total for outcome, weight in weights.items()}


def _digest(*parts: str) -> int:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[0]


def by_option_name(request, key, question):
    """Weights from the state and each option's *name*: different per case, blind to their order.

    `uniform_probs` answers every request identically, so a probe driven by it passes however it
    pairs the responses up - deleting the re-alignment in `option_order` leaves it green. This
    kernel varies case by case, so a probe that compares the wrong pair sees a difference.
    """
    state = compact_json(request.state)
    return _spread({outcome: 1 + _digest(state, outcome) for outcome in question.outcomes()})


def by_position(request, key, question):
    """Most of the mass on whichever option is listed first: a position-biased model."""
    outcomes = question.outcomes()
    return _spread({outcome: 0.5**index for index, outcome in enumerate(outcomes)})


def swayed_by_neighbours(request, key, question):
    """Leans harder on the first option the more personas share the state."""
    crowd = len(request.state.get("respondents", [None])) if isinstance(request.state, dict) else 1
    return _spread({outcome: float(crowd if index == 0 else 1) for index, outcome in enumerate(question.outcomes())})


def about_the_respondent_asked(request, key, question):
    """Answers about the persona the question names, whoever else shares the state."""
    packed = request.state.get("respondents")
    who = packed[int(key.removeprefix("q"))] if packed else request.state.get("respondent")
    return _spread({outcome: 1 + _digest(compact_json(who), outcome) for outcome in question.outcomes()})


class NoisyKernel:
    """Answers the same request differently every time, like the real model does."""

    name, model_id = "noisy", "noisy-1"

    def __init__(self, spread: float = 0.06, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._spread = spread

    async def evaluate(self, request):
        answers = {}
        for key, question in request.questions.items():
            outcomes = question.outcomes()
            weights = [1.0 + self._rng.uniform(0, self._spread) for _ in outcomes]
            total = sum(weights)
            answers[key] = KernelAnswer.from_probs(question, {o: w / total for o, w in zip(outcomes, weights, strict=True)})
        return KernelResponse(answers=answers, usage=Usage(input_tokens=100), model=self.model_id, backend=self.name)


def test_describe_and_summarize():
    assert probes.describe([]) == {"n": 0, "median": None, "p95": None, "max": None}
    assert probes.describe([0.1, 0.2, 0.3])["median"] == 0.2
    # a manipulation that moves the answer no further than re-asking does
    assert probes.summarize([0.04, 0.05, 0.06], [0.03, 0.05, 0.07])["passed"] is True
    # one that moves it much further
    verdict = probes.summarize([0.3, 0.3, 0.3], [0.01, 0.01, 0.01])
    assert verdict["passed"] is False and verdict["excess_median"] == 0.29
    assert probes.summarize([], [0.1])["passed"] is None


def test_repeat_noise_is_zero_for_a_deterministic_kernel_and_positive_for_a_noisy_one():
    quiet = asyncio.run(probes.repeat_noise(MockKernel(), CASES, repeats=3))
    assert quiet["noise"]["max"] == 0.0 and quiet["noise"]["n"] == 2 * len(CASES)
    loud = asyncio.run(probes.repeat_noise(NoisyKernel(), CASES, repeats=3))
    assert loud["noise"]["median"] > 0.005


def test_irrelevant_field_is_caught_when_the_kernel_reads_the_whole_state():
    # MockKernel hashes the entire state, so an added field changes the answer completely
    caught = asyncio.run(probes.irrelevant_field(MockKernel(), CASES))
    assert caught["noise"]["max"] == 0.0
    assert caught["excess_median"] > 0.02 and caught["passed"] is False
    # a kernel that ignores the state is unaffected
    clean = asyncio.run(probes.irrelevant_field(MockKernel(fn=uniform_probs), CASES))
    assert clean["excess_median"] == 0.0 and clean["passed"] is True


def test_a_noisy_kernel_passes_because_the_effect_is_no_bigger_than_its_own_noise():
    """The point of the redesign: plain jitter must not be reported as sensitivity."""
    verdict = asyncio.run(probes.irrelevant_field(NoisyKernel(), CASES))
    assert verdict["noise"]["median"] > 0.005  # the kernel really is noisy
    assert verdict["passed"] is True  # yet the manipulation adds nothing on top


def test_batching_invariance_holds_for_a_kernel_that_isolates_questions():
    kernel = MockKernel()
    verdict = asyncio.run(probes.batching_invariance(kernel, CASES, group_size=4))
    assert verdict["effect"]["max"] == 0.0 and verdict["passed"] is True
    assert verdict["effect"]["n"] == 16  # 16 distinct questions, four groups of four
    assert len(kernel.seen[0].questions) == 4  # asked together first
    assert len(kernel.seen[-1].questions) == 1  # and one at a time afterwards


def test_batching_invariance_needs_a_full_group():
    assert asyncio.run(probes.batching_invariance(MockKernel(), CASES[:2], group_size=5))["passed"] is None


def test_flatten_gives_each_group_its_own_position_even_when_groups_are_equal():
    """Looking the group up by value collapses equal groups onto the first one's response, and TV = 0.

    Deduplicating cases by question text currently keeps `batching_invariance` from ever building two
    equal groups, but that is a non-local invariant: this pins the indexing itself.
    """
    case = ProbeCase(persona=None, question="Same question?", options=["Yes", "No"])
    groups = [[case, case], [case, case]]  # value-equal, and `groups.index` would answer 0 for both
    assert [at for at, _, _, _ in probes._flatten(groups)] == [0, 0, 1, 1]
    assert [index for _, _, index, _ in probes._flatten(groups)] == [0, 1, 0, 1]
    assert all(group is groups[at] for at, group, _, _ in probes._flatten(groups))


class BatchSensitiveKernel(MockKernel):
    """Answers a question differently depending on how many questions shared its request."""

    async def evaluate(self, request):
        response = await super().evaluate(request)
        if len(request.questions) > 1:
            for key, answer in response.answers.items():
                flipped = {option: prob for option, prob in zip(answer.probs, reversed(list(answer.probs.values())), strict=True)}
                response.answers[key] = KernelAnswer.from_probs(request.questions[key], flipped)
        return response


def test_batching_invariance_fails_for_a_kernel_whose_answers_depend_on_their_neighbours():
    verdict = asyncio.run(probes.batching_invariance(BatchSensitiveKernel(), CASES, group_size=4))
    assert verdict["noise"]["max"] == 0.0  # the kernel is still deterministic
    assert verdict["effect"]["median"] > 0.02 and verdict["passed"] is False


def test_packing_perturbation_reports_each_group_size():
    report = asyncio.run(probes.packing_perturbation(MockKernel(fn=uniform_probs), CASES, sizes=(2, 4, 32)))
    assert set(report) == {"k=2", "k=4"}  # 32 personas do not fit in 16 cases
    assert report["k=4"]["effect"]["max"] == 0.0 and report["k=4"]["effect"]["n"] == 16
    assert report["k=4"]["passed"] is True


def test_packing_perturbation_stays_quiet_for_a_kernel_that_answers_about_the_persona_it_was_asked_about():
    """The quiet case with a kernel that actually reads the state: packing moves nothing."""
    kernel = MockKernel(fn=about_the_respondent_asked)
    report = asyncio.run(probes.packing_perturbation(kernel, CASES, sizes=(2, 4)))
    assert {size: report[size]["effect"]["max"] for size in report} == {"k=2": 0.0, "k=4": 0.0}
    assert all(report[size]["passed"] for size in report)
    # and it does not answer everyone the same way, so there was something for the probe to get wrong
    answers = [asyncio.run(kernel.evaluate(probes._solo(case))).answers["answer"].probs for case in CASES[:2]]
    assert answers[0] != answers[1]


def test_packing_perturbation_flags_a_kernel_swayed_by_the_other_personas():
    report = asyncio.run(probes.packing_perturbation(MockKernel(fn=swayed_by_neighbours), CASES, sizes=(2, 4)))
    for size in ("k=2", "k=4"):
        assert report[size]["noise"]["max"] == 0.0  # the kernel is deterministic
        assert report[size]["effect"]["median"] > 0.02 and report[size]["passed"] is False
    assert report["k=4"]["effect"]["median"] > report["k=2"]["effect"]["median"]  # a bigger crowd moves it further


def test_option_order_flags_a_position_biased_kernel():
    verdict = asyncio.run(probes.option_order(MockKernel(fn=by_position), CASES))
    assert verdict["noise"]["max"] == 0.0  # the kernel is deterministic
    assert verdict["effect"]["median"] > 0.02 and verdict["passed"] is False


def test_option_order_stays_quiet_for_a_kernel_that_reads_the_option_names():
    verdict = asyncio.run(probes.option_order(MockKernel(fn=by_option_name), CASES))
    assert verdict["effect"]["max"] == 0.0 and verdict["effect"]["n"] == len(CASES)
    assert verdict["passed"] is True


def test_option_order_and_latency():
    order = asyncio.run(probes.option_order(MockKernel(fn=uniform_probs), CASES))
    assert order["passed"] is True
    timing = asyncio.run(probes.latency(MockKernel(), CASES))
    assert set(timing) == {"n", "concurrency", "p50_ms", "p95_ms", "mean_input_tokens"}
    assert timing["mean_input_tokens"] > 0
    assert set(probes.PROBES) == {
        "repeat_noise",
        "irrelevant_field",
        "batching_invariance",
        "packing_perturbation",
        "option_order",
        "latency",
    }
