"""The strings below are real SocSci210 `stimuli` endings, sampled from the dataset."""

import pytest

from kite.eval.scales import Scale, parse_scale, strip_response_instruction

LIKERT = (
    'You read "Jaime is 20 years old." and then were asked: "How likely do you think it is that Jaime will '
    "still identify as non-binary in 5 years? Only return an integer from 1 to 7, where 1 means Extremely "
    'unlikely and 7 means Extremely likely, nothing else."'
)


def test_range_with_endpoint_labels():
    scale = parse_scale(LIKERT)
    assert (scale.lo, scale.hi, scale.source) == (1, 7, "parsed")
    assert scale.labels == {1: "Extremely unlikely", 7: "Extremely likely"}


def test_range_without_labels():
    scale = parse_scale("“The U.S. government should act.” Only return an integer from 1 to 7, nothing else.")
    assert (scale.lo, scale.hi, scale.labels) == (1, 7, {})


def test_enumerated_numbers():
    scale = parse_scale("Did winnings increase (1), decrease (2), or not change (3) it? Only return 1, 2, or 3, nothing else.")
    assert (scale.lo, scale.hi) == (1, 3)


def test_for_labels_with_sentences_and_semicolons():
    scale = parse_scale(
        "Only return 1 for 'Because the doctor felt it was the right thing to do.'; 2 for 'Because the doctor "
        "wanted to maintain a strong doctor-patient relationship with me.'; 3 for 'Because the doctor was hoping "
        "I would not file a malpractice lawsuit.', nothing else."
    )
    assert (scale.lo, scale.hi) == (1, 3)
    assert scale.labels[1] == "Because the doctor felt it was the right thing to do"
    assert scale.labels[3] == "Because the doctor was hoping I would not file a malpractice lawsuit"


def test_for_labels_joined_by_or():
    assert parse_scale("Did you choose it? Only return 1 for Yes or 2 for No.").labels == {1: "Yes", 2: "No"}
    scale = parse_scale(
        "Only return 1 for mostly by human activity; 2 for mostly by natural causes; 3 for about equally by human "
        "activity and natural causes; or 4 for global warming has not been occurring, nothing else."
    )
    assert scale.labels[4] == "global warming has not been occurring"
    assert len(scale.labels) == 4


def test_labels_after_nothing_else_with_curly_quotes():
    scale = parse_scale(
        "Only return an integer from 1 to 5, nothing else, where 1 means “Agree strongly,” 2 means “Agree,” "
        "3 means “Disagree,” 4 means “Disagree strongly,” and 5 means “Not sure.”"
    )
    assert scale.labels == {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Disagree strongly", 5: "Not sure"}


def test_equals_sign_labels_and_labels_that_start_with_digits():
    scale = parse_scale("Only return an integer from 1 to 4, nothing else, where 1 = “Very serious” and 4 = “Not a threat at all”.")
    assert scale.labels == {1: "Very serious", 4: "Not a threat at all"}
    scale = parse_scale("How often? Only return an integer from 1 to 5, where 1 = '0 days' and 5 = '7 days'.")
    assert scale.labels == {1: "0 days", 5: "7 days"}


def test_hundred_point_scale_with_a_midpoint_label():
    scale = parse_scale("Only return an integer from 0 to 100 where 0 means strongly disagree, 50 means neutral, and 100 means strongly agree.")
    assert (scale.lo, scale.hi) == (0, 100)
    assert scale.labels == {0: "strongly disagree", 50: "neutral", 100: "strongly agree"}


def test_no_range_means_no_scale():
    assert parse_scale("How much would you pay? Only return an integer representing dollars, nothing else.") is None
    assert parse_scale("A question with no instruction at all?") is None


def test_strip_response_instruction():
    stripped = strip_response_instruction(LIKERT)
    assert stripped.endswith("in 5 years?")
    assert "Only return" not in stripped
    assert strip_response_instruction("No instruction here.") == "No instruction here."


def test_short_scales_get_one_described_option_per_level():
    scale = parse_scale(LIKERT)
    assert scale.bins() == [(k, k) for k in range(1, 8)]
    descriptions = scale.descriptions()
    assert descriptions[0] == "1: Extremely unlikely"
    assert descriptions[3] == "4: exactly the middle on the scale from 1 (Extremely unlikely) to 7 (Extremely likely)"
    assert descriptions[6] == "7: Extremely likely"
    assert len(set(descriptions)) == 7


def test_long_scales_are_binned_into_ten_options():
    scale = Scale(lo=0, hi=100)
    bins = scale.bins()
    assert len(bins) == 10
    assert bins[0] == (0, 10) and bins[1] == (11, 20) and bins[-1] == (91, 100)
    assert sum(last - first + 1 for first, last in bins) == 101
    descriptions = scale.descriptions()
    assert descriptions[0].startswith("0 to 10: ")
    assert len(set(descriptions)) == 10


def test_expand_spreads_bin_mass_evenly():
    scale = Scale(lo=0, hi=100)
    levels = scale.expand([1.0] + [0.0] * 9)
    assert len(levels) == 101
    assert levels[0] == pytest.approx(1 / 11) and levels[10] == pytest.approx(1 / 11) and levels[11] == 0.0
    assert Scale(lo=1, hi=3).expand([0.2, 0.3, 0.5]) == [0.2, 0.3, 0.5]
    with pytest.raises(ValueError):
        scale.expand([1.0])


def test_index_of_checks_the_range():
    scale = Scale(lo=1, hi=7)
    assert scale.index_of(1) == 0 and scale.index_of(7) == 6
    with pytest.raises(ValueError):
        scale.index_of(8)
