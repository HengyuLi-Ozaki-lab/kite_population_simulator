import json

import numpy as np
import pandas as pd
import pytest

from kite.eval import arechar


@pytest.fixture
def fake_csv(tmp_path):
    """Three countries, two respondents each; every respondent rates every item so leaks would show."""
    rng = np.random.default_rng(0)
    rows = []
    for index, country in enumerate(["US", "US", "GB", "GB", "BR", "BR"]):
        row = {"id": index, "country": country, "condition": 1 + index % 4, "age": 30 + index}
        row |= {f"rating_{k}": int(rng.integers(1, 7)) for k in range(1, arechar.N_ITEMS + 1)}
        rows.append(row)
    path = tmp_path / "CR.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_halves_are_odd_and_even_indices():
    assert [arechar.half_of(k) for k in (1, 2, 3, 44, 45)] == ["A", "B", "A", "B", "A"]
    assert sum(arechar.half_of(k) == "A" for k in range(1, 46)) == 23
    with pytest.raises(ValueError):
        arechar.half_of(46)


def test_calibration_sees_only_us_respondents_and_half_a(fake_csv):
    long = arechar.load_ratings(fake_csv, "calibration")
    assert set(long["country"]) == {"US"}
    assert set(long["half"]) == {"A"}
    assert len(long) == 2 * 23
    assert set(long["condition"]) <= set(arechar.CONDITIONS.values())


def test_held_out_is_refused_without_committed_criteria(fake_csv, tmp_path):
    with pytest.raises(PermissionError, match="refusing the held-out"):
        arechar.load_ratings(fake_csv, "held_out", criteria=tmp_path / "no_such_criteria.yaml")


def test_the_split_declaration_names_the_same_calibration_part():
    split = arechar.split_declaration()
    assert split["calibration"] == {"countries": ["US"], "items": "A", "conditions": "all four"}
    assert set(split["held_out"]) == {"new_headlines", "new_countries", "both_new"}


def fake_qsf(tmp_path, *, break_row=None):
    rows = {str(k): {"1": f"<b>Headline number {k}</b>&nbsp;", "2": str(k), "3": f"${{e://Field/h{k}}}"} for k in range(1, 46)}
    if break_row:
        rows[str(break_row)]["2"] = "99"
    survey = {
        "SurveyElements": [
            {
                "Element": "BL",
                "Payload": [
                    {"Description": "CRT", "Options": {"LoopingOptions": {"Static": {"1": {"1": "x"}}}}},
                    {"Description": "Headline", "Options": {"LoopingOptions": {"Static": rows, "Randomization": "All"}}},
                ],
            }
        ]
    }
    path = tmp_path / "survey.qsf"
    path.write_text(json.dumps(survey))
    return path


def test_headlines_come_from_the_45_row_loop_with_markup_removed(tmp_path):
    headlines = arechar.load_headlines(fake_qsf(tmp_path))
    assert sorted(headlines) == list(range(1, 46))
    assert headlines[7] == "Headline number 7"
    assert [arechar.is_true(k) for k in (1, 30, 31, 45)] == [False, False, True, True]


def test_a_loop_whose_rows_do_not_carry_their_own_index_is_refused(tmp_path):
    with pytest.raises(ValueError, match="row-to-item mapping"):
        arechar.load_headlines(fake_qsf(tmp_path, break_row=12))


def test_capitalised_column_names_are_accepted(fake_csv):
    """The real file writes Country and Condition where the codebook writes country and condition."""
    frame = pd.read_csv(fake_csv).rename(columns={"country": "Country", "condition": "Condition"})
    frame["Country"] = frame["Country"].str.lower()  # and the codes are lower case in the real file
    frame.to_csv(fake_csv, index=False)
    long = arechar.load_ratings(fake_csv, "calibration")
    assert set(long["country"]) == {"US"} and len(long) == 2 * 23


def full_qsf(tmp_path):
    """A questionnaire with everything load_materials and ArecharTask read."""
    rows = {str(k): {"1": f"Headline number {k}", "2": str(k), "3": f"${{e://Field/h{k}}}"} for k in range(1, 46)}

    def sq(tag, text, choices=None):
        payload = {"DataExportTag": tag, "QuestionText": text}
        if choices:
            payload["Choices"] = {str(i + 1): {"Display": c} for i, c in enumerate(choices)}
        return {"Element": "SQ", "PrimaryAttribute": tag, "Payload": payload}

    flow = {"Type": "Root", "Flow": [{"Type": "BlockRandomizer", "SubSet": 1, "Flow": [
        {"Type": "EmbeddedData", "EmbeddedData": [{"Field": "msg2", "Value": f"Neutral headline {i}"}]} for i in range(4)
    ]}]}  # fmt: skip
    survey = {"SurveyElements": [
        {"Element": "BL", "Payload": [{"Description": "Headline", "Options": {"LoopingOptions": {"Static": rows}}}]},
        {"Element": "FL", "Payload": flow},
        sq("SMInst", "We are interested in whether you would share them."),
        sq("AInst", "We are interested in whether you think they are accurate."),
        sq("AccInst", "First, rate the accuracy of one headline."),
        sq("c4img", "Think carefully:<br>Be skeptical.<br>Check the evidence."),
        sq("headlineS", "share?", arechar.QUESTIONS["share"][1]),
        sq("headlineA", "accurate?", arechar.QUESTIONS["accuracy"][1]),
    ]}  # fmt: skip
    path = tmp_path / "full.qsf"
    path.write_text(json.dumps(survey))
    return path


@pytest.fixture
def respondents_csv(fake_csv):
    frame = pd.read_csv(fake_csv)
    frame = frame.assign(sex=1, edu=6, ses=4, urbanrural=6, minority=2, fb=1, tw=0, sn=0, ig=1, wh=0, ti=0, ot=0, no=0,
                         political=1, sports=0, celebrity=0, science=1, business=0, other=0, none=0)  # fmt: skip
    for k in range(1, 46):  # each respondent rates ten false and ten true items, as in the real design
        keep = frame.index.map(lambda i, k=k: (k + i) % 3 == 0 or k in (1, 31))
        frame.loc[~keep, f"rating_{k}"] = np.nan
    frame.to_csv(fake_csv, index=False)
    return fake_csv


def test_persona_uses_words_and_only_allowed_fields():
    row = {"age": 34, "sex": 2, "edu": 6, "ses": 9, "urbanrural": 1, "minority": 1, "fb": 1, "wh": 1, "no": 0,
           "political": 1, "science": 1, "none": 0, "imp_accuracy": 5, "trust": 3}  # fmt: skip
    persona = arechar.render_persona(row, "PN")
    assert persona["country"] == "the Philippines" and persona["age"] == "in their 30s"
    assert persona["social_media"] == "uses Facebook and WhatsApp"
    assert persona["news_sharing"] == "would consider sharing political and science or technology news on social media"
    assert not any(ch.isdigit() for value in persona.values() for ch in value if value != persona["age"])
    assert "imp_accuracy" not in arechar.PERSONA_COLUMNS and "trust" not in arechar.PERSONA_COLUMNS


def test_contexts_follow_the_order_participants_read_things(tmp_path):
    materials = arechar.load_materials(full_qsf(tmp_path))
    assert materials["tips"] == "Think carefully:\nBe skeptical.\nCheck the evidence."
    prompt = arechar.context_for("prompt", materials, 6)
    assert prompt.index("rate the accuracy") < prompt.index("Neutral headline 2") < prompt.index("would share")
    assert "Neutral" not in arechar.context_for("share_only", materials, 0)
    assert "accurate" in arechar.context_for("accuracy", materials, 0)


def test_task_runs_end_to_end_on_the_calibration_part(respondents_csv, tmp_path):
    import asyncio

    from kite.eval.runner import read_predictions, run_task
    from kite.kernel.mock import MockKernel

    task = arechar.ArecharTask(respondents_csv, full_qsf(tmp_path), "calibration")
    items = list(task.items())
    assert items and {i.meta["country"] for i in items} == {"US"} and {i.meta["half"] for i in items} == {"A"}
    accuracy = [i for i in items if i.meta["condition"] == "accuracy"]
    assert all(i.meta["kind"] == "accuracy" for i in accuracy)
    asyncio.run(run_task(task, MockKernel(), tmp_path / "run", concurrency=2))
    predictions = read_predictions(tmp_path / "run")
    scores = arechar.score_arechar(predictions)
    assert scores["n_predictions"] == len(items) and scores["n_respondents"] == 2
    assert 0 <= scores["distance"] <= 1 and 0 <= scores["distance_uniform"] <= 1


def synthetic_panel(rng, countries=("US", "GB"), people=120, signal=1.0, prompt_effect=0.0):
    """Ratings where headline k has a known appeal, true headlines are rated higher, and the prompt lowers false sharing."""
    appeal = {k: rng.normal(0, 0.8) for k in range(1, 46)}
    rows = []
    for country in countries:
        for person in range(people):
            condition = ["share_only", "prompt", "accuracy", "tips"][person % 4]
            taste = rng.normal(0, 0.5)
            for k in rng.choice(np.arange(1, 46), 20, replace=False):
                truth = arechar.is_true(int(k))
                base = 3 + appeal[k] + 0.6 * truth + taste - (prompt_effect if condition == "prompt" and not truth else 0)
                rating = int(np.clip(round(base + rng.normal(0, 0.8)), 1, 6))
                model = 3 + signal * (appeal[k] + 0.6 * truth + 0.5 * taste) + rng.normal(0, 0.2)
                rows.append({"country": country, "id": f"{country}{person}", "condition": condition, "item": int(k), "true": truth,
                             "rating": rating, "expected": model})  # fmt: skip
    return pd.DataFrame(rows)


def test_held_out_measures_find_real_signal_and_reject_noise():
    rng = np.random.default_rng(0)
    frame = synthetic_panel(rng)
    share = frame[(frame["condition"] == "share_only") & (frame["country"] == "US")]
    level = arechar.headline_level(share, share, rng, n_perm=300)
    assert level["r"] > level["null_p95"] and level["p"] < 0.01 and 0 < level["human_reliability"] <= 1
    assert arechar.discernment_test(share, share, rng, n_perm=200)["model"] > 0
    individual = arechar.individual_level(frame, rng, n_perm=50)
    assert individual["r"] > individual["null_p95"]

    blind = frame.assign(expected=rng.normal(3, 0.1, len(frame)))
    blind_share = blind[(blind["condition"] == "share_only") & (blind["country"] == "US")]
    assert arechar.headline_level(blind_share, blind_share, rng, n_perm=300)["p"] > 0.01


def test_treatment_effect_sees_a_real_prompt_effect_in_people_and_none_in_a_blind_model():
    rng = np.random.default_rng(1)
    frame = synthetic_panel(rng, people=400, prompt_effect=0.8)
    result = arechar.treatment_effect(frame, frame, "prompt", rng, n_perm=100, n_boot=100)
    assert result["human_ci"][0] > 0  # people: the prompt raises discernment
    assert result["model_pooled"] < result["model_null_p95"]  # this model has no prompt term, so its effect is noise
    assert set(result["per_country"]) == {"US", "GB"}
