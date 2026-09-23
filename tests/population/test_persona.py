from kite.population.persona import Persona, age_words, count_words

DEMOGRAPHIC = {
    "age": 34,
    "education": "Post grad study/professional degree",
    "employment": "Employed as paid employee",
    "ethnicity": None,
    "gender": "Female",
    "household_size": 4,
    "housing_ownership": "Owned or being bought by you or someone in your household",
    "housing_type": "A building with 2 or more apartments",
    "ideology": "Moderate",
    "income": "75-99K",
    "internet_access": "Internet Household",
    "location": "New Jersey",
    "marital_status": "Married",
    "metro_status": "Metro Area",
    "party_id": "Moderate Democrat",
    "phone_service": "Cellphone only",
}


def test_age_and_count_words():
    assert age_words(34) == "in their 30s"
    assert age_words(17) == "under 18"
    assert age_words(85) == "80 or older"
    assert count_words(4) == "four"
    assert count_words(40) == "more than twelve"


def test_socsci210_persona_is_ordered_wordy_and_skips_missing_fields():
    rendered = Persona.from_socsci210(DEMOGRAPHIC).render()
    assert list(rendered)[:3] == ["age", "gender", "education"]  # ethnicity is None, so it is skipped
    assert rendered["age"] == "in their 30s"
    assert rendered["household_size"] == "four people"
    assert rendered["state_of_residence"] == "New Jersey"
    assert rendered["party_identification"] == "Moderate Democrat"
    assert "internet_access" not in rendered and "phone_service" not in rendered
    assert all(isinstance(value, str) for value in rendered.values())


def test_opinionqa_group_persona():
    assert Persona.from_opinionqa_group("POLPARTY", "Democrat").render() == {"political_party": "Democrat"}
