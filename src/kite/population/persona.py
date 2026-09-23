"""Personas: ordered attribute -> words records that go into the kernel state.

Numbers become words here because Jev reads numerals as text, not as quantities.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

_COUNT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

# SocSci210 `demographic` fields we pass on, in a fixed order, with readable names.
# `internet_access` and `phone_service` are survey-panel bookkeeping and are left out.
SOCSCI210_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "age"),
    ("gender", "gender"),
    ("ethnicity", "race_or_ethnicity"),
    ("education", "education"),
    ("employment", "employment"),
    ("marital_status", "marital_status"),
    ("household_size", "household_size"),
    ("housing_type", "housing_type"),
    ("housing_ownership", "housing_ownership"),
    ("location", "state_of_residence"),
    ("metro_status", "metro_status"),
    ("income", "household_income"),
    ("party_id", "party_identification"),
    ("ideology", "political_ideology"),
)

# OpinionQA (Pew American Trends Panel) demographic attribute codes.
OPINIONQA_ATTRIBUTES: dict[str, str] = {
    "CREGION": "census_region",
    "AGE": "age_group",
    "SEX": "sex",
    "EDUCATION": "education",
    "CITIZEN": "us_citizen",
    "MARITAL": "marital_status",
    "RELIG": "religion",
    "RELIGATTEND": "religious_service_attendance",
    "POLPARTY": "political_party",
    "INCOME": "family_income",
    "POLIDEOLOGY": "political_ideology",
    "RACE": "race",
}


def age_words(age: int) -> str:
    if age < 18:
        return "under 18"
    if age >= 80:
        return "80 or older"
    return f"in their {age // 10 * 10}s"


def count_words(count: int) -> str:
    if 0 <= count < len(_COUNT_WORDS):
        return _COUNT_WORDS[count]
    return "more than twelve"


class Persona(BaseModel):
    model_config = ConfigDict(frozen=True)

    attributes: dict[str, str]

    @classmethod
    def from_socsci210(cls, demographic: Mapping[str, Any]) -> Persona:
        attributes: dict[str, str] = {}
        for source, name in SOCSCI210_FIELDS:
            value = demographic.get(source)
            if value is None or value == "":
                continue
            if source == "age":
                attributes[name] = age_words(int(value))
            elif source == "household_size":
                attributes[name] = f"{count_words(int(value))} people"
            else:
                attributes[name] = str(value)
        return cls(attributes=attributes)

    @classmethod
    def from_opinionqa_group(cls, attribute: str, group: str) -> Persona:
        return cls(attributes={OPINIONQA_ATTRIBUTES[attribute]: group})

    def render(self) -> dict[str, str]:
        return dict(self.attributes)
