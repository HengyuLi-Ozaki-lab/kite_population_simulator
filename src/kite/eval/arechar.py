"""Arechar et al. 2023 (OSF g65qu): accuracy prompts across sixteen countries, headlines shown as plain text.

The loader enforces configs/eval/arechar_split.yaml. Until the held-out criteria file exists and is
committed, it returns only the calibration part - US respondents, odd-numbered headlines - and the other
rating columns are never read from disk. The split was committed before the data was downloaded.
"""

from __future__ import annotations

import json
import re
import subprocess
import zlib
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

SPLIT = Path("configs/eval/arechar_split.yaml")
CRITERIA = Path("configs/eval/arechar_criteria.yaml")
N_ITEMS = 45
CONDITIONS = {1: "share_only", 2: "prompt", 3: "accuracy", 4: "tips"}

Phase = Literal["calibration", "held_out"]

# Verified from the US questionnaire (docs/data/arechar-schema.md): the debrief lists loop rows 31-45 as
# the true headlines, and two randomisers draw ten items each from rows 1-30 and rows 31-45.
TRUE_ITEMS = range(31, N_ITEMS + 1)
QUESTIONS = {
    "share": (
        "If you were to see the above headline online, how likely would you be to share it?",
        ["Extremely unlikely", "Moderately unlikely", "Slightly unlikely", "Slightly likely", "Moderately likely", "Extremely likely"],
    ),
    "accuracy": (
        "To the best of your knowledge, is the above headline accurate?",
        ["Extremely inaccurate", "Moderately inaccurate", "Slightly inaccurate", "Slightly accurate", "Moderately accurate", "Extremely accurate"],
    ),
}


def is_true(item: int) -> bool:
    return item in TRUE_ITEMS


def load_headlines(qsf: str | Path) -> dict[int, str]:
    """The 45 headline texts from a questionnaire's Loop & Merge block, keyed by the loop row = the k of rating_k.

    A row's second field holds its own row number and its third field the embedded-data flag h<k> that
    the randomisers set, which is what ties loop row k to rating_k.
    """
    survey = json.loads(Path(qsf).read_text(encoding="utf-8"))
    payload = next(e for e in survey["SurveyElements"] if e["Element"] == "BL")["Payload"]
    blocks = payload.values() if isinstance(payload, dict) else payload
    loops = [b for b in blocks if len(((b.get("Options") or {}).get("LoopingOptions") or {}).get("Static") or {}) == N_ITEMS]
    if len(loops) != 1:
        raise ValueError(f"{qsf}: expected one loop of {N_ITEMS} rows, found {len(loops)}")
    rows = loops[0]["Options"]["LoopingOptions"]["Static"]
    out = {}
    for key, fields in rows.items():
        row = int(key)
        if fields.get("2") != key or fields.get("3") != f"${{e://Field/h{key}}}":
            raise ValueError(f"{qsf}: loop row {key} does not carry its own index and flag; the row-to-item mapping is not safe")
        out[row] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>|&nbsp;", " ", fields["1"])).strip()
    return out


def half_of(item: int) -> str:
    """Half A is the odd item indices, half B the even ones - fixed before any text or rating was seen."""
    if not 1 <= item <= N_ITEMS:
        raise ValueError(f"item index {item} is outside 1..{N_ITEMS}")
    return "A" if item % 2 else "B"


def _committed(path: Path) -> bool:
    if not path.exists():
        return False
    return not subprocess.run(["git", "status", "--porcelain", str(path)], capture_output=True, text=True).stdout.strip()


def load_ratings(path: str | Path, phase: Phase = "calibration", *, criteria: Path = CRITERIA) -> pd.DataFrame:
    """Ratings in long form: one row per (respondent, item), with country, condition and half.

    calibration  US respondents and half-A items only; nothing else is read from the file
    held_out     everything, and only once the criteria file exists and is committed
    """
    # The file capitalises some names the codebook writes in lower case (Country, Condition, Minority).
    actual = {column.lower(): column for column in pd.read_csv(path, nrows=0).columns}
    ratings = [f"rating_{k}" for k in range(1, N_ITEMS + 1)]
    missing = [c for c in ["id", "country", "condition", *ratings] if c not in actual]
    if missing:
        raise ValueError(f"{path} is missing columns {missing[:5]}")

    def read(columns: list[str]) -> pd.DataFrame:
        frame = pd.read_csv(path, usecols=[actual[c] for c in columns])
        frame = frame.rename(columns={actual[c]: c for c in columns})
        # codes are lower case in the file, and two are not ISO: "pn" is the Philippines, "uk" the United Kingdom
        frame["country"] = frame["country"].astype(str).str.upper()
        return frame

    if phase == "calibration":
        frame = read(["id", "country", "condition", *[f"rating_{k}" for k in range(1, N_ITEMS + 1) if half_of(k) == "A"]])
        frame = frame[frame["country"] == "US"]
    elif phase == "held_out":
        if not _committed(criteria):
            raise PermissionError(f"refusing the held-out ratings: {criteria} does not exist or is not committed")
        frame = read(["id", "country", "condition", *ratings])
    else:
        raise ValueError(f"unknown phase {phase!r}")

    long = frame.melt(id_vars=["id", "country", "condition"], var_name="column", value_name="rating").dropna(subset=["rating"])
    long["item"] = long["column"].str.removeprefix("rating_").astype(int)
    long["half"] = long["item"].map(half_of)
    long["condition"] = long["condition"].map(CONDITIONS)
    if phase == "held_out":
        in_calibration = (long["country"] == "US") & (long["half"] == "A")
        long["part"] = "calibration"
        long.loc[(long["country"] == "US") & (long["half"] == "B"), "part"] = "new_headlines"
        long.loc[(long["country"] != "US") & (long["half"] == "A"), "part"] = "new_countries"
        long.loc[(long["country"] != "US") & (long["half"] == "B"), "part"] = "both_new"
        assert (long.loc[in_calibration, "part"] == "calibration").all()
    else:
        long["part"] = "calibration"
    return long.drop(columns="column").reset_index(drop=True)


def split_declaration() -> dict:
    return yaml.safe_load(SPLIT.read_text())


# --- the model's view of a respondent and a trial ---------------------------------------------------

COUNTRY_NAMES = {
    "US": "the United States", "UK": "the United Kingdom", "AU": "Australia", "ZA": "South Africa", "NG": "Nigeria",
    "IN": "India", "PN": "the Philippines", "BR": "Brazil", "MX": "Mexico", "AR": "Argentina", "ES": "Spain",
    "IT": "Italy", "CN": "China", "EG": "Egypt", "SA": "Saudi Arabia", "RU": "Russia",
}  # fmt: skip
EDUCATION = {
    1: "no formal education", 2: "less than a secondary school degree", 3: "less than a high school degree",
    4: "a high school diploma", 5: "some college", 6: "a bachelor's degree", 7: "a graduate degree",
}  # fmt: skip
COMMUNITY = {
    1: "a village or very small town", 2: "a small town", 3: "a town", 4: "a small city", 5: "a mid-sized city", 6: "a large city",
}  # fmt: skip
PLATFORMS = {"fb": "Facebook", "tw": "Twitter", "sn": "Snapchat", "ig": "Instagram", "wh": "WhatsApp", "ti": "TikTok", "ot": "another platform"}
NEWS = {
    "political": "political",
    "sports": "sports",
    "celebrity": "celebrity",
    "science": "science or technology",
    "business": "business",
    "other": "other",
}
# Only what the treatment could not have moved: see the addendum in configs/eval/arechar_split.yaml.
PERSONA_COLUMNS = ["age", "sex", "edu", "ses", "urbanrural", "minority", *PLATFORMS, "no", *NEWS, "none"]


def _listing(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _present(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and value != value)


def render_persona(row: dict, country: str) -> dict[str, str]:
    """Words, not codes: the kernel reads numbers badly, so every value is turned into a phrase here."""
    from kite.population.persona import age_words

    out = {"country": COUNTRY_NAMES.get(country, country)}
    if _present(row.get("age")) and 18 <= int(row["age"]) < 100:
        out["age"] = age_words(int(row["age"]))
    if _present(row.get("sex")) and int(row["sex"]) in (1, 2):
        out["sex"] = "male" if int(row["sex"]) == 1 else "female"
    if _present(row.get("edu")) and int(row["edu"]) in EDUCATION:
        out["education"] = EDUCATION[int(row["edu"])]
    if _present(row.get("ses")) and 1 <= int(row["ses"]) <= 10:
        place = ["near the bottom", "below the middle", "in the middle", "above the middle", "near the top"][(int(row["ses"]) - 1) // 2]
        out["social_standing"] = f"places themselves {place} of their country's social ladder"
    if _present(row.get("urbanrural")) and int(row["urbanrural"]) in COMMUNITY:
        out["community"] = f"lives in {COMMUNITY[int(row['urbanrural'])]}"
    if _present(row.get("minority")) and int(row["minority"]) in (1, 2):
        out["ethnicity"] = (
            "sees themselves as part of an ethnic minority" if int(row["minority"]) == 1 else "sees themselves as part of the ethnic majority"
        )
    used = [name for column, name in PLATFORMS.items() if row.get(column) == 1]
    if used:
        out["social_media"] = f"uses {_listing(used)}"
    elif row.get("no") == 1:
        out["social_media"] = "does not use social media"
    shared = [name for column, name in NEWS.items() if row.get(column) == 1]
    if shared:
        out["news_sharing"] = f"would consider sharing {_listing(shared)} news on social media"
    elif row.get("none") == 1:
        out["news_sharing"] = "would not consider sharing any kind of news on social media"
    return out


def _text(html: str | None) -> str:
    """Visible text of a Qualtrics field: block tags become line breaks, other markup goes."""
    text = re.sub(r"<\s*(br|/p|/div|/li)\s*/?>", "\n", html or "", flags=re.I)
    text = re.sub(r"<[^>]+>|&nbsp;", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_materials(qsf: str | Path) -> dict:
    """The wording participants read, taken from the questionnaire rather than retyped."""
    survey = json.loads(Path(qsf).read_text(encoding="utf-8"))
    elements = survey["SurveyElements"]
    questions = {e["Payload"].get("DataExportTag"): e["Payload"] for e in elements if e["Element"] == "SQ"}
    neutral = []

    def walk(node: dict) -> None:
        if node.get("Type") == "EmbeddedData":
            neutral.extend(_text(f.get("Value")) for f in node.get("EmbeddedData", []) if f.get("Field") == "msg2")
        for child in node.get("Flow", []) or []:
            walk(child)

    walk(next(e for e in elements if e["Element"] == "FL")["Payload"])
    materials = {
        "share_instructions": _text(questions["SMInst"]["QuestionText"]),
        "accuracy_instructions": _text(questions["AInst"]["QuestionText"]),
        "prompt_instructions": _text(questions["AccInst"]["QuestionText"]),
        "tips": _text(questions["c4img"]["QuestionText"]),
        "neutral_headlines": neutral,
    }
    for kind, tag in (("share", "headlineS"), ("accuracy", "headlineA")):
        options = [_text(c["Display"]) for c in questions[tag]["Choices"].values()]
        if options != QUESTIONS[kind][1]:
            raise ValueError(f"{qsf}: the {kind} options differ from the ones this module expects: {options}")
    if len(neutral) != 4:
        raise ValueError(f"{qsf}: expected four neutral headlines for the accuracy prompt, found {len(neutral)}")
    return materials


def context_for(condition: str, materials: dict, neutral_index: int) -> str:
    """What the respondent had read before the headlines, in the order they read it."""
    if condition == "share_only":
        return f'The survey began: "{materials["share_instructions"]}"'
    if condition == "accuracy":
        return f'The survey began: "{materials["accuracy_instructions"]}"'
    if condition == "prompt":
        neutral = materials["neutral_headlines"][neutral_index % len(materials["neutral_headlines"])]
        return (
            f'The survey began: "{materials["prompt_instructions"]}" The respondent was then shown the headline "{neutral}" '
            f'and asked whether it was accurate. Next the survey said: "{materials["share_instructions"]}"'
        )
    if condition == "tips":
        return f'The survey began with a screen reading: "{materials["tips"]}" Next the survey said: "{materials["share_instructions"]}"'
    raise ValueError(f"unknown condition {condition!r}")


def load_respondents(path: str | Path, phase: Phase = "calibration", *, criteria: Path = CRITERIA) -> pd.DataFrame:
    """Persona columns, one row per respondent, under the same gate as the ratings."""
    actual = {column.lower(): column for column in pd.read_csv(path, nrows=0).columns}
    columns = ["id", "country", *PERSONA_COLUMNS]
    if phase == "held_out" and not _committed(criteria):
        raise PermissionError(f"refusing the held-out respondents: {criteria} does not exist or is not committed")
    frame = pd.read_csv(path, usecols=[actual[c] for c in columns]).rename(columns={actual[c]: c for c in columns})
    frame["country"] = frame["country"].astype(str).str.upper()
    return frame[frame["country"] == "US"] if phase == "calibration" else frame


class ArecharTask:
    """One request per (respondent, headline) rating, scored against what that respondent answered.

    `max_respondents` simulates a fixed number of respondents per (country, condition), chosen by a hash of
    their id, so that a smaller run is nested in a larger one; human references always use everybody.
    """

    name = "arechar"

    def __init__(
        self,
        csv: str | Path,
        qsf: str | Path,
        phase: Phase = "calibration",
        *,
        max_respondents: int | None = None,
        parts: list[str] | None = None,
        seed: int = 0,
        criteria: Path = CRITERIA,
    ):
        from kite.eval.phrasing import build_request, decode

        self._build, self._decode = build_request, decode
        self.phase = phase
        self.materials = load_materials(qsf)
        self.headlines = load_headlines(qsf)
        ratings = load_ratings(csv, phase, criteria=criteria)
        if parts is not None:
            ratings = ratings[ratings["part"].isin(parts)]
        people = load_respondents(csv, phase, criteria=criteria).set_index(["country", "id"])
        if max_respondents is not None:
            keys = ratings[["country", "condition", "id"]].drop_duplicates()
            keys = keys.assign(order=[zlib.crc32(f"{seed}:{c}:{i}".encode()) for c, i in zip(keys["country"], keys["id"], strict=True)])
            chosen = keys.sort_values("order").groupby(["country", "condition"]).head(max_respondents)
            ratings = ratings.merge(chosen[["country", "id"]], on=["country", "id"])
        self.ratings = ratings.sort_values(["country", "id", "item"]).reset_index(drop=True)
        self.people = people

    def members(self) -> list[str]:
        return sorted(self.ratings["country"].unique())

    def items(self):
        from kite.eval.task import Item

        for row in self.ratings.itertuples(index=False):
            person = self.people.loc[(row.country, row.id)].to_dict()
            kind = "accuracy" if row.condition == "accuracy" else "share"
            question, options = QUESTIONS[kind]
            request = self._build(
                phrasing="p3",
                primitive="choice",
                persona=render_persona(person, row.country),
                question=f"{self.headlines[row.item]}\n\n{question}",
                options=options,
                context=context_for(row.condition, self.materials, zlib.crc32(f"neutral:{row.country}:{row.id}".encode())),
            )
            meta = {
                "country": row.country, "id": int(row.id), "condition": row.condition, "item": int(row.item), "true": is_true(int(row.item)),
                "half": row.half, "part": row.part, "rating": int(row.rating), "kind": kind,
            }  # fmt: skip
            yield Item(item_id=f"{row.country}:{row.id}:{row.item}", request=request, meta=meta)

    def decode(self, item, response) -> list[float]:
        return self._decode(response, phrasing="p3", options=QUESTIONS[item.meta["kind"]][1])

    def target(self, prediction) -> list[float]:
        one_hot = [0.0] * 6
        one_hot[prediction.meta["rating"] - 1] = 1.0
        return one_hot

    def score(self, predictions) -> dict:
        return score_arechar(predictions)


def score_arechar(predictions) -> dict:
    """The measures the Arechar gates are built from, overall and for each part of the split."""
    import numpy as np

    from kite.eval import metrics

    rows = []
    for p in predictions:
        probs = np.asarray(p.probs, dtype=float)
        probs = probs / probs.sum()
        rows.append({**p.meta, "expected": float(np.dot(probs, np.arange(1, 7))), "probs": probs})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n_predictions": 0}

    def measures(d: pd.DataFrame) -> dict:
        out = {"n_predictions": len(d), "n_respondents": int(d[["country", "id"]].drop_duplicates().shape[0]), "n_items": int(d["item"].nunique())}
        cells = d.groupby(["country", "condition", "item"])
        model_mean, human_mean = cells["expected"].mean(), cells["rating"].mean()
        for condition in ("share_only", "accuracy"):
            m = model_mean.xs(condition, level="condition") if condition in d["condition"].values else pd.Series(dtype=float)
            h = human_mean.xs(condition, level="condition") if condition in d["condition"].values else pd.Series(dtype=float)
            out[f"item_r_{condition}"] = float(np.corrcoef(m, h)[0, 1]) if len(m) >= 3 and m.std() > 0 else None
        discern = {}
        for condition, g in d.groupby("condition"):
            by_truth = g.groupby("true")[["expected", "rating"]].mean()
            if {True, False} <= set(by_truth.index):
                discern[condition] = {
                    "model": float(by_truth.loc[True, "expected"] - by_truth.loc[False, "expected"]),
                    "human": float(by_truth.loc[True, "rating"] - by_truth.loc[False, "rating"]),
                }
        out["discernment"] = discern
        for treatment in ("prompt", "tips"):
            if treatment in discern and "share_only" in discern:
                out[f"{treatment}_effect"] = {side: discern[treatment][side] - discern["share_only"][side] for side in ("model", "human")}
        centred = d.assign(e=d["expected"] - cells["expected"].transform("mean"), r=d["rating"] - cells["rating"].transform("mean"))
        out["individual_r"] = float(np.corrcoef(centred["e"], centred["r"])[0, 1]) if centred["e"].std() > 0 else None
        distance, uniform = [], []
        for _, g in cells:
            human = np.bincount(g["rating"] - 1, minlength=6).astype(float)
            distance.append(metrics.wasserstein_unit(np.mean(np.stack(g["probs"].to_list()), axis=0), human))
            uniform.append(metrics.wasserstein_unit(np.ones(6), human))
        out["distance"], out["distance_uniform"] = float(np.mean(distance)), float(np.mean(uniform))
        return out

    result = measures(frame)
    if frame["part"].nunique() > 1:
        result["by_part"] = {part: measures(g) for part, g in frame.groupby("part")}
    return result


# --- held-out analysis: the measures configs/eval/arechar_criteria.yaml is written in terms of -------------


def _discernment(frame: pd.DataFrame, column: str) -> float:
    by_truth = frame.groupby("true")[column].mean()
    return float(by_truth.get(True, float("nan")) - by_truth.get(False, float("nan")))


def split_half_reliability(human: pd.DataFrame, rng, n_splits: int = 200) -> float:
    """Spearman-Brown reliability of the per-headline human mean, from random halves of the respondents."""
    import numpy as np

    ids = human["id"].unique()
    values = []
    for _ in range(n_splits):
        half = set(rng.choice(ids, len(ids) // 2, replace=False))
        in_half = human["id"].isin(half)
        a, b = human[in_half].groupby("item")["rating"].mean(), human[~in_half].groupby("item")["rating"].mean()
        a, b = a.align(b, join="inner")
        if len(a) >= 3:
            values.append(np.corrcoef(a, b)[0, 1])
    r = float(np.nanmean(values))
    return 2 * r / (1 + r)


def headline_level(model: pd.DataFrame, human: pd.DataFrame, rng, n_perm: int = 2000) -> dict:
    """Across headlines: the model's mean expected rating against the human mean, with an item-shuffle null."""
    import numpy as np

    m = model.groupby("item")["expected"].mean()
    h = human.groupby("item")["rating"].mean().reindex(m.index)
    r = float(np.corrcoef(m, h)[0, 1])
    null = np.array([np.corrcoef(rng.permutation(m.to_numpy()), h.to_numpy())[0, 1] for _ in range(n_perm)])
    reliability = split_half_reliability(human, rng)
    return {"r": r, "null_p95": float(np.percentile(null, 95)), "p": float((np.sum(null >= r) + 1) / (n_perm + 1)), "items": len(m),
            "human_reliability": reliability, "share_of_reliability": r / reliability}  # fmt: skip


def discernment_test(model: pd.DataFrame, human: pd.DataFrame, rng, n_perm: int = 1000) -> dict:
    """Does the model rate true headlines higher than false ones, beyond what shuffling the labels gives?"""
    import numpy as np

    items = model["item"].unique()
    truth = [is_true(int(i)) for i in items]
    observed = _discernment(model, "expected")
    null = []
    for _ in range(n_perm):
        relabel = dict(zip(items, rng.permutation(truth), strict=True))
        null.append(_discernment(model.assign(true=model["item"].map(relabel)), "expected"))
    return {"model": observed, "human": _discernment(human, "rating"), "null_p95": float(np.percentile(null, 95))}


def _effect(frame: pd.DataFrame, treatment: str, column: str) -> float:
    return _discernment(frame[frame["condition"] == treatment], column) - _discernment(frame[frame["condition"] == "share_only"], column)


def _person_sums(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per respondent: the sum and count of `column` over true and over false headlines."""
    sums = frame.groupby(["country", "id", "condition", "true"])[column].agg(["sum", "count"]).unstack("true", fill_value=0)
    sums.columns = [f"{stat}_{'T' if truth else 'F'}" for stat, truth in sums.columns]
    for column_name in ("sum_T", "count_T", "sum_F", "count_F"):
        if column_name not in sums:
            sums[column_name] = 0.0
    return sums.reset_index()


def _discern(sums: pd.DataFrame, weights) -> float:
    """Rating-weighted true-minus-false difference over respondents with the given weights (same as pooling their rows)."""
    import numpy as np

    w = np.asarray(weights, dtype=float)
    return float(
        (w @ sums["sum_T"].to_numpy()) / (w @ sums["count_T"].to_numpy()) - (w @ sums["sum_F"].to_numpy()) / (w @ sums["count_F"].to_numpy())
    )


def treatment_effect(model: pd.DataFrame, human: pd.DataFrame, treatment: str, rng, n_perm: int = 1000, n_boot: int = 1000) -> dict:
    """Effect on sharing discernment, pooled with equal weight per country, for the model and for people.

    The model's null shuffles condition labels between treated and share-only respondents within each
    country; the human interval resamples respondents within each (country, condition). Both work on
    per-respondent sums, which gives the same discernment as pooling the rating rows.
    """
    import numpy as np

    countries = sorted(set(model["country"]) & set(human["country"]))
    pair = [treatment, "share_only"]
    m_sums = _person_sums(model[model["condition"].isin(pair)], "expected")
    h_sums = _person_sums(human[human["condition"].isin(pair)], "rating")

    def effect(sums: pd.DataFrame, treated, weights) -> float:
        return _discern(sums, weights * treated) - _discern(sums, weights * ~treated)

    per_country, model_null, human_boot = {}, np.zeros((n_perm, len(countries))), np.zeros((n_boot, len(countries)))
    for index, country in enumerate(countries):
        m = m_sums[m_sums["country"] == country].reset_index(drop=True)
        h = h_sums[h_sums["country"] == country].reset_index(drop=True)
        m_treated, h_treated = (m["condition"] == treatment).to_numpy(), (h["condition"] == treatment).to_numpy()
        ones_m, ones_h = np.ones(len(m)), np.ones(len(h))
        per_country[country] = {"model": effect(m, m_treated, ones_m), "human": effect(h, h_treated, ones_h)}
        for i in range(n_perm):
            model_null[i, index] = effect(m, rng.permutation(m_treated), ones_m)
        groups = [np.flatnonzero(h_treated), np.flatnonzero(~h_treated)]
        for i in range(n_boot):
            weights = np.zeros(len(h))
            for members in groups:
                weights[members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)))
            human_boot[i, index] = effect(h, h_treated, weights)
        per_country[country]["human_ci"] = [float(np.percentile(human_boot[:, index], 2.5)), float(np.percentile(human_boot[:, index], 97.5))]
    pooled_boot = human_boot.mean(axis=1)
    return {
        "model_pooled": float(np.mean([v["model"] for v in per_country.values()])),
        "model_null_p95": float(np.percentile(model_null.mean(axis=1), 95)),
        "human_pooled": float(np.mean([v["human"] for v in per_country.values()])),
        "human_ci": [float(np.percentile(pooled_boot, 2.5)), float(np.percentile(pooled_boot, 97.5))],
        "per_country": per_country,
    }


def individual_level(model: pd.DataFrame, rng, n_perm: int = 500) -> dict:
    """Within each (country, headline, condition): does the model's ranking of people match their answers?"""
    import numpy as np

    keys = ["country", "item", "condition"]
    e = model["expected"] - model.groupby(keys)["expected"].transform("mean")
    r = model["rating"] - model.groupby(keys)["rating"].transform("mean")
    observed = float(np.corrcoef(e, r)[0, 1])
    null = []
    for _ in range(n_perm):
        shuffled = e.groupby([model[k] for k in keys]).transform(lambda x: rng.permutation(x.to_numpy()))
        null.append(np.corrcoef(shuffled, r)[0, 1])
    return {"r": observed, "null_p95": float(np.percentile(null, 95))}
