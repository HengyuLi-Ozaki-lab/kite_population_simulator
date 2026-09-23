"""Epstein et al. 2021 (HKS Misinformation Review): eight accuracy-prompt-style interventions across five waves.

Research plan v3 §5 (the D1 test) and docs/data/epstein-schema.md. Every participant rated 20 COVID-19 news
cards (items 1-10 false, 11-20 true) for sharing (binary), or for accuracy in the accuracy-only arm; arms were
randomised within their wave, each wave with its own control.

Two reading modes, as for Arechar: the design (who was in which wave and arm, pretreatment covariates) can be
read at any time; the ratings are refused until the criteria file is committed, so the frozen test cannot
have seen an outcome.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

CSV = Path("data/epstein/data_to_post.csv")
HEADLINES = Path("data/epstein/headlines.json")
CRITERIA = Path("configs/eval/epstein_criteria.yaml")
ENCODING = "latin-1"  # Windows-1252 with stray bytes in free-text columns we never read

TREATMENTS = {
    0.0: "accuracy_only",
    1.0: "control",
    2.0: "evaluation",
    3.0: "long_evaluation",
    4.0: "generic_norms",
    4.5: "partisan_norms",
    5.0: "tips",
    6.0: "tips_norms",
    7.0: "importance",
    8.0: "importance_norms",
}
WAVES = {1: "acc", 2: "frank", 3: "horserace", 4: "normstips", 5: "normsorder"}
DESIGN_COLUMNS = ["id", "wave", "treatment", "completed", "age", "gender", "white", "college", "education", "demrep_c", "region", "hispanic"]
RATING_COLUMNS = ["id", "wave", "treatment", "completed", "item_num", "real", "rating"]

GENDER = {1: "male", 2: "female"}  # 3-7: transgender, non-binary, not listed, prefer not to answer -> "another gender"
EDUCATION = {  # Lucid panel codes; college == education >= 6 in the data, which corroborates them
    1: "some high school or less",
    2: "a high school diploma",
    3: "post-high-school vocational training",
    4: "some college but no degree",
    5: "an associate's degree",
    6: "a bachelor's degree",
    7: "a master's or professional degree",
    8: "a doctorate",
}
PARTY = {1: "a strong Democrat", 2: "a Democrat", 3: "leaning Democratic", 4: "leaning Republican", 5: "a Republican", 6: "a strong Republican"}
REGION = {1: "the Northeast", 2: "the Midwest", 3: "the South", 4: "the West"}


def _committed(path: Path) -> bool:
    if not path.exists():
        return False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], capture_output=True, text=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", str(path)], capture_output=True, text=True).stdout.strip()
    return tracked and not dirty


def load_design(path: str | Path = CSV) -> pd.DataFrame:
    """One row per completed participant-wave with its arm and pretreatment covariates. No rating is read."""
    frame = pd.read_csv(path, usecols=DESIGN_COLUMNS, encoding=ENCODING, low_memory=False)
    frame = frame[frame["completed"] == 1].drop_duplicates(["id", "wave"]).drop(columns="completed")
    frame["arm"] = frame["treatment"].map(TREATMENTS)
    if frame["arm"].isna().any():
        raise ValueError(f"unknown treatment codes: {sorted(frame.loc[frame['arm'].isna(), 'treatment'].unique())}")
    frame["wave"] = frame["wave"].astype(int)
    return frame.reset_index(drop=True)


def load_ratings(path: str | Path = CSV, *, criteria: Path = CRITERIA) -> pd.DataFrame:
    """Long ratings (participant x item) for completed participants. Refused until the criteria file is committed."""
    if not _committed(criteria):
        raise PermissionError(f"{criteria} is not committed: the Epstein outcomes stay closed until the D1 criteria are on record")
    frame = pd.read_csv(path, usecols=RATING_COLUMNS, encoding=ENCODING, low_memory=False)
    frame = frame[frame["completed"] == 1].drop(columns="completed")
    frame["arm"] = frame["treatment"].map(TREATMENTS)
    frame["true"] = frame["real"].astype(int) == 1
    frame["wave"] = frame["wave"].astype(int)
    frame["item_num"] = frame["item_num"].astype(int)
    return frame.reset_index(drop=True)


def headlines(path: str | Path = HEADLINES) -> dict[int, dict]:
    """item_num -> {source, text, true}; the card text as the participants saw it (see the schema note)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    return {int(k): {"source": v["source"], "text": v["text"], "true": bool(v["true"])} for k, v in raw.items()}


def _age_words(age: float | None) -> str | None:
    if age is None or pd.isna(age) or age < 18 or age > 90:  # the authors' own validity range
        return None
    return f"in their {int(age) // 10 * 10}s" if age < 80 else "80 or older"


def render_persona(row: pd.Series) -> dict[str, str]:
    """Pretreatment fields only, as words; missing or out-of-range values are left out rather than guessed."""
    persona: dict[str, str] = {}
    if (age := _age_words(row.get("age"))) is not None:
        persona["age"] = age
    if not pd.isna(row.get("gender")):
        persona["gender"] = GENDER.get(int(row["gender"]), "another gender")
    if not pd.isna(row.get("white")):
        persona["race"] = "white" if int(row["white"]) == 1 else "not white"
    if not pd.isna(row.get("hispanic")):
        persona["hispanic_or_latino"] = "no" if int(row["hispanic"]) == 1 else "yes"
    if not pd.isna(row.get("education")) and int(row["education"]) in EDUCATION:
        persona["education"] = EDUCATION[int(row["education"])]
    if not pd.isna(row.get("demrep_c")) and int(row["demrep_c"]) in PARTY:
        persona["party"] = PARTY[int(row["demrep_c"])]
    if not pd.isna(row.get("region")) and int(row["region"]) in REGION:
        persona["region_of_the_us"] = REGION[int(row["region"])]
    return persona


SCREENS = Path("configs/eval/epstein_screens.yaml")


def load_screens(path: str | Path = SCREENS) -> dict:
    """The transcribed screens, with `ref` entries resolved into the screens they point to."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    screens = {}
    for arm, entries in raw["screens"].items():
        screens[arm] = [raw["screens"][e["ref"]][0] if "ref" in e else e for e in entries]
    return {**raw, "screens": screens}


def _screen_sentence(entry: dict) -> str:
    """One screen as a sentence of the narrative the model reads, in the style of the Arechar contexts."""
    kind = entry["kind"]
    if kind == "pretest_instruction":
        return f'The survey began: "{entry["text"]}"'
    if kind == "headline_rating":
        return f'The respondent was then shown the headline "{entry["headline"]}" (from {entry["source"]}) and asked: "{entry["question"]}"'
    if kind == "feedback":
        return f'After answering, the respondent was told: "{entry["text"]}"'
    if kind == "question":
        return f'The survey began by asking the respondent: "{entry["text"]}" ({" / ".join(entry["options"])})'
    if kind == "screen":
        text = f"{entry['title']}. {entry['text']}" if "title" in entry else entry["text"]
        return f'The survey began with a screen reading: "{text}"'
    if kind == "described":
        return f"The survey began with a task: {entry['text']}"
    raise ValueError(f"unknown screen kind {kind!r}")


def context_for(arm: str, screens: dict) -> str:
    """What the respondent had read before the cards, in the order they read it, then the task instruction."""
    if arm not in screens["screens"]:
        raise KeyError(f"no screens recorded for arm {arm!r}")
    instruction = screens["instruction"]["accuracy" if arm == "accuracy_only" else "sharing"]
    parts = [_screen_sentence(e) for e in screens["screens"][arm]]
    if not parts:
        return f'The survey began: "{instruction}"'
    joined = " ".join(p if i == 0 else p.replace("The survey began", "The survey continued", 1) for i, p in enumerate(parts))
    return f'{joined} Next the survey said: "{instruction}"'


def question_for(arm: str, screens: dict) -> tuple[str, list[str]]:
    q = screens["question"]["accuracy" if arm == "accuracy_only" else "sharing"]
    return q["text"], list(q["options"])


class EpsteinTask:
    """One request per (persona, arm, headline) on the model side.

    Personas are completed participants of a wave (any arm - their fields are pretreatment), chosen by a hash of
    their id so that a smaller pool is nested in a larger one. Every persona is asked under every arm of its
    wave, which is what pairs a control prediction with each treated one; the human data stay between-subjects.
    The first `personas_per_wave` form the application pool and the next `audit_per_wave` the disjoint audit pool.
    """

    name = "epstein"

    def __init__(
        self,
        csv: str | Path = CSV,
        *,
        personas_per_wave: int = 200,
        audit_per_wave: int = 30,
        seed: int = 0,
        screens: str | Path = SCREENS,
        headlines_path: str | Path = HEADLINES,
        arms: dict[int, list[str]] | None = None,
    ):
        import zlib

        from kite.eval.phrasing import build_request, decode

        self._build, self._decode = build_request, decode
        self.screens = load_screens(screens)
        self.headlines = headlines(headlines_path)
        design = load_design(csv)
        design["order"] = [zlib.crc32(f"{seed}:{w}:{i}".encode()) for w, i in zip(design["wave"], design["id"], strict=True)]
        design = design.sort_values(["wave", "order"])
        pools = []
        for _wave, group in design.groupby("wave", sort=True):
            chosen = group.head(personas_per_wave + audit_per_wave).copy()
            chosen["pool"] = ["application"] * min(personas_per_wave, len(chosen)) + ["audit"] * max(0, len(chosen) - personas_per_wave)
            pools.append(chosen)
        self.people = pd.concat(pools, ignore_index=True)
        present = design.groupby("wave")["arm"].agg(lambda a: sorted(set(a)))
        first = ("control", "accuracy_only")
        self.arms = arms or {int(w): [a for a in first if a in v] + [a for a in v if a not in first] for w, v in present.items()}

    def members(self) -> list[str]:
        return [f"wave{w}:{','.join(a)}" for w, a in sorted(self.arms.items())]

    def items(self):
        from kite.eval.task import Item

        for person in self.people.itertuples(index=False):
            persona = render_persona(pd.Series(person._asdict()))
            for arm in self.arms[int(person.wave)]:
                question, options = question_for(arm, self.screens)
                context = context_for(arm, self.screens)
                for item_num, card in self.headlines.items():
                    request = self._build(
                        phrasing="p3",
                        primitive="choice",
                        persona=persona,
                        question=f"{card['source']}\n{card['text']}\n\n{question}",
                        options=options,
                        context=context,
                    )
                    meta = {
                        "wave": int(person.wave), "id": int(person.id), "pool": person.pool, "arm": arm, "item_num": int(item_num),
                        "true": bool(card["true"]), "kind": "accuracy" if arm == "accuracy_only" else "share",
                    }  # fmt: skip
                    yield Item(item_id=f"{int(person.wave)}:{int(person.id)}:{arm}:{int(item_num)}", request=request, meta=meta)

    def decode(self, item, response) -> list[float]:
        return self._decode(response, phrasing="p3", options=question_for(item.meta["arm"], self.screens)[1])

    def score(self, predictions) -> dict:
        """Model-side summaries only: nothing here reads a human rating."""
        import numpy as np

        frame = pd.DataFrame([{**p.meta, "p_yes": float(np.asarray(p.probs)[1] / max(np.asarray(p.probs).sum(), 1e-12))} for p in predictions])
        by = frame[frame["kind"] == "share"].groupby(["wave", "arm", "true"])["p_yes"].mean().unstack("true")
        return {
            "n_predictions": len(frame),
            "mean_p_yes_by_wave_arm_true": {f"{w}:{a}": {"false": float(r[False]), "true": float(r[True])} for (w, a), r in by.iterrows()},
        }
