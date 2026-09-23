"""OpinionQA (Santurkar et al., ICML 2023): 1,498 Pew American Trends Panel questions, 60 groups.

Raw layout after unpacking the CodaLab bundle into `data/opinionqa/`:
  human_resp/American_Trends_Panel_W{wave}/info.csv       key, question, references, option_ordinal
  human_resp/American_Trends_Panel_W{wave}/metadata.csv   key (attribute), options (groups)
  human_resp/American_Trends_Panel_W{wave}/responses.csv  answers as text, attribute columns, WEIGHT_W{wave}
`references` lists the ordinal options first and the refusal options last; `option_ordinal` gives one
number per ordinal option. List-valued cells are Python literals. Human distributions are survey-weighted
and exclude refusals, exactly as in the authors' helpers.extract_human_opinions.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from kite.eval import metrics
from kite.eval.phrasing import Phrasing, Primitive, build_request, decode
from kite.eval.protocol import Mode, Split
from kite.eval.task import Item, Prediction, ProbeCase
from kite.kernel.types import KernelResponse
from kite.population.persona import OPINIONQA_ATTRIBUTES, Persona

WAVES = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]
DEV_WAVES = [26]
TEST_WAVES = [wave for wave in WAVES if wave not in DEV_WAVES]
OVERALL = "Overall"


def waves_of(split: Split) -> list[int]:
    """The waves a split resolves to. One function, so no caller can spell the test set by accident."""
    return DEV_WAVES if Split(split) is Split.dev else TEST_WAVES


class SurveyQuestion(BaseModel):
    wave: int
    key: str
    text: str
    options: list[str]
    ordinal: list[float]


def wave_dir(root: str | Path, wave: int) -> Path:
    return Path(root) / "human_resp" / f"American_Trends_Panel_W{wave}"


def check_layout(root: str | Path, waves: Sequence[int] = WAVES) -> list[str]:
    """Paths that should exist but do not. Empty means the bundle is unpacked correctly."""
    expected = [wave_dir(root, wave) / name for wave in waves for name in ("info.csv", "metadata.csv", "responses.csv")]
    return [str(path) for path in expected if not path.exists()]


def load_wave(root: str | Path, wave: int) -> tuple[list[SurveyQuestion], dict[str, list[str]], pd.DataFrame]:
    folder = wave_dir(root, wave)
    info = pd.read_csv(folder / "info.csv")
    metadata = pd.read_csv(folder / "metadata.csv")
    responses = pd.read_csv(folder / "responses.csv", low_memory=False)
    groups = {
        row.key: [str(option) for option in ast.literal_eval(row.options)]
        for row in metadata.itertuples(index=False)
        if row.key in OPINIONQA_ATTRIBUTES
    }
    questions = []
    for row in info.itertuples(index=False):
        ordinal = [float(value) for value in ast.literal_eval(row.option_ordinal)]
        options = [str(option) for option in ast.literal_eval(row.references)][: len(ordinal)]
        if row.key in responses.columns and len(options) >= 2 and len(set(options)) == len(options):
            questions.append(SurveyQuestion(wave=wave, key=row.key, text=str(row.question), options=options, ordinal=ordinal))
    return questions, groups, responses


def human_distributions(
    questions: Sequence[SurveyQuestion], groups: dict[str, list[str]], responses: pd.DataFrame, wave: int
) -> dict[tuple[str, str, str], np.ndarray]:
    """{(question key, attribute, group): weighted distribution over the ordinal options}."""
    weights = responses[f"WEIGHT_W{wave}"].astype(float)
    result: dict[tuple[str, str, str], np.ndarray] = {}
    for question in questions:
        answers = responses[question.key]
        indicator = np.stack([(answers == option).to_numpy() for option in question.options], axis=1)
        weighted = indicator * weights.to_numpy()[:, None]
        masks = {(OVERALL, OVERALL): np.ones(len(responses), dtype=bool)}
        for attribute, names in groups.items():
            column = responses[attribute].astype(str)
            for name in names:
                masks[(attribute, name)] = (column == name).to_numpy()
        for (attribute, name), mask in masks.items():
            counts = weighted[mask].sum(axis=0)
            if counts.sum() > 0:
                result[(question.key, attribute, name)] = counts / counts.sum()
    return result


class OpinionQATask:
    name = "opinionqa"

    def __init__(
        self,
        root: str | Path,
        *,
        waves: Sequence[int],
        mode: Mode = "default",
        phrasing: Phrasing = "p1",
        primitive: Primitive = "choice",
        max_questions_per_wave: int | None = None,
    ) -> None:
        self._mode = mode
        self._phrasing = phrasing
        self._primitive = primitive
        self._waves = sorted(waves)
        self._questions: dict[tuple[int, str], SurveyQuestion] = {}
        self._human: dict[tuple[int, str, str, str], np.ndarray] = {}
        for wave in waves:
            questions, groups, responses = load_wave(root, wave)
            questions = questions[:max_questions_per_wave]
            for question in questions:
                self._questions[(wave, question.key)] = question
            for (key, attribute, group), dist in human_distributions(questions, groups, responses, wave).items():
                self._human[(wave, key, attribute, group)] = dist

    def members(self) -> list[str]:
        """The waves this task actually loaded, for the run to record instead of the split's name."""
        return [f"W{wave}" for wave in self._waves]

    def _targets(self) -> Iterator[tuple[SurveyQuestion, str, str]]:
        for wave, key, attribute, group in self._human:
            if (self._mode == "default") == (attribute == OVERALL):
                yield self._questions[(wave, key)], attribute, group

    def _persona(self, attribute: str, group: str) -> dict[str, str] | None:
        return None if attribute == OVERALL else Persona.from_opinionqa_group(attribute, group).render()

    def _identify(self, question: SurveyQuestion, attribute: str, group: str) -> tuple[str, dict[str, Any]]:
        item_id = f"W{question.wave}|{question.key}|{attribute}|{group}"
        return item_id, {"wave": question.wave, "key": question.key, "attribute": attribute, "group": group}

    def items(self) -> Iterator[Item]:
        for question, attribute, group in self._targets():
            request = build_request(
                phrasing=self._phrasing,
                primitive=self._primitive,
                persona=self._persona(attribute, group),
                question=question.text,
                options=question.options,
            )
            item_id, meta = self._identify(question, attribute, group)
            yield Item(item_id=item_id, request=request, meta=meta)

    def probe_cases(self, limit: int) -> list[ProbeCase]:
        cases = []
        for question, attribute, group in self._targets():
            cases.append(ProbeCase(persona=self._persona(attribute, group), question=question.text, options=question.options))
            if len(cases) == limit:
                break
        return cases

    def decode(self, item: Item, response: KernelResponse) -> list[float]:
        question = self._questions[(item.meta["wave"], item.meta["key"])]
        return decode(response, phrasing=self._phrasing, options=question.options)

    def target(self, prediction: Prediction) -> list[float]:
        meta = prediction.meta
        return self._human[(meta["wave"], meta["key"], meta["attribute"], meta["group"])].tolist()

    def baseline(self, kind: str) -> list[Prediction]:
        if kind not in ("uniform", "pooled"):
            raise ValueError(f"unknown baseline: {kind!r}")
        predictions = []
        for question, attribute, group in self._targets():
            if kind == "uniform":
                probs = [1.0 / len(question.options)] * len(question.options)
            else:  # the whole population's answer, whoever we were asked about
                probs = self._human[(question.wave, question.key, OVERALL, OVERALL)].tolist()
            item_id, meta = self._identify(question, attribute, group)
            predictions.append(Prediction(item_id=item_id, probs=probs, meta=meta))
        return predictions

    def score(self, predictions: Sequence[Prediction]) -> dict[str, Any]:
        own, divergences, ratios = [], [], []
        low_entropy = 0
        by_group: dict[str, list[float]] = defaultdict(list)
        for prediction in predictions:
            meta = prediction.meta
            question = self._questions[(meta["wave"], meta["key"])]
            target = self.target(prediction)
            alignment = metrics.opinionqa_alignment(prediction.probs, target, question.ordinal)
            own.append(alignment)
            divergences.append(metrics.js_divergence(prediction.probs, target))
            low_entropy += int(metrics.entropy(target) < metrics.LOW_ENTROPY_BITS)
            ratio = metrics.entropy_ratio(prediction.probs, target)
            if ratio is not None:
                ratios.append(ratio)
            if self._mode == "steered":
                by_group[f"{meta['attribute']}|{meta['group']}"].append(alignment)
            else:  # whose opinions does the un-steered model reflect?
                for (wave, key, attribute, group), human in self._human.items():
                    if (wave, key) == (meta["wave"], meta["key"]) and attribute != OVERALL:
                        by_group[f"{attribute}|{group}"].append(metrics.opinionqa_alignment(prediction.probs, human, question.ordinal))
        return {
            "mode": self._mode,
            "n_predictions": len(predictions),
            "alignment": float(np.mean(own)),
            "js_mean": float(np.mean(divergences)),
            # the median is the headline; see metrics.describe_entropy_ratios
            **metrics.describe_entropy_ratios(ratios),
            "n_low_human_entropy": low_entropy,
            "entropy_ratios": [round(ratio, 4) for ratio in ratios],
            "alignment_by_group": {name: float(np.mean(values)) for name, values in sorted(by_group.items())},
        }
