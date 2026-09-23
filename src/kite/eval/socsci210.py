"""SocSci210 (Kolluri et al., EMNLP 2025): 2.9M individual responses from 210 TESS survey experiments.

Raw layout (HuggingFace `socratesft/SocSci210`): `data/*.parquet` with columns sample_id, participant,
demographic (struct), stimuli, response (int), condition_num, task_num, prompt, reasoning, study_id;
`metadata/participant_mapping.json` with the study-level {"seen": [...170], "unseen": [...40]} split.

`prepare` filters studies, attaches a response scale to every (study, condition, task) cell, and
tabulates that cell's human answer distribution. The scale is per *cell*, not per task: 71 of the
dataset's (study, task) groups word their answer instruction differently in different conditions —
14 change the numeric range and 57 change the option labels (study `3ydty` swaps "Brian / Matt" for
"Amy / Jennifer", which *is* its manipulation). `SocSci210Task` turns rows into requests.
Metrics follow the paper: accuracy = 1 - mean |prediction - response| / (max - min); distribution =
Wasserstein distance on responses rescaled to [0, 1], per (condition, task) cell, averaged within a
study and then across studies.
"""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads

from kite.eval import metrics
from kite.eval.phrasing import Phrasing, Primitive, build_request, decode
from kite.eval.protocol import Split
from kite.eval.scales import Scale, parse_scale, strip_response_instruction
from kite.eval.task import Item, Prediction, ProbeCase
from kite.kernel.types import KernelResponse
from kite.population.persona import Persona

COLUMNS = ["sample_id", "participant", "demographic", "stimuli", "response", "condition_num", "task_num", "study_id"]
MAX_LEVELS = 101  # covers 0-100 scales; longer (unbounded dollar amounts) are reported as unsupported
CELL = ["study_id", "condition_num", "task_num"]
SCALES = "scales.json"  # keyed "study|condition|task", the same key as cells.json


def split_studies(raw_dir: str | Path) -> dict[str, list[str]]:
    return json.loads((Path(raw_dir) / "metadata" / "participant_mapping.json").read_text(encoding="utf-8"))


def dev_studies(seen: Sequence[str], n: int = 20, seed: int = 0) -> list[str]:
    """A fixed random sample of training studies for choosing the phrasing and fitting the temperature."""
    return sorted(random.Random(seed).sample(sorted(seen), n))


def studies_of(split: Split, raw_dir: str | Path) -> list[str]:
    """The studies a split resolves to. One function, so no caller can spell the test set by accident."""
    studies = split_studies(raw_dir)
    return dev_studies(studies["seen"]) if Split(split) is Split.dev else sorted(studies["unseen"])


def _disagreements(scales: dict[str, Scale]) -> dict[str, int]:
    """How many (study, task) groups word their answer scale differently in different conditions.

    Keying the scale by task and reading it off whichever row came first — what we used to do —
    silently asks every condition with condition 0's range and labels. These counts make the size
    of that hazard visible in every prepared split instead of leaving it to be rediscovered.
    """
    by_task: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for key, scale in scales.items():
        study, _, task = key.split("|")
        by_task[(study, task)].append(((scale.lo, scale.hi), tuple(sorted(scale.labels.items()))))
    varies = [(len({span for span, _ in group}) > 1, len({labels for _, labels in group}) > 1) for group in by_task.values()]
    return {
        "n_tasks": len(by_task),
        "n_tasks_range_varies": sum(1 for span, _ in varies if span),
        "n_tasks_labels_vary": sum(1 for _, labels in varies if labels),
        "n_tasks_conditions_disagree": sum(1 for span, labels in varies if span or labels),
    }


def prepare(raw_dir: str | Path, out_dir: str | Path, studies: Sequence[str]) -> dict[str, Any]:
    table = pads.dataset(Path(raw_dir) / "data", format="parquet").to_table(columns=COLUMNS, filter=pc.field("study_id").isin(list(studies)))
    frame = table.to_pandas().reset_index(drop=True)
    frame["demographic"] = frame["demographic"].map(lambda value: json.dumps(dict(value), ensure_ascii=False))

    scales: dict[str, Scale] = {}
    unsupported: dict[str, str] = {}
    keep = np.zeros(len(frame), dtype=bool)
    dropped_out_of_range = 0
    # The fallback range is read across the whole task, not within the condition. A condition whose text
    # states no range states nothing we could differ on, so inventing a per-condition range would only
    # hand each cell the exact span of the answers it is scored against: the two conditions of a task
    # would then both span their own answers, which erases the very condition difference G1b measures.
    observed_range = {key: (int(rows["response"].min()), int(rows["response"].max())) for key, rows in frame.groupby(["study_id", "task_num"])}
    for (study, condition, task), group in frame.groupby(CELL, sort=True):
        key = f"{study}|{condition}|{task}"
        scale = parse_scale(group["stimuli"].iloc[0])
        if scale is None:
            low, high = observed_range[(study, task)]
            if low == high:
                unsupported[key] = "no parseable range and constant responses"
                continue
            scale = Scale(lo=low, hi=high, source="observed")
        if scale.n_levels > MAX_LEVELS:
            unsupported[key] = f"{scale.n_levels} levels"
            continue
        inside = group["response"].between(scale.lo, scale.hi)
        dropped_out_of_range += int((~inside).sum())
        keep[group.index[inside].to_numpy()] = True
        scales[key] = scale

    frame = frame[keep].reset_index(drop=True)
    cells: dict[str, list[int]] = {}
    for (study, condition, task), group in frame.groupby(CELL, sort=True):
        key = f"{study}|{condition}|{task}"
        scale = scales[key]
        counts = np.bincount(group["response"].to_numpy() - scale.lo, minlength=scale.n_levels)
        cells[key] = counts.tolist()
    scales = {key: scale for key, scale in scales.items() if key in cells}  # every kept scale has rows behind it

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "rows.parquet", index=False)
    (out / SCALES).write_text(json.dumps({key: scale.model_dump() for key, scale in scales.items()}, ensure_ascii=False), encoding="utf-8")
    (out / "tasks.json").unlink(missing_ok=True)  # the pre-2026-09-20 name, keyed by task and so wrongly keyed now
    (out / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    summary = {
        "n_rows": len(frame),
        "n_studies": int(frame["study_id"].nunique()),
        # which studies, not only how many: a prepared directory has to say what is inside it
        "studies": sorted(str(study) for study in frame["study_id"].unique()),
        "n_cells": len(cells),
        **_disagreements(scales),
        "n_cells_observed_range": sum(1 for scale in scales.values() if scale.source == "observed"),
        "dropped_out_of_range": dropped_out_of_range,
        "unsupported_cells": unsupported,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class SocSci210Task:
    name = "socsci210"

    def __init__(
        self,
        prepared_dir: str | Path,
        *,
        phrasing: Phrasing = "p1",
        primitive: Primitive = "choice",
        max_per_cell: int | None = None,
        max_participants: int | None = None,
        seed: int = 0,
    ) -> None:
        root = Path(prepared_dir)
        self._phrasing = phrasing
        self._primitive = primitive
        self._seed = seed
        self._scales = {key: Scale(**raw) for key, raw in json.loads((root / SCALES).read_text()).items()}
        self._cells = {key: np.asarray(counts, dtype=float) for key, counts in json.loads((root / "cells.json").read_text()).items()}
        frame = pd.read_parquet(root / "rows.parquet")
        if max_per_cell is not None and max_participants is not None:
            raise ValueError("max_per_cell and max_participants select rows differently; pass at most one")
        if max_per_cell is not None:
            frame["_order"] = [zlib.crc32(f"{seed}:{sample}".encode()) for sample in frame["sample_id"]]
            frame = frame.sort_values([*CELL, "_order"]).groupby(CELL, sort=False).head(max_per_cell)
        if max_participants is not None:
            # Keeping whole people, not whole rows: anything that asks how one person's answers hang
            # together needs that person present on every task. Subsampling by row would hand each
            # task a different slice of the panel and leave nothing to correlate.
            keys = frame["study_id"].astype(str) + "|" + frame["condition_num"].astype(str) + "|" + frame["participant"].astype(str)
            order = {key: zlib.crc32(f"{seed}:{key}".encode()) for key in keys.unique()}
            frame = frame.assign(_person=keys, _order=keys.map(order))
            chosen = (
                frame[["study_id", "condition_num", "_person", "_order"]]
                .drop_duplicates("_person")
                .sort_values(["study_id", "condition_num", "_order"])
                .groupby(["study_id", "condition_num"], sort=False)
                .head(max_participants)["_person"]
            )
            frame = frame[frame["_person"].isin(set(chosen))]
        self._frame = frame.sort_values("sample_id").reset_index(drop=True)

    def members(self) -> list[str]:
        """The study ids this task actually loaded, for the run to record instead of the split's name."""
        return sorted(str(study) for study in self._frame["study_id"].unique())

    def _scale(self, study: str, condition: int, task: int) -> Scale:
        """The scale this condition states. Conditions of one task may differ, and the labels may be the manipulation."""
        return self._scales[f"{study}|{condition}|{task}"]

    def _meta(self, row: Any) -> dict[str, Any]:
        scale = self._scale(row.study_id, int(row.condition_num), int(row.task_num))
        return {
            "study_id": row.study_id,
            "condition_num": int(row.condition_num),
            "task_num": int(row.task_num),
            "participant": int(row.participant),
            "true_index": scale.index_of(int(row.response)),
            "n_levels": scale.n_levels,
        }

    def _case(self, row: Any) -> ProbeCase:
        scale = self._scale(row.study_id, int(row.condition_num), int(row.task_num))
        return ProbeCase(
            persona=Persona.from_socsci210(json.loads(row.demographic)).render(),
            question=strip_response_instruction(row.stimuli),
            options=scale.descriptions(),
        )

    def items(self) -> Iterator[Item]:
        for row in self._frame.itertuples(index=False):
            case = self._case(row)
            request = build_request(
                phrasing=self._phrasing,
                primitive=self._primitive,
                persona=case.persona,
                question=case.question,
                options=case.options,
            )
            # sample ids restart in every study, so the study is part of the id
            yield Item(item_id=f"{row.study_id}:{row.sample_id}", request=request, meta=self._meta(row))

    def probe_cases(self, limit: int) -> list[ProbeCase]:
        return [self._case(row) for row in self._frame.head(limit).itertuples(index=False)]

    def decode(self, item: Item, response: KernelResponse) -> list[float]:
        scale = self._scale(item.meta["study_id"], item.meta["condition_num"], item.meta["task_num"])
        return scale.expand(decode(response, phrasing=self._phrasing, options=scale.descriptions()))

    def target(self, prediction: Prediction) -> list[float]:
        one_hot = [0.0] * prediction.meta["n_levels"]
        one_hot[prediction.meta["true_index"]] = 1.0
        return one_hot

    def _human(self, study: str, condition: int, task: int) -> np.ndarray:
        return self._cells[f"{study}|{condition}|{task}"]

    def _pooled(self) -> dict[tuple[str, str, int], np.ndarray]:
        """Each task's human answers with the conditions merged, one vector per scale length it uses.

        Conditions of one task may state different numeric ranges, so their level indices are not
        commensurable and cannot simply be added. They are merged on the shared [0, 1] axis instead,
        which keeps this baseline exactly condition-blind: every cell of a task is predicted the same
        shape, and `metrics.regrid` leaves its mean position unchanged. Splitting the pool by range
        would instead turn the baseline into a per-cell oracle on precisely the tasks that vary.
        """
        merged: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for key, counts in self._cells.items():
            study, _, task = key.split("|")
            merged[(study, task)].append(counts)
        pooled: dict[tuple[str, str, int], np.ndarray] = {}
        for (study, task), group in merged.items():
            for n_levels in {len(counts) for counts in group}:
                pooled[(study, task, n_levels)] = sum(metrics.regrid(counts, n_levels) for counts in group)
        return pooled

    def baseline(self, kind: str) -> list[Prediction]:
        if kind not in ("uniform", "pooled"):
            raise ValueError(f"unknown baseline: {kind!r}")
        pooled = self._pooled() if kind == "pooled" else {}
        predictions = []
        for row in self._frame.itertuples(index=False):
            meta = self._meta(row)
            if kind == "uniform":
                probs = [1.0 / meta["n_levels"]] * meta["n_levels"]
            else:
                probs = metrics.as_dist(pooled[(row.study_id, str(row.task_num), meta["n_levels"])]).tolist()
            predictions.append(Prediction(item_id=f"{row.study_id}:{row.sample_id}", probs=probs, meta=meta))
        return predictions

    def score(self, predictions: Sequence[Prediction]) -> dict[str, Any]:
        rng = np.random.default_rng(self._seed)
        errors: dict[str, list[float]] = defaultdict(list)
        draw_errors: dict[str, list[float]] = defaultdict(list)
        by_cell: dict[tuple[str, int, int], list[list[float]]] = defaultdict(list)
        sampled: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        briers, confidences, corrects = [], [], []
        for prediction in predictions:
            meta, probs = prediction.meta, metrics.as_dist(prediction.probs)
            true, levels = meta["true_index"], meta["n_levels"]
            errors[meta["study_id"]].append(metrics.normalized_abs_error(metrics.median_index(probs), true, levels))
            draw_errors[meta["study_id"]].append(metrics.random_draw_abs_error(probs, true))
            cell = (meta["study_id"], meta["condition_num"], meta["task_num"])
            by_cell[cell].append(probs.tolist())
            sampled[cell].append(int(rng.choice(levels, p=probs)))
            briers.append(metrics.brier(probs, true))
            confidences.append(float(probs.max()))
            corrects.append(int(probs.argmax()) == true)

        distance: dict[str, list[float]] = defaultdict(list)
        distance_published: dict[str, list[float]] = defaultdict(list)
        distance_sampled: dict[str, list[float]] = defaultdict(list)
        movement: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        ratios, low_entropy = [], 0
        for cell, rows in by_cell.items():
            human = self._human(*cell)
            mean = np.mean(rows, axis=0)
            distance[cell[0]].append(metrics.wasserstein_unit(mean, human))
            distance_published[cell[0]].append(metrics.wasserstein_observed(mean, human))
            histogram = np.bincount(sampled[cell], minlength=len(human)).astype(float)
            distance_sampled[cell[0]].append(metrics.wasserstein_unit(histogram, human))
            movement[(cell[0], cell[2])].append((metrics.mean_position(mean), metrics.mean_position(human)))
            low_entropy += int(metrics.entropy(human) < metrics.LOW_ENTROPY_BITS)
            ratio = metrics.entropy_ratio(mean, human)
            if ratio is not None:
                ratios.append(ratio)

        sensitivity = metrics.condition_sensitivity(movement, seed=self._seed)
        calibration_error, bins = metrics.ece(confidences, corrects)
        all_errors = [error for study in errors.values() for error in study]
        all_draw_errors = [error for study in draw_errors.values() for error in study]
        return {
            "n_predictions": len(predictions),
            "n_cells": len(by_cell),
            "n_studies": len(errors),
            # the median of the predicted distribution: the best single point prediction
            "accuracy_micro": 1.0 - float(np.mean(all_errors)),
            "accuracy_macro": float(np.mean([1.0 - np.mean(study) for study in errors.values()])),
            # one answer drawn from the predicted distribution: the convention the paper's Uniform Guess
            # row uses, and 0.095 lower on a flat 7-level scale. Never put the two in one column.
            "accuracy_micro_random_draw": 1.0 - float(np.mean(all_draw_errors)),
            "accuracy_macro_random_draw": float(np.mean([1.0 - np.mean(study) for study in draw_errors.values()])),
            "distribution": float(np.mean([np.mean(study) for study in distance.values()])),
            # the convention the published SocSci210 figures use; the column to put next to them
            "distribution_published_convention": float(np.mean([np.mean(study) for study in distance_published.values()])),
            "distribution_sampled": float(np.mean([np.mean(study) for study in distance_sampled.values()])),
            # G1b: does the prediction move across conditions the way people do? See metrics.condition_sensitivity.
            "condition_sensitivity_r": sensitivity["r"],
            "condition_sensitivity_slope": sensitivity["slope"],
            "condition_spread_ratio": sensitivity["spread_ratio"],
            "condition_sensitivity_permutation_p": sensitivity["permutation_p"],
            "condition_sensitivity_null_p95": sensitivity["null_p95"],
            "condition_sensitivity_ci_low": sensitivity["ci_low"],
            "condition_sensitivity_ci_high": sensitivity["ci_high"],
            "condition_sensitivity_n_tasks": sensitivity["n_tasks"],
            "condition_sensitivity_n_cells": sensitivity["n_cells"],
            "brier": float(np.mean(briers)),
            "ece": calibration_error,
            "reliability_bins": bins,
            # the median is the headline: the ratio is unbounded above, so its mean tracks the few cells
            # where the humans were nearly unanimous. n_low_human_entropy counts those cells.
            **metrics.describe_entropy_ratios(ratios),
            "n_low_human_entropy": low_entropy,
            "entropy_ratios": [round(ratio, 4) for ratio in ratios],
        }

    def parity_uniform(self) -> dict[str, float]:
        """The paper's 'Uniform Guess' row, computed analytically: accuracy 61.2, distribution 0.203.

        Matching it shows our metric code, study split and preprocessing agree with the paper's. The
        published figure is reproduced by `distribution_published_convention`, not by `distribution`:
        the two rescale responses by the observed and the stated range respectively.
        """
        errors: dict[str, list[float]] = defaultdict(list)
        for row in self._frame.itertuples(index=False):
            meta = self._meta(row)
            errors[row.study_id].append(metrics.uniform_expected_abs_error(meta["true_index"], meta["n_levels"]))
        distance: dict[str, list[float]] = defaultdict(list)
        distance_published: dict[str, list[float]] = defaultdict(list)
        for key, human in self._cells.items():
            flat = np.ones(len(human))
            distance[key.split("|")[0]].append(metrics.wasserstein_unit(flat, human))
            distance_published[key.split("|")[0]].append(metrics.wasserstein_observed(flat, human))
        all_errors = [error for study in errors.values() for error in study]
        return {
            # the random-draw convention, matching score()'s accuracy_*_random_draw. The median rule on a
            # flat distribution is a different and much better predictor - see metrics.random_draw_abs_error.
            "accuracy_micro": 1.0 - float(np.mean(all_errors)),
            "accuracy_macro": float(np.mean([1.0 - np.mean(study) for study in errors.values()])),
            "distribution": float(np.mean([np.mean(study) for study in distance.values()])),
            "distribution_published_convention": float(np.mean([np.mean(study) for study in distance_published.values()])),
        }
