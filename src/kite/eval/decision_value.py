"""Decision value: what is a simulated experiment worth to someone choosing between conditions?

A wind tunnel is not asked to reproduce flight; it is asked to get the *direction and ranking* of
design changes right. The equivalent questions here, for every task that was run under several
experimental conditions:

  sign accuracy   for a pair of conditions, does the simulation move the same way people did?
  captured gain   someone who picks the condition the simulation ranks best (or worst, if the goal
                  is to lower the outcome) - how much of the gain that was there to be had do they
                  capture, against picking at random (0) and picking with hindsight (1)?

Both need references on either side, because real treatment effects are small and the human data is
itself noisy:

  floor       shuffle the simulated condition means within each task - a simulation that carries no
              information about which condition is which
  reference   a human pilot with half the sample: one random half of the respondents chooses, the
              other half judges. The simulation is judged by that same other half, so the two are on
              the same footing. This is a reference, not a ceiling - a good simulation may beat a
              half-sample pilot.

Which contrasts count as reliable, and which effect-size bin a contrast falls in, is always decided
from the **judge's** data alone. Deciding it from the full sample looks harmless and is not: a
full-sample difference near zero means the two halves differ in opposite directions, so binning on it
forces the halves to disagree (a first version of this analysis showed half-sample agreement of 0.24
in the smallest bin), and filtering on full-sample reliability keeps exactly the contrasts on which
the halves happen to agree. Either would flatter or punish the human pilot while leaving the
simulation, whose errors are independent of human sampling noise, untouched.

Standard errors cluster on the respondent: in about half of SocSci210 a person contributes several
rows to one cell (the same question about several stimuli), and treating rows as independent would
overstate how reliable the human effects are. A respondent who appears in both cells of a contrast
makes the difference *more* precise than the formula here assumes, so the reliability filter errs
towards keeping fewer contrasts.

Only tasks whose conditions all state the same scale take part: where the options themselves differ
between conditions, a difference in mean position is not a treatment effect.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from kite.eval import metrics
from kite.eval.scales import Scale
from kite.eval.task import Prediction

RELIABLE_Z = 3.0  # a human contrast this many standard errors from zero has the wrong sign about once in 700
EFFECT_BINS = (0.0, 0.01, 0.02, 0.04, 0.08, 1.0)  # |difference in mean position| on the [0, 1] scale


def comparable_tasks(scales: dict[str, Scale]) -> dict[tuple[str, int], list[int]]:
    """(study, task) -> its conditions, for tasks whose conditions all state one and the same scale."""
    by_task: dict[tuple[str, int], list[tuple[int, Scale]]] = defaultdict(list)
    for key, scale in scales.items():
        study, condition, task = key.split("|")
        by_task[(study, int(task))].append((int(condition), scale))
    out = {}
    for task, entries in by_task.items():
        first = entries[0][1]
        if len(entries) >= 2 and all((s.lo, s.hi, s.labels) == (first.lo, first.hi, first.labels) for _, s in entries):
            out[task] = sorted(condition for condition, _ in entries)
    return out


@dataclass
class HumanCells:
    """Per-cell human means with respondent-clustered standard errors, and what split-halves need."""

    keys: list[str]
    mean: np.ndarray
    se: np.ndarray
    n_people: np.ndarray
    group_cell: np.ndarray  # one entry per (cell, respondent): which cell
    group_person: np.ndarray  # ... which respondent, numbered within the whole table
    group_sum: np.ndarray  # ... the sum of that respondent's positions in the cell
    group_count: np.ndarray  # ... and how many rows that was
    n_persons: int

    def subset(self, member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell means and respondent-clustered standard errors over the (cell, respondent) groups flagged in `member`."""
        n_cells = len(self.keys)
        total = np.bincount(self.group_cell, weights=self.group_sum * member, minlength=n_cells)
        count = np.bincount(self.group_cell, weights=self.group_count * member, minlength=n_cells)
        people = np.bincount(self.group_cell, weights=member.astype(float), minlength=n_cells)
        mean = np.divide(total, count, out=np.full(n_cells, np.nan), where=count > 0)
        residual = np.where(member, self.group_sum - np.nan_to_num(mean)[self.group_cell] * self.group_count, 0.0)
        squares = np.bincount(self.group_cell, weights=residual**2, minlength=n_cells)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.where(people > 1, people / (people - 1) * squares / count**2, np.nan)
        return mean, np.sqrt(variance)

    def halves(self, rng: np.random.Generator) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """(mean, se) in two random halves of the respondents. A respondent is wholly in one half."""
        in_a = rng.integers(0, 2, self.n_persons)[self.group_person].astype(bool)
        return self.subset(in_a), self.subset(~in_a)


def human_cells(rows: pd.DataFrame, scales: dict[str, Scale]) -> HumanCells:
    """`rows` needs study_id, participant, condition_num, task_num, response. Cells without a scale are dropped."""
    key = rows["study_id"].astype(str) + "|" + rows["condition_num"].astype(str) + "|" + rows["task_num"].astype(str)
    lo, hi = key.map({k: s.lo for k, s in scales.items()}), key.map({k: s.hi for k, s in scales.items()})
    keep = lo.notna() & (hi > lo)
    table = pd.DataFrame(
        {
            "key": key[keep],
            "person": rows.loc[keep, "study_id"].astype(str) + "|" + rows.loc[keep, "participant"].astype(str),
            "position": (rows.loc[keep, "response"] - lo[keep]) / (hi[keep] - lo[keep]),
        }
    )
    grouped = table.groupby(["key", "person"], sort=True)["position"].agg(["sum", "count"]).reset_index()
    cell, keys = pd.factorize(grouped["key"], sort=True)
    person, persons = pd.factorize(grouped["person"])
    total = np.bincount(cell, weights=grouped["sum"], minlength=len(keys))
    count = np.bincount(cell, weights=grouped["count"], minlength=len(keys))
    mean = total / count
    n_people = np.bincount(cell, minlength=len(keys))
    residual = grouped["sum"].to_numpy() - mean[cell] * grouped["count"].to_numpy()
    squares = np.bincount(cell, weights=residual**2, minlength=len(keys))
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(n_people > 1, n_people / (n_people - 1) * squares / count**2, np.nan)
    return HumanCells(
        keys=list(keys),
        mean=mean,
        se=np.sqrt(variance),
        n_people=n_people,
        group_cell=cell,
        group_person=person,
        group_sum=grouped["sum"].to_numpy(dtype=float),
        group_count=grouped["count"].to_numpy(dtype=float),
        n_persons=len(persons),
    )


def predicted_means(predictions: Iterable[Prediction], *, n_per_cell: int | None = None, seed: int = 0) -> dict[str, float]:
    """Mean predicted position per cell, optionally from the first `n_per_cell` simulated respondents.

    "First" follows the task's own subsampling order, so the cell means for n are exactly what a run
    with `--max-per-cell n` would have produced: smaller experiments are nested in larger ones.
    """
    by_cell: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for prediction in predictions:
        meta = prediction.meta
        sample = prediction.item_id.rsplit(":", 1)[-1]  # ids are "study:sample"; older runs stored the bare sample id
        order = zlib.crc32(f"{seed}:{sample}".encode())
        by_cell[f"{meta['study_id']}|{meta['condition_num']}|{meta['task_num']}"].append((order, metrics.mean_position(prediction.probs)))
    out = {}
    for key, entries in by_cell.items():
        entries.sort()
        chosen = entries if n_per_cell is None else entries[:n_per_cell]
        out[key] = float(np.mean([value for _, value in chosen]))
    return out


@dataclass
class TaskBlock:
    study: str
    task: int
    cells: np.ndarray  # indices into the HumanCells arrays, one per condition
    predicted: np.ndarray
    human: np.ndarray
    se: np.ndarray


def build_blocks(comparable: dict[tuple[str, int], list[int]], human: HumanCells, predicted: dict[str, float]) -> list[TaskBlock]:
    index = {key: position for position, key in enumerate(human.keys)}
    blocks = []
    for (study, task), conditions in sorted(comparable.items()):
        keys = [f"{study}|{condition}|{task}" for condition in conditions]
        keys = [k for k in keys if k in index and k in predicted and np.isfinite(human.se[index[k]])]
        if len(keys) < 2:
            continue
        cells = np.array([index[k] for k in keys])
        blocks.append(TaskBlock(study, task, cells, np.array([predicted[k] for k in keys]), human.mean[cells], human.se[cells]))
    return blocks


def _pairs(mean: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every pair of conditions, with the judge's difference and how many standard errors from zero it is."""
    a, b = np.triu_indices(len(mean), k=1)
    difference = mean[a] - mean[b]
    error = np.sqrt(se[a] ** 2 + se[b] ** 2)
    z = np.divide(difference, error, out=np.zeros_like(difference), where=np.isfinite(error) & (error > 0))
    return a, b, difference, np.nan_to_num(z)


def _agreement(chooser: np.ndarray, judge: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pair: 1 if the chooser orders the two conditions as the judge does, 0.5 if it cannot tell, NaN if the judge cannot."""
    chose, judged = np.sign(chooser[a] - chooser[b]), np.sign(judge[a] - judge[b])
    return np.where(judged == 0, np.nan, np.where(chose == 0, 0.5, (chose == judged).astype(float)))


def _gain(chooser: np.ndarray, judge: np.ndarray) -> tuple[float, float]:
    """Gain captured and gain available, summed over the two goals (raise the outcome; lower it).

    Ties in the chooser are broken at random in expectation, so a simulation that cannot tell the
    conditions apart captures exactly nothing rather than whatever condition happens to come first.
    """
    if len(judge) < 3 or not np.all(np.isfinite(judge)) or not np.all(np.isfinite(chooser)):
        return 0.0, 0.0
    centre = judge.mean()
    picked_high = judge[chooser == chooser.max()].mean()
    picked_low = judge[chooser == chooser.min()].mean()
    return float((picked_high - centre) + (centre - picked_low)), float(judge.max() - judge.min())


@dataclass
class Parts:
    """Per-block sums, kept separate so that studies can be resampled without recomputing anything."""

    study: np.ndarray
    sign_hits: np.ndarray
    sign_n: np.ndarray
    gain: np.ndarray
    available: np.ndarray

    def summary(self, weights: np.ndarray | None = None) -> dict[str, float]:
        w = np.ones(len(self.study)) if weights is None else weights
        sign_n, available = float((w * self.sign_n).sum()), float((w * self.available).sum())
        return {
            "sign_accuracy": float((w * self.sign_hits).sum() / sign_n) if sign_n else float("nan"),
            "captured_gain": float((w * self.gain).sum() / available) if available else float("nan"),
        }


def evaluate(
    blocks: Sequence[TaskBlock],
    *,
    chooser: Sequence[np.ndarray] | None = None,
    judge: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    reliable_z: float = RELIABLE_Z,
) -> Parts:
    """Score a chooser against a judge, given as (mean, se) per block; by default the simulation against the full sample.

    The contrasts that count towards sign accuracy are the ones the judge's own data makes reliable.
    """
    hits, counts, gains, available = [], [], [], []
    for position, block in enumerate(blocks):
        chose = block.predicted if chooser is None else chooser[position]
        judged, error = (block.human, block.se) if judge is None else judge[position]
        a, b, _, z = _pairs(judged, error)
        agreement = _agreement(chose, judged, a, b)[np.abs(z) >= reliable_z]
        agreement = agreement[np.isfinite(agreement)]
        hits.append(agreement.sum())
        counts.append(len(agreement))
        gain, room = _gain(chose, judged)
        gains.append(gain)
        available.append(room)
    return Parts(np.array([b.study for b in blocks]), np.array(hits), np.array(counts, dtype=float), np.array(gains), np.array(available))


def permutation_null(blocks: Sequence[TaskBlock], *, n_perm: int, rng: np.random.Generator, reliable_z: float = RELIABLE_Z) -> dict[str, np.ndarray]:
    """What a simulation with no idea which condition is which would score: its condition means, shuffled within each task."""
    out: dict[str, list[float]] = {"sign_accuracy": [], "captured_gain": []}
    for _ in range(n_perm):
        summary = evaluate(blocks, chooser=[rng.permutation(b.predicted) for b in blocks], reliable_z=reliable_z).summary()
        for name, value in summary.items():
            out[name].append(value)
    return {name: np.asarray(values) for name, values in out.items()}


def bootstrap_by_study(parts: Parts, *, n_boot: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    """95% intervals from resampling whole studies: tasks within a study share respondents and design."""
    studies, membership = np.unique(parts.study, return_inverse=True)
    draws: dict[str, list[float]] = {"sign_accuracy": [], "captured_gain": []}
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(0, len(studies), len(studies)), minlength=len(studies))[membership].astype(float)
        for name, value in parts.summary(weights).items():
            draws[name].append(value)
    return {name: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for name, v in draws.items()}


def half_sample_comparison(blocks: Sequence[TaskBlock], human: HumanCells, *, n_splits: int, rng: np.random.Generator) -> dict[str, float]:
    """A half-sample human pilot and the simulation, both judged by the other half of the respondents."""
    pilot, simulation = defaultdict(list), defaultdict(list)
    for _ in range(n_splits):
        (mean_a, _), (mean_b, se_b) = human.halves(rng)
        judge = [(mean_b[b.cells], se_b[b.cells]) for b in blocks]
        for name, value in evaluate(blocks, chooser=[mean_a[b.cells] for b in blocks], judge=judge).summary().items():
            pilot[name].append(value)
        for name, value in evaluate(blocks, judge=judge).summary().items():
            simulation[name].append(value)
    return {
        **{f"pilot_{name}": float(np.nanmean(values)) for name, values in pilot.items()},
        **{f"simulation_{name}": float(np.nanmean(values)) for name, values in simulation.items()},
    }


def effect_curve(
    blocks: Sequence[TaskBlock], human: HumanCells, *, n_splits: int, rng: np.random.Generator, bins: Sequence[float] = EFFECT_BINS
) -> pd.DataFrame:
    """Sign agreement with the judging half, against the size of the effect *in that half*, for the pilot and the simulation."""
    edges = np.asarray(bins)
    totals = {name: np.zeros(len(edges) - 1) for name in ("pilot", "simulation", "n")}
    for _ in range(n_splits):
        (mean_a, _), (mean_b, se_b) = human.halves(rng)
        for block in blocks:
            a, b, difference, _ = _pairs(mean_b[block.cells], se_b[block.cells])
            which = np.digitize(np.abs(difference), edges[1:-1])
            for name, chooser in (("pilot", mean_a[block.cells]), ("simulation", block.predicted)):
                agreement = _agreement(chooser, mean_b[block.cells], a, b)
                valid = np.isfinite(agreement)
                np.add.at(totals[name], which[valid], agreement[valid])
                if name == "pilot":
                    np.add.at(totals["n"], which[valid], 1.0)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if totals["n"][index]:
            rows.append(
                {
                    "effect_from": low,
                    "effect_to": high,
                    "contrasts_per_split": totals["n"][index] / n_splits,
                    "simulation": totals["simulation"][index] / totals["n"][index],
                    "pilot": totals["pilot"][index] / totals["n"][index],
                }
            )
    return pd.DataFrame(rows)


def reliability_curve(blocks: Sequence[TaskBlock], edges: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 5.0, np.inf)) -> pd.DataFrame:
    """The simulation against the full-sample sign, by how reliable that sign is.

    `ceiling` is what a simulation that knew the true direction would score: the full-sample sign is
    itself right with probability about Phi(|z|), so nothing can agree with it more often than that.
    """
    z, agreement = [], []
    for block in blocks:
        a, b, _, score = _pairs(block.human, block.se)
        z.append(np.abs(score))
        agreement.append(_agreement(block.predicted, block.human, a, b))
    z, agreement = np.concatenate(z), np.concatenate(agreement)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (z >= low) & (z < high) & np.isfinite(agreement)
        if inside.any():
            rows.append(
                {
                    "z_from": low,
                    "z_to": high,
                    "contrasts": int(inside.sum()),
                    "simulation": float(agreement[inside].mean()),
                    "ceiling": float(norm.cdf(z[inside]).mean()),
                }
            )
    return pd.DataFrame(rows)
