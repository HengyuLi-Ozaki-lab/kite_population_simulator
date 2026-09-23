"""Numeric response scales (SocSci210): parse them from the prompt text, describe them in words.

SocSci210 rows end with an LLM-facing instruction such as
"Only return an integer from 1 to 7, where 1 means Extremely unlikely and 7 means Extremely likely,
nothing else." Jev needs a closed list of described options instead, so we parse the range and any
labels, turn levels into descriptions, and bin long scales (0-100) into at most ten options.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

MAX_OPTIONS = 10

_RANGE = re.compile(r"integer from (-?\d+) to (-?\d+)", re.IGNORECASE)
_NEXT_MEANS = r"\s*,?\s*(?:and\s+)?-?\d+\s*(?:means|=)"
_MEANS = re.compile(
    rf"(-?\d+)\s*(?:means|=)\s*(.+?)(?={_NEXT_MEANS}|\s*,?\s*nothing else|[\s.”\"']*\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NEXT_FOR = r"\s*[;,]?\s*(?:or\s+)?-?\d+ for\s"
_FOR = re.compile(
    rf"(-?\d+) for\s+(.+?)(?={_NEXT_FOR}|\s*,?\s*nothing else|[\s.”\"']*\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ENUMERATION = re.compile(r"only return ((?:-?\d+\s*,\s*)*-?\d+\s*,?\s*or\s+-?\d+)", re.IGNORECASE)
_LABEL_TRIM = " \t\n,.;:'\"“”‘’"


class Scale(BaseModel):
    model_config = ConfigDict(frozen=True)

    lo: int
    hi: int
    labels: dict[int, str] = {}
    source: Literal["parsed", "observed"] = "parsed"

    @property
    def n_levels(self) -> int:
        return self.hi - self.lo + 1

    def index_of(self, response: int) -> int:
        if not self.lo <= response <= self.hi:
            raise ValueError(f"response {response} is outside {self.lo}..{self.hi}")
        return response - self.lo

    def bins(self) -> list[tuple[int, int]]:
        """Inclusive (first, last) level ranges: one per level, or ten near-equal bins for long scales."""
        if self.n_levels <= MAX_OPTIONS:
            return [(level, level) for level in range(self.lo, self.hi + 1)]
        base, extra = divmod(self.n_levels, MAX_OPTIONS)
        bins, start = [], self.lo
        for index in range(MAX_OPTIONS):
            width = base + (1 if index < extra else 0)
            bins.append((start, start + width - 1))
            start += width
        return bins

    def descriptions(self) -> list[str]:
        """One self-contained description per bin. Each starts with its numerals, so all are distinct."""
        span = self.hi - self.lo
        low, high = self.labels.get(self.lo), self.labels.get(self.hi)
        if low and high:
            frame = f"on the scale from {self.lo} ({low}) to {self.hi} ({high})"
        else:
            frame = f"on the response scale from {self.lo} to {self.hi}"
        described = []
        for first, last in self.bins():
            if first == last and first in self.labels:
                described.append(f"{first}: {self.labels[first]}")
                continue
            name = str(first) if first == last else f"{first} to {last}"
            position = _position_words(((first + last) / 2 - self.lo) / span)
            described.append(f"{name}: {position} {frame}")
        return described

    def expand(self, bin_probs: list[float]) -> list[float]:
        """Spread each bin's probability evenly over its levels."""
        bins = self.bins()
        if len(bin_probs) != len(bins):
            raise ValueError(f"expected {len(bins)} bin probabilities, got {len(bin_probs)}")
        levels: list[float] = []
        for (first, last), mass in zip(bins, bin_probs, strict=True):
            width = last - first + 1
            levels.extend([mass / width] * width)
        return levels


def _position_words(t: float) -> str:
    if t <= 0.0:
        return "the lowest point"
    if t >= 1.0:
        return "the highest point"
    if t < 0.2:
        return "very near the low end"
    if t < 0.4:
        return "toward the low end"
    if t < 0.5:
        return "slightly below the middle"
    if t == 0.5:
        return "exactly the middle"
    if t <= 0.6:
        return "slightly above the middle"
    if t <= 0.8:
        return "toward the high end"
    return "very near the high end"


def _tail(text: str) -> str | None:
    matches = list(re.finditer(r"only return\b", text, re.IGNORECASE))
    return text[matches[-1].start() :] if matches else None


def strip_response_instruction(text: str) -> str:
    """Drop the trailing 'Only return ...' sentence: it instructs an LLM, not a survey respondent."""
    tail = _tail(text)
    if tail is None:
        return text.strip()
    return text[: len(text) - len(tail)].rstrip(" \t\n\"“'‘")


def parse_scale(text: str) -> Scale | None:
    """Read the response range and labels from the trailing instruction. None when there is no range."""
    tail = _tail(text)
    if tail is None:
        return None
    labels: dict[int, str] = {}
    for pattern in (_MEANS, _FOR):
        for number, label in pattern.findall(tail):
            cleaned = " ".join(label.split()).strip(_LABEL_TRIM)
            if cleaned:
                labels.setdefault(int(number), cleaned)

    span = _RANGE.search(tail)
    if span:
        lo, hi = int(span.group(1)), int(span.group(2))
    elif labels:
        lo, hi = min(labels), max(labels)
    else:
        listed = _ENUMERATION.search(tail)
        if not listed:
            return None
        numbers = [int(number) for number in re.findall(r"-?\d+", listed.group(1))]
        lo, hi = min(numbers), max(numbers)
    if lo >= hi:
        return None
    return Scale(lo=lo, hi=hi, labels={k: v for k, v in labels.items() if lo <= k <= hi})
