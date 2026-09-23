"""The choices a run may make, and the facts those choices resolve to.

Every option here has a small closed set, so it is an enum and a typo fails at parse time. That
matters most for `--split`: `dev` and anything else used to mean "test", which made `--split Dev`
evaluate the 40 unseen studies under a development label with nothing downstream able to tell.

A run directory also has to be self-describing. `config.yaml` records what a run was invoked with
*and* what that resolved to - the study ids or wave numbers actually evaluated, the frozen phrasing
it was held to, the commits involved - so a later reader never has to re-derive them from a flag.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kite.eval.phrasing import Phrasing, Primitive

CONFIG = "config.yaml"


class ProtocolError(RuntimeError):
    """A run would break the dev/test discipline. Always refuse; never warn and continue."""


class Split(StrEnum):
    dev = "dev"
    test = "test"


class TaskName(StrEnum):
    socsci210 = "socsci210"
    opinionqa = "opinionqa"


class Mode(StrEnum):
    default = "default"
    steered = "steered"


class BaselineKind(StrEnum):
    uniform = "uniform"
    pooled = "pooled"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class FrozenChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrasing: Phrasing
    primitive: Primitive


class Freeze(BaseModel):
    """`configs/eval/frozen.yaml`: what the development comparison chose, and when it was frozen."""

    model_config = ConfigDict(extra="forbid")

    commit: str
    tasks: dict[TaskName, FrozenChoice]
    dev_report: str | None = None
    note: str | None = None


TEMPLATE = """commit: <`git rev-parse --short HEAD` at the moment of freezing>
dev_report: results/dev-grid/report.md
tasks:
  socsci210: {phrasing: <chosen on dev>, primitive: <chosen on dev>}
  opinionqa: {phrasing: <chosen on dev>, primitive: <chosen on dev>}"""


def load_freeze(path: str | Path) -> Freeze:
    """The frozen choice. A missing or malformed file blocks a test run; it never falls back to defaults."""
    path = Path(path)
    if not path.exists():
        raise ProtocolError(
            f"{path} does not exist, so nothing holds this test run to a choice made on the development split.\n"
            "`--split test` requires the frozen phrasing and primitive; `--split dev` does not.\n"
            "Run the development comparison first (docs/superpowers/plans/2026-09-20-sp1-kernel-and-fidelity-eval.md, step 3),\n"
            f"then write {path} and commit it:\n\n{TEMPLATE}\n\n"
            "Fill it in from the dev comparison only - never from a test run."
        )
    try:
        return Freeze.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except Exception as error:
        raise ProtocolError(f"{path} is not a usable freeze file: {error}\n\nIt must look like:\n\n{TEMPLATE}") from None


def check_frozen(task: TaskName, phrasing: Phrasing, primitive: Primitive, path: str | Path, *, override: bool = False) -> dict[str, Any]:
    """What a test run must record about the freeze, refusing unless it matches or is explicitly overridden."""
    freeze = load_freeze(path)
    chosen = freeze.tasks.get(task)
    if chosen is None:
        raise ProtocolError(
            f"{path} has no entry for {task.value}, so its phrasing and primitive were never frozen; "
            f"add `tasks.{task.value}: {{phrasing: ..., primitive: ...}}` from the dev comparison before running the test split."
        )
    used = {"phrasing": phrasing, "primitive": primitive}
    differs = {field: value for field, value in used.items() if getattr(chosen, field) != value}
    if differs and not override:
        raise ProtocolError(
            f"refusing `--split test` for {task.value} with phrasing={phrasing.value} primitive={primitive.value}: "
            f"{path} (frozen at {freeze.commit}) says phrasing={chosen.phrasing.value} primitive={chosen.primitive.value}.\n"
            "The test split is evaluated once, with the values chosen on dev. Drop the flags to use the frozen values, "
            "or pass --override-frozen to run anyway - the override is then recorded in config.yaml."
        )
    return {
        "source": str(path),
        "commit": freeze.commit,
        "phrasing": chosen.phrasing.value,
        "primitive": chosen.primitive.value,
        "override": {field: value.value for field, value in differs.items()} or None,
    }


class TaskConfig(BaseModel):
    """The `task` block of a run's config.yaml: enough to rebuild the task, plus what it resolved to."""

    model_config = ConfigDict(extra="forbid")

    name: TaskName
    split: Split
    phrasing: Phrasing
    primitive: Primitive
    mode: Mode
    max_per_cell: int | None = None
    max_questions_per_wave: int | None = None
    limit: int | None = None
    # The resolved membership: the study ids (SocSci210) or waves (OpinionQA) this run covers, read
    # off the data the task actually loaded rather than off the name of the split.
    members: list[str] | None = None


class RunConfig(BaseModel):
    """A run directory's config.yaml. Extra keys are kept so older runs still read back."""

    model_config = ConfigDict(extra="allow")

    task: TaskConfig
    kernel: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    frozen: dict[str, Any] | None = None
    resumed: list[dict[str, Any]] = Field(default_factory=list)
    recalibration: dict[str, Any] | None = None

    @classmethod
    def read(cls, run_dir: str | Path) -> RunConfig:
        path = Path(run_dir) / CONFIG
        if not path.exists():
            raise ProtocolError(f"{path} does not exist, so this directory does not say what it is; only a run written by `eval` can be used here")
        try:
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except ProtocolError:
            raise
        except Exception as error:  # a hand-edited or pre-enum config
            raise ProtocolError(f"{path} is not a valid run config: {error}") from None

    def write(self, run_dir: str | Path) -> None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / CONFIG).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
