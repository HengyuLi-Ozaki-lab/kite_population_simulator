"""Command line entry point: `kite <command>`."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from kite.config import Settings
from kite.eval import probes as probe_module
from kite.eval.opinionqa import OpinionQATask, check_layout, waves_of
from kite.eval.phrasing import Phrasing, Primitive
from kite.eval.protocol import BaselineKind, Mode, ProtocolError, RunConfig, Split, TaskConfig, TaskName, check_frozen, git_commit
from kite.eval.recalibrate import apply_to_run, fit_from_run, load_fit_for
from kite.eval.report import build_report
from kite.eval.runner import run_task, write_predictions
from kite.eval.socsci210 import SocSci210Task, prepare, studies_of
from kite.eval.task import FidelityTask
from kite.kernel.build import AnswerMode, Backend, KernelSpec, Provider, build_kernel
from kite.kernel.cache import CacheStore
from kite.kernel.ledger import Ledger

app = typer.Typer(no_args_is_help=True, add_completion=False)
cache_app = typer.Typer(no_args_is_help=True)
app.add_typer(cache_app, name="cache")

SOCSCI210_REPO = "socratesft/SocSci210"
# Published in Kolluri et al. (EMNLP 2025), unseen-studies table.
SOCSCI210_UNIFORM_REFERENCE = {"accuracy": 0.612, "distribution": 0.203}

# Exit codes: 1 too many failed items, 2 stopped at the dollar cap (resumable), 3 refused on protocol.
BUDGET_STOP = 2
REFUSED = 3


@app.callback()
def _load_environment() -> None:
    load_dotenv()


def _refuse(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(REFUSED)


def _run_dir(settings: Settings, name: str) -> Path:
    return settings.results_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"


def make_task(config: TaskConfig, settings: Settings) -> FidelityTask:
    """Rebuild a task from the `task` block of a run's config.yaml."""
    if config.name is TaskName.socsci210:
        return SocSci210Task(
            settings.data_dir / "socsci210" / "prepared" / config.split.value,
            phrasing=config.phrasing,
            primitive=config.primitive,
            max_per_cell=config.max_per_cell,
        )
    return OpinionQATask(
        settings.data_dir / "opinionqa",
        waves=waves_of(config.split),
        mode=config.mode,
        phrasing=config.phrasing,
        primitive=config.primitive,
        max_questions_per_wave=config.max_questions_per_wave,
    )


def _resolved(config: TaskConfig, task: FidelityTask) -> TaskConfig:
    """The same config, with the split's name resolved to the studies or waves actually loaded."""
    return config.model_copy(update={"members": task.members()})


def _protocol_facts(task: TaskConfig, spec: KernelSpec) -> dict[str, Any]:
    """Everything that decides what a run *means*, keyed by the flag that sets it.

    `--max-usd` and `--resume-dir` are deliberately absent: they steer one invocation, and a resume
    may legitimately raise the cap. Everything here, changed halfway through, would leave one
    predictions.jsonl holding two different experiments.
    """
    return {
        "task": task.name.value,
        "split": task.split.value,
        "phrasing": task.phrasing.value,
        "primitive": task.primitive.value,
        "mode": task.mode.value,
        "max_per_cell": task.max_per_cell,
        "max_questions_per_wave": task.max_questions_per_wave,
        "limit": task.limit,
        "kernel": spec.backend.value,
        "model": spec.model_id,
        "provider": spec.provider.value if spec.provider else None,
        "answer_mode": spec.answer_mode.value,
        "n_samples": spec.n_samples,
    }


def _given(ctx: typer.Context, names: Iterable[str]) -> list[str]:
    """Which of these options the operator actually typed, as opposed to taking their default.

    Compared by name because typer vendors its own copy of click: the context's ParameterSource is
    `typer._click.core.ParameterSource`, which is never identical to `click.core`'s member.
    """
    return [name for name in names if getattr(ctx.get_parameter_source(name), "name", None) == "COMMANDLINE"]


def _resume(ctx: typer.Context, resume_dir: Path, requested: TaskConfig, spec: KernelSpec) -> tuple[RunConfig, KernelSpec]:
    """Continue an interrupted run with *its* configuration, refusing any flag that contradicts it.

    Resuming used to take the new invocation's options and rewrite config.yaml with them, so
    `--resume-dir` without the original `--phrasing p3` appended default-phrasing predictions to a
    file the config then described as one phrasing throughout.
    """
    previous = RunConfig.read(resume_dir)
    try:
        stored_spec = KernelSpec.model_validate(previous.kernel)
    except ValidationError as error:  # e.g. a baseline directory, whose backend is not a kernel
        raise ProtocolError(f"{resume_dir} does not record a kernel this command can continue: {error}") from None
    stored, asked = _protocol_facts(previous.task, stored_spec), _protocol_facts(requested, spec)
    clashes = [
        f"--{name.replace('_', '-')}: the run used {stored[name]!r}, you passed {asked[name]!r}"
        for name in _given(ctx, stored)
        if stored[name] != asked[name]
    ]
    if clashes:
        raise ProtocolError(
            f"refusing to resume {resume_dir} with a different configuration - its predictions.jsonl would hold two experiments:\n  "
            + "\n  ".join(clashes)
            + "\nDrop those flags to continue with the run's own configuration, or start a new run."
        )
    return previous, stored_spec.model_copy(update={"max_usd": spec.max_usd})


@app.command("fetch-socsci210")
def fetch_socsci210() -> None:
    """Download the SocSci210 parquet shards and split metadata from HuggingFace (about 1.4 GB)."""
    from huggingface_hub import snapshot_download

    target = Settings().data_dir / "socsci210" / "raw"
    snapshot_download(SOCSCI210_REPO, repo_type="dataset", local_dir=target, allow_patterns=["data/*.parquet", "metadata/*", "README.md"])
    typer.echo(f"downloaded to {target}")


@app.command("prepare-socsci210")
def prepare_socsci210(split: Split = typer.Option(Split.dev, help="dev (20 training studies) or test (the 40 unseen studies)")) -> None:
    settings = Settings()
    raw = settings.data_dir / "socsci210" / "raw"
    summary = prepare(raw, settings.data_dir / "socsci210" / "prepared" / split.value, studies_of(split, raw))
    typer.echo(json.dumps(summary, indent=2))


@app.command("parity-socsci210")
def parity_socsci210(split: Split = Split.test) -> None:
    """Recompute the paper's model-free 'Uniform Guess' row to check that our metrics line up with theirs."""
    ours = SocSci210Task(Settings().data_dir / "socsci210" / "prepared" / split.value).parity_uniform()
    typer.echo(json.dumps({"ours": ours, "paper": SOCSCI210_UNIFORM_REFERENCE}, indent=2))


@app.command("check-opinionqa")
def check_opinionqa() -> None:
    missing = check_layout(Settings().data_dir / "opinionqa")
    if missing:
        typer.echo("missing:\n" + "\n".join(missing))
        raise typer.Exit(1)
    typer.echo("OpinionQA layout OK")


@app.command("eval")
def evaluate(
    ctx: typer.Context,
    task: TaskName = typer.Argument(..., help="socsci210 or opinionqa"),
    kernel: Backend = typer.Option(Backend.mock, help="jev, adapter, mock or uniform"),
    split: Split = typer.Option(Split.dev),
    phrasing: Phrasing = typer.Option(Phrasing.p1),
    primitive: Primitive = typer.Option(Primitive.choice),
    mode: Mode = typer.Option(Mode.steered, help="opinionqa only: default or steered"),
    max_per_cell: int | None = typer.Option(None, help="socsci210 only: participants per (study, condition, task) cell"),
    max_questions_per_wave: int | None = typer.Option(None, help="opinionqa only"),
    limit: int | None = typer.Option(None),
    max_usd: float = typer.Option(5.0),
    provider: Provider | None = typer.Option(None, help="adapter only: openai or anthropic"),
    model: str | None = typer.Option(None),
    answer_mode: AnswerMode = typer.Option(AnswerMode.probabilities),
    n_samples: int = typer.Option(1),
    resume_dir: Path | None = typer.Option(None, help="continue an interrupted run in this directory, with that run's own configuration"),
    override_frozen: bool = typer.Option(False, "--override-frozen", help="run the test split with values the freeze does not hold; recorded"),
) -> None:
    settings = Settings()
    task_config = TaskConfig(
        name=task,
        split=split,
        phrasing=phrasing,
        primitive=primitive,
        mode=mode,
        max_per_cell=max_per_cell,
        max_questions_per_wave=max_questions_per_wave,
        limit=limit,
    )
    spec = KernelSpec(backend=kernel, model_id=model, provider=provider, answer_mode=answer_mode, n_samples=n_samples, max_usd=max_usd)
    config = RunConfig(task=task_config, kernel=spec.model_dump(mode="json"), git_commit=git_commit())
    if resume_dir is not None:
        # a resume re-runs nothing of the freeze: the run was held to it when it started, and its
        # config - carried over whole, override and all - is what the remainder must match
        try:
            previous, spec = _resume(ctx, resume_dir, task_config, spec)
        except ProtocolError as error:
            _refuse(str(error))
        task_config = previous.task
        config = previous.model_copy(
            update={
                "kernel": spec.model_dump(mode="json"),
                "resumed": [*previous.resumed, {"at": f"{datetime.now():%Y-%m-%dT%H:%M:%S}", "git_commit": git_commit(), "max_usd": max_usd}],
            }
        )
    elif split is Split.test:
        try:
            frozen = check_frozen(task, phrasing, primitive, settings.freeze_path, override=override_frozen)
        except ProtocolError as error:
            _refuse(str(error))
        config = config.model_copy(update={"frozen": frozen})

    ledger = Ledger()
    out = resume_dir or _run_dir(settings, f"{task}-{split}-{kernel}-{phrasing}-{primitive}")
    built_task = make_task(task_config, settings)
    resolved = _resolved(task_config, built_task)
    if resume_dir is not None and task_config.members not in (None, resolved.members):
        _refuse(
            f"refusing to resume {out}: it evaluated {task_config.members} and the same split now loads {resolved.members}. "
            "The prepared data under that split changed; start a new run."
        )
    config = config.model_copy(update={"task": resolved})

    async def main():
        built = build_kernel(spec, settings, ledger)
        try:
            return await run_task(
                built_task,
                built.kernel,
                out,
                concurrency=settings.max_concurrency,
                limit=task_config.limit,
                ledger=ledger,
                config=config.model_dump(mode="json"),
            )
        finally:
            await built.aclose()

    summary = asyncio.run(main())
    typer.echo(f"{out}\n{summary.model_dump_json()}\n{json.dumps(ledger.summary())}")
    if summary.stopped == "budget":
        typer.echo(
            "stopped at the dollar cap. Resume with exactly this - every other option is read back from the run:\n"
            f"  uv run kite eval {task_config.name.value} --resume-dir {out} --max-usd {max_usd:g}"
        )
        raise typer.Exit(BUDGET_STOP)
    if summary.failure_rate > 0.01:
        raise typer.Exit(1)


@app.command("baseline")
def baseline(
    task: TaskName,
    kind: BaselineKind = typer.Argument(..., help="uniform or pooled"),
    split: Split = Split.dev,
    mode: Mode = Mode.steered,
    max_per_cell: int | None = None,
    max_questions_per_wave: int | None = None,
) -> None:
    settings = Settings()
    task_config = TaskConfig(
        name=task,
        split=split,
        phrasing=Phrasing.p1,
        primitive=Primitive.choice,
        mode=mode,
        max_per_cell=max_per_cell,
        max_questions_per_wave=max_questions_per_wave,
    )
    built_task = make_task(task_config, settings)
    predictions = built_task.baseline(kind.value)
    out = _run_dir(settings, f"{task}-{split}-baseline-{kind}")
    write_predictions(out, predictions)
    metrics = {"task": built_task.name, **built_task.score(predictions)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    RunConfig(task=_resolved(task_config, built_task), kernel={"backend": f"baseline-{kind}"}, git_commit=git_commit()).write(out)
    typer.echo(str(out))


@app.command("probe")
def probe(
    task: TaskName = TaskName.opinionqa,
    split: Split = Split.dev,
    mode: Mode = Mode.steered,
    kernel: Backend = Backend.jev,
    n_cases: int = 100,
    max_usd: float = 2.0,
    max_per_cell: int | None = 5,
) -> None:
    """Run every E0 probe against the uncached kernel and write probes.json."""
    settings = Settings()
    probe_config = TaskConfig(
        name=task,
        split=split,
        phrasing=Phrasing.p1,
        primitive=Primitive.choice,
        mode=mode,
        max_per_cell=max_per_cell,
        max_questions_per_wave=20,
    )
    cases = make_task(probe_config, settings).probe_cases(n_cases)
    ledger = Ledger()

    async def main():
        built = build_kernel(KernelSpec(backend=kernel, use_cache=False, max_usd=max_usd), settings, ledger)
        try:
            return {name: await run(built.kernel, cases) for name, run in probe_module.PROBES.items()}
        finally:
            await built.aclose()

    results = asyncio.run(main())
    out = _run_dir(settings, f"probes-{task}-{kernel}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "probes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ledger.write(out / "ledger.json")
    typer.echo(f"{out}\n{json.dumps(results, indent=2)}")


@app.command("recalibrate-fit")
def recalibrate_fit(run_dir: Path) -> None:
    """Fit a temperature on a dev run. Writes temperature.json into that run directory."""
    try:
        config = RunConfig.read(run_dir)
        temperature = fit_from_run(make_task(config.task, Settings()), run_dir)
    except ProtocolError as error:
        _refuse(str(error))
    typer.echo(f"temperature = {temperature:.4f}")


@app.command("recalibrate-apply")
def recalibrate_apply(test_run: Path, fitted_on: Path) -> None:
    """Apply the temperature fitted on `fitted_on` (a dev run) to `test_run`, into a new run directory."""
    settings = Settings()
    try:
        fit, config = load_fit_for(fitted_on, test_run)
    except ProtocolError as error:
        _refuse(str(error))
    out = _run_dir(settings, f"{test_run.name}-recalibrated")
    metrics = apply_to_run(make_task(config.task, settings), test_run, out, fit)
    config.model_copy(update={"recalibration": fit.model_dump(mode="json")}).write(out)
    typer.echo(f"{out}\n{json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, indent=2)}")


@app.command("report")
def report(run_dirs: list[Path], out: Path = typer.Option(Path("results/report"))) -> None:
    typer.echo(str(build_report(run_dirs, out)))


@cache_app.command("stats")
def cache_stats() -> None:
    typer.echo(json.dumps(CacheStore(Settings().cache_path).stats(), indent=2))


@cache_app.command("export")
def cache_export(path: Path) -> None:
    typer.echo(f"{CacheStore(Settings().cache_path).export_jsonl_gz(path)} entries written to {path}")


@cache_app.command("import")
def cache_import(path: Path) -> None:
    typer.echo(f"{CacheStore(Settings().cache_path).import_jsonl_gz(path)} entries read from {path}")


if __name__ == "__main__":
    app()
