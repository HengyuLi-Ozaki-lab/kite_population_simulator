# KITE

**K**ernel · **I**ntervention anchors · **T**abulated execution · **E**rror propagation.
Population experiments on a typed behavioral kernel (TypeSafe's Jev, `jev-1.13.0`), with sparse flagship-LLM
corrections and propagated human–model discrepancy.

**Showcase:** [hengyuli-ozaki-lab.github.io/kite_population_simulator](https://hengyuli-ozaki-lab.github.io/kite_population_simulator/),
the architecture and the held-out results on one page.

This repository accompanies the preprint

> Hengyu Li (2026). *KITE: Scaling Jev Population Experiments with Sparse Flagship Calibration.* arXiv preprint
> (identifier to be added on posting).

It contains the `kite` Python package, the script behind every evaluation, the frozen criteria files with their
appended errata, the evaluation records, the manuscript source, and the aggregate results that every table and figure
is drawn from. It does not contain survey microdata or the Jev response cache (see [Data and terms](#data-and-terms)).

## The architecture

![KITE's architecture running the method once: the inputs feed the kernel (K), which fills the state table once per unique state; the flagship's paired anchor calls (I) tilt the treated row; the population runs from the table (T); the shared human–model gap widens the interval (E).](docs/readme/kite-architecture.gif)

- **Kernel.** Jev returns a typed answer distribution for a described respondent in a described situation. KITE asks
  it once per unique state (persona × content × situation × condition) and stores the answer in a state table.
- **Intervention anchors.** A flagship LLM predicts a sparse set of *paired* states — the same persona and content
  under control and under treatment. Within declared cells, the kernel's treated distributions are exponentially
  tilted so that their mean equals the kernel's control mean plus the flagship's paired effect
  (`kite.operators.tilt`).
- **Tabulated execution.** Populations of any size run from the table with event-keyed random numbers, common random
  numbers across conditions and Rao–Blackwell expectations (`kite.engine`).
- **Error propagation.** The measured human–model discrepancy on condition effects,
  θ = β·x + b(study) + u(arm), is drawn once per study and arm and shared by every agent, so intervals describe
  uncertainty about people rather than Monte Carlo noise (`scripts/d2_discrepancy_model.py`).

## Results at a glance

![KITE results at a glance. A: intervention-effect error −41%, kernel 0.0305 to hybrid 0.0180 (Epstein 2021, held out). B: captured decision gain +0.12, 0.27 to 0.39 (SocSci210, 37 unseen experiments). C: held-out coverage of nominal 90% intervals 96%, against 36% from sampling error alone. D: 10⁶ agents × 20 steps from the state table in 0.90 s.](docs/readme/kite-results.png)

Every number is read on held-out human experiments, with the criteria committed before the results were read. The
values are in [`aggregates/`](aggregates/), and the
[showcase](https://hengyuli-ozaki-lab.github.io/kite_population_simulator/) walks through each result.

## Layout

| Path | Contents |
|---|---|
| `src/kite/` | the package: kernel adapters, cache and ledger (`kernel/`), evaluation harness and tasks (`eval/`), engine (`engine/`), operators (`operators/`), personas (`population/`), CLI |
| `scripts/` | one script per evaluation, named after its record (`d1_epstein.py`, `b_paired_anchors.py`, `c_scale_demo.py`, …); each docstring gives the command |
| `configs/eval/` | criteria files, committed before the held-out run they govern; corrections are appended as errata, never edited in |
| `docs/decisions/` | evaluation records written as each evaluation ran (in Chinese; tables and numbers are language-neutral) |
| `docs/data/` | data provenance: schemas, file hashes, audits, what was seen before each freeze |
| `docs/paper/arxiv/` | manuscript source, bibliography and figures (`figures/fig1_architecture.tex` is TikZ, compiled with LuaLaTeX) |
| `aggregates/` | the aggregate results behind the paper — see [`aggregates/README.md`](aggregates/README.md) |
| `tests/` | unit tests (no API key or data needed) |

Records and the paper sections they support: G1, G-A (decision value) and A2–A6 → §5; C2 (Arechar, incl. the flagship
comparison C3b) → §6; D0, D1 (Epstein) and D3 (SocSci210 paired anchors) → §7; D2 → §8; A3/A3b, A3c and S1 → §9;
B1/B2 and E3–E3d (individual coherence and memory) → §5 and the limitations.

## Install and test

```bash
uv sync
uv run pytest -q
uv run kite --help
```

## Reproducing the evaluations

1. **Data.** Obtain each dataset from its source and place it under `data/` (never committed):
   SocSci210 (`uv run kite fetch-socsci210`, HuggingFace `socratesft/SocSci210`), Arechar et al. 2023 (OSF `g65qu`),
   Epstein et al. 2021 (Harvard Dataverse, doi:10.7910/DVN/18SHLJ; stimuli on OSF `hu4k2`). `docs/data/` records the
   files and hashes used.
2. **Keys.** Put `TYPESAFE_API_KEY` in `.env` (never committed). Flagship runs use the local Codex CLI.
3. **Runs.** Each script refuses to run on held-out data while its criteria file or the script itself is
   uncommitted. Kernel runs are resumable and stop at `--max-usd`; the whole project's Jev spend was about $20.
4. **Figures.** `PYTHONPATH=scripts uv run python scripts/paper_figures.py` redraws Figures 1–5 from the run
   directories under `results/`; the same numbers are in `aggregates/`.

Outputs are not bit-for-bit reproducible: the kernel and the flagship are hosted models, so version pins and recorded
run hashes document what was computed but cannot guarantee identical answers later.

## Data and terms

Only aggregates are redistributed. No respondent-level row, no survey microdata and no Jev response is included.
The full response cache is not released, and model outputs are not used for distillation or to train an imitation
model, in line with the Jev terms of use. Dataset licences: Epstein et al.'s Dataverse data are CC0; SocSci210 and
the Arechar OSF project state no licence, so their original terms apply.

## Use of generative AI

Beyond the models evaluated here, generative AI tools were used throughout the project: Claude (Anthropic, via Claude Code) for analysis code and evaluation records. The author directed the work and takes full responsibility for all content.

## Citation

```bibtex
@misc{li2026kite,
  author = {Li, Hengyu},
  title  = {{KITE}: Scaling {Jev} Population Experiments with Sparse Flagship Calibration},
  year   = {2026},
  note   = {arXiv preprint},
  url    = {https://github.com/HengyuLi-Ozaki-lab/kite_population_simulator}
}
```

## License

Code: MIT (see `LICENSE`). Data remain under their sources' terms.
