# SocSci210: schema and how we use it

Source: HuggingFace dataset `socratesft/SocSci210` (Kolluri, Wu, Park, Bernstein, EMNLP 2025, arXiv 2509.05830).
Checked on 2026-09-20. The dataset card states no license; we do not redistribute any of it.

## Files

| Path | Content |
|---|---|
| `data/train-000NN-of-00017.parquet` | 17 shards, 2,901,390 rows, about 1.4 GB to download |
| `metadata/participant_mapping.json` | `{"seen": [170 study ids], "unseen": [40 study ids]}`, the study-level split used in the paper's "unseen studies" table |
| `metadata/task_mapping.json`, `metadata/condition_mapping.json` | sample-level 75/25 splits for the paper's other two settings; we do not use them |

## Columns

| Column | Type | Notes |
|---|---|---|
| `sample_id` | int | unique per row; our `item_id` |
| `participant` | int | participant index within a study |
| `demographic` | struct | age (int), education, employment, ethnicity (often null), gender, household_size (int), housing_ownership, housing_type, ideology, income, internet_access, location (US state), marital_status, metro_status, party_id, phone_service |
| `stimuli` | string | one text holding the vignette, the question, and an LLM-facing answer-format sentence ("Only return an integer from 1 to 7, where 1 means ... , nothing else.") |
| `response` | int | the participant's answer on that scale |
| `condition_num` | int | experimental condition |
| `task_num` | int | outcome question within the study |
| `study_id` | string | five-character TESS study id |
| `prompt`, `reasoning` | string | the authors' fine-tuning prompt and a synthetic rationale; long, never read by us |

## What `prepare` does

1. Reads only the eight columns we need, filtered to the requested studies.
2. Attaches a response scale to every **(study, condition, task)** cell, parsed from that cell's own
   answer-format sentence (`windtunnel.eval.scales.parse_scale`). Where the sentence gives no range at all
   (for example "Only return an integer representing dollars"), the min..max observed **across the whole
   (study, task)** stands in: a condition whose text states no range states nothing we could differ on, and
   reading it per condition would hand every cell the span of the answers it is scored against. Written to
   `scales.json`, keyed exactly like `cells.json`.
3. Marks cells with more than 101 levels as unsupported (unbounded dollar amounts), and drops rows whose
   response lies outside their cell's scale.
4. Tabulates the human answer counts of every (study, condition, task) cell from all remaining rows.

### Why the scale is per condition, not per task

The answer-format sentence is part of the stimulus, so a study can word it differently in different
conditions. Scanning all 210 studies (5,998 cells), **71 (study, task) groups disagree across their
conditions**: 14 state a different numeric range (`53kjy` task 0 is 0-4 in condition 0 and 0-5 in
condition 1; `cdgp2` task 0 is 0-7 and 0-8) and 57 keep the range but change the option labels.

The clearest case is `3ydty`, a name-choice experiment: conditions 0-1 offer `{1: Brian, 2: Matt}`,
conditions 2-3 offer `{1: Amy, 2: Jennifer}` and conditions 4-7 offer `{1: Brian, 2: Jennifer}`. The
labels *are* the manipulation. Reading one scale per (study, task) off whichever row came first asked
every respondent about Brian and Matt, and flattened the two conditions of `53kjy` task 0 onto one
range. That corrupts exactly the studies the condition-sensitivity gate (G1b) depends on, so
`prepare`'s summary now reports `n_tasks_range_varies`, `n_tasks_labels_vary` and
`n_tasks_conditions_disagree` on every split.

One consequence: the "pooled" baseline merges a task's conditions only when they state the same
numeric range. A five-level and a six-level histogram are not commensurable.

Prepared splits (2026-09-20, this repository's real data):

| Split | studies | rows | cells | tasks | range varies | labels vary | unsupported cells | dropped out of range |
|---|---|---|---|---|---|---|---|---|
| dev (20 training studies) | 20 | 275,644 | 467 | 151 | 4 | 2 | 8 | 888 |
| test (the 40 unseen studies) | 39 | 456,587 | 723 | 175 | 2 | 4 | 75 | 9,809 |

Scale lengths seen: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 101.

## Metrics (as defined in the paper, section 3.2)

- accuracy = 1 - mean(|prediction - response| / (max - min)), reported under **both** conventions because
  they differ by about 0.09 and are easily confused. `accuracy_micro` / `accuracy_macro` take the median of
  the predicted distribution, the point prediction that minimizes expected absolute error;
  `accuracy_micro_random_draw` / `accuracy_macro_random_draw` take one answer **drawn** from it. The paper's
  Uniform Guess row (0.612) is the drawing convention, and so is `parity_uniform`. On the real dev split a
  uniform predictor scores 0.700 by the median rule and 0.614 by drawing — putting "Uniform 0.612 (as
  published)" next to a median-rule model number would understate the floor by enough to flip the sign of
  "did the model beat uniform".
- distribution = Wasserstein distance between predicted and human response distributions on responses
  rescaled to [0, 1], per (condition, task) cell, averaged within a study and then across studies. Lower is better.
- condition sensitivity (ours, not the paper's; the G1b gate). Per cell, the predicted and human mean
  response, each rescaled to [0, 1]. Both are centred within their own (study, task), which removes the
  task's overall level and leaves only movement across conditions; the centred pairs are pooled across
  every task with at least two conditions. `score` reports Pearson `r`, the least-squares slope of
  human-on-predicted (one unit of predicted movement is this many units of human movement), the spread
  ratio std(predicted) / std(human), and a 95% bootstrap interval for `r` that resamples **whole tasks** —
  cells within a task share a question and a respondent pool, so resampling cells would understate it.
  A predictor that does not move across a task's conditions reports `r = None`, not a number fitted to
  its own rounding noise.
- Published reference values on the 40 unseen studies: uniform guess 0.203 (accuracy 61.2), GPT-4o 0.174,
  GPT-4o few-shot 0.161, Socrates-LLaMA3-8B SFT 0.153, Socrates-Qwen2.5-14B SFT 0.151, empirical bound 0.125.

## Parity check against the published Uniform Guess row (2026-09-20)

Run on all 40 unseen studies (472,833 responses, 798 cells), with `prepare`'s level cap removed so
nothing is excluded. "Uniform guess" is model-free, so any mismatch is ours, not the model's.
Recomputed 2026-09-20 after the scale was keyed by condition; the earlier per-task figures are in
the last column.

| Convention for the [0,1] rescaling | accuracy | distribution | before the per-condition fix |
|---|---|---|---|
| Stated scale range, parsed from the question text | 0.6006 | 0.1678 | 0.1674 |
| **Observed response range (min/max of the answers in the cell)** | — | **0.2015** | 0.2024 |
| Published Uniform Guess row | 0.612 | **0.2030** | 0.2030 |

**The distribution metric reproduces the published number to 0.0015 once responses are rescaled by the
range respondents actually used** rather than by the scale the question states. That is the natural
thing to compute from a response column alone, with no scale parser, which is what the authors had.
This is inferred from the match, not from their code, which is not public.

Accuracy does not depend on the rescaling convention here, so `parity_uniform` reports one figure;
the earlier table's second accuracy row came from also renormalising the absolute error by the
observed range, which no metric in this repository does.

With the 101-level cap that `windtunnel parity-socsci210` actually applies (723 cells), the same run
gives 0.1564 under the stated scale and 0.1771 under the published convention.

**The no-range fallback has to be read across the task.** Where a question states no range at all we
invent one from the answers. Reading it *within the condition* sets each cell's scale to exactly the
span of the answers it is scored against, which collapses the two conventions on those cells (they
come out identical, 0.2536 flat-mean on the 104 such cells, against 0.6025 task-wide) and pulls the
parity number to 0.1873. Worse, it erases real condition differences: two conditions answering 0-2
and 4-6 both become "the full range", with the same mean position. `prepare` therefore parses the
scale per condition but falls back per task.

Two hypotheses were tested and ruled out first: excluding high-cardinality tasks accounts for only
0.011 of the 0.047 gap (0.1561 with our 101-level cap, 0.1674 with no cap), and drawing one uniform
answer per participant instead of using the smooth uniform distribution accounts for 0.0007 (with
hundreds of participants per cell the histogram converges). Averaging flat over cells instead of per
study, and rescaling to the union of options anyone used, both move the number the wrong way.

The accuracy row stays about 0.016 low under either convention. Unexplained; most likely task
inclusion or a micro/macro aggregation difference. It does not affect the gate, which reads the
distribution metric.

**What we do about it.** `score()` reports the distribution distance under both conventions, as
`distribution` and `distribution_published_convention`. The
stated-scale number is primary — it does not depend on which answers happened to occur — and the
observed-range number is what we put next to the published GPT-4o, Socrates and bound figures.
Any table that cites those figures uses the observed-range column, and says so.

## Differences from the paper's setup that readers must be told about

- We score only supported tasks (see step 3). `windtunnel parity-socsci210` reports how our model-free
  "uniform guess" numbers compare with the paper's, which shows how much this matters.
- The "pooled" baseline (the task's human distribution with conditions merged) is an oracle that uses the same
  study's answers. It is a yardstick for condition sensitivity, not a competitor. Where a task's conditions
  state different numeric ranges their level indices are not commensurable, so the merge happens on the
  shared [0, 1] axis (`metrics.regrid`). Adding the histograms range by range instead would make this
  baseline a per-cell oracle on exactly the tasks that vary — measured at r = 0.56 on the dev split, which
  would destroy its role as G1b's floor. Merged on the shared axis it reports `r = None` on both splits.

## Within-person repeats: `task_num` is not a question identifier everywhere (found 2026-09-21)

In roughly half the studies one person has several rows with the **same** `task_num` and **different**
`stimuli`: a within-subject design in which each respondent answers the same question about several
stimuli (vignettes, candidates, images), and the reconstruction gave every one of those items the
same task number.

| Split | Studies affected | Rows affected |
|---|---|---|
| Test (unseen) | 19 of 40 | 202,023 of 482,642 (41.9%) |
| Dev pool (seen) | 77 of 170 | 1,156,355 of 2,418,748 (47.8%) |

The rows share demographics (it is the same person) and differ in stimulus text; in nj5dx, 865 of
1,326 repeated pairs also differ in the answer.

**What this does not affect: the cell-level metrics (G1, G1b).** A cell is (study, condition, task),
as in the published paper. The predicted cell distribution averages the rows in the cell and the
human one pools the same rows, so both sides pool across stimuli in the same way.

**What it does affect: anything that treats (person, task) as one answer** - the inter-item
correlation experiments. There, a task becomes a blend of stimuli, and in an autoregressive
simulation a person answers "the same question about a different stimulus" several times in a row,
which manufactures local structure in the history. Individual-level analyses must either restrict
themselves to unaffected studies or identify an item by (task, stimulus). 31 dev studies with at least
six questions and 150 complete respondents are unaffected.
