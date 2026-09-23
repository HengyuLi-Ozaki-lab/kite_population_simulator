# OpinionQA: schema and how we use it

Source: Santurkar, Durmus, Ladhak, Lee, Liang, Hashimoto, "Whose Opinions Do Language Models Reflect?" (ICML 2023),
code at `github.com/tatsu-lab/opinions_qa`. The data are Pew Research Center American Trends Panel microdata, which
Pew does not allow us to redistribute: nothing under `data/opinionqa/` is ever committed.

## Getting the data (manual)

The bundle lives on CodaLab, worksheet `0x6fb693719477478aac73fc07db333f69`
(https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69). CodaLab's API timed out when we
tried to script the download on 2026-09-20, so this step is manual: open the worksheet in a browser, download the
dataset bundle, and unpack it so that `data/opinionqa/human_resp/` exists. Then run `uv run windtunnel check-opinionqa`.

## Availability as of 2026-09-20: blocked

The CodaLab REST API timed out on two attempts several hours apart, so the bundle could not be
fetched from this machine. Before falling back, the HuggingFace mirrors were checked and **none of
them is usable**:

| Mirror | What it has | Why it does not work |
|---|---|---|
| `andrewsiah/opinions_qa_users_responses` | 80,098 rows x 1,495 question columns, answers as integer indices | No demographic columns and no `WEIGHT_W*` columns |
| `d42me/opinions_qa` | 5.7M (user, question, prompt, response) rows | Demographics only as prose inside the prompt; no weights |
| `timchen0618/OpinionQA`, `RiverDong/OpinionQA` | Reformatted question/answer pairs | Same omissions |

Both omissions are fatal: the benchmark's human distributions are **survey-weighted** and computed
**per demographic group**, so respondent weights and group membership are both required. A mirror
without them cannot reproduce the reference numbers.

Options, in the order worth trying: retry CodaLab from another network; ask the authors for the
bundle; or obtain the American Trends Panel waves from Pew directly, which requires registration and
means rebuilding the question metadata ourselves.

Until then E1 does not run. This does not affect the G1 or G1b gates, which read SocSci210 only; it
costs the paper its second dataset and the "whose opinions does the model reflect" analysis.

## Layout we read

```
data/opinionqa/human_resp/American_Trends_Panel_W{wave}/info.csv
data/opinionqa/human_resp/American_Trends_Panel_W{wave}/metadata.csv
data/opinionqa/human_resp/American_Trends_Panel_W{wave}/responses.csv
```

Waves: 26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92. Dev split: wave 26. Test split: the other fourteen.

| File | Columns we use |
|---|---|
| `info.csv` | `key` (question id), `question` (text), `references` (Python list: ordinal options first, refusal options last), `option_ordinal` (Python list: one number per ordinal option) |
| `metadata.csv` | `key` (demographic attribute code), `options` (Python list of group names) |
| `responses.csv` | one row per respondent: a column per question key holding the answer text, a column per attribute, and `WEIGHT_W{wave}` |

Attributes (12): CREGION, AGE, SEX, EDUCATION, CITIZEN, MARITAL, RELIG, RELIGATTEND, POLPARTY, INCOME, POLIDEOLOGY, RACE;
60 groups in total.

## Human distributions and the metric

Distributions are survey-weighted, exclude refusals, and are renormalized over the ordinal options, as in the authors'
`helpers.extract_human_opinions`. Alignment is `1 - WD(model, human) / (max ordinal - min ordinal)` with the Wasserstein
distance computed on the `option_ordinal` positions, as in their `process_results.ipynb`.

`scripts/check_opinionqa_parity.py` imports the authors' own function from a clone of their repository and checks that
every distribution we compute equals theirs. The metric formula is covered by
`tests/eval/test_metrics.py::test_opinionqa_alignment_matches_the_original_formula`.

## Two evaluation modes

- `default`: no persona. Reports whose opinions the un-steered model reflects (alignment with each of the 60 groups).
- `steered`: persona = one demographic attribute. Reports alignment with that group. The "pooled" baseline answers every
  group with the whole population's distribution: a model that cannot beat it is not using the persona.
