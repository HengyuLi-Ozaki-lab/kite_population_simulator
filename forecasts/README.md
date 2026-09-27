# KITE prospective forecast ledger

Forecasts of experiments whose results are not yet public, sealed before the results exist and scored after.
This is the test that separates inference from memory: a model cannot have seen results that did not exist when the
forecast was sealed.

## How a forecast is sealed

1. A forecast is one JSON record (fields below).
2. `uv run python scripts/e2_ledger.py seal <record.json> --source "<where the study came from>"` draws a random
   salt and appends one line to `LEDGER.jsonl`: the forecast id, `SHA-256(salt + canonical JSON)`, the UTC time and
   the source. Canonical JSON: keys sorted, no spaces, UTF-8.
3. The record and its salt stay in `sealed/`, which is never published before the reveal. Study materials that are
   not public are never published by us at any point.
4. After the study's results are public, `scripts/e2_ledger.py reveal <id>` copies the record and salt to
   `revealed/`. Anyone can recompute the digest and compare it with the ledger line.
5. Each batch of sealed forecasts is published as a GitHub release of this repository, which Zenodo archives as a
   new version with its own DOI. That fixes the time independently of git history.

A sealed forecast is never replaced. A corrected forecast gets a new id and its own line, and both are scored.

## Record fields

| Field | Content |
|---|---|
| `forecast_id` | `<source>-<study>-<n>`, unique |
| `study` | source, title or identifier, link to the public protocol or materials, date the materials were read |
| `population` | who the study samples, as described in its materials |
| `conditions` | every condition: id, the stimulus as the model read it (or a hash, if the materials are not public), predicted answer distribution and mean on the unit scale |
| `effects` | each condition against the control named in the materials: predicted effect, 80% and 90% intervals from the discrepancy model (D2), and the probability of each sign |
| `best_condition` | the probability that each condition is the best one, where the study has three or more |
| `models` | kernel (`jev-1.13.0`), flagship and reasoning effort when anchors are used, KITE commit hash |
| `created` | UTC time the record was generated |

## How forecasts are scored (fixed 2026-09-27, before any forecast)

- Sign accuracy on reliable contrasts (|z| ≥ 3 in the study's own data, as in D3).
- Spearman rank correlation of the condition effects within each study.
- Coverage of the 80% and 90% intervals.
- Mean absolute error of the effects on the unit scale.
- Whether the forecast's most probable best condition is the study's best condition.
- Where the study has human forecasts (for example on the Social Science Prediction Platform), the same scores for
  the median human forecast.

Scores are computed for every sealed forecast, including those that turn out badly.

## Sources

- TESS studies fielded before TESS suspended operations and still within the one-year embargo (their proposals and
  data are posted together when it ends).
- Studies on the Social Science Prediction Platform whose authors opt in, as a separate and clearly labelled model
  forecaster.
- Registered reports with in-principle acceptance whose materials are public.
- Studies whose results became public after the training cutoffs of the models used (retrospective, but outside
  what the models could have seen).
