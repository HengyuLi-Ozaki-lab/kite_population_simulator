# Coding protocol: were a TESS study's results public? (A3c)

Fixed before coding started. Criteria: `configs/eval/publication_moderator.yaml`. Coders are agents with web
search; each codes every study independently, in its own random order, and sees only the public metadata
below - never a model result, never the other coder's work, and nothing in this repository except its own
input file.

## Input per study

`study_id` (the TESS project's OSF id; the project page is `https://osf.io/<study_id>/`), the OSF title and
creation date, and - where the TESS past-studies index matched the title - the TESS study page URL, its
title, PIs, field period and the text of its sections (Abstract, Hypotheses, Summary of Results, ...),
collected by `scripts/a3c_tess_metadata.py`. Where no TESS page matched, the OSF wiki text (PIs, field
period) is given instead.

## What to decide

**1. Does the study's TESS page state what was found?** (`tess_page_reports_findings`)
- `true`: a Summary of Results (or similarly named) section that reports findings - directions of effects,
  which hypotheses held - or an abstract that reports what was found ("we find", "respondents were more
  likely to ..."). A section that only points elsewhere ("see the OSF page") does not count.
- `false`: the page describes design, hypotheses or measures only, or says no further information is available.
- `null`: no TESS page could be tied to the study.

**2. Was the experiment's result published before 2025-01-01?** (`publication`: `yes` / `no` / `unclear`)

Counts: a journal article, book or chapter, dissertation, conference paper, or working paper / preprint
(SSRN, OSF Preprints, arXiv, a university repository, a PI's website) that **reports results of this
experiment**. It is this experiment when the document says its data came from TESS (or from the TESS
sample: Knowledge Networks, GfK or NORC AmeriSpeak) **and** describes the same manipulation, or matches
the design, field period and sample size.

Does not count: a paper by the same PIs about the same topic that reports a different experiment; a paper
that only cites the TESS study or its data; the TESS proposal or the TESS page itself (that is question 1);
a conference programme listing without results.

Search, at minimum, in this order, and record every query:
1. the PIs' surnames with two or three distinctive words of the title;
2. distinctive title words with "TESS" or "Time-sharing Experiments for the Social Sciences";
3. a PI's full name with the topic and "survey experiment";
4. if a candidate appears: open it (publisher page, repository, PDF) and check the data section.
Stop when a match is verified, or after about six searches without a candidate (`no`). Use `unclear` only
when a candidate exists but cannot be verified (for example the data section is behind a paywall and the
abstract is ambiguous) - and say why.

## Output per study (JSON)

```json
{"study_id": "sffyb",
 "tess_page_reports_findings": false,
 "tess_evidence": "why, in at most 25 words",
 "publication": "no",
 "citation": null,
 "match_evidence": null,
 "earliest_public_year": null,
 "queries": ["..."],
 "notes": ""}
```

When `publication` is `yes`: `citation` = {"authors", "year", "title", "venue", "url", "doi"} (doi may be
null), `match_evidence` = what identifies it as this experiment (at most 40 words), and
`earliest_public_year` = the year the result first appeared in any public form (a working paper counts).

## Reconciliation (by a third agent, after both coders finish)

- Agreement is computed first, on `results_public` and on `publication`, and reported (percent and Cohen's kappa).
- A publication one coder found and the other did not: the third agent opens the cited source and accepts
  it if it meets the rule above.
- `tess_page_reports_findings` disagreements: the third agent reads the page text and decides.
- Both `unclear`: coded `no`, and the study is flagged `unclear` (a sensitivity analysis drops it).
- `results_public` = `tess_page_reports_findings` is true **or** `publication` is `yes` with
  `earliest_public_year` < 2025. `published` = the latter alone.
- Citation counts for accepted publications come from OpenAlex (`cited_by_count`), by DOI where there is
  one and otherwise by exact title.

The reconciled labels are written to `docs/data/tess_publication_status.csv` and committed before the
analysis runs.
