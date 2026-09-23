# Epstein et al. 2021: what was downloaded, what was read, and how the items are identified

Epstein, Berinsky, Cole, Gully, Pennycook & Rand (2021), "Developing an accuracy-prompt toolkit to reduce
COVID-19 misinformation online", *HKS Misinformation Review* 2(3). Data on Harvard Dataverse
(doi:10.7910/DVN/18SHLJ, **CC0 1.0**); questionnaires and stimuli on OSF `hu4k2` (no licence stated).
Downloaded 2026-09-22 with the owner's approval into `data/epstein/` (not committed).

| File | Source | Bytes | SHA-256 |
|---|---|---|---|
| `data_to_post.csv` | Dataverse file 4625647 | 299,630,356 | `6e151c0015c743817d014595d4ad668911bcaa8475a6605e11f0572c23c55a37` |
| `code.do` | Dataverse file 4625648 | 17,700 | `73ddac73898970d416f30f08c822cf7786ea3a0653bb35c122c79666e905997d` |
| `stimulii.zip` | OSF `fybg7` | 3,703,058 | `acbcb85f65dcaafe83de59ab2380d8b34112335efb5f8a87d47dbdf3ce974c90` |
| `wave1survey.qsf` … `wave5survey.qsf` | OSF `f2hwm`, `y7zgs`, `ex5k4`, `s68f3`, `h7vwx` | 849,854 / 904,149 / 909,410 / 919,243 / 922,461 | `b78c3157…2712`, `6a803614…df89`, `f69c43fd…a15e`, `a8361c61…d645`, `059ab45b…0404` |

## What has and has not been read

The audit for research plan v3 §5 (D1) must not see any outcome before the criteria are frozen. Columns read
so far: `id`, `wave`, `treatment`, `completed`, `item_num`, `real`, `study`, `condition`, and the pretreatment
covariates listed below. **`rating` (the sharing / accuracy answer) has not been read**, and the loader will
refuse it until `configs/eval/epstein_criteria.yaml` is committed, as `arechar.load_ratings` does. The
Qualtrics export also carries respondent metadata (`ipaddress`, `locationlatitude`, `locationlongitude`,
`recipientemail`, `zipcode`, …); those columns are never read by any code in this repository. The only
outcome-level numbers seen are the paper's and `code.do`'s published aggregates (pooled accuracy-prompt effect
on discernment 0.036, tips 0.035), which no per-wave decision can be tuned to.

The file is Windows-1252 / mixed; read it with `encoding="latin-1"` (the affected bytes are in free-text
columns that are not used). 555 columns; the long format has one row per participant × headline (224,740
rows, 11,237 participant-waves; 9,070 completed, each with all 20 items).

## Design as read from the data (completed participants)

| Wave (`study`) | Control (1) | Other arms (treatment code: n) | Total |
|---|---|---|---|
| 1 (`acc`) | 195 | accuracy-only (0): 208 | 403 |
| 2 (`frank`) | 387 | evaluation (2): 395 · long evaluation (3): 410 | 1,192 |
| 3 (`horserace`) | 533 | evaluation (2): 540 · generic norms (4): 510 · tips (5): 498 | 2,081 |
| 4 (`normstips`) | 487 | partisan norms (4.5): 949 · tips (5): 408 · tips + norms (6): 934 | 2,778 |
| 5 (`normsorder`) | 498 | importance (7): 1,046 · importance + norms (8): 1,072 | 2,616 |

Pooled control 2,100, matching the paper's Figure 3. `condition` (1–4) is the Qualtrics assignment slot and
maps onto `treatment` within each wave; `treatment` is the analysis variable. Arms were randomised only
within their wave: every comparison is arm versus **that wave's** control.

## Items

`item_num` 1–10 are false (`real` = 0), 11–20 true (`real` = 1); every participant rated all 20. Each wave's
questionnaire loops a block over rows 30–49 whose field is an image URL on `yourfeed.social`: rows 30–39 are
`wilma1` … `wilma10`, rows 40–49 `betty1` … `betty10`, so `item_num` = loop row − 29 and wilma = false,
betty = true. Those URLs are no longer served (2026-09-22). `stimulii.zip` holds the same twenty cards as
`false1` … `false10` and `true1` … `true10`; the card text (source domain + headline as shown, two false
headlines truncated with an ellipsis on the card itself) was transcribed by viewing each image and is stored
locally in `data/epstein/headlines.json` with this provenance. **The one unverifiable step** is that the
posted `false_k` / `true_k` is the same card as `wilma_k` / `betty_k`. It does not affect any pooled
(true / false) quantity, since every participant rated all items of both kinds; it could only permute items
within a veracity group in item-level analyses, which are secondary. The paper's Table 2 is an image in the
article and was not used to corroborate the order (the fetch tool's summary of it was invented and discarded).

## Overlap with Arechar (stimulus text only, before any outcome is opened)

None of the 20 card texts matches any of Arechar's 45 headlines: best normalised similarity 0.57 (threshold
for a near-duplicate 0.6), no exact match; one pair shares three long words but states different claims. So
the D1 test is on **new content, new participants, new intervention variants** relative to the development
data.

## Pretreatment covariates for personas

Only fields fixed before the treatment screen are eligible (research plan v3 §5): `age`, `gender`, `white`,
`college`, `demrep_c` (the fields the authors used as moderators) plus the Lucid panel demographics
(`education`, `hhi`, `region`, `ethnicity`, `hispanic`). Their codings are recorded in the loader once
inspected. Post-treatment measures (`accimp`, `crtacc`, `attention`, `covid_news`, `covid_concern_1`,
`sharing`, `socialmedia`) are never given to a model.

## What participants read (from the survey flow of the questionnaires, 2026-09-22)

Read from the `FL` flow element and the block list of `wave3survey.qsf` and `wave5survey.qsf` (waves 2 and 4
share the same components). In order: consent → social-media-use questions → **the treatment screen (if
any)** → instructions ("You will be presented with a series of news headlines about the Coronavirus
(COVID-19). We are interested in whether you would consider sharing the information on social media (such as
Facebook or Twitter).") → the 20 cards, each with **"Would you consider sharing this story online (for
example, through Facebook or Twitter)?"** and the options No / Yes (order randomised) → attention screeners,
COVID concern and news-checking items, CRT, media items, demographics, debrief. The accuracy-only arm sees
"We are interested in whether you think the information accurate" and, per card, "To the best of your
knowledge, is the claim in the above headline accurate?" (No / Yes).

Treatment screens:

| Arm | What the questionnaire holds | Text available verbatim? |
|---|---|---|
| evaluation | block `acc_inst`: "First, we would like to pretest an actual news headline for future studies. We are interested in whether people think it is accurate or not. We only need you to give your opinion about the accuracy of a single headline. We will then continue on to the primary task." Then one of four **neutral true headlines shown as an image** (`rn1`; "R2 seinfeld coming to netflix"; "R3 social media use may harm teens mental health"; "R4 youtube channel accused of deceiving kids into watching sponsored content" are the questionnaire's descriptors), the question "To the best of your knowledge, is the above headline accurate?" (No / Yes) and **feedback**: "Correct! That was a true news headline." / "Incorrect! That was a true news headline." | instructions and feedback yes; the neutral headline only as a descriptor |
| long_evaluation | the same with all four neutral headlines (wave 2) [I, from the block layout] | as above |
| tips | block `literacy`: one image (`Literacy2`) | **no** — text is inside the image; the paper's Figure 2 shows it |
| generic_norms | block `social_nudge`: one image ("Social nudge turk") | **no** — the paper's Figure 2 caption gives the sentence |
| partisan_norms, tips_norms | wave 4 blocks of the same kind | **no** |
| importance | block `imp_acc`: "How important is it to you that you only share news articles on social media (such as Facebook and Twitter) if they are accurate?" — Not at all / Slightly / Moderately / Very / Extremely important | yes |
| importance_norms | the importance item plus the norms image, in either order | partly |
| control | no treatment screen | — |

Two differences from Arechar's prompt condition matter for the model's state: participants here were told
whether their accuracy judgement of the neutral headline was right, and the sharing answer is binary. The
tips and norms screens exist only as images; their wording has to be transcribed from the article's Figure 2
(an image in the CC BY 4.0 article) before those arms can be simulated verbatim — until then they are
described, not quoted, and the record must say which.

## Article files (downloaded 2026-09-22 with approval; CC BY 4.0)

| File | Bytes | SHA-256 |
|---|---|---|
| `article/f2-1024x759.jpg` (Figure 2: evaluation / importance / tips / partisan norms screens) | 62,789 | `6c61027f…9c9c` |
| `article/Fig.2-rev-1024x619.png` (Table 2 as an image: the twenty headlines) | 96,117 | `27b90921…9141` |
| `article/epstein_toolkit_covid_19_misinformation_20210518.pdf` (12 pages) | 419,606 | `41eb31cf…4887` |

**Item order corroborated.** Table 2 lists the headlines in exactly the order `false1` … `false10`, `true1` …
`true10` of the posted images, so `item_num` 1–20 follows that order; the residual doubt noted above is
resolved to the extent a published table can resolve it. Two card texts differ from the table by a letter or
a word ("Robertson" / "Roberston"; "past 2 years" / "last 2 years"); the card text is used.

**Treatment screens, verbatim where the material is text** (`configs/eval/epstein_screens.yaml`): tips ("Think
carefully about the news with these tips" — "Be skeptical of headlines. Investigate the source. Watch for
unusual formatting. Check the evidence."), partisan norms ("In an earlier study, we found that more than 8 out
of 10 individuals think that it is "very important" or "extremely important" to *only* share news articles on
social media if they are accurate. This was true for *both* Democrats and Republicans."), generic norms (the
caption's sentence: "Did you know, over 80% of past survey respondents say it's important to think about
accuracy before sharing news on social media?"), importance (the questionnaire item), evaluation (the pretest
instruction from the questionnaire, the Figure 2 sample headline from ABCNEWS.GO.COM about the most massive
neutron star, the accuracy question, and the feedback sentence). Tips+Norms is partisan norms then tips;
Importance+Norms is importance and partisan norms in random order (the model gets importance first). **Long
Evaluation is the one arm that is described rather than quoted**: its eight non-COVID headlines with feedback
are images that are not in the public files.

**Seen while transcribing (recorded so the freeze is honest):** the article's Figure 3 gives each treatment's
pooled effect (percent change in discernment relative to control, with intervals) and its Figure 1 the wave-1
control sharing rates (true 54.1%, false 48.3%). These are published, cross-wave-adjusted aggregates; the
per-wave arm outcomes that D1 scores have not been opened. Nothing in the D1 pipeline is chosen by a person
after this point except the wording above, which is copied, and the representation rule, which is fixed on
Arechar.
