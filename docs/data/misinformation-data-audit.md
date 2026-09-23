# Misinformation and megastudy data: availability audit

Audit date: 2026-09-21. Purpose: find public human-subject datasets on which a text-only behavioural
prediction model can be tested for (a) item-level validity, (b) intervention direction, (c) intervention
ranking with K >= 5 arms, (d) individual-level heterogeneity.

## How the audit was done, and how to read the evidence tags

- Only web search and page reading were used. No participant-level data file (CSV, XLSX, ZIP, parquet) was
  opened or downloaded. No outcome data has been seen, so whichever split is later declared "test" is untouched.
- What was opened: repository metadata through public JSON APIs (OSF `api.osf.io/v2`, Harvard Dataverse,
  Hugging Face, GitHub, Europe PMC), article pages, and small documentation files (papers, codebooks,
  read-mes, Stata `.do` scripts, Qualtrics `.qsf` questionnaires; all under 3.3 MB). The fetch tool keeps a
  cached copy of such documents in the Claude session's `tool-results` folder, and a few were converted to
  plain text in the session scratchpad. Nothing was written into the project except this file.
- Evidence tags. **[D]** = I read the primary text, or the raw API JSON, directly. **[S]** = the content came
  back through the fetch tool's summarising model (it reads the page and answers a question). It is reliable for
  structured listings, but it made two errors during this audit that I caught and discarded: it invented an
  intervention list for one paper, and it echoed back a country list that I had put in the question.
  **[I]** = my inference. **UNVERIFIED** = could not be checked.
- Access problems. nature.com article URLs redirect to a cookie-authorisation step that the fetch tool cannot
  complete; the same URL with `?error=cookies_not_supported` appended returned the page. *Nature* and
  *Nature Human Behaviour* articles showed only abstract, figures and availability statements (paywalled);
  *Nature Communications* and *Scientific Data* are open. pnas.org returned HTTP 403, so the PNAS papers were
  read through the Europe PMC full-text XML. `osf.io` project pages are JavaScript shells; all OSF facts below
  come from the OSF API.
- No page contained text addressed to an AI agent or any instruction-like content. (The summariser once
  flagged a README note about Windows compilers as such; on inspection it is an ordinary replication caveat.)
- Quotation policy: statements are paraphrased. One short verbatim quotation is used in the whole report
  (section 8). File names, variable names and condition codes are given as identifiers, not as quotations.
  Headline lists are not reproduced.

**Correction to the brief.** Epstein et al. (2021) was published in *Harvard Kennedy School (HKS)
Misinformation Review*, volume 2, issue 3, not in *Harvard Data Science Review* [D].

## Summary table

Usability codes: Y = usable now, P = partly / with extra work, N = not usable.

| # | Dataset | Repository | Licence / terms | Participant-level data? | Stimuli as text? | Intervention arms | (a) / (b) / (c) / (d) | Main blocker |
|---|---|---|---|---|---|---|---|---|
| 1 | Pennycook et al. 2021, *Nature* | https://osf.io/p6u8k/ | none stated | Yes for Studies 1-6 (seven CSVs, 0.1-2.2 MB). Study 7 (Twitter) not included: codebook and code only | No. Headlines are PNG/JPEG cards; file names hold abbreviated descriptors, not verbatim headlines | S1: 2 (accuracy vs sharing); S3, S4: 2; S5: 4; S6: 2 | P / Y / N / Y | Headline text exists only inside images; humans saw image + lede + source; no licence |
| 2 | Pennycook & Rand 2022, *Nat. Commun.* | https://osf.io/4mv9z/ | none stated (article CC BY) | No. One 26 KB study-level spreadsheet | No headline text and no headline ids | 6 prompt types across 20 experiments | N / P / P / N | Study-level coefficients only; 11 of 20 experiments are unpublished |
| 3 | Epstein et al. 2021, *HKS Misinformation Review* | https://doi.org/10.7910/DVN/18SHLJ (data, code); https://osf.io/hu4k2/ (same plus questionnaires and stimuli) | **CC0 1.0** on Dataverse; none on OSF; article CC BY 4.0 | Yes. One long-format CSV, 299.6 MB | Images in the survey; all 20 headline texts are printed in the paper (Table 2) | 8 treatments + control + accuracy-only, spread over 5 waves | P / Y / **Y** / Y | `item_num` to headline-text mapping is undocumented; arms were not all randomised in one wave |
| 4 | Voelkel et al. 2024, *Science* (Strengthening Democracy Challenge) | https://osf.io/jzbnt/, data in component https://osf.io/2sv7p/ | none stated | Yes. `SDC - Data - Anonymized.csv` 9.2 MB, `SDC - Data - Recoded.csv` 25.4 MB | About 16 of 25 treatments are text-based (full wording in a 374-page questionnaire PDF); 7 are audio/video; 2 are live-interactive | 25 treatments + null control + alternative control | N / Y / **Y** (backup domain) / N | Not misinformation; long treatments; 9 arms cannot be rendered as text; no licence |
| 5 | Milkman et al. 2021 and 2022, *PNAS* (vaccination text messages) | OSF view-only links published in the papers (section 5) | none stated on OSF; articles CC BY-NC-ND | **No.** 2021: only under a signed non-disclosure agreement on a protected server. 2022: not shareable. Aggregated arm-level data are public | Yes, SMS texts (said to be in the SI Appendices, which I could not open) | 19 + control; 22 + control | N / P / P (arm level only) / N | No individual data; outcome is real vaccination, not a survey answer; many arms statistically tied |
| 6 | Arechar et al. 2023, *Nat. Hum. Behav.* | https://osf.io/g65qu/ | none stated | Yes. `CR.csv` 10.4 MB, wide, one row per participant, with codebook | **Yes.** 45 headlines are plain text in the questionnaire, and participants saw text only | 4 conditions = 2 interventions (Prompt, Tips) + sharing control + accuracy baseline | **Y** / **Y** / N / **Y** | Japan is not among the 16 countries; only 2 interventions; no licence |
| 7a | Fazio et al. 2024 preprint, nine-intervention misinformation megastudy | https://osf.io/zvnjb/ (materials), https://osf.io/preprints/psyarxiv/uyjha | none stated | **Not yet.** Only pretest data are posted | UNVERIFIED (materials ZIP is 457 MB) | 9 interventions, N = 33,233 (US) | N now; would be the best (c) dataset | Main data not public; still a preprint as far as I could find |
| 7b | Offer-Westort, Rosenzweig & Athey 2024, *Nat. Hum. Behav.* (Kenya, Nigeria) | https://github.com/gsbDBI/infodemic-replication | none (GitHub licence field is null) | Yes. `data/cleaned-data_2023-03-28.csv` 6.7 MB | UNVERIFIED (posts shown in a Messenger chatbot) | 7 respondent-level + 4 headline-level treatments + controls (40 combinations) | N / Y / P / N | Adaptive (unequal-probability) assignment; 4 + 4 posts per person; no licence |
| 7c | Vlasceanu et al. 2024, *Sci. Adv.*; data descriptor Doell et al. 2024, *Sci. Data* (climate tournament, 63 countries) | https://osf.io/ytf89/ | **CC0 1.0 Universal** | Yes. `data_notimers.csv` 66.4 MB, `data_countries.csv` 167.8 MB | Interventions are reading / writing tasks; 76 per-country questionnaires incl. `japan_1.qsf`, `japan_2.qsf` | 11 interventions + control | N / Y / **Y** (backup domain) / N | Climate, not misinformation; sharing outcome is a single item |
| 8 | SocSci210 (already in use) | HF `socratesft/SocSci210`; GitHub `akaashkolluri/socrates` | **No dataset licence anywhere**; see section 8 | n/a | n/a | n/a | n/a | Licence remains unstated as of 2026-09-21 |

## 1. Pennycook, Epstein, Mosleh, Arechar, Eckles & Rand (2021), *Nature* 592, 590-595

URLs opened: `https://api.osf.io/v2/nodes/p6u8k/` and its `files/osfstorage/...` listings;
`https://www.nature.com/articles/s41586-021-03344-2` (paywalled; availability statements visible);
two open copies of the manuscript: the December 2020 working paper at
`https://ide.mit.edu/sites/default/files/publications/Pennycook%20et%20al%20-%20Shifting%20attention%20to%20accuracy.pdf`
and an accepted-manuscript copy at
`https://static.poder360.com.br/2022/03/pesquisa-compartilhamento-fake-news-estudo-pennycook-universidade-regina.pdf`;
`Study_3_and_4_code.do` from OSF.

What is on OSF [D, raw API JSON]. Project is public, last modified 2021-10-13, no child components,
`node_license` is null. Four top-level folders:

- `/Data and Code/` (15 files): `study_1_data.csv` 2,205,728 bytes; `Study_2_data.csv` 133,839;
  `Study_3_data.csv` 1,080,522; `Study_4_data.csv` 1,214,839; `Study_5_data.csv` 1,456,678;
  `Study_6_b1_data.csv` 512,639; `Study_6_b2_data.csv` 1,277,955; Stata code for Studies 1, 2, 3-4, 5, 6
  and 7; `twitter data codebook.xlsx` 10,788; `agent_based_network_sims_code.zip` 110 MB.
  There is **no Study 7 data file**.
- `/News items/`: Study 1 (36 image files), Study 3 & 6 (24), Study 4 (24), Study 5 (20).
- `/Materials/`: for each of Studies 1, 3, 4, 5, 6 a Qualtrics `.qsf` (0.1-0.8 MB) and a Word export (5-10 MB,
  so almost certainly image-heavy). `/Preregistrations/` was not listed.

Twitter field experiment [D, accepted manuscript; [S] for the Nature page, same content]: data and materials
for Studies 1-6 and code for all studies are at the OSF project; Study 7 data are available only on request,
for privacy reasons.

Studies, N and conditions [D, accepted manuscript]:

| Study | Sample | N | Design | Headlines, scale |
|---|---|---|---|---|
| 1 | MTurk | 1,015 recruited (1,002 in the figure caption) | Accuracy vs Sharing, between subjects | 36 (half false), binary |
| 2 | Lucid | 401 | survey on what matters when sharing; no headline task [I] | n/a |
| 3 | MTurk | 1,254 began; 1,158 full sample; **727** in the preregistered main analysis (those who would share political content) | Control vs Treatment | 24, 6-point sharing likelihood |
| 4 | MTurk | 1,328 began; 1,248 full; **780** main | Control vs Treatment | a different 24, 6-point |
| 5 | Lucid, quota-matched | 1,628 began; 1,287 full; main text reports 1,268 | Control, Active Control, Treatment, Importance Treatment | 20, 6-point |
| 6 | MTurk, two rounds (218 + 542 began) | 711 full (main text says 710); 398 restricted | Control vs Full Attention Treatment (rate accuracy of every headline before the sharing question) | same 24 as Study 3 |
| 7 | Twitter | 5,379 users | direct message asking to rate one non-political headline; stepped-wedge roll-out | quality of subsequently shared links |

For Study 5 the Methods give a restricted sample of 671 (333 + 338) while the main text gives 1,268; I did
not resolve this. The two Study 6 files match the two collection rounds [I].

Intervention wording [D]: the Methods print the exact Treatment instructions (pretest framing, then one
politically neutral headline rated on a 4-point accuracy scale; one of two neutral headlines in Studies 3-4, one
of four in Study 5), the Active Control (same framing, rate how funny the headline is) and the Importance
Treatment (one agree/disagree item about sharing only accurate and unbiased content). All are describable in text.
The Active Control is a ready-made placebo arm for test (b).

Data structure [D, `Study_3_and_4_code.do`]: CSVs are wide, one row per participant (the code creates `id`
from the row number), with `condition`, item columns in `fake*` / `real*` families that the code reshapes to long
(`item_num`, `real`), and `age`, `sex`, `demrep`, `socialmedia_chk`, `fb`, `accimp`. Row counts and exact column
names were not checked (no data opened): UNVERIFIED.

Headline text. The belief that file names contain the headline text is **only partly right** [D]. Names are
short descriptors with a veracity prefix, for example `f15_trump to deport melania.png` (Study 1) and
`f c 5_Clinton divorce.png` (Study 3 & 6; the middle letter is presumably the political lean [I]). Study 4 and 5
names are longer and closer to a headline, some with the outlet appended, but they are lower-cased, abbreviated
and sometimes misspelt. The lede sentence and source that participants saw are not in the file name. Verbatim
text would have to be transcribed from 104 images. Whether the `.qsf` or Word materials carry headline text:
UNVERIFIED (not opened). The mapping from image file to data column is UNVERIFIED.

Licence: none stated on OSF [D].

## 2. Pennycook & Rand (2022), *Nature Communications* 13, 2333

URLs opened: `https://www.nature.com/articles/s41467-022-30073-5` plus `/tables/1` and `/tables/2`;
`https://api.osf.io/v2/nodes/4mv9z/` and its file listing; `meta_code.do`; Europe PMC record (PMC9051116).

- Data availability [S]: study-level data and code are at https://osf.io/4mv9z/. The OSF project describes itself
  as study-level data plus Stata replication code [S]; title "Accuracy prompt meta-analysis", public, modified
  2021-12-16, no components, `node_license` null [D].
- Files [D]: `study level data.xlsx` 26,177 bytes; `meta_code.do` 11,337; `example interventions.docx`
  1,563,832; `PSA-Video.mp4` 9,754,665. Nothing else.
- `meta_code.do` loads the spreadsheet and works with one row per study-by-treatment coefficient and standard
  error (discernment effect, effect on false sharing, baseline sharing, moderator interactions) [S].
  So there is **no pooled participant-level file, no headline ids and no headline text**.
- 20 experiments, 26,863 participants, 2017-2020 [S]. Table 1 names six prompt types: Evaluation, Importance,
  Norms, PSA video, Reason, Tips (plus long / with-feedback variants and combinations) [S]. Table 2 lists the
  experiments [S]: three are Pennycook et al. 2021 Studies 3-5, four are Epstein et al. 2021 waves 2-5, one is
  Pennycook et al. 2020 Study 2, one is Guay et al. 2022, and 11 are marked unpublished.
- Licence: none on OSF [D]; the article is CC BY (Europe PMC metadata) [S].
- Use: the spreadsheet gives benchmark effect sizes by prompt type for (b), and a coarse, study-confounded
  ordering for (c). Raw data exist separately only for the published components (sections 1 and 3 here;
  locations for Pennycook et al. 2020 and Guay et al. 2022 are UNVERIFIED).

## 3. Epstein, Berinsky, Cole, Gully, Pennycook & Rand (2021), *HKS Misinformation Review* 2(3)

URLs opened: article page
`https://misinforeview.hks.harvard.edu/article/developing-an-accuracy-prompt-toolkit-to-reduce-covid-19-misinformation-online/`;
article PDF (read in full, 12 pages) [D];
`https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/18SHLJ` [D];
`https://api.osf.io/v2/nodes/hu4k2/` and file listing [D]; `code.do` [D]; `wave1survey.qsf` [D, parsed].

- Design [D]: Lucid, quota-matched to the US; five waves from 27 April to 21 May 2020; 11,237 began, 9,070
  completed. Every participant saw 20 COVID-19 news cards (headline, image, source), 10 true and 10 false.
  Sharing question is binary (would you consider sharing, no / yes); the Accuracy-Only condition asks a
  binary accuracy question instead.
- Conditions by wave [D, Table 1]: wave 1 (N 403) Control, Accuracy-Only; wave 2 (1,192) Control, Evaluation,
  Long Evaluation; wave 3 (2,081) Control, Evaluation, Tips, Generic Norms; wave 4 (2,778) Control, Tips,
  Partisan Norms, Tips+Norms; wave 5 (2,616) Control, Importance, Importance+Norms. Arm sizes from Figure 3:
  Evaluation 935, Long Evaluation 410, Importance 1,046, Tips 906, Generic Norms 510, Partisan Norms 949,
  Tips+Norms 934, Importance+Norms 1,072, pooled control 2,100.
- **Eight distinct interventions** [D]. Wording: each is described in the text; Figure 2 shows the
  Evaluation, Tips, Importance and Partisan Norms screens, and its caption gives the Generic Norms sentence;
  the full Qualtrics files for all five waves are on OSF. Tips and the two norms screens include a decorative
  illustration; the operative content is text.
- Headline text [D]: **Table 2 of the paper lists all 20 headlines verbatim with veracity.** In the survey the
  items are images: the wave-1 `.qsf` loops over 20 image URLs named `wilma1`-`wilma10` and `betty1`-`betty10`
  (loop rows 30-49). Which name family is true and which is false, and how `item_num` in the data maps to the
  20 texts, is not documented anywhere I looked: UNVERIFIED. `stimulii.zip` (3.7 MB, OSF) holds the images, so
  the mapping is a 20-image manual step.
- Data [D]: Dataverse version 1.0, released 2021-05-08, licence CC0 1.0, no restricted files, no terms of use
  field. Files: `data_to_post.csv` 299,630,356 bytes (file id 4625647), `code.do` 17,700 (id 4625648). OSF `hu4k2`
  holds the same two files plus `wave1survey.qsf` ... `wave5survey.qsf` (0.85-0.92 MB each) and `stimulii.zip`;
  no licence on OSF. The article itself is CC BY 4.0.
- Variables [D, `code.do`]: `id` (made unique as id + wave x 1,000,000), `wave`, `treatment` (0 accuracy-only,
  1 control [I], 2 evaluation, 3 long evaluation, 4 generic norms, 4.5 partisan norms, 5 tips, 6 tips+norms,
  7 importance, 8 importance+norms), `item_num`, `real`, `rating`, `completed`, `age`, `gender`, `white`,
  `college`, `demrep_c`, `attention`, `crtacc`, `accimp`, `covid_news`, `covid_concern_1`. Long format, one row
  per participant-headline. Why the file is 300 MB for about 180,000 participant-item rows is UNVERIFIED
  (probably many carried-over survey columns [I]).
- The paper reports a headline-level analysis (treatment effect against perceived accuracy, r(18) = 0.74), which
  is a published human benchmark for an item-level test.

## 4. Voelkel et al. (2024), *Science* 386, eadh4764

URLs opened: open copy of the article PDF
`https://bpb-us-w2.wpmucdn.com/web.sas.upenn.edu/dist/9/244/files/2024/10/science.adh4764.pdf` (read in full) [D];
`https://api.osf.io/v2/nodes/jzbnt/`, its children, and component `2sv7p` listings [D]; `Read Me.pdf` [D];
`SDC - Data - Intervention Names.csv` [S]; `SDC - Questionnaire.pdf` (converted to text; block headers and
two treatment blocks inspected) [D]; `https://www.strengtheningdemocracychallenge.org/winning-interventions` [S].

- Availability [D]: the article says preregistration, materials, anonymised data and code are on OSF and cites
  DOI 10.17605/OSF.IO/JZBNT. Root project `jzbnt` holds `Supplementary Materials.pdf` (3.6 MB) and four public
  components: Main Survey (`2sv7p`), Durability Survey (`utmde`), Attriter Survey (`g69qd`), Forecasting (`v2c5n`).
  No licence on the root or on Main Survey [D].
- Main Survey files [D]: `/Data/SDC - Data - Anonymized.csv` 9,177,726 bytes; `SDC - Data - Recoded.csv`
  25,374,043; `SDC - Data - Intervention Names.csv`; `SDC - Data - Outcome Names.csv`; two small coding
  spreadsheets. `/Materials/SDC - Questionnaire.pdf` 1.7 MB (374 pages) and `.qsf` 2.0 MB. `/Code/`,
  `/Other Results/`, `/Preregistration/`. The read-me says the posted file is an anonymised version of the raw
  data (exclusions applied, identifying variables removed, demographics merged in) and names the R scripts
  that produce each result; it states no terms of use.
- Design [D]: N = 32,059 US partisans (Bovitz panel, quota-matched); random assignment to a null control, an
  alternative control, or one of 25 treatments; 27 arm labels in the names file [S]. Eight outcomes on 0-100
  scales: partisan animosity, support for undemocratic practices, support for partisan violence (the three
  preregistered ones), support for undemocratic candidates, opposition to bipartisan cooperation, social
  distrust, social distance, biased evaluation of politicised facts. Follow-up about two weeks later, n = 8,644.
  Forecasts of the treatment effects by 98 academics and 51 practitioners are in the Forecasting component:
  a ready human-expert baseline for a ranking test.
- Text-describability, classified by me from Table 1 of the article [D for the table, I for the classification]:
  - Reading, or reading plus a short response (10): Common exhausted majority identity; Common national
    identity; Democratic system justification; Moral similarities and differences; Outpartisans' experiences of
    harm; Outpartisans' willingness to learn; Political violence inefficacy; Pro-democracy inparty elite cues;
    Reducing outpartisan electoral threat; Utility of outparty empathy.
  - Writing or perspective-taking tasks (2): Describing a likable outpartisan; Counterfactual partisan selves.
  - Estimate-then-feedback quizzes, describable in text (4): Correcting democracy misperceptions; Correcting
    opportunism misperceptions; Correcting oppositional misperceptions; Party overlap on policies.
  - Audio or video (7): Befriending meditation (audio); Correcting division misperceptions; Common economic
    interests; Democratic collapse threat; Positive contact video; Pro-democracy bipartisan elite cues;
    Sympathetic personal narratives.
  - Live-interactive (2): Bipartisan joint trivia quiz; Correcting policy misperceptions chatbot.
  So roughly **16 of 25 can be given to a text-only model**, 12 of them as plain passages. The questionnaire PDF
  contains the full wording; I confirmed this for two blocks (Experiences of harm and Violence inefficacy are
  both timed reading tasks). The project website's one-line summary of the harm treatment suggested audio; the
  questionnaire shows that is wrong. Whether reading treatments embed images or charts is UNVERIFIED, and the
  passages are long (the whole questionnaire converts to about 13,000 lines of text).
- For (d): one row per participant with single composite outcomes, so it does not provide many items per
  person. Demographic columns in the anonymised file: UNVERIFIED (not opened).

## 5. Milkman et al. (2021) and Milkman et al. (2022), *PNAS*

URLs opened: Europe PMC full-text XML for PMC8157982 (2021), PMC8833156 (2022) and PMC12582327 (2025
correction) [S]; OSF API for both projects through the view-only links printed in the papers [D]; the 1 KB
data-availability read-me in the 2021 project [D]. pnas.org itself returned HTTP 403, so the SI Appendix PDFs
were **not** opened.

2021, doctor's-appointment megastudy (DOI 10.1073/pnas.2101165118):

- N = 47,306 patients at Penn Medicine and Geisinger; 19 message arms (2,295-2,397 each) plus usual care [S].
- Message texts: the paper says all intervention content is in the SI Appendix [S]; not opened, so UNVERIFIED by
  me. Three image files on OSF (a joke image, two reply-prompt images) show that a few arms included a picture [D].
- Arm-level results: Figure 1 in the paper; numeric tables in the OSF web appendix [S]. OSF project `tucjs`
  (not public; reachable at `https://osf.io/tucjs/?view_only=c491df37a33840abbdedda4e60176f34`) holds
  `Web Appendix.docx` 4.4 MB, `Image Files of Web Appendix Tables.zip` 3.5 MB, and `/Code and Aggregate Data/`
  with `Aggregate Data.csv` 18,461 bytes and analysis folders [D]. No licence.
- Participant-level restriction [D, read-me]: the authors say they have no legal permission to post
  individual-level vaccination data; researchers must contact Behavior Change for Good
  (bcfg@wharton.upenn.edu) and sign a non-disclosure agreement to use the data on a protected medical server.
- The 2025 correction only repairs broken image links in the SI [S].

2022, Walmart pharmacy megastudy (DOI 10.1073/pnas.2115126119):

- N = 689,693; 22 message arms plus a business-as-usual control; outcome is a flu shot at a Walmart pharmacy
  between 25 September and 31 December 2020 [S].
- Message texts: said to be in the SI Appendix [S]; not opened. Table 1 of the main text lists all 22 arms with
  descriptive names, coefficients, standard errors and p-values [S].
- Data [S]: individual-level data cannot be posted; aggregated summary data are on OSF at
  `https://osf.io/rn8tw/?view_only=546ed2d8473f4978b95948a52712a3c5`. That project (not public) holds
  `Code & Data.zip` 102,829 bytes, `James-Stein Shrinkage R Code and Data.zip` 9,940,
  `Attribute Analysis Study Stimuli.pdf` 2.8 MB and a folder of `.qsf` files for the prediction and attribute
  surveys [D]. No licence. Both articles are CC BY-NC-ND.
- Both projects ran prediction surveys (lay people and experts forecasting arm effects), a possible baseline.

Fit to the project [I]: arm-level ranking only. The outcome is field behaviour with small effects and many
statistically tied arms, so a rank correlation would mostly measure noise unless shrinkage estimates and their
uncertainty are used. No test of (a) or (d) is possible.

## 6. Arechar et al. (2023), *Nature Human Behaviour* 7, 1502-1513

URLs opened: `https://www.nature.com/articles/s41562-023-01641-6` (paywalled; availability statement
visible) [S]; `https://api.osf.io/v2/nodes/g65qu/` and listings [D]; `Codebook.pdf` [D];
`CRUS.docx` and `CRUS.qsf` (US questionnaire; parsed) [D]; `230412 CR.do` [D]; PsyArXiv preprint
`https://osf.io/preprints/psyarxiv/a9frz/` version 1, 53 pages, converted to text [D].

- **Countries [D, preprint Methods]: Argentina, Australia, Brazil, China, Egypt, India, Italy, Mexico, Nigeria,
  the Philippines, Russia, Saudi Arabia, Spain, United Kingdom, United States, South Africa. Japan is not
  included.** Nine languages. The OSF materials folder has 18 questionnaires (16 country codes; India and the
  Philippines have two language versions each), none for Japan [D].
- Sample [D]: Lucid, target 2,000 per country with age and sex quotas, 22 February to 25 April 2021; 54,757
  began, 34,286 gave at least one rating (676,605 ratings), 33,480 complete; no country below 1,928 complete.
- Conditions [D, codebook and questionnaire]: codes 1 = Share only, 2 = Prompt, 3 = Accuracy, 4 = Tips.
  Prompt: pretest framing, then rate the accuracy of one of four neutral headlines (text) on the 6-point scale.
  Tips: a screen with four short digital-literacy tips (text, with one decorative image). Both wordings are in
  the questionnaire files as text.
- **Headlines [D, `CRUS.qsf` parsed]: the headline block loops over 45 rows whose field is plain text; no row
  contains an image tag or URL. The question page shows the headline in bold text, nothing else, then the
  question.** So the stimulus the model would receive is the stimulus the participants received. 30 false and 15
  true; each participant saw 10 false and 10 true, one at a time. Rows 31-45 are the true ones in the debrief
  list, and the analysis code sets `real = 1` for `item_num > 30`, so loop row k should be `rating_k` [I, consistent
  on both sides]. Topics are COVID-19 claims (for instance, one false item about RNA vaccines and DNA, one true
  item about a dexamethasone trial).
- Outcomes [D]: accuracy, 6-point from extremely inaccurate to extremely accurate (Accuracy condition); sharing
  likelihood, 6-point from extremely unlikely to extremely likely (other three conditions).
- Data [D]: `/Data/CR.csv` 10,388,608 bytes, `/Data/Codebook.pdf` 82,642. Wide format: `id`, `country`
  (alpha-2), `condition`, screeners and attention checks, `rating_1` ... `rating_45` (20 filled per person), CRT
  scores, five importance-of-content items, vaccine items, need for cognition, risk, trust, `edu`, `ses`, four
  World Values Survey items, religiosity, `minority`, urbanicity, treatment-liking items, `age`, `sex`, `mobile`.
  `/Analysis/` has the Stata script (37,896 bytes) and two R files.
- Licence: none on OSF [D]. The consent text in the questionnaire tells participants that anonymised data may
  be placed in repositories such as OSF for further analysis by other researchers [D].
- Not checked: non-US `.qsf` files (translated text assumed [I]); per-country N by condition; the 2023 Author
  Correction; the final typeset article. An Institute for Replication comment (I4R DP 277) reports a largely
  successful computational reproduction with incomplete plotting code [S].

## 7. Other candidates for (c) (three, as requested)

### 7a. Fazio, Rand, Lewandowsky, Susmann, Berinsky, Guess, Kendeou, Lyons, Miller, Newman, Pennycook, Swire-Thompson: nine-intervention misinformation megastudy (PsyArXiv, June 2024)

URLs opened: `https://api.osf.io/v2/preprints/uyjha/` and `/versions/` [S]; `https://api.osf.io/v2/nodes/zvnjb/`
and listings [D].

- 33,233 US participants; nine interventions (accuracy prompts, digital-literacy tips, inoculation, debunking
  and others) tested in one design on true, false and misleading health and political headlines, for both
  accuracy and sharing judgments [S]. This is exactly the design test (c) needs.
- **Blocker: the preprint's data field says the full data will be posted once the manuscript is submitted for
  publication** [S]; there is a single preprint version, no journal DOI attached, and my searches found no
  published version: status UNVERIFIED beyond that. The linked OSF project "Mercury Project Misinformation
  Intervention Comparison Study" (public, last modified 2024-06-23, no licence) has only
  `Intervention details.docx` 9.4 MB, `Mercury_Project_Main_Study_Survey.qsf` 2.2 MB,
  `Materials Updated.zip` 457 MB, `Pretest 12 (Updated).zip` 28 MB, `Pretest Data fall 2022.xlsx` 0.3 MB and a
  design/analysis folder [D]. Stimulus format UNVERIFIED.
- Action: write to the authors, or re-check OSF `zvnjb` periodically.

### 7b. Offer-Westort, Rosenzweig & Athey (2024), *Nature Human Behaviour* 8, 823-834

URLs opened: `https://www.nature.com/articles/s41562-023-01810-7` [S]; arXiv 2212.13638 v6 (text) [D];
GitHub API for `gsbDBI/infodemic-replication` [D].

- Facebook Messenger chatbot; Kenya n = 7,498, Nigeria n = 7,794. Learning stage: factorial of seven
  respondent-level treatments (accuracy nudge, Facebook tips, AfricaCheck tips, video training, emotion
  suppression, pledge, deliberation nudge) and four headline-level treatments (related articles, fact check, real
  information, more information), each with a control: 40 combinations, assigned adaptively (contextual
  Thompson sampling). Evaluation stage: best arms plus a learned targeting policy [D].
- Outcome: intention to share each post on the timeline and by Messenger; four posts before and four after
  treatment, two true and two false each time [D].
- Data and code: the GitHub repository (statement [S]); `data/cleaned-data_2023-03-28.csv` 6,715,682 bytes plus
  RDS versions; repository licence is null; last push 2025-12-07 [D]. arXiv version is CC BY 4.0.
- Caveats: unequal assignment probabilities need the supplied weights; six of the seven respondent-level arms
  look text-describable and video training does not [I]; stimuli format UNVERIFIED; the population is far from
  the model's validation population.

### 7c. Vlasceanu et al. (2024), *Science Advances* 10, eadj5778, with data descriptor Doell et al. (2024), *Scientific Data*

URLs opened: Europe PMC records and XML (PMC10849597, PMC11445540) [S];
`https://www.nature.com/articles/s41597-024-03865-1` and `/tables/1` [S]; OSF API for `ytf89` [D].

- 59,440 participants (59,508 passing attention checks in the descriptor) in 63 countries; 11 crowd-sourced
  interventions plus a control that reads a literary excerpt [S]. Interventions (descriptor Table 1 [S]): Dynamic
  social norms, Work-together norm, Effective collective action, Psychological distance, System justification,
  Future-self continuity, Negative emotions, Pluralistic ignorance, Letter to future generation, Binding moral
  foundations, Scientific consensus. All are reading tasks; three add a writing task.
- Outcomes: climate belief, policy support, **willingness to share a climate fact on social media**, and a
  tree-planting effort task [S].
- **Japan is included**: `/ClimateManylabs_QSF/` holds 76 questionnaire files, among them `japan_1.qsf`
  (666,014 bytes) and `japan_2.qsf` (665,138 bytes) [D], and the author list has several Japanese institutions
  [S]. Japan sample size and the content of those two files: UNVERIFIED.
- Data [D]: OSF `ytf89`, public, **licence CC0 1.0 Universal** (read from the API). `/ClimateManylabs_Data/` has
  `codebook.xlsx` 56 KB, `cleaned/data_countries.csv` 167,820,138 bytes,
  `cleaned_notimers/data_notimers.csv` 66,355,376 bytes, plus `raw/`. The Science Advances article is CC BY and
  also cites Zenodo record 10345806 and two GitHub repositories (search snippet only, UNVERIFIED).
- Fit: not misinformation, but it is the only audited megastudy with a licence, a sharing outcome, text
  treatments in local languages and Japanese respondents.

Considered and dropped: Spampatti et al. (2024, *Nat. Hum. Behav.*; six text inoculations against climate
disinformation, OSF `m58zx`, CC-BY 4.0 [D]); smaller than the three above and its country list is UNVERIFIED.

## 8. SocSci210 licence check

URLs opened (all [D] unless marked): `https://huggingface.co/api/datasets/socratesft/SocSci210`;
`https://huggingface.co/datasets/socratesft/SocSci210/raw/main/README.md`;
`https://api.github.com/repos/akaashkolluri/socrates`;
`https://raw.githubusercontent.com/akaashkolluri/socrates/main/README.md`;
`https://arxiv.org/abs/2509.05830` and the HTML full text [S]; `https://stanfordhci.github.io/socrates/` [S].

- Hugging Face: no `license:` tag, `cardData` is null, the README has no YAML header and no sentence about
  licence, terms, permission or citation. Not gated. Last modified 2025-09-09. Unchanged from the project's
  note of 2026-09-20.
- GitHub: API `license` field is null, no LICENSE file; last push 2025-09-09. The README does have a
  "License" section. About the data it says only: "All derivative works retain the license of their original
  source." (README, License section). The rest of the section concerns the fine-tuned models (Llama 3 Community
  License; Qwen licence). It names no licence for the dataset.
- Paper (EMNLP 2025; arXiv v2 of 2025-11-05; the paper itself is CC BY 4.0): no dataset licence. The ethics
  statement says materials were handled according to the applicable data and API usage policies and that the
  data come from publicly available TESS experiments [S].
- Project website: rendered only a title through the fetch tool; nothing could be read. UNVERIFIED.
- Consequence [I]: by the README's own rule the governing terms are those of TESS. TESS terms were not opened
  (search snippets mention a one-year embargo followed by public release, and a requested citation):
  UNVERIFIED. Keep the current practice of not redistributing any of it.

## Recommendation

| Test | Use | Why | First retrieval step |
|---|---|---|---|
| (a) item-level validity | **Arechar et al. 2023**, US first, then the other English-language countries | 45 headlines as plain text, shown to humans as plain text; accuracy and sharing versions; about 2,000 people per country | `https://osf.io/download/69tu5/` (`CR.csv`, 10.4 MB), `https://osf.io/download/ght26/` (`Codebook.pdf`), `https://osf.io/download/ru78y/` (`CRUS.qsf`; headline texts are Loop & Merge rows 1-45 of the block named Headline) |
| (a) secondary | Epstein et al. 2021 (20 items, text in the paper's Table 2); Pennycook et al. 2021 Study 1 (36 items) | Both need a manual image-to-text step, and humans saw image and source as well | Epstein: `https://osf.io/download/fybg7/` (`stimulii.zip`, 3.7 MB) to map `item_num` to Table 2 |
| (b) direction | **Arechar** (Prompt and Tips against Share only, 16 countries = 32 direction tests) plus **Pennycook 2021 Studies 3-5** (Treatment, Importance, and the Active Control placebo) plus Epstein (two norms arms that did not work) | Exact wording available as text for every arm; includes arms where the right answer is "no effect" | Pennycook: `https://osf.io/download/sd643/`, `https://osf.io/download/5ta42/`, `https://osf.io/download/nu3qp/` (Study 3, 4, 5 CSVs) and `https://osf.io/download/vdne6/` (`Study_3_and_4_code.do`) |
| (c) ranking | **Epstein et al. 2021** as the in-domain test (K = 8, CC0). Backup: **Voelkel et al. 2024** restricted to the 16 text arms (K = 16, three preregistered outcomes, expert forecasts available) and **Vlasceanu et al. 2024** (K = 11, CC0, Japan) | Fazio et al. would be better but has no public data | Epstein: `https://dataverse.harvard.edu/api/access/datafile/4625647` (`data_to_post.csv`, 299.6 MB) and `.../datafile/4625648` (`code.do`). Voelkel: `https://osf.io/download/ta3gc/`, `https://osf.io/download/mbhe5/` (questionnaire), `https://osf.io/download/uh8vn/`. Vlasceanu: `https://osf.io/download/8q6ue/`, `https://osf.io/download/jpm7a/` |
| (d) heterogeneity | **Arechar** (20 six-point items per person, rich covariates, 16 countries) and **Epstein** (20 binary items per person, N 9,070, CC0); Pennycook Studies 3-5 as a third | Participant id, many items, demographics all present | same files as above |

Points to settle before any outcome data is opened:

1. Human rankings are noisy. In Epstein et al. most of the effective arms are statistically indistinguishable
   from one another, Long Evaluation is larger than most, and the two norms-only arms are null; in the
   vaccination megastudies most arms are tied. Fix the ranking metric and a noise ceiling (for example
   split-half reliability of the human ranking) in advance, in line with the project's calibrate-then-freeze rule.
2. Epstein's arms were randomised within waves, each with its own control. Rank on within-wave contrasts
   against that wave's control, not on raw arm means.
3. Decide now which part is exploration and which is test (for example Arechar US for calibration, the other 15
   countries held out). This audit has not looked at any outcome data.
4. Licences: only Epstein (CC0) and Vlasceanu/Doell (CC0) state one. For Pennycook, Arechar and Voelkel the
   OSF projects are public with no licence; analyse, do not redistribute, cite, and ask the authors if
   anything derived from the raw files is to be published.
5. Japan: no audited misinformation dataset has Japanese respondents. The climate tournament does. A
   Japanese misinformation dataset would need a separate search.
6. Contamination [I]: none of these studies is a TESS study, so they should not overlap SocSci210; the base
   language model may still have seen the papers and headline texts.

## What remained unverified

- Row counts, column names and value codings of every CSV (none opened); structure is inferred from code and
  codebooks.
- Pennycook 2021: verbatim headline text; image-to-column mapping; whether `.qsf` / Word materials carry
  text; the Study 5 sample-size discrepancy; the final typeset *Nature* text (two open manuscript copies read).
- Pennycook & Rand 2022: spreadsheet contents (not opened); repositories for the 2020 and Guay et al. components.
- Epstein 2021: `item_num` to headline mapping and which image family is true; why the CSV is 300 MB.
- Voelkel 2024: images inside reading treatments; demographic columns; the Science supplement was not opened.
- Milkman 2021 / 2022: SI Appendix message texts (pnas.org 403); contents of the aggregate files.
- Arechar 2023: non-US questionnaires; per-country cell sizes; Author Correction; final typeset article.
- Fazio et al.: publication status and whether data exist elsewhere; stimulus format.
- Offer-Westort et al.: stimulus format; file structure.
- Vlasceanu / Doell: Japan N; content of `japan_2.qsf`; Zenodo and GitHub mirrors; intervention table came
  through the summariser.
- SocSci210: TESS terms of use; the project website.
- All [S] items carry summariser risk; the facts that drive the recommendation (file listings, licences,
  Arechar text stimuli and country list, Epstein design and headline table, Voelkel Table 1, SocSci210 licence
  text) were each read directly.
