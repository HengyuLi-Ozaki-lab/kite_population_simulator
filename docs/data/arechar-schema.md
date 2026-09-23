# Arechar et al. 2023 — data schema, verified

Source: Arechar, A. A., Allen, J., Berinsky, A. J., et al. (2023). Understanding and combatting misinformation
across 16 countries on six continents. *Nature Human Behaviour* 7, 1502–1513. OSF project `g65qu`.
Downloaded 2026-09-21 with the user's approval into `data/arechar/` (gitignored). The OSF project states no
licence: the data are analysed here and not redistributed. The split (`configs/eval/arechar_split.yaml`) was
committed before the download.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `CR.csv` | 10,388,608 | `acc45a28efc71155ca472eb3a78bdf36ccf4239484dbcb3a7bdf4ff18872b83b` |
| `Codebook.pdf` | 82,642 | `3c3035722035ac169bdd8ac5e4a1982a774c62bccb0c2b06ab79db45999d365f` |
| `CRPNe.qsf` | 124,554 | `80e3b0c487802bc13cc46cabca322c662579b4b4f7613f443208948dd9863d8a` |
| `CREG.qsf` | 277,781 | `d35159ff8b53425b69185102cc4c471df9d34422de27dd3ba872c0e569b22ea7` |
| `CRNG.qsf` | 122,535 | `17cec2b245f0dc9170262c1832a4af4d5c146621b234c5765674afb6d6d1e861` |
| `CRUS.qsf` | 136,529 | `6f6cf8be7f871f498b5d8a31a082ae9ee649243b30ace38d39e9ca4997811eb3` |
| `CRAR.qsf` | 137,131 | `13e02adabebddaeeab9fa74c652ec1820ed79e75eda5db934017b2286ca814f9` |
| `CPUK.qsf` | 136,413 | `0ebe27b439c11e5e45938cf182a6f2186c9c2cf5c810957942eb7ec45cf6342f` |
| `CRSA.qsf` | 277,735 | `7f95257270434210a604beb3aa2ed765da06c390946dc0cf1add3f5958060655` |
| `CRCN.qsf` | 160,652 | `f02cd6a0f38360202ad59f287de75f50b87b5badab5b28fe06d65b4b5d8f0483` |
| `CRPNt.qsf` | 126,456 | `4bb45f36691e1167b28467ee1285750e183521fb03655834009b89aedebf4e82` |
| `CRAU.qsf` | 136,452 | `48f76520aeefaa892a7766e789e533636bb3d16caedf21dbf7f257f1ff0c0aa8` |
| `CRIT.qsf` | 144,357 | `37ffc5008efd4c23e3c5b0768d6243cb521cb414b6e2bf9f84fedc984a90b336` |
| `CRMX.qsf` | 136,810 | `87b76473ea82c0fb3be4faaee2c5619c3c2bfce3abe9fd5141dd02006a7922cb` |
| `CRES.qsf` | 142,696 | `f0c9848d3f50831d7803b530e3533cbbd72908dd4692c746b8609af8113c0d05` |
| `CPINh.qsf` | 206,091 | `f9529623325e6600424266f9c79842ca51e2dd0742f13a390ed44f3f80eb9980` |
| `CPBR.qsf` | 142,782 | `7f125a5e365e284a2fe5f2db2d7ab4c96f3dded26af53ecfdc2f767ef8a7d001` |
| `CRZA.qsf` | 122,347 | `de7f3d40a3303e297f214bff6e44af9cbba2c0acf5fb6d6a86c68103dc935a65` |
| `CPINe.qsf` | 124,689 | `0b7ea2efaf84743c10ad78e9b9116249386bb2fa7a1e96e4b132e57ed5d6282a` |
| `CRRU.qsf` | 298,808 | `8e9576986ce22e997c18b540db7c112e6e80dfb429f22558eed99a35375a5ea9` |

Every size matched the OSF listing. Tools: `windtunnel.eval.arechar.load_headlines`, `load_ratings`.

## Verified from the US questionnaire (`CRUS.qsf`)

- **Headlines are plain text.** The block `Headline` is a Loop & Merge of 45 rows; field 1 is the headline
  (5–26 words, median 11), with no image tag or URL in any row. Participants saw the text only.
- **Loop row k is `rating_k`.** Field 2 of every row equals the row number, and field 3 is the embedded-data
  flag `h<k>` that the randomisers set; the loader refuses a questionnaire where this does not hold.
- **Which items a person saw.** Two block randomisers each set ten flags: ten of `h1`–`h30` and ten of
  `h31`–`h45`, evenly presented. The loop shows a row only if its flag is set, in random order.
- **Which items are true.** The debrief question (`debrief`, QID1822) lists exactly rows 31–45 as the true
  headlines and says every other headline was false. (The audit had inferred this from the analysis code; it
  is now read from the questionnaire.)
- **Conditions.** A randomiser assigns `Condition` 1–4 evenly. The sharing question (`headlineS`) is shown
  when Condition ≠ 3, the accuracy question (`headlineA`) when Condition = 3; both are six-point scales
  (extremely unlikely … extremely likely; extremely inaccurate … extremely accurate).
- **Survey order.** Consent → social media (platforms used, kinds of news shared) → screener → *the prompt
  (Condition 2) or the tips (Condition 4)* → instructions → **headlines** → CRT → screener → demographics and
  attitudes (importance items, vaccine, need for cognition, risk, trust, education, income, WVS, …) →
  vignette → comments (party cues, urban/rural, ethnic minority, treatment ratings). Age and sex come from the
  panel at entry. This order is what the persona rule in the split's addendum rests on.

## Questionnaire language by country (compared with the US headline text)

| Questionnaire | Identical to the US text | Script | Group |
|---|---|---|---|
| CRUS, CPINe, CRPNe | 45 / 45 | Latin | English |
| CPUK, CRAU | 44 / 45 (fetus → foetus) | Latin | English |
| CRZA | 44 / 45 (one spelling) | Latin | English |
| CRNG | 31 / 45 (14 reworded in plainer English, e.g. "corpse" → "dead body", "Hoaxes" → "Rumors") | Latin | English |
| CPINh | 0 / 45 | Devanagari | Hindi |
| CRPNt | 0 / 45 | Latin | Tagalog |
| CPBR | 0 / 45 | Latin | Portuguese |
| CRAR, CRES, CRMX | 0 / 45 | Latin | Spanish |
| CRIT | 0 / 45 | Latin | Italian |
| CRCN | 0 / 45 | CJK | Chinese |
| CREG, CRSA | 0 / 45 | Arabic | Arabic |
| CRRU | 0 / 45 | Cyrillic | Russian |

The codebook has no variable recording whether an Indian or Filipino respondent took the English or the
local-language version, so those two countries are "mixed" for the language moderator.

## The ratings file (`CR.csv`)

One row per respondent, 101 columns, 54,757 rows (every respondent who started, as the codebook's id range says;
many have no ratings). Country codes are lower case and two are not ISO: `pn` is the Philippines (the
questionnaire is CRPN) and `uk` the United Kingdom; the loader upper-cases them. Rows per country: pn 4,585,
ng 4,382, sa 4,343, in 4,269, eg 4,008, za 3,478, us 3,385, mx 3,215, ar 3,121, au 2,984, br 2,908, ru 2,906,
uk 2,893, cn 2,856, it 2,752, es 2,672. Some names are capitalised where the codebook writes them in lower case
(`Country`, `Condition`, `pass_Screener1`, `pass_Screener2`, `pass_Attention1`, `CRT`, `CRTi`, `Minority`); the
loader matches names without regard to case. `rating_1` … `rating_45` hold the six-point answer to the sharing
or the accuracy question, depending on the condition; 20 are filled per person.

Only the header has been read so far. Until `configs/eval/arechar_criteria.yaml` is committed, the loader
reads US respondents and odd-numbered items only.
