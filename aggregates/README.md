# Aggregate results

Every file here is an aggregate written by a script in `scripts/`: condition means, effects, correlations,
intervals, costs and timings. None holds a respondent-level row, a survey item's text or a model response.

| Folder | Files | Written by | Record | Paper |
|---|---|---|---|---|
| `epstein_d1/` | `report.json` (endpoints, human tables), `policies.json` (locked kernel/hybrid/flagship tables, per-cell corrections, input hashes), `robustness.json` (bootstrap intervals, calibrated parents, sampling floor, persona-count curve), `headlines.json` (headline-level heterogeneity), `human_effect_intervals.json` (per-effect human SEs and intervals) | `d1_analysis.py`, `d1_robustness.py`, `d1_headlines.py` | `docs/decisions/D1.md` | §7.1, Fig. 2A–B, Fig. 4, App. B |
| `socsci210_d3/` | `test_report.json` (frozen primary and secondaries), `dev_report.json` | `b_paired_anchors.py` | `docs/decisions/D3.md` | §7.2, Fig. 2C, App. C |
| `discrepancy_d2/` | `discrepancy_model.json` (β, τ, σ, split-half and test coverage, Monte Carlo versus discrepancy) | `d2_discrepancy_model.py` | `docs/decisions/D2.md` | §8, Fig. 3, App. D |
| `scale_s1/` | `execute.json` (throughput, CRN, exact effects), `live_throughput.json`, `live_ledger.json` | `c_scale_demo.py` | `docs/decisions/S1-scale.md` | §9, Fig. 5 |
| `three_models_a3b/` | `three_models.json` (Jev, GPT-5.6 Luna, GPT-6 Astra on the same test items) | `a3b_three_models.py` | `docs/decisions/A3.md` | §9 |
| `arechar_c2_c3b/` | `c2_calibration.json`, `c2_held_out.json` (frozen gates by part and country), `c3b_report*.json`, `c3b_paired.json` (flagship versus kernel) | `c2_arechar.py`, `c3_llm_arechar.py`, `c3b_flagship_paired.py` | `docs/decisions/C2.md` | §6 |
| `prototype_d0/` | `prototype.json` (correction-cell prototype on Arechar) | `d0_correction_prototype.py` | `docs/decisions/D0.md` | §3.3 |
| `publication_a3c/` | `analysis.json` (recall-versus-inference moderator test) | `a3c_publication_moderator.py` | `docs/decisions/A3c.md` | §6, §11, App. A |
| `memory_b2/` | `per_study.csv` (memory-tier replication) | `b2_replication.py` | `docs/decisions/B2.md` | §5 |
| `socsci210_runs/<run>/` | `metrics.json`, `decision_value.json` for the kernel, Luna, Astra and the two model-free baselines on the SocSci210 test split | `kite eval`, `a1_decision_value.py` | `docs/decisions/G1.md`, `G-A.md` | §5, §9 |

Paths inside the files (for example `results/d1/...`) are the run directories on the machine that produced them;
the files were copied here unchanged.
