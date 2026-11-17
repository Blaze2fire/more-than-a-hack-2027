# Current state

Read this first in a new session, then `DATA_CONTRACT.md`. Details are in `RESEARCH_LOG.md`, `ELVERN_EXPERIMENTS_SUMMARY.md`.
Last updated: 2026-11-17. The internal deadline review controls newer
methodology and source corrections; historical scores below are retained for continuity.

## Final submission (2026-11-17 night)

- **Shipped artifact:** `solution/configs/pipeline_final.pkl` (+ `.provenance.json`), assembled by
  `python -m solution.build_final` from an `ens3` + P3 + ball-free-fallback GBDT (no boundary, no
  burst/stoppage augmentation, no Metrica) and a 3-seed 25 Hz `w128` OwnNet
  (`solution/out/final_nn2_w128_25hz_sc16_3seed.pkl`: 7 DFL + Groningen + SC0–15, `w_sc=0.25`, 2800 steps).
  Blend `nn0.7_ownership` (no switch signal), decoder `restart_order`
  (`solution/configs/decode_final_restart_order.json`). Verified `dead_leak == 0.0` and `swap_diff == 0.0`.
- GBDT component: DFL4 ens3+P3 retrained with published-rate degradations only (`solution/configs/benchmark/gbdt_dfl4_p3_ens3_clean/`), plus the line snap (2026-11-18 03:34). A full-external
  `train_v4 --mode ens3 --use-p3 --sc-limit 20` refit ran out of memory twice in OOF fold 2 (RSS 15.3 GB of 23 GB), so it is not shipped.
- Next lever (not done): use the observed ball only to *snap* dead-ball onset to a line crossing near a player-detected stoppage.
  With the true dead mask, held-out transitions within 0.5 s rise from 0.50 to 0.82 and SpellF1 rises by about 0.08 (`solution/out/ablate_nn2_oracle.json`).
- `evaluation/results/more_than_a_hack_2027.csv`: validated OK, sanity PASS/PASS, regeneration byte-identical.
  Technical document: [more_than_a_hack_2027.md](more_than_a_hack_2027.md). `pipeline_v4.pkl` untouched.
- Do not run inference while GPU training is active: XGBoost CUDA prediction fails when it runs out of GPU memory.

## November 17 audit and implementation

- Incoming E015/G0/NN-v2 work was further ahead than the earlier summary. The
  historical master-cache ablation has higher-order stacker leakage and cannot
  substantiate strict nested-LOMO/provider-zero-shot claims. It is now explicitly
  exploratory. Do not promote its blend/tuner results as independent scores.
- Priorities 1–4 have implementation paths and guardrail tests. New bounded
  comparisons fit current classifiers/imputers/stackers on DFL first4 only and
  evaluate the other DFL3, Groningen and Metrica. Inherited decoder development
  and reused-fixture history are disclosed. See the internal experiment record.
- `train_v4.py` is a corrected, fixture-group OOF training recipe. It does **not**
  reproduce the legacy E014 artifact byte-for-byte. Preserve that artifact and
  save new runs to a separate path. SC contributes ownership-only Stage-2 data;
  `--no-metrica` removes Metrica from every fit, including imputers.
- `run.py` uses the saved artifact's embedded decoder unless `--decode-cfg` is
  explicit; it prints validation/hash only, fails on missing artifacts, and
  never silently selects an older method.

## Where we are
- Branch `solution`. Legacy E014 `PipelineV4` (`ball_tracker` + `features_v4` + Stage-1 `ens3` + Stage-2 `TemporalStacker` `ens3` + 4-state restart-aware `viterbi4`) retained as the frozen baseline.
- An external audit (2026-11-10) invalidated E001–E005. Read `AUDIT_RESPONSE.md`. Only E006+ numbers may be quoted, and only held-out/zero-shot external numbers.
- **Historical recorded E014 benchmarks (new recipe requires fresh evaluation):**
  - **Metrica v3 3-fold LOMO (held-out):** `0.5993` objective (`0.6149` merged, `0.5837` raw) vs `0.5509` (E008) vs `0.4316` (E007) vs `0.2498` (E006 ±50 baseline). `macroF1=0.8707`, `aliveDeadF1=0.9281`, `teamAcc=0.9325`, `poss_ratio=0.8353`. Zero-shot (`DFL+8SC->Met`): `0.4463` (`0.4569` merged, `0.4357` raw).
  - **Sportec DFL v1 7-fold LOMO:** `natural` `0.6429` raw / `0.6750` merged (`0.6590` obj; `J03WQQ+WR9=0.6692`); `atd_real` `0.5767` raw / `0.6151` merged (`0.5959` obj; `J03WQQ+WR9=0.5883`; `teamAcc=0.8984`, `aliveDeadF1=0.9539`).
  - **Groningen zero-shot (`DFL+8SC->Gron`):** `natural` `0.6278` raw / `0.6460` merged (`teamAcc=0.9427`); `atd_real` `0.5594` raw / `0.5878` merged (`teamAcc=0.9102`).
- **Legacy frozen artifact:** `solution/configs/pipeline_v4.pkl` (31 external matches: 7 DFL + Groningen + 20 SkillCorner + 3 Metrica, `seed=0`, `--no-tags`). Its embedded decoder belongs to that artifact. The edited `viterbi4.json` sidecar is an experimental override, not automatically promoted.
- **Submission file:** `evaluation/results/more_than_a_hack_2027.csv`, generated from `pipeline_final.pkl`; see "Final submission" above. (The legacy E014 file is kept as `more_than_a_hack_2027_e014_bak.csv`.)

## Standing rules (from the audit)
- No ball-based cue for dead vs alive: verified `dead_leak = 0.0` (`atd_blind` vs `atd_carried_dead`) in unit tests and on reloaded `PipelineV4`.
- Equivariant H/A scoring: verified `swap_diff = 0.0` in `test_core.py` and on reloaded `PipelineV4`.
- Smart Tagging stays OFF (`--no-tags` default): no external game has matching tags.
- Never print a public-ground-truth score. `python -m solution.sanity <csv>` gives PASS/FAIL only.
- Do not compute unpublished statistics of the competition match.

## How to run things
| What | Command |
|---|---|
| Install solution environment | `pip install -r solution/requirements.txt` (evaluation keeps root `requirements.txt` minimal) |
| Tests | `.venv/bin/python -m solution.tests.test_core` |
| Train a separate corrected candidate | `.venv/bin/python -m solution.train_v4 --cpu --no-metrica --out solution/out/pipeline_v4_v2.pkl` |
| Write final submission (PipelineV4 default) | `.venv/bin/python -m solution.run --learned --name more_than_a_hack_2027` |
| Format check | `.venv/bin/python evaluation/validate_submission.py evaluation/results/more_than_a_hack_2027.csv` |
| Alignment & orientation check | `.venv/bin/python -m solution.sanity evaluation/results/more_than_a_hack_2027.csv` |
| Video + public GT viewer (no predictions, by rule; local only) | `.venv/bin/python -m solution.gt_video_viewer` → open `solution/out/gt_video_viewer.html` |
| Benchmark rule-based / E008 legacy | `.venv/bin/python -m solution.bench_learned --exp <id>` |

## Notes & open items
- See `DATA_CONTRACT.md` for verified local `external/` & `cache/` (`31` `.npz` files) inventory, migrated Elvern `ORGANIZER-CLARIFICATION.md` + `research/` docs (`15` files), and section 5 organizer questions.
- See `ELVERN_EXPERIMENTS_SUMMARY.md` for teammate Elvern's approaches `01`–`08` (incl. `04.46`–`04.51`, `05.23 ConvBiGRU`, `08.05`, `08.17` long-gap) and `TEAM_AND_PARTICIPANT_SYNTHESIS.md` for full impartial head-to-head evals and Nov 16 participant review.
- On `atd_real` DFL LOMO (`0.5767` raw / `0.6151` merged), short deflections (`<= 35f`) and ~0.58s burst gaps remain the gap to `oracle_dead` (`+0.029` full-31) vs `oracle_team` (`+0.303` full-31; `team_acc` drops `0.9206` visible -> `0.8530` burst on full-31 `atd_real`).
- **Pre-deadline priorities & open code items:** (1) add equivariant `PlayerSetTCN`/`ConvBiGRU` + `5–30s` long-gap ownership stream (`08.04`/`08.17`) alongside `PipelineV4`'s `dead_leak=0` dead head (`J03WPY` held-out `08.17` `0.8437` vs `PipelineV4_full31` in-sample `0.7657`); (2) fix `configs/viterbi4.json` inverted restart penalty (`c_res_same=16.0 > c_res_flip=15.1642`) & aggressive blip suppression (`min_dead=59, min_live_blip=26, min_turn=10`), and test dynamic changepoint switch discounting; (3) add `04.48` centroid-axis motion (`+0.0279`) + `04.46` `2–12s` ball-free fallback; (4) address `train_v4.py` Stage-2 SkillCorner exclusion (`if sp[2]`), `external.degrade(..., seed=0)`, and `--no-metrica` license option.
