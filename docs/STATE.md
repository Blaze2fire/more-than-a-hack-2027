# Current state

Read this first in a new session, then `DATA_CONTRACT.md`. Details are in `RESEARCH_LOG.md`.
Last updated: 2026-11-14.

## Where we are
- Branch `solution`. E014 `PipelineV4` (`ball_tracker` + `features_v4` + Stage-1 `ens3` + Stage-2 `TemporalStacker` `ens3` + 4-state restart-aware `viterbi4`) implemented, verified, and trained cleanly.
- An external audit (2026-11-10) invalidated E001–E005. Read `AUDIT_RESPONSE.md`. Only E006+ numbers may be quoted, and only held-out/zero-shot external numbers.
- **Current best (`E014`):**
  - **Metrica v3 3-fold LOMO (held-out):** `0.5993` objective (`0.6149` merged, `0.5837` raw) vs `0.5509` (E008) vs `0.4316` (E007) vs `0.2498` (E006 ±50 baseline). `macroF1=0.8707`, `aliveDeadF1=0.9281`, `teamAcc=0.9325`, `poss_ratio=0.8353`. Zero-shot (`DFL+8SC->Met`): `0.4463` (`0.4569` merged, `0.4357` raw).
  - **Sportec DFL v1 7-fold LOMO:** `natural` `0.6429` raw / `0.6750` merged (`0.6590` obj; `J03WQQ+WR9=0.6692`); `atd_real` `0.5767` raw / `0.6151` merged (`0.5959` obj; `J03WQQ+WR9=0.5883`; `teamAcc=0.8984`, `aliveDeadF1=0.9539`).
  - **Groningen zero-shot (`DFL+8SC->Gron`):** `natural` `0.6278` raw / `0.6460` merged (`teamAcc=0.9427`); `atd_real` `0.5594` raw / `0.5878` merged (`teamAcc=0.9102`).
- **Final trained artifacts:** `solution/configs/pipeline_v4.pkl` (trained deterministically on all 31 confirmed matches: 7 Sportec DFL + WCSF Groningen + 20 SkillCorner Current + 3 Metrica Sample Games, `seed=0`, `--no-tags`) and `solution/configs/viterbi4.json`.
- **Submission file:** `evaluation/results/more_than_a_hack_2027.csv` (validated OK, `solution.sanity` PASS/PASS, `368` possessions, mean `9.0 s`, `D 48.1%` graded / `38.3%` total, home share `0.61`, deterministic duplicate `sha256`: `1f5c7ea4f4fbdc29...`).

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
| Reproduce final PipelineV4 weights | `.venv/bin/python -m solution.train_v4` |
| Write final submission (PipelineV4 default) | `.venv/bin/python -m solution.run --learned --name more_than_a_hack_2027` |
| Format check | `.venv/bin/python evaluation/validate_submission.py evaluation/results/more_than_a_hack_2027.csv` |
| Alignment & orientation check | `.venv/bin/python -m solution.sanity evaluation/results/more_than_a_hack_2027.csv` |
| Benchmark rule-based / E008 legacy | `.venv/bin/python -m solution.bench_learned --exp <id>` |

## Notes & open items
- See `DATA_CONTRACT.md` for full external dataset inventory (Sportec DFL v1, ELASTIC, WCSF Groningen, SkillCorner Current, Metrica 1-3) and section 5 organizer questions.
- On `atd_real` DFL LOMO (`0.5767` raw / `0.6151` merged), short deflections (`<= 35f`) and ~0.58s burst gaps remain the gap to `oracle_dead=0.6465` / `oracle_team=0.8952`.
