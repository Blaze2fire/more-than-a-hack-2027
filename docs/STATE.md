# Current state

Read this first in a new session, then `DATA_CONTRACT.md`. Details are in `RESEARCH_LOG.md`.
Last updated: 2026-11-10.

## Where we are
- Branch `solution`. Nothing committed yet.
- An external audit (2026-11-10) invalidated E001–E005. Read `AUDIT_RESPONSE.md`. Only E006+
  numbers may be quoted, and only the held-out column.
- **Current best:** E008, two-stage learned model + Viterbi. Held-out 0.551 vs 0.432 rule-based
  (E007) vs 0.250 organisers' baseline. Models `solution/configs/gbm.pkl`, decoder
  `solution/configs/gbm_decode.json`.
- **Candidate file:** `evaluation/results/ours_E008_learned.csv` (valid, sanity PASS/PASS,
  251 possessions, D 48.7%). Fallback: `ours_E007_rules.csv`.
- **Team notebook artifact:** https://claude.ai/artifact/STvW9zFXtn74ia4XrfgH6E
  Source: `docs/pipeline-notebook.html`. After each kept experiment, edit that file and republish
  to the same URL (Artifact tool, `url` = the link above).

## Standing rules (from the audit)
- No ball-based cue for dead vs alive: the external games have no ball during stoppages.
- A component whose parameters cannot be validated on external data is not used (Smart Tagging).
- Never print a public-ground-truth score. `python -m solution.sanity <csv>` gives PASS/FAIL.
- Do not compute unpublished statistics of the competition match.
- Quote held-out numbers only.

## How to run things
| What | Command |
|---|---|
| Tests | `python -m solution.tests.test_core` |
| Tune + benchmark the rule-based model | `python -m solution.bench --exp <id> --method hmm --tune 150 --set decouple=1 --save <json>` |
| Benchmark the learned model | `python -m solution.bench_learned --exp <id> [--final]` |
| Write a submission (learned) | `python -m solution.run --learned --name <name>` |
| Write a submission (rule-based) | `python -m solution.run --params <json> --name <name>` |
| Alignment check | `python -m solution.sanity evaluation/results/<name>.csv` |

## Next steps
1. Team decision: report the tagging episode to the organisers (AUDIT_RESPONSE.md, finding 1).
2. Stability of E008 across degradation seeds and training seeds.
3. Hand-over detection is still the main loss (possessions ×0.74): team-model features, decoder.
4. Technical document (must disclose the audit and the tagging episode), pinned dependencies,
   one reproduction command (download external data, train, predict), clean-clone check.

## Open questions
- Organisers: is unsupervised fitting on the provided tracking allowed? May an AI assistant see
  aggregate outputs? (DATA_CONTRACT.md section 5)
- Team name for the output files.

## Known risks
- Proxy labels on external games are our derivation, not the organisers' definition.
- D share on the competition match (48.7%) is above the external games (32–41%) and cannot be
  checked within the rules.
- Three external games only.
