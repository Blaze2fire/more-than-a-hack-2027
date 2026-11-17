# Reproduce, evaluate and retrain

How to rerun our submission, check our held-out numbers and train your own variant. Branch `solution`.

## 0. Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r solution/requirements.txt   # torch 2.15 + CUDA, xgboost 3.2, lightgbm 4.7, catboost 1.2.10
```

Data is never committed. Put it here:

| Path | What | Source |
|---|---|---|
| `data/` | competition bundle (BAR-ATH ATD etc.) | organiser download |
| `external/sportec-dfl-v1/` | 7 DFL matches: `DFL_02_01_matchinformation`, `DFL_04_03_positions_raw_observed` XML | Figshare 28196177 v1, CC BY 4.0 |
| `external/wcsf-groningen/` | Groningen match zips | `MatKempeGroningen/WCSF_SolutionRoom` |
| `external/skillcorner-current/data/matches/` | 20 A-League matches | `SkillCorner/opendata` @ `4340d274` |
| `external/metrica/` | Metrica sample games 1–3 | `metrica-sports/sample-data` |

Loaders build `cache/*.npz` on first use. Sources, licences, versions and amounts are in the dataset register,
`docs/DATA_CONTRACT.md` §1–2. The root `EXTERNAL_TRAINING_DATA.md` is a teammate's inventory, with another folder layout.

**Rules we hold ourselves to** (`docs/DATA_CONTRACT.md`, `ORGANIZER-CLARIFICATION.md`): train, tune and select on
external data only. Never print rows of `data/`, `cache/BAR-ATH*`, `evaluation/gt/` or prediction CSVs.
The public 5-minute sample is used only as the PASS/FAIL check in `solution.sanity`, never as a score.

## 1. Regenerate the submission (about 1 min on GPU)

```bash
.venv/bin/python -m solution.run --learned --pipeline solution/configs/pipeline_final.pkl --name more_than_a_hack_2027
.venv/bin/python evaluation/validate_submission.py evaluation/results/more_than_a_hack_2027.csv
.venv/bin/python -m solution.sanity evaluation/results/more_than_a_hack_2027.csv
```

The SHA-256 must equal `submission_csv.sha256` in `solution/configs/pipeline_final.pkl.provenance.json`.
Don't run inference while GPU training is going on: XGBoost CUDA prediction fails when it runs out of GPU memory.

## 2. Reproduce our held-out benchmark (about 45 min)

Strict protocol: every fitted part (classifiers, imputer, stacker, NN) sees only the first four DFL
matches (`J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`). Scoring is on DFL `J03WPY`, `J03WQQ` and `J03WR9`,
Groningen, and Metrica 1–3, each under 6 scenarios. The two committed DFL4 components are in
`solution/configs/benchmark/`.

```bash
.venv/bin/python -m solution.deadline_benchmark evaluate \
  --pipeline solution/configs/benchmark/gbdt_dfl4_p3_ens3/pipeline.pkl \
  --neural   solution/configs/benchmark/nn2_dfl4_w128_25hz/models_4_5_6_g.pkl \
  --scenarios natural,atd_real,atd_blind,atd_carried_dead,gap-stress-burst-01,moving_dead \
  --out-dir solution/out/my_eval --save-predictions        # add --skip-saved to resume after a crash
.venv/bin/python -m solution.snap_eval --pred-dir solution/out/my_eval --out solution/out/my_eval/snap.json
```

`snap_eval` prints provider-balanced SpellF1 for the shipped decoder (`restart_order`), with and
without the line snap, plus a disclosed grid of gentler decoders. Add
`--neural solution/configs/benchmark/nn2_dfl4_w128_25hz_sc16_2seed/models_4_5_6_g.pkl` to re-blend the
saved GBDT output with a different held-out-safe NN in a few minutes. Expected `restart_order`:
DFL 0.6678, Groningen 0.6625, Metrica 1 0.6779.

**To test your own model the same way**, write per-case `lp` (log P[H, A, D], shape n×3) for the same
regenerated cases (`snap_eval.case_match`) and score them with `bench_v4.score_match`. That function
wraps the organisers' `possession_metrics`.

## 3. Retrain

```bash
# NN, 25 Hz OwnNet: shipped recipe, all DFL + Groningen + SC0-15 (about 5 min per seed on an RTX 4050)
.venv/bin/python -m solution.nn2_lomo --tag final_w128_25hz --groups "" --no-metrica --n-sc 16 --w-sc 0.25 \
  --frame-stride 1 --gap-policy mixed --width 128 --gru-layers 2 --steps 2800 --batch 12 --lr 0.002 \
  --seeds 0,1,2 --save-final solution/out/final_nn2_w128_25hz_sc16_3seed.pkl --out-dir solution/out/final_nn2
# Benchmarkable GBDT, DFL4 only (about 40 min GPU); --sc-limit N adds SkillCorner ownership to Stage 2
.venv/bin/python -m solution.deadline_benchmark train --mode ens3 --use-p3 --ball-free --gpu \
  --published-degradation --out solution/out/my_gbdt/pipeline.pkl
# Assemble and verify (refuses to save unless dead_leak == 0 and swap_diff == 0)
.venv/bin/python -m solution.build_final --gbdt <gbdt.pkl> --neural <nn.pkl> \
  --decode-cfg solution/configs/decode_final_restart_order.json --line-snap --out solution/configs/pipeline_final.pkl
```

RAM limit: `train_v4` on all external fixtures (7 DFL + Groningen + 20 SC) needs more than 23 GB.
Use fewer SC matches (`--sc-limit`) on a 24 GB machine.

## 4. Tests

```bash
.venv/bin/python -m unittest discover -s solution/tests -p 'test_*.py'
.venv/bin/python -m solution.tests.test_core
```
