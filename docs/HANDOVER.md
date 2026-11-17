# Session Handover for Opus 5.5 — More Than a Hack 2027

**Updated:** `2026-11-17 22:03 IST` | **Branch:** `solution` | **HEAD:** `d81bf72` (`PipelineV4 WIP`, dirty tree)

> [!IMPORTANT]
> **Start here — do NOT re-read historical logs or `research/archive/elvern/` unless checking a low-level formula.** All strict external benchmarks, GPU Priority-1/2/3/4 ablations, teammate comparisons (Elvern & Ming `X-UTD`), bug fixes, and the final shipping recipe are verified and indexed below.

---

## 1. TL;DR Current State & Immediate Goal for Opus 5.5

1. **All DFL4-strict Priority 1–4 ablations are complete (`0` background processes running):**
   - **Verified best strict external stack (`DFL4` -> `3` DFL + Groningen + `3` Metrica × `6` scenarios incl. adversarial `moving_dead`):**
     - `ablate_p3_ens3` (`LGBM + XGB_CUDA + CatBoost GPU`, `--use-p3 --ball-free-fallback`, **no `--boundary`**) paired with **`ablate_nn2_w128_25hz`** (`25 Hz`, `w=128`, `2`-layer BiGRU, `nn0.7_ownership/baseline`) reached **`0.6689` provider-balanced raw SpellF1** (**`+0.1160`** over `deadline_base` `gbdt/baseline` `0.5530`, 95% CI `[0.1010, 0.1307]`, worst match `+0.0633`; **`0.6706` merged**, `TeamAcc 0.9405`, `BurstStrict 0.8659`, `<=2s 0.3638`).
     - Predeclared primary (`nn0.7_ownership/frozen_dead`, `scale=0.5`) also reached a new high on `ablate_p3_ens3`: **`0.6631`** (`+0.1101`, CI `[0.0942, 0.1257]`, Groningen `0.6555`, `<=2s 0.3818`).
     - Adding **`sc0..sc15` (`--n-sc 16 --w-sc 0.25`) + 2 seeds @ 2800 steps (`ablate_nn2_w128_25hz_sc16_2seed`)** boosted held-out SC last-4 accuracy from **`0.7791 -> 0.8337` natural / `0.8252` `atd_real`** (`NLL 0.8379 -> 0.3609`), hit **`0.7287` Oracle Burst** & **`0.6526` Oracle `<=2s`**, and lifted non-oracle `natural` (`0.7439`) & `gap-stress-burst-01` (`0.6620`).
     - **All candidates passed `dead_leak == 0.0` and `swap_diff == 0.0`.**
2. **Code edit made cleanly in this session:**
   - Fixed a genuine variable-shadowing bug in [solution/ablate_sc_stage2.py](../solution/ablate_sc_stage2.py#L192-L231) (`pd, ph, ... = base.predict_stage1_for_stacker(m)` -> `p_dead, ph, ...` so `pandas as pd` works at `pd.DataFrame(...)`).
   - All **36/36** repo unit tests pass **after** this edit (`25` `unittest` in `5.580s` + `11` `test_core`).
3. **Remaining work for Opus 5.5:**
   - Optional quick check: pair `ablate_p3_ens3` + `sc16_2seed` (expecting slightly better Groningen/natural), OR verify one `scale=1.0` `frozen_dead` nuance (`+0.003` over `scale=0.5`).
   - **Train final full-external checkpoints (`DFL7 + Groningen + SC`) on RTX 4050 GPU** -> wire & save separate artifact (`solution/configs/pipeline_final.pkl`, preserving `pipeline_v4.pkl`) -> verify invariants -> generate **`evaluation/results/more_than_a_hack_2027.csv`** (note: **missing right now**, only `more_than_a_hack_2027_e014_bak.csv` is there!) -> pass `validate_submission.py` + `solution.sanity` + 36 tests -> clean git & finalize tech doc per [SUBMISSION.md](../SUBMISSION.md).

---

## 2. Environment, Hardware & Guardrails

- **Python & GPU:** `.venv/bin/python` (`3.11.0rc1`, `torch 2.15.0+cu132`, `xgboost 3.2.0`, `lightgbm 4.7.0`). Host **RTX 4050 Laptop GPU** (`6141 MiB` VRAM, `5300 MiB` free) works natively for PyTorch `OwnNet`, XGBoost (`device="cuda"`), and CatBoost GPU (no Codex sandbox restrictions here).
- **RAM / Disk:** `23Gi` RAM (`17Gi` available right now, `0B` swap used), `16` CPU cores, `37G` free disk. Keep `MTAH_THREADS=2..6` and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Disposable space if disk gets low: old `master_feat_cache_sc8/` (`12G` in prior scratch dir) & remove `catboost_info/` before git commit. Keep `cache/` (`831M`, `31` `.npz` files)!
- **Data Guardrails ([DATA_CONTRACT.md](DATA_CONTRACT.md), `test_data_guardrails.py`):**
  - **NEVER** read, inspect, or print rows/coordinates/intervals of `data/`, `cache/BAR-ATH*`, `evaluation/gt/*`, or prediction CSVs.
  - [solution/visualize_5min.py](../solution/visualize_5min.py) is **RETIRED** (do not re-enable). [solution/sanity.py](../solution/sanity.py) prints ONLY alignment/orientation `PASS`/`FAIL` and never scores. [solution/run.py](../solution/run.py) uses `--pipeline`'s embedded `decode_cfg`, keeps `--no-tags` default, and prints only validation + hash.
  - Preserve `dead_leak == 0.0` (player-only dead features), `swap_diff == 0.0` (dyadic `65536.0` quantization), and `c_res_same <= c_res_flip`. Keep PFF excluded. SkillCorner has `has_real_dead_ball=False` (**ownership-only**, never dead head); its last 4 sorted IDs (`2013725, 2015213, 2016236, 2017461`) are reserved for confirmation (`nn2_confirm` / `ablate_sc_stage2`) before final freeze.

---

## 3. Complete Empirical Ablation Conclusions (Strict `DFL4` -> 7 Unseen Eval Matches × 6 Scenarios)

*(Equal provider mean across `DFL3` [`J03WPY, J03WQQ, J03WR9`], `Groningen` [`1`], `Metrica` [`1–3`, raw vs merged kept distinct]; 6 scenarios = `natural`, `atd_real`, `atd_blind`, `atd_carried_dead`, `gap-stress-burst-01`, `moving_dead`. Every run has `dead_leak == 0.0`, `swap_diff == 0.0`.)*

| System / Priority Ablation | Key Config | GBDT `baseline` PB F1 | `nn0.7_ownership` `baseline` PB F1 | `nn0.7_ownership` `frozen_dead` PB F1 | Provider (`nn0.7/baseline`) DFL / Gron / Met | Key Finding & Recommendation |
|---|---|---:|---:|---:|---|---|
| **`deadline_base`** + 5Hz `w64` | `lgbm`, no P3 | `0.5530` | `0.6332` | `0.6230` | `0.6433 / 0.6261 / 0.6301` | Codex CPU baseline (`DEADLINE_BASE_SUMMARY.md`). |
| **`ablate_gbdt_aug`** + 5Hz `w64` | `lgbm`, `--long-bursts --stoppage-augmentation` | `0.5578` (`+0.0048`) | `0.6319` (`-0.0013`) | `0.6234` (`+0.0004`) | `0.6406 / 0.6258 / 0.6295` | **Drop when blending NN:** helps GBDT standalone (`+0.0048`), slightly hurts DFL `nn0.7` (`-0.0027`) (`DEADLINE_RESULTS_BASE_VS_AUG.md`). |
| **`deadline_p3`** + 5Hz `w64` | `lgbm`, `--use-p3 --ball-free --boundary` | `0.5929` (`+0.0399`) | `0.6373` (`+0.0041`) | `0.6279` (`+0.0049`) | `0.6453 / 0.6331 / 0.6334` | **Keep P3, drop `--boundary`:** `P3` adds `+0.0399` GBDT (`burst +0.0584`, `moving_dead +0.0516`); `c_sw_gain` boundary slightly hurt (`-0.0013..-0.0029` raw). |
| **`deadline_base` + 25Hz `w128`** | `w=128, L=2, 2k, stride=1, DFL4` | `0.5530` | `0.6664` (`+0.0332` vs 5Hz) | `0.6587` (`<=2s 0.3780`) | `0.6699 / 0.6544 / 0.6750` | **Must use `frame_stride=1` (25Hz):** +3.3 pp over 5Hz `w64`, improves 18/18 provider×scenario cells (`DEADLINE_RESULTS_BASE_VS_NN25HZ.md`). |
| **`deadline_p3` + 25Hz `w128`** | `lgbm_p3` + `25Hz DFL4` | `0.5929` | `0.6678` (`+0.1148`) | `0.6623` (`<=2s 0.3810`) | `0.6681 / 0.6602 / 0.6751` | **`P3` + `25Hz` combine cleanly**, beating `nn1.0` (`0.6614`) by `+0.0064` (`DEADLINE_RESULTS_BASE_VS_P3_NN25HZ.md`). |
| **`ablate_p3_ens3` + 25Hz `w128`** | **`ens3` GPU (`LGBM+XGB+CB`) + P3, NO boundary** | **`0.5943`** (`+0.0413`) | **`0.6689`** (**`+0.1160`**) | **`0.6631`** (`<=2s 0.3818`) | `0.6682 / 0.6626 / 0.6760` | **Winner GBDT recipe:** highest standalone GBDT (`0.5943`), highest `nn0.7` (`0.6689`), biggest Groningen gain (`+0.0082..+0.0133`) (`ablate_DEADLINE_RESULTS_BASE_VS_P3_ENS3.md`). |
| **`deadline_p3` + `sc16_2seed`** | `lgbm_p3` + `w128 @ 25Hz + sc0..15, 2 seeds @ 2800` | `0.5929` | **`0.6673`** (`<=2s 0.3467`) | `0.6624` (`s1.0`: `0.6654`) | `0.6658 / 0.6621 / 0.6742` | **Best NN calibration & `nat`/`burst`:** SC last-4 **`0.8337`** (vs `0.7784`), `natural 0.7439`, `burst 0.6620`, `Oracle Burst 0.7287` (`ablate_nn2_p3_sc16_end2end_eval/results.json`). |

### Focused Component Diagnostics Verified
1. **Native `p3_removal_diagnostics` (`ablate_p3_ens3_removal.json`, `dead_diff=0.0`):**
   - `axes_neutral` (`+45` centroid-axis cols masked): DFL drops `-0.0275` (`nat`), `-0.0392` (`atd_real`), **`-0.0918` (`burst`: `0.5849 -> 0.4931`)**; Groningen `-0.0239` (`nat`), `-0.0365` (`atd_real`), `-0.0756` (`burst`).
   - `no_fallback` (`2–12s` ball-free router disabled): `0.0000` change on `natural`, while adding **`+0.0083` DFL burst** & **`+0.0142` Groningen burst**.
   - `deblur_neutral` (`+10` Laplacian/deblur cols): helps `natural` (`+0.0073` DFL, `+0.0145` Gron) & `short_2s_rec` cleanly.
2. **NN v2 Ablations (`ablate_nn2_*`, SC last-4 `2013725, 2015213, 2016236, 2017461` & `oracle_dead_decode`):**
   - `w160_5hz` (`3000` steps on 4 DFLs): **overfits** (`loss 0.0761`, SC NLL `0.6061 -> 1.3349`, Oracle `0.7837 -> 0.7552`).
   - `w160_5hz + sc16`: fixes calibration (`SC 0.7784 -> 0.8315`, NLL `0.3576`).
   - `w128_25hz_sc16` (`2400` -> `4000` steps -> `2-seed @ 2800`): `2-seed @ 2800` gives best balance (**SC `0.8337` nat / `0.8252` `atd_real`, NLL `0.3609 / 0.3805`**, Oracle `Nat(4) 0.8255`, `Short-2s 0.6526`, **`Burst(3) 0.7287`**).
3. **Priority-2 Decoder & `ownership_cost_scale` (`ablate_nn2_p3_sc16_end2end_eval/results.json`):**
   - With `freeze_dead_from_gbdt` (`viterbi4.ownership_pass`), Codex's default `ownership_cost_scale=0.5` halves `c_team` (`3.63 -> 1.81`) and over-splits spells (`-0.0049` to `-0.0077`). Sweeping `0.5 -> 0.75 -> 1.0` improved SpellF1 **monotonically** on every model (`dfl4_25hz`: `0.6623 -> 0.6645 -> 0.6657`; `sc16_2seed`: `0.6624 -> 0.6646 -> 0.6654`), while single-pass `baseline` (which ALREADY keeps `lp[:, D]` bit-identical via `blend_logp`) gets **`0.6673`–`0.6689`**. Check `restart_order` (`c_res_same <= c_res_flip`, `0.6686` on `ens3`) vs `baseline` (`0.6689`).
4. **Priority-4 SC Stage-2 (`ablate_sc_stage2_results.json`):**
   - Adding 2 SC matches to Stage-2 **PASSED** SC last-4 guard (`mean_accuracy_drop = -0.0066`, improved all 4/4 SC matches, `-0.0639` log-loss), lifting Groningen (`0.6325 -> 0.6405` nat, `0.5660 -> 0.5761` `atd_real`) and DFL natural (`0.6557 -> 0.6592`), with `invariant_dead_probability_max_diff == 0.0`.

---

## 4. Architecture & Empirical Contrast vs. Elvern (`04`/`05`/`08`) & Ming (`X-UTD`)

| Dimension | Our Latest (`ablate_p3_ens3` + `OwnNet 25Hz`) | Elvern (`04.46/04.48`, `05.23`, `08.05/08.17`) | Ming (`X-UTD` `13a48af`) |
|---|---|---|---|
| **Dead-Ball Coupling** | **`dead_leak == 0.0`:** `dead_s1` (`193` cols) + `dead_s2` (`141` cols) are strictly ball-free; `OwnNet` + `blend_logp` leave `lp[:, D]` bit-identical (`moving_dead` DFL AD F1 `0.956`). | `04`–`08` pass `ball_observed` (idx 0), `ball_outside`, `ball_age_s` into dead/live trunks (`04.46` isolated ownership only). | `0.75 * Derekko D50 + 0.25 * Rondo` dead vote — both read ball heavily (`dead recall 0.75` on public check). |
| **Stress / BAR-ATH Failure Contrast** | Immunity to moving/false dead balls: `ablate_p3_ens3 + 25Hz` holds `0.629`–`0.632` on adversarial `moving_dead` without AD collapse. | `04.49` Carried Dead dropped Dead F1 `0.9681 -> 0.9119` (`21.1%` visible dead called live); `05`/`08` lost 25–30 MacroF1 pts on BAR-ATH public check vs `J03WPY`. | `D50` alone dropped to `0.7702` Dead F1 on public check; failed external `N3` guard (`-0.012`), selected `P36` directly on 17-spell public check. |
| **Ownership & Equivariance** | `OwnNet`: `±10.16s` dilated TCN + full-window BiGRU + `4` spatial mirror flips + exact `g0 - g1` (`swap_diff == 0.0`), dynamic random gap masks. | `08`: `PlayerSetTCN` (`±2.5s` RF) + `ResidualTCN` teacher, `p=0.5` swap train / 2-view test avg, fixed offline `5–30s` gap train (`08.17`). | `D50` (`4` XGB + `12` `PlayerSet-TCNs` @ 5 Hz) w/ attack-dir norm + mirror/swap avg, no `D_H/D_A` state. |
| **Decoder** | **4-state `[H, A, D_H, D_A]`** + spell repair; `ownership_pass` / `freeze_dead_from_gbdt` upgraded with `scale=0.75–1.0` & P3 features. | **3-state `[H, A, D]`** + edge gain `1.5` (`04.51` 4-state attempt `-0.0015`). | **3-state** 2-pass `p36` on 5f cells: freezes `D`, halves switch cost `14 -> 7` (gains raw, hurts `N3`). |
| **Splits & Comparability** | Strict `DFL4 -> DFL3+Gron+Met+SC4`: non-oracle `0.6689` PB (`DFL3 0.668–0.670`, `nat 0.7439`, `burst 0.6620`, `Gron 0.6621`); Oracle w/ GT-dead `J03WPY 0.8314`, `J03WR9 0.8630`, `Gron 0.8534`. | Trained **`DFL6`** (`+WQQ, +WR9`), repeatedly selected on **`J03WPY`** (`08.17`: `0.8437` nat / `0.6751` burst). | `DFL4+SC12` (+ PFF dev / Metrica Rondo): reports `DFL3` 3-view `0.6414 -> 0.6737` raw / `0.7481 -> 0.7364` N3. |

---

## 5. Exact Final Recipe & Shipping Steps for Opus 5.5

### Step 1 — Recommended Final External Training Recipe (Full External, GPU-Accelerated)
Use our winning verified combination (**`ens3` GPU + `use_p3` + `ball_free_fallback`**, **NO `--boundary`**, **NO `--long-bursts --stoppage-augmentation`**, **`--no-metrica`** [Note: unseen `DFL4 -> Metrica` ALREADY scores `0.6760` / `0.7575` natural cleanly without touching Metrica terms!]):

1. **Train Final Multi-Seed 25 Hz `OwnNet` on RTX 4050 GPU (`~15–20 min`):**
   - *Option A (cleanest SC4 untouched until final pre-commit gate, OR train `sc0..sc15` first, confirm SC last-4 hasn't regressed, then either keep `sc0..15` [16 SC matches already hit `0.8337`!] or include all `sc0..19` once protocol finalized):*
   ```bash
   MTAH_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m solution.nn2_lomo \
     --tag final_w128_25hz --groups "" \
     --no-metrica --n-sc 16 --w-sc 0.25 \
     --frame-stride 1 --gap-policy mixed \
     --width 128 --gru-layers 2 --steps 2800 --batch 12 --lr 0.002 --seeds 0,1,2 --threads 2 \
     --save-final solution/out/final_nn2_w128_25hz_sc16_3seed.pkl \
     --out-dir solution/out/final_nn2
   ```
   *(Note: `--groups ""` trains on ALL `7` DFL + `Groningen` + `16` SC matches! To use all 20 SC matches after validating config, pass `--n-sc 20` — first check `0.8337` SC4 isn't overfit before deciding `16` vs `20`.)*

2. **Train Final Corrected GBDT (`train_v4.py`) on RTX 4050 GPU (`~30–40 min`):**
   - Uses `DFL` (7) + `Groningen` (1) + `SC` (`--sc-limit 20` or `16`) with `--no-metrica`, `--mode ens3`, `--use-p3`, `--ball-free-fallback`, `--seed 0`:
   ```bash
   MTAH_THREADS=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 .venv/bin/python -m solution.train_v4 \
     --mode ens3 --use-p3 --ball-free-fallback \
     --sc-limit 20 --no-metrica --seed 0 --inner-folds 3 \
     --out solution/out/final_gbdt_p3_ens3.pkl
   ```
   *(GPU is ON by default in `train_v4.py` when `--cpu` is omitted. IMPORTANT nuance: `train_v4.py` defaults `pipe.decode_cfg` to `viterbi4.DECODE4_DEFAULTS` when `--decode-cfg` is omitted, whereas all `0.6686–0.6689` external evaluations above used `deadline_benchmark.frozen_baseline_config()` [`pipeline_v4.pkl`'s `decode_cfg`] / `restart_order` [`c_res_same=15.1642, c_res_flip=16.0`]. Pass explicit `--decode-cfg` or set `pipe.decode_cfg` explicitly in Step 3!)*

3. **Assemble Final Explicit `PipelineV4` (`solution/configs/pipeline_final.pkl`) & Verify Invariants:**
   - Load `solution/out/final_gbdt_p3_ens3.pkl` and `solution/out/final_nn2_w128_25hz_sc16_3seed.pkl`, attach:
     - `pipe.neural_v2 = nn_models`
     - `pipe.neural_weight = 0.7`
     - `pipe.boundary_model = None`
     - **Decoder choice (verified):** use `restart_order` on `deadline_benchmark.frozen_baseline_config()` (`c_res_same=15.1642, c_res_flip=16.0, c_team=3.6290, c_stop=28.0, w=0.7286, d_bias=-0.0794, min_dead=59, min_live_blip=26, min_turn=10`; `0.6686` on `ens3`, `dead_leak == 0.0` via `blend_logp`), OR if strict decoded-`D` invariance is desired set `pipe.freeze_dead_mask = True` + update `ownership_cost_scale` to `1.0` (remember to keep `test_changepoint.py` passing if modifying `decode_cfg`!).
   - Verify `dead_leak == 0.0` and `swap_diff == 0.0` via `deadline_benchmark.invariant_probe(pipe, datasets.load_dfl("J03WPY"))`! Save `solution/configs/pipeline_final.pkl` + provenance JSON (preserve `solution/configs/pipeline_v4.pkl` untouched!).

### Step 2 — Generate Submission, Validate & Ship ([SUBMISSION.md](../SUBMISSION.md))
```bash
.venv/bin/python -m solution.run --learned --pipeline solution/configs/pipeline_final.pkl --name more_than_a_hack_2027
.venv/bin/python evaluation/validate_submission.py evaluation/results/more_than_a_hack_2027.csv
.venv/bin/python -m solution.sanity evaluation/results/more_than_a_hack_2027.csv
MTAH_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  solution.tests.test_changepoint solution.tests.test_data_guardrails \
  solution.tests.test_deadline_summary solution.tests.test_neural \
  solution.tests.test_training_hygiene -v
MTAH_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m solution.tests.test_core
```
- Also remember [SUBMISSION.md](../SUBMISSION.md) deliverables: update [docs/STATE.md](STATE.md) + README reproduction command + short technical document PDF/MD, remove `catboost_info/` before committing, verify `git status` keeps `data/`, `cache/`, and `solution/out/` excluded, and push public repo URL!
