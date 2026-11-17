# Elvern Experiments & Implementations Summary (`01`–`08`)

> **Scope & Provenance:** Summarizes teammate Elvern's local submission workspace (`Elvern-latest-submission/`, git-ignored in the main repo) verified on 2026-11-16. Strictly excludes competition-match tracking inputs, ground truth CSVs, and prediction CSVs (`data/`, `evaluation/gt/`, `evaluation/results/`). All claims, file paths, symbol names, test counts, and metrics come from direct inspection and verification of the source code and experiment records.

---

## 1. Governance, Contract, and Checkpoints Storage

Elvern's workspace enforces a strict provenance contract documented in `Elvern-latest-submission/implementations/CONTRACT.md`, `RULES_AUDIT.md`, and `ORGANIZER-CLARIFICATION.md` (copied to the main repo root at `ORGANIZER-CLARIFICATION.md`):

- **Competition timeline & contract:** 25 fps, `199,319` frames (`video_frame = frame_id - 1`), inclusive graded blocks `11185..90651` and `110808..190197`. Smart Tagging stays unused in tactical inference.
- **External-only fitting:** Following the 2026-11-04 organizer clarification, all model training, hyper-parameter search, calibration, and decoder tuning use external data only (`Sportec Open DFL IDSSE v1`, `WCSF Groningen`, `SkillCorner`, `ELASTIC`, `SoccerTrack v2`). Metrica Sample Games 1–3 are **excluded from training** in Elvern's workspace (only used in an external replay script matching Sigue's Metrica split).
- **Mandatory remote training & SHA-256 manifests:** Learned experiments (`04`, `05`, `08`) run training remotely on Kaggle notebooks (`elvernneylmavtanny/...`) and pin raw dataset checksums, feature schemas, source hashes, native checkpoints (`.cbm`, `.txt`, `.json`, `.pt`), and output CSV hashes in JSON manifests.
- **Where checkpoints live:** Zero model binary weights (`.pt`, `.pth`, `.cbm`, `.pkl`, `.joblib`, `.onnx`, `.safetensors`, `.npz`) are committed in `Elvern-latest-submission/`. All trained checkpoints and downloaded external data live under git-ignored `data/implementations/<approach_id>/<run_id>/`. `04_classical_ml/config.json` ships `"checkpoint": null`, and external baselines (`CoreOrigin`, `Sigue`) are fetched/verified by SHA-256 inside Kaggle staging scripts (`08_neural_network/public_baseline/`).
- **Unit test verification in `.venv` (2026-11-16):** Running `python -B -m unittest discover` in separate processes verified:
  - `01_rule_based/tests/`: **74 tests OK**
  - `02_viterbi/tests/`: **68 tests OK**
  - `03_semi_markov/tests/`: **77 tests OK** (`+ 3` benchmark tests in source)
  - `04_classical_ml/tests/`: **37 tests OK** (`+ 157` tests across 45 experiment test files in source)
  - `05_tcn/tests/` (`57` tests) and `08_neural_network/tests/` (`40 + 3` tests) exist with comprehensive schema/invariance checks (PyTorch environment not installed in main `.venv`).

---

## 2. Inventory of Approaches `01`–`08`

| ID | Approach | Status | Feature Count / Inputs | Decoder | Key External Validation Headline |
|---|---|---|---|---|---|
| `01_rule_based` | Causal hysteresis (`engine.py`) & IFAB candidate (`experimental/engine.py`) | Implemented (non-ML) | Geometric proximity, streaks, slowdown vote | Causal FSM | 3 IDSSE matches: `0.3053` -> `0.3137` nat SpellF1; `0.2472` -> `0.2272` burst-masked |
| `02_viterbi` | Heuristic evidence + exact 3-state `[H, A, D]` Viterbi (`run.py`, `run_v2.py`) | Implemented (non-ML) | Geometry, memory, relative motion | Exact 3-state Viterbi (`decoder.py`, reused by `04`, `05`, `08`) | `J03WOH`: `0.3668` base vs `0.4006` candidate dynamic SpellF1 (`0.6775` MacroF1) |
| `03_semi_markov` | Exact monotone-queue explicit-duration MAP (`run.py`, `reliable/reliable_run.py`) | Implemented (non-ML) | Proximity/motion/exit cues + transition boundaries | Exact semi-Markov MAP (no hard bounds or beam pruning) | 2 DFL matches (`J03WOH+J03WMX`): `0.3699` -> `0.4039` clean, `0.3081` -> `0.3327` masked |
| `04_classical_ml` | Factorized GBDTs (`live`, `home_given_live`, + optional `restart`) (`04.01`–`04.51`) | Implemented (Kaggle CPU) | `35` -> `67` -> `76` -> `94` -> `166`/`189` tabular features | 3-state Viterbi / `2–12s` blend / 4-state `H/A/D_H/D_A` | Best historical DFL (`04.40` CB `J03WQQ+WR9` 7-cond mean): `0.576694`; Inner `04.48 motion`: `0.5808` (`04.46 gradual_2_12`: `0.5621` qualified) |
| `05_tcn` | Dilated TCN (`TemporalEvidence`) & `ConvBiGRUEvidence` (`05.01`–`05.23`) | Implemented (Kaggle GPU) | `35` -> `58` -> `85` aggregate sequence channels | Viterbi + 126-col boundary MLP (`gain=1.5`) | `05.23 ConvBiGRU` (`J03WPY` val): **`0.828571` nat (`0.966440` MacroF1) / `0.659852` 6-gap / `0.631669` burst** |
| `06_relational_crf` | Attention/graph player interactions + CRF | **Documented only** | — | — | Not implemented, not run, no tests |
| `07_video_fusion` | Visual ball detection fusion | **Documented only** | — | — | Not implemented, not run, no tests |
| `08_neural_network` | `ResidualTCN` (`08.01`/`08.02`) + `PlayerSetTCN` (`08.03`/`08.04`) blend (`08.05`) + refits (`08.17`–`08.19`) | Implemented (Kaggle GPU) | `85` aggregate + `3` teacher logits (`ResidualTCN`); `11×8` per-player tokens (`PlayerSetTCN`) | Viterbi + boundary edge (`gain=(1.5, 0)`), ordered touch (`08.18`), restart advance (`08.19`) | **`08.17` long-gap (`J03WPY`)**: **`0.843735` nat / `0.711725` 6-gap / `0.675132` burst** (`08.05 s17`: `0.839803 / 0.707034 / 0.666310`) |

---

## 3. Deep Dive: `04_classical_ml` (`04.01`–`04.51`)

### 3.1 Architecture & Feature Progression
`04_classical_ml` factorizes framewise prediction into binary `live` vs `D` and `home_given_live` heads (`inference.py`: `live(features)`, `home_given_live(features)`), decoded via `02_viterbi/decoder.py` (`04_classical_ml/decoding.py` with `LOG_PROBABILITY_FLOOR = 1e-12`):

1. **Base (`35` features, `features.py`):** `9` `_BALL_NAMES` (`ball_observed`, `ball_motion_observed`, `ball_speed_mps`, `ball_endline_clearance_m`, `ball_touchline_clearance_m`, `ball_outside`, `recent_ball_observed_fraction`, `recent_ball_outside_fraction`, `ball_age_s`) + `13` `_TEAM_NAMES` × `2` (`home`/`away`).
2. **Temporal windows (`67` features, `temporal_features/temporal.py`):** Adds `32` multi-scale centered/past distance, near-player, speed-advantage, and ball-observed summary columns. `67`-feature XGBoost raised historical DFL test SpellF1 to `0.6210` natural / `0.5243` fixed-gap.
3. **Gap anchors (`76` features, `lightgbm_gap_anchors/anchors.py`):** Adds `9` `ANCHOR_NAMES` recording the previous/next unambiguous ownership anchor within `8.0 s` (`MAX_NEAR_M=2.5`, `MIN_MARGIN_M=0.75`), span duration, agreement, and linear interpolation.
4. **Player relative motion (`94` `BASE_NAMES`, `lightgbm_short_spell_weights/player_motion.py` & `participant_sweep/sweep.py`):** Adds `18` ball-free motion columns (`home`, `away`, `home_minus_away` × `velocity_along`, `velocity_across`, `agreement`, `speed_change`, `along_axis_spread_change`, `matched_players` over `0.4 s` windows along the Home→Away centroid axis).
5. **Ball-free motion-context specialist (`166` cols, `groningen_mix/run_mixtures.py`):** `94` base columns with **all ball observations masked** (`match.absent()`) + `72` directional past/future mean columns (`0.4 s` and `1.2 s` × `18` motion cols). Used for `gradual_2_12` ownership fallback.
6. **Sweep evidence blocks (`04.48`, `participant_sweep/evidence.py`):** Appends `+15` `contact`, `+8` `interception`, `+72` `motion` (`166` total), or `+95` `all_evidence` (`189` total) onto `94` `BASE_NAMES`.

### 3.2 Key Experiment Milestones & `participant_sweep` (`04.46`–`04.51`)
Early sweeps (`04.01`–`04.45`) established:
- Static `c019` doubled Viterbi transition penalties `[[0, -12, -8], [-12, 0, -8], [-6, -6, 0]]` (used by `04.05`/`04.06`) outperformed adaptive transitions (`adaptive_transitions`). The shipped costs used by `05`/`08` are `[[0,-6,-4],[-6,0,-4],[-3,-3,0]]` (`02_viterbi/config.json`), emission scale 1.0, no minimum-duration filter.
- Uncalibrated probabilities transferred across tracking domains better than OOF sigmoid/beta calibration (`probability_calibration`).
- `04.40` CatBoost DFL (`J03WQQ+J03WR9` 7-condition mean) reached **`0.576694` SpellF1** (`0.905326` MacroF1; `0.680955` natural / `0.596681` fixed). Adding Groningen ownership at 10% helped stability, whereas mixing SkillCorner or SoccerTrack into `04` regressed overall DFL validation.

The structured 4-fold inner DFL sweep (`J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`) across **7 archive views** (Natural, Fixed, Varied, Varied seeds 1–3, Burst) and **4 stress views** (`Carried dead ball`, `Player tracking degradation`, `False ball detections`, `Combined tracking degradation`) in `participant_sweep/review.md` yielded:

- **Oracle ceiling (`04.46`):**
  - True live/dead oracle: `0.6071` SpellF1 / `0.9205` MacroF1 (Restart F1 `0.9692`).
  - True ownership oracle: **`0.9185` SpellF1 / `0.9742` MacroF1** (Short-spell recall `0.9069`). **Conclusion:** live-play ownership and gap/turnover attribution — not just live/dead boundaries — dominate GBDT SpellF1 loss on DFL.
- **Gradual `2–12 s` player-only ownership blend (`04.46`, Qualified):** Linearly routes the **ownership head** (`home_given_live`) toward the ball-free `166`-feature `motion-context` specialist across `50..300`-frame ball gaps. Improved 4/4 folds from `0.5542` -> **`0.5621` (`0.562081`) SpellF1** (`0.8978` MacroF1), driven by Burst (`0.4989` -> **`0.5159`**, `+0.0170`).
- **Model family & capacity (`04.47`):** Across balanced trees, **CatBoost depth 8** (`0.5558`) > CatBoost depth 6 (`0.5529`) > XGBoost depth 7 (`0.5450`) ≈ LightGBM leaves 63 (`0.5450`, highest short-spell recall `0.2467`) > LightGBM leaves 31 (`0.5425`) >> Logistic Regression (`0.4281`). CatBoost also beat LightGBM on **every stress condition** (`0.410950` vs `0.389999` 4-stress mean).
- **Evidence blocks (`04.48`):** **`CatBoost · motion`** (`166` cols) achieved **`0.5808` archive SpellF1 / `0.9019` MacroF1** (`0.6653` clean, `0.5904` fixed, `0.5376` burst, `0.9713` Dead F1, `+0.0279` across 4/4 folds; Placed on **Hold** only by a strict `-0.0145` fixed-gap short-recall gate). `CatBoost · all evidence` (`189` cols) scored `0.5785` archive and led False-ball (`0.3525`) and Combined (`0.3282`) stress. Three boundary-weight runs **Qualified** (`boundary both 04s x4` lifted short-spell recall `0.2328 -> 0.3025`).
- **Tracking robustness (`04.49`, all Hold at gate):**
  - Masking visibility flags (`LIVE_DROPS`: `live_no_visibility`) slightly hurt (`0.5472`).
  - Completely removing ball from the `live` head (`Player-only live/dead head`, `match.absent` on `35–94` tabular stats) collapsed to **`0.4763` SpellF1** (`0.9091` Dead F1, calling `8.1%` of visible dead frames live).
  - Augmenting with carried dead-ball / combined tracking corruptions dramatically rescued stress performance (`0.4836 -> 0.5407` on Carried, Dead F1 `0.9119 -> 0.9731`; `0.3043 -> 0.4748` on Combined), **but degraded clean/archive DFL (`0.5529 -> 0.5401 / 0.5352`)** and failed retention gates.
- **Additional datasets (`04.50`) & 4-state Viterbi (`04.51`):** `25% Groningen ownership` hit `0.5571` (`0.6553` natural, Hold on fixed short recall); SkillCorner (`0.5488`/`0.5437`) and SoccerTrack (`0.5471`) regressed. Adding a learned `restart_team` head + 4-state `H/A/D_H/D_A` Viterbi (`04.51`) scored `0.5514` (`-0.0015`, 0/4 folds).

---

## 4. Deep Dive: `05_tcn` (`05.01`–`05.23`) and `08_neural_network` (`08.01`–`08.19`)

### 4.1 `05_tcn` Progression
- `05.01`–`05.18` used 4 DFL train matches (`J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`) with `TemporalEvidence` (1D dilated residual TCN, `width=48`, `dilations=[1,2,4,8,16,32,64,128,256]`, radius `511`, `dropout=0.1`). Adding motion + contact features (`58` ch, `05.17`) reached `0.745685` natural SpellF1 on `J03WPY`.
- `05.19`–`05.23` expanded to **6 DFL train matches** (`+J03WQQ`, `+J03WR9`) validating on `J03WPY` at 25 Hz:
  - `05.19` (58 ch TCN, ownership gain `0.5`): `0.767179` nat / `0.658751` 6-gap / `0.630854` burst.
  - `05.20` (**85 ch** adding `27` `MULTISCALE_NAMES` at lags `3, 8, 20` f, ownership gain `1.5`): **`0.800496` nat / `0.669935` 6-gap / `0.642261` burst**.
  - `05.21` (turnover head, 58 ch): `0.777693` nat / `0.657434` 6-gap.
  - `05.22` (85 ch TCN + player-track dropout & jitter): `0.794416` nat / `0.645630` 6-gap; best BAR-ATH `05` public pipeline check at **`0.590000` SpellF1 / `0.696559` MacroF1**, and lifted player-corrupted burst stress over `05.20` (`0.543276 -> 0.583429`).
  - **`05.23` `ConvBiGRUEvidence`** (`six_bigru.py`: `Conv1d(85,48,1)->Conv1d(48,48,3,p=1)->GELU` + bidirectional `GRU(48,24)` residual + 126-col 7-lag boundary MLP, ownership gain `1.5`): **`0.828571` nat (`0.966440` MacroF1) / `0.659852` 6-gap / `0.631669` burst** (`0.500000 / 0.665125` public check).

### 4.2 `08_neural_network` Architecture & Results
Approach `08` combines two heterogeneous neural families trained on the 6-match Sportec split (`network08.py`, `prepare.py`, `train08.py`, `ensemble08.py`):

1. **`ResidualTCN` (`08.01`, `08.02`, `08.02-control`):** `88` channels = `85` motion features + `3` `05.23 ConvBiGRU` teacher channels (frozen `05.23` only in `08.01`; in `08.02`/`08.02-control` the teacher is warmed up, refit 12 epochs on Sportec and given a refitted boundary head before freezing, `train08.py:404-421`) `[log_live - log_dead, log_home - log_away, clip(edge - prior, -4, 4)]` -> zero-initialized 6-layer `TemporalEvidence` (`width=64, dilations=[1,2,4,8,16,32], dropout=0.2`). Warm-up (`512` steps at `10 Hz`, owner-only loss) uses SkillCorner (`08.02`) or downsampled Sportec (`08.02-control`).
2. **`PlayerSetTCN` (`08.03`, `08.04`, `08.04-control`):** Encodes per-frame `player_tokens` (up to `11` slots/team × `8` fields: centered `xy/30`, `vel/12`, `ball_relative_xy/30`, `ball_relative_vel/12`) via 2-layer `32-d` GELU MLP + per-team masked mean & max (`128` ch) + `85` aggregate channels = **`213` channels** -> `TemporalEvidence` (`width=96, dilations=[1,2,4,8,16,32], dropout=0.2`).
3. **`08.05 BlendPredictor` (seed 17 & 29):** Equal `0.5/0.5` average of `(probabilities, edges)` from `08.02-control` + `08.04-control` decoded via Viterbi + boundary gain `(1.5, 0)`:
   - **`J03WPY` (seed 17):** **`0.839803` nat (`0.970215` MacroF1) / `0.707034` 6-gap / `0.666310` burst**; recovers **`75/124` short spells (`<= 2 s`)**, Transition F1 `0.841791`, p90 `1 frame`.
   - Public 5-minute pipeline check: `08.04-control` s17 reached `0.625641 / 0.715926`; `08.05` s17 scored `0.577778 / 0.685151` (`s29`: `0.600000 / 0.683957`).
4. **`08.17` Long-gap refit (`experiments/long_gap/`):** Adapts Derekko D49's idea — replaces `burst_train` with disjoint `5–30 s` gaps (allocating `69.56%` of masked frames to `>= 25 s` gaps while preserving total masked frames) and refits `08.02-control` + `08.04-control` at seed 17:
   - Raises `J03WPY` across the board: **`0.843735` nat (`0.969513` MacroF1), `0.711725 / 0.935585` 6-gap mean, `0.675132 / 0.915798` long burst**, and lifts the frozen public check **`0.577778 -> 0.582857` SpellF1** (`0.685151 -> 0.701772` MacroF1).
5. **`08.18` Ordered-touch cue & `08.19` Restart confirmation:** Inspired by Rondo v5. Both **lock `08.05`'s decoded `D` mask** (`emissions[dead, :2] = -inf`). `08.18` (`gain=1.0` H↔A cue; changed `54` natural / `19` fixed / `25` burst frames on `J03WPY`, `15` on the full competition match) lifted `J03WPY` natural `0.839803 -> 0.842365` (`3-view` mean `0.750272 -> 0.751476`); `08.19` (`<=0.5 s` restart advance) was essentially neutral (`0.839803` nat).
6. **`08.06`–`08.16` Public baseline replays (`public_baseline/`):**
   - **CoreOrigin run-48 (`08.07`, `08.15`):** `0.857005 / 0.858041` natural on `J03WPY` (`0.974079` MacroF1, 780 features + ConvBiGRU + Viterbi `(0,2,1)` + OOB rule), **but dropped to `0.505263 / 0.651007` on BAR-ATH public check (`D F1 = 0.311494`)**.
   - **Sigue (`08.10`, `08.11`):** Metrica `0.854471`, Sportec refit `0.811210` on `J03WPY` (`0.494737 / 0.666856` public check, `08.16`).
   - `08.08`, `08.09`, `08.12`, `08.14` froze run-48's live/dead mask and targeted live H↔A switches; none surpassed run 48 on `J03WPY`.

---

## 5. Critical Soundness Finding: `ball_observed` & Ball Cues in Live/Dead Prediction

A code-level audit of `04_classical_ml`, `05_tcn`, and `08_neural_network` confirms:

1. **`04_classical_ml` (`04.01`–`04.51`):** `inference.py` passes identical feature columns starting with `FEATURE_NAMES[:9]` (`ball_observed` at index `0`, `ball_outside`, `ball_age_s`, distances, speeds) to both `live` and `home_given_live`. In `04.46`–`04.51`, `gradual_2_12` only blends the **`owner`** head (`run_sweep.py`); the **`live`** head still sees all ball features. Only two `04.49` ablations altered `live`: `live_no_visibility` (still kept ball coordinates/distances/clearances) and `live_players` (`0.4763` SpellF1, failed retention).
2. **`05_tcn` (`05.01`–`05.23`):** `TemporalEvidence` (`05_tcn/model.py`) and `ConvBiGRUEvidence` (`05_tcn/experiments/six_bigru.py`) ingest all `35`/`58`/`85` channels (with `ball_observed = FEATURES[0]`, `ball_outside`, `ball_age_s`, ball kinematics) into a **single shared trunk** predicting `[live, home_given_live]` (or 3 classes) jointly.
3. **`08_neural_network` (`08.01`–`08.19`):** `ResidualTCN` consumes all `85` motion channels + `3` `05.23` teacher logits `[log_live-log_dead, log_home-log_away, edge]`, and `PlayerSetTCN` consumes all `85` channels + `player_tokens` (`ball_relative_xy/vel` sorted by ball distance) in a shared `TemporalEvidence` trunk outputting `[live, home_given_live, switch]`. Notice SkillCorner warm-up (`owner_only=True` in `train08.py`) trains `logits[:, 1]` only, leaving `logits[:, 0]` (live/dead) governed entirely by Sportec DFL where stoppages strongly correlate with clean ball absence/OOB/stillness.

**Empirical Consequence on Stress Views & ATD Domain Transfer:**
- On clean multi-camera DFL, `04` (`0.9681`), `05` (`0.966+`), and `08.05`/`08.17` (`0.9686+`) achieve ~0.97 Dead-ball F1 with `<=0.2%` visible dead frames called live.
- Under `04.49` **Carried dead ball** (ball placed near a player for 1–5 s during stoppages), CatBoost `Dead-ball F1` drops from **`0.9681` -> `0.9119`** (**`31,007 / 147,008 = 21.1%`** visible dead frames called live); under **False ball detections**, `Dead F1` drops to **`0.9062`** (`15.9%` visible dead called live); under **Combined tracking degradation**, `Dead F1` collapses to **`0.8524`** (**`51,274 / 160,803 = 31.9%`** visible dead called live; SpellF1 `0.3043`).
- On `BAR-ATH` (where tactical tracking misses ~41% of graded ball frames while players carry or leave a false ball visible during stoppages), `CoreOrigin` run-48's `Dead F1` collapsed to `0.311494`, and all `05`/`08` models drop 25–30 MacroF1 points relative to `J03WPY` (`0.96–0.97` -> `0.651–0.718`). Compare this with the main repo's `PipelineV4` (`E014`), which enforces exact `dead_leak = 0.0` (193 dead + 141 Stage-2 features strictly ball-free), albeit at the cost of non-neural GBDT boundary resolution on short spells (`<= 35 f`).


---

## 6. Corrections and additions from the 2026-11-17 code-level deep dive

Source: a second read of `Elvern-latest-submission/implementations/` on 2026-11-17 (code and docs only). Items 1–5 of the earlier text were corrected in place above.

### 6.1 Things the earlier summary did not say
- **`08` decoder edge:** in `08` the Viterbi H↔A bonus is the network's own third (switch) logit, `clip(logit - prior, ±4) × 1.5`, added to H↔A edges only (live↔dead gain 0). The 126-column MLP edge is used only inside the `05.23` teacher signal.
- **Receptive fields and sizes:** `08` TCNs see ±63 frames (±2.5 s); the `05` TCN ±511 frames; the BiGRU the whole half. `ResidualTCN` ≈ 80k parameters, `PlayerSetTCN` ≈ 189k, `05.23` ConvBiGRU ≈ 22k (computed from the layer definitions, not stated in his docs). Inside any gap longer than about 5 s the player-set model sees no ball at all.
- **Training recipe (`08`):** AdamW `lr 1e-3`, weight decay `1e-4`, no schedule, 12 epochs, batch 8, 1024-frame cores plus halo, grad clip 2, seeds 17 and 29. Loss = BCE live + BCE owner (×2 on complete H/A runs ≤ 50 frames) + 0.15 × BCE switch (`pos_weight = sqrt(neg/pos)`). Controls use a 512-step 10 Hz owner-only warm-up. Train views per match: `clean, fixed, varied_train, burst_train` (`08.17` swaps the last for `long_gap_train`). No player dropout, jitter or mirroring in `08`; team swap p=0.5 in training and two-view averaging at inference.
- **Selection protocol:** checkpoint (epoch 4/8/12), edge gain and promotion are all chosen on the single match `J03WPY`; the six-match runs have no independent test. Seed variance is about ±0.01 (`08.05`: `0.8398` s17 vs `0.8312` s29). `08.05` seed 29 fails his own promotion rule against the player-set control and the blend "remains exploratory". `08.17` + `08.18` was never combined.
- **Stress masks:** `gap-stress-burst-01` = gaps of 12/18/24 s separated by 12–24 s, seed `sha256("<fixture>/gap-stress-burst-01")[:4]` little-endian, one rng across blocks. For `J03WPY` the mask hides 73,155 frames (`sha256(hidden.tobytes()) = ad27c912…e21c`). `08.17` long-gap training: 5–30 s gaps, same hidden-frame budget as `burst_train`, 40% of gaps forced ≥ 25 s. `05.22` player stress: 8% of each slot's frames dropped in 10–30-frame spans, AR(1) jitter 0.08 m RMS, ρ = 0.85.
- **Transfer evidence he recorded:** `08` models score 0.59–0.66 SpellF1 on Groningen. On the BAR-ATH five-minute pipeline check (recorded after freeze, never a selection target) his `05`/`08` runs sit at 0.45–0.63 SpellF1 / 0.65–0.72 MacroF1, while the participant replays that are much weaker on `J03WPY` (Derekko D50 `0.688`, Rondo v5 `0.619`) reach `0.676 / 0.875` and `0.627 / 0.887`. The Sportec ranking inverts on the real match, and the gap is in dead-ball frames.
- **Running his code locally:** `require_kaggle()` hard-fails off Kaggle and the entry points are hash-pinned. Remote-only training is his team convention, not an organiser rule. Compute is small (a T4 fits three 05 models in about 9 minutes).

### 6.2 Negative results worth not repeating
| Area | Tried | Result |
|---|---|---|
| `08.02` / `08.04` | SkillCorner owner warm-up | Fails against update-matched controls (`08.04`: −0.018 six-gap, −0.031 burst) |
| `08.08` / `08.09` / `08.12` / `08.14` | Owner-correction, edit-ranking, event-actor auxiliary loss, contact-candidate edits on a frozen run | All below control (−0.005 to −0.079) despite oracle headroom of 0.95+; extra predicted spells cost more than they recover |
| `08.11` | Blending a Sportec-refit Sigue ConvBiGRU into run 48 | Every blend weight 5–50% regressed |
| `08.19` | Restart-advance rule | Neutral |
| `05.05` | Edge gain ≥ 2 | SpellF1 collapses (0.72 → 0.55 at gain 2) |
| `05.11` | Re-tuning the gain on one match | +0.010 validation, −0.005 on the historical pair: single-match tuning does not transfer |
| `05.13`–`05.15` | Seed ensembles | Gaps +0.011–0.017, natural −0.002 to −0.003 |
| `05.21` | Switch head on the plain TCN | +0.011 natural, −0.012 burst |
| `05.09` | Transition loss | Short recall up, pooled natural −0.003 |
| `04.49` | Player-only live head; carried-dead / combined-corruption augmentation | −0.077; fixes its stress view but −0.013 / −0.018 archive |
| `04.50` | SkillCorner, SoccerTrack, ELASTIC as extra training data | All fail; Groningen 25% is a Hold |
| `04.51` | Learned restart-team head + four-state Viterbi | −0.0015 |
| `04.47` | LightGBM / XGBoost vs CatBoost | CatBoost best (LightGBM −0.010) |
| `04.04`, `04.26`–`04.27` | Probability calibration | Does not transfer |
