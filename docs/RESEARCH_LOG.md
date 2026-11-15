# Research log

Newest first. Numbers are from the **external benchmark v3**: three Metrica sample games × three
pictures of what the ball does during a stoppage, each scored against two label variants (with
and without our uncontrolled-touch rule) with the organisers' `possession_metrics.py`.

**The number we quote is the held-out objective**: mean spellF1 over both label variants, on a
game that neither the parameters nor the models were fitted on, averaged over the three games.
Raw rows: `solution/experiments.csv`. Benchmark scores are not predictions of the real-match
score: different matches, labels derived by us.

| Reference on v3 (E006) | held-out | macroF1 | possessions vs truth |
|---|---|---|---|
| Organisers' baseline, ±50 | 0.250 | 0.556 | ×2.01 |
| Organisers' baseline, ±125 (its best) | 0.264 | 0.537 | ×1.06 |

Noise: gaps under about 0.02 are not treated as real.

> **E001–E005 are superseded.** An external audit on 2026-11-10 found that their benchmark
> rewarded dead-ball cues tuned on a ball track we had simulated ourselves, and other faults.
> See `AUDIT_RESPONSE.md`. Their entries are kept below for the record; do not quote their
> numbers. Their rows are in `solution/experiments_v1v2_superseded.csv`.

---

## E014 · PipelineV4 (ball reconstruction + v4 features + Stage-2 contact/gap TemporalStacker + Viterbi4) · 2026-11-14 · current best
**Change.** Multi-dataset `PipelineV4` architecture:
- `solution/ball_tracker.py`: physical jump filtering + dead-ball clamping + Hermite/constellation ball reconstruction (`BallConstellationModel` fitted on clean full-pitch matches) with dual visibility/confidence encoding (preserving visible vs imputed uncertainty cues that naive replacement in `E009` destroyed).
- `solution/features_v4.py`: strictly ball-free dead-ball kinematics (`193` cols) + `H<->A` swap-equivariant ownership features (`156` cols: multi-scale `T` Voronoi potentials, Closing Velocity, Time-to-Intercept, forward projections, last/next-touch physics).
- `solution/temporal.py`: `TemporalStacker` (`ens3`: LightGBM + XGBoost-CUDA + CatBoost-GPU) with one-sided past/future window contrasts, 3-level causal/anticausal contact timers, and visibility-gap bridges; bit-for-bit `0.0` `H<->A` equivariance via dyadic margin quantization and `Xs_base`.
- `solution/viterbi4.py`: 4-state restart-aware decoder `{H, A, D_H, D_A}` with separate `c_res_same` (`16.0`) vs `c_res_flip` (`15.16`) restart costs + spell repair (`min_dead=59`, `min_live_blip=26`, `min_turn=10`).

**Results across benchmarks (`E010`..`E014`):**
- **Metrica v3 3-fold LOMO (held-out, `standardize_match` before `degrade`):** **`0.5993`** objective (vs `0.5509` in `E008`), `spellF1_merged=0.6149`, `spellF1_raw=0.5837`, `macroF1=0.8707`, `aliveDeadF1=0.9281`, `teamAcc=0.9325`, `poss_ratio=0.8353`; per-game `0.6165 / 0.5955 / 0.5859`, per-scenario `path=0.5998 / hold=0.6072 / hidden=0.5908` (spread `0.0164`). Zero-shot (`DFL+8SC->Met`): `0.4463` (`0.4569` merged, `0.4357` raw).
- **Sportec DFL v1 7-fold LOMO (nested `viterbi4`):**
  - `natural`: `0.6429` raw / `0.6750` merged (`0.6590` obj; `J03WQQ+WR9=0.6692`), `macroF1=0.9005`, `aliveDeadF1=0.9535`, `teamAcc=0.9247`.
  - `atd_real`: `0.5767` raw / `0.6151` merged (`0.5959` obj; `J03WQQ+WR9=0.5883`), `macroF1=0.8840`, `aliveDeadF1=0.9539`, `teamAcc=0.8984` (burst-gap `teamAcc=0.8926`).
  - `atd_blind` / `atd_carried_dead`: `0.5223` / `0.5373` raw (`AD_F1=0.9530` / `0.9532`); live `PipelineV4` `max|logp_D(blind)-logp_D(carried_dead)| = 0.0`.
- **WCSF Groningen zero-shot (`DFL+8SC->Gron`):** `natural` `0.6278` raw / `0.6460` merged (`teamAcc=0.9427`); `atd_real` `0.5594` raw / `0.5878` merged (`teamAcc=0.9102`).
- **Progression on DFL 7-fold LOMO `atd_real` (raw `spellF1`):** `E008-dfl` `0.5134` -> `E009` `0.5037` -> `E010` `0.5195` -> `E011` `0.5371` -> `E012` `0.5570` -> `E013` `0.5589` -> `E014` **`0.5767`** (`0.6151` merged).

**Final model & submission (`solution/configs/pipeline_v4.pkl`):**
- Trained deterministically (`seed=0`, `--no-tags`) on all 31 confirmed matches (`7` Sportec DFL + `1` Groningen + `20` SkillCorner Current ownership + `3` Metrica Sample Games, 3-fold group OOF `TemporalStacker`). Verified reloaded `dead_leak=0.0`, `swap_diff=0.0`.
- Submission `evaluation/results/more_than_a_hack_2027.csv`: valid format (`199,319` rows), `solution.sanity` PASS/PASS, `368` possessions (mean `9.0 s`), graded shares `H 31.7% / A 20.2% / D 48.1%`, duplicate runs produce identical `sha256` `1f5c7ea4f4fbdc297fcf23fd535225e1c8e1999a2dc002ad0e39bb882964c221`.

## E008 · Two-stage learned model + Viterbi · 2026-11-10 · superseded by E014
**Change.** `learned.py` split in two. Dead model: 23 player-kinematics features, never sees the
ball. Team model: 74 ball-and-player features, trained on live frames only, both team
orientations. Combined as P(D), (1−P(D))·P(H|alive), (1−P(D))·P(A|alive); same decoder.
**Result.** Held-out **0.551** (models and decoder both fitted without the scored game; per game
0.564 / 0.551 / 0.549). In-sample 0.555. Against merged labels 0.574, raw labels 0.536.
macroF1 0.853, possessions ×0.74, alive/dead F1 0.914, team accuracy 0.928.
**Dependence on the invented stoppage picture.** path / hold / hidden: 0.557 / 0.562 / 0.545.
Spread 0.017, inside the noise band. This is the check E005 could not pass by construction.
**Decoder.** w 0.60, c_team 2.2, c_dead 24.8, c_live 23.8, min_spell 29
(`solution/configs/gbm_decode.json`). Final models trained on all three games:
`solution/configs/gbm.pkl`.
**Competition match** (`ours_E008_learned.csv`): valid; sanity PASS/PASS; 251 possessions, mean
13.0 s; D on 48.7% of graded frames; home share 0.62; two runs give the same sha256.
**Concern.** D at 48.7% is above the 32–41% of the external games. The true share is unpublished
and we may not measure target-match statistics to investigate, so this stays an unverified risk:
the dead model reads player speeds, and the ATD speed channel may be distributed differently from
the speeds we derive for the external games.

## E007 · Rule-based model after the audit fixes · 2026-11-10 · fallback
**Change.** Dead-ball evidence from player speed only (1 s and 6 s means); `absorb_short` fixed;
two scoring variants tuned separately, 150 iterations each, 12 parameters.
**Result.**

| Variant | held-out | in-sample | path / hold / hidden |
|---|---|---|---|
| Symmetric (team evidence also counts as "alive") | 0.423 | 0.437 | 0.410 / 0.441 / 0.459 |
| Decoupled (only dead-ball evidence decides alive vs dead) | **0.432** | 0.426 | 0.435 / 0.427 / 0.416 |

The two are within noise of each other held-out. Decoupled is kept because it depends less on the
invented stoppage picture (spread 0.019 against 0.049), which is the property the audit was about.
Parameters: `solution/configs/E007-decoupled.json`.
**Competition match** (`ours_E007_rules.csv`): valid; sanity PASS/PASS; 235 possessions, mean
15.1 s; D 44.2%; home share 0.61.

## E006 · Audit fixes and benchmark v3 · 2026-11-10 · benchmark and process change
Point-by-point record in `AUDIT_RESPONSE.md`. In short:
- No model uses the ball to decide dead vs alive.
- Benchmark runs three stoppage pictures and two label variants; held-out is the headline.
- `run.py` no longer reads the public ground truth; `sanity.py` prints PASS/FAIL only.
- `absorb_short` bug fixed; tag stacking capped; unpublished match statistics removed from
  `atd.summary()`.
- Smart Tagging off under a rule that needs no score: a component whose parameters cannot be
  validated on external data is not used.

**Baseline on v3.** ±0 / ±25 / ±50 / ±125: 0.021 / 0.157 / 0.250 / 0.264.

---

# Superseded entries (benchmark v1/v2, before the audit)

## E005 · Learned emissions + Viterbi (Option B) · 2026-11-10 · promising, robustness pending
**Hypothesis.** Hand-overs are the main loss (E003 oracle ablation). A classifier that sees who
controlled the ball before and after each frame should segment teams better than per-frame
hand-written evidence.
**Change.** `solution/learned.py`: 74 identity-free features, gradient-boosted trees
(`HistGradientBoostingClassifier`, 300 rounds, seed 0), trained on external games only, both team
orientations. Same decoder on top. `bench_learned.py` trains leave-one-game-out, then tunes the
decoder (weight, three costs, minimum spell; 200 iterations) on the out-of-fold probabilities.
**Result (v2).** spellF1 **0.6547** (rule-based 0.5250, baseline 0.2432), macroF1 0.9041,
possessions ×0.74, alive/dead F1 0.959, team accuracy 0.927. Frame accuracy before decoding 0.902.
**Held-out (model and decoder both tuned without the game).** 0.672 / 0.675 / 0.607, mean
**0.651**.
**Decoder.** w 0.447, c_team 4.21, c_dead 20.2, c_live 13.1, min_spell 28.
**Not yet done.** Not run on the competition match. A learned model can latch onto the simulated
dead-ball ball track, so before it is used: re-score on new degradation seeds and with the
dead-ball track removed or altered, and compare the drop with the rule-based model.

## E004 · Benchmark v2: dead ball runs past the line · 2026-11-10 · benchmark change
**Why.** In v1 the simulated ball during a stoppage moved straight from where it went out to the
restart spot, so it was almost never outside the pitch. A real tracker sees the ball roll past
the line. Dead-ball parameters tuned on v1 would have learned from an unrealistic picture.
**Change.** `external._fill_dead_ball`: if play stopped within 3 m of a line, the ball runs 1–8 m
past it, lies there, then is brought to the restart spot.
**Correction (audit finding 2).** This entry originally called the change "a reasoned assumption,
not fitted to any tracking file". The size was not fitted, but the idea came from seeing an
unpublished statistic of the competition tracking (share of ball positions outside the pitch).
**Result (v2 reference points).** Baseline ±50 **0.2432**, ±125 0.2691. E003 parameters,
unchanged: **0.5250**. All later experiments use v2.

## E003 · HMM, random-search tuning · 2026-11-10 · keep (current best rule-based)
**Change.** 150-iteration random search over all 15 parameters (`bench.py --tune 150 --logo`).
Parameters in `solution/configs/E003.json`.
**Result (v1).** spellF1 **0.5074** (baseline ±50: 0.2382), macroF1 0.8431, possessions ×0.66.
**Held-out.** Tuned on two games, scored on the third: 0.519 / 0.505 / 0.444, mean **0.489**.
The gain is not an artefact of tuning on the scored games.
**Oracle ablation with these parameters.** True dead mask + our team decoding 0.59; our dead mask
+ true team 0.88. Hand-over detection is where the remaining score is; dead-ball detection is
nearly solved *on the benchmark* (alive/dead F1 0.93), which is the least trustworthy part of the
benchmark because the dead-ball ball track is simulated.
**Competition match, first full run** (`ours_E003_notags.csv`, valid): 283 possessions, mean
12.8 s, D on 43.0% of graded frames, home share 0.60. Public-window sanity check: 17 possessions
predicted vs 17 true, spellF1 0.48, macroF1 0.82. Pipeline is aligned and labels mean what we
think. (Sanity only; not a selection signal.)

### E003b · Smart Tagging priors · 2026-11-10 · dropped for now
**Change.** `solution/tagging.py`: fixed per-frame priors from phase tags (team) and set-piece
windows (dead, then alive).
**Observation on the competition match.** Switching the priors on moved D from 43.0% to 34.1% of
graded frames and the possession count from 283 to 259. A stage with untunable weights is
overriding the tracking evidence: 0.15 per frame over a 10-second phase is 37 units, six times
the tuned team-switch cost.
**Decision.** Off. No external game has Smart Tagging, so there is no rule-compliant way to set
the weights, and a prior this strong cannot be justified by argument.
**Honesty note.** The public-window sanity score also fell with tags on (0.31 vs 0.48). We are not
allowed to select on that number, so the decision rests on the argument above; but it is what
prompted the second look. Flagged to the team as a judgment call near the line.
**Possible revival.** Use tags only as weak restart hints (set-piece windows), with weights small
relative to the switch costs, decided by argument before looking at any score.

## E002 · HMM with untuned defaults · 2026-11-10 · keep
**Hypothesis.** Decoding possessions as intervals (three-state Viterbi over evidence for team
control and dead ball) beats a fixed majority-vote window even before tuning.
**Change.** `solution/model.py`: team evidence from the distance margin between the nearest home
and away player, faded when nobody is near the ball or the ball is moving fast; dead-ball
evidence from mean player speed, ball outside the pitch, ball stationary; switching costs.
**Result.** spellF1 **0.3521**, macroF1 0.6254, possessions ×0.52.
**Oracle ablation** (`python -m solution.diagnose`, default parameters): true dead mask + our
team decoding 0.57; our dead mask + true team 0.78. Team hand-overs are the larger loss.

## E001 · Organisers' baseline on the benchmark · 2026-11-10 · reference
**Purpose.** Check the benchmark behaves like the real task before trusting it.
**Result.** spellF1 by window ±0 / ±25 / ±50 / ±125: 0.0171 / 0.1426 / 0.2382 / 0.2586.
Published full-match values: 0.0486 / 0.1996 / 0.2433 / 0.2242. Same shape (raw near zero,
plateau from ±50), same magnitude; the benchmark prefers ±125 slightly where the real match
prefers ±50. macroF1 is lower on the benchmark (0.43 vs 0.67) because the proxy games have about
40% dead ball and the degradation keeps the ball visible during stoppages.
**Check on the competition match.** Our loader reproduces the committed `baseline_raw.csv` and
`baseline_smooth50.csv` labels on 100% of graded frames.
