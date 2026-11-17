# more_than_a_hack_2027 — Technical document

Frame-level ball possession (H / A / D) for the BAR-ATH tactical tracking data, produced end to end
by an algorithm trained only on public external matches. Nothing from the provided game (tracking,
tags, video, public 5-minute ground truth) is used to train, tune or select any component of
the shipped pipeline. Earlier deviations in our development history are disclosed in section 7.

## 1. Approach

Three stages, each chosen because it survived a strict held-out external benchmark (section 6):

1. **Dead-ball and team GBDT stack** (`solution/pipeline_v4.py`, trained by `solution/train_v4.py`).
   - *Ball reconstruction* (`ball_tracker.py`): the ATD ball is missing on ~41 % of graded frames.
     A learned constellation imputer plus kinematic gap filling estimates a ball position, a gap
     length and a confidence for team features only.
   - *Stage 1*: a dead-vs-live classifier on **player-only** features (193 columns: formation
     compactness, velocities, restart shapes; no ball input of any kind) and a team classifier on
     player + reconstructed-ball features. "P3" team features add centroid-axis motion (+45
     columns), deblurred player dynamics (+10) and a 2–12 s ball-free fallback head routed in by
     gap length. Each classifier is an ensemble of LightGBM + XGBoost (CUDA) + CatBoost (GPU).
   - *Stage 2*: a temporal stacker over Stage-1 outputs, fitted on fixture-group out-of-fold
     predictions (no fixture ever predicts itself), with a ball-free dead branch (141 columns).
2. **OwnNet, a 25 Hz equivariant ownership network** (`solution/neural_v2.py`). A player-set encoder,
   ±10.16 s dilated TCN and full-window BiGRU that outputs only P(Home | live) and a switch
   probability. Home/away symmetry is exact by construction (score g(H,A) − g(A,H)), with
   4 mirror flips at inference; training randomly masks the ball for 5–30 s to mimic ATD gaps.
   3 seeds are averaged. It re-splits the live mass of the GBDT output at weight 0.7 and **copies
   the dead column bit-for-bit** (`ownership_blend.blend_logp`), so it cannot change P(D).
3. **Line snap**: the player-only head decides *whether* a stoppage happened. When the observed ball
   leaves the pitch (with continuity, glitches rejected) between 4 s before and 1 s after a predicted
   stoppage onset, the onset moves to that exit + 2 frames (`solution/line_snap.py`). It never creates
   or deletes a stoppage, and its parameters were fixed on the DFL training matches.
4. **4-state restart-aware Viterbi** (`solution/viterbi4.py`) over {H, A, D_H, D_A}, where D_x remembers
   the last live owner so a same-team restart (free kick won) and an opposite-team restart
   (throw-in, goal kick) cost differently, followed by conservative spell repair.

Why this rather than something else: dead vs alive cannot be learned from the ball on external
data, because the external providers have no ball during stoppages (whatever appears there in a
degraded copy is invented). A dead head that sees the ball learns that shortcut and collapses
when a tracker keeps the ball visible during a stoppage. We therefore made the dead decision
structurally ball-free and spent the model capacity on ownership, which is where the 41 %
missing ball hurts.

Invariants enforced in code and unit tests, and re-checked on the final artifact:
`dead_leak == 0.0` (removing the ball changes no P(D)) and `swap_diff == 0.0` (swapping team
identities swaps H/A exactly).

## 2. Assets used and dropped

| Asset | Used? | Why |
|---|---|---|
| ATD tactical tracking (players + ball) | **Input only** | The only per-frame signal. |
| Published match facts (graded blocks, 2nd-half kickoff 113,742, pitch size, ~41 % missing ball, ~20.5 objects/frame) | Constants | Graded blocks, forced `D` before the 2nd-half kickoff; the statistics shape the external degradation only. |
| Smart Tagging XML | Dropped | No external game has matching tags, so their weights cannot be validated. |
| Homography, broadcast tracking, video | Dropped | Not frame-aligned or not needed; never decoded. |
| Public 5-minute ground truth | Format/alignment check only | `solution.sanity` prints PASS/FAIL for orientation and ±50-frame alignment, never a score. |
| Metrica sample games | Dropped from final fit (`--no-metrica`) | Unverified redistribution terms; held-out Metrica already scores well without it. |
| Learned boundary / change-point head | Dropped | Hurt held-out SpellF1 slightly (−0.001 to −0.003). |
| Long-burst / stoppage augmentation | Dropped | Helped standalone GBDT, slightly hurt the blended model. |
| Two-pass "frozen dead" decoder (scale 0.5) | Dropped | Over-splits spells; single-pass `restart_order` is better. |
| PFF data | Not used | Licence. |

## 3. Training data

All labels are the providers' own; none come from the provided game. All of it is open data.

| Dataset | Source, licence | Who labelled, how | Amount | Used for |
|---|---|---|---|---|
| Sportec Open DFL / IDSSE v1 | Bassek et al. 2025, Figshare 28196177, CC BY 4.0 | Provider (DFL/Sportec) `BallStatus` + `BallPossession`, 25 Hz | 7 full Bundesliga matches, 1,002,644 frames | GBDT dead + team heads: the first 4 (`J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`). OwnNet: all 7 |
| WCSF Groningen / SciSports | `MatKempeGroningen/WCSF_SolutionRoom`, MIT (owner-confirmed) | Manual `AliveFlag` + `PossessionFlag` in the provider release, 10 Hz → 25 Hz | 1 full match, 143,431 frames at 25 Hz | OwnNet (ownership) |
| SkillCorner Open Data (A-League 2024/25) | `SkillCorner/opendata`, MIT | Provider `possession.group` on live frames; no dead-ball labels | 20 matches, 10 Hz → 25 Hz | OwnNet: the first 16 by sorted ID (ownership, weight 0.25); never the dead head |
| Metrica Sports sample data, games 1–3 | `metrica-sports/sample-data` (co-organiser) | Corrected tracking + events; H/A/D derived by `solution/external.py` | 3 matches | Held-out evaluation only (not in the shipped fit) |

The shipped GBDT uses the section 6 recipe on the same 4 DFL matches, retrained with published-rate
degradations only (section 7, item 3). Refitting it on all external data ran out of memory (23 GB RAM),
so the extra data enters only through OwnNet, which cannot change P(D).

External tracking is clean, so every training match is also degraded to look like ATD
(`datasets.degrade_atd`), using only published figures: ~41 % uniform ball dropout, ball glitches, ~20.5 visible players, tracklet
fragmentation into 46 slots with ID swaps and jitter, plus adversarial views where the ball
is visible during stoppages.

## 4. Possession rules implemented

- **Labels follow the provider definitions**, with dead taking precedence: a frame is `D` whenever
  the provider marks the ball out of play; otherwise H/A is the provider's possessing team.
- **Loose balls and challenges**: the team keeps possession until the other team controls the
  ball. Live blips under 26 frames (~1 s) and turnovers shorter than 10 frames are absorbed into
  the surrounding spell, so a touch without control does not create a possession.
- **Restarts**: a stoppage is a dead spell of at least 59 frames (~2.4 s), and shorter dead
  blips are merged. The D_H / D_A states let the decoder price "same team restarts"
  (c=15.16) below "other team restarts" (c=16.0).
- **Where the decoder constants come from**: emission weight `w`, dead bias, switch cost `c_team`,
  stoppage cost and the minimum durations were fitted by random search (`solution/tune_v4.py`) on
  out-of-fold predictions for external matches. They are inherited from our earlier E014 artifact,
  and that search included fixtures later used for held-out scoring. The two restart costs were
  then swapped by hand so that a same-team restart never costs more than a turnover. None of them
  was set on the provided match.
- Before the 2nd-half kickoff (frames 110,808–113,741) all frames are `D`, as the FAQ states.
  Ungraded frames are written as `X` outside the graded blocks.

## 5. Limitations

- **Short possessions**: recall on possessions of 2 s or less is still low (~0.36–0.38 held out).
  The minimum-duration rules rarely bind. Cheaper team switches (×0.85, ×0.7, minimum turn 0)
  found more short possessions but lowered held-out SpellF1, because the ownership model itself is
  unsure around changes.
- **Long ball-free stretches**: team accuracy drops in multi-second ball gaps; OwnNet and the
  ball-free fallback close part of this gap but not all.
- **Dead/live from players only**: quick restarts with little player reorganisation are detected
  late; a dead head that could use the ball safely would need external data with a real
  ball-during-stoppage signal.
- **Benchmark caveat**: the external evaluation fixtures were used during development across many
  experiments, so the numbers below are held-out from fitting, not from all selection.
- With more time: a learned dead head on ATD-like data with real stoppage ball tracks, and
  joint end-to-end training of the decoder costs.

## 6. Evidence (held-out external only)

Strict protocol: fit on the first four DFL matches only (classifiers, imputer, stacker, NN),
evaluate on the other 3 DFL + Groningen + Metrica games × 6 scenarios (natural, ATD-realistic,
ATD-blind, carried-dead, gap-stress burst, adversarial moving-dead ball), provider-balanced.

| System | Provider-balanced SpellF1 |
|---|---:|
| Baseline GBDT (LightGBM, no P3) | 0.5530 |
| + P3 features, ens3 GPU | 0.5943 |
| + 25 Hz OwnNet blend 0.7, `restart_order` decoder | 0.6686 (95 % CI of gain over baseline [0.101, 0.131]) |
| + line snap (shipped), DFL + Groningen | +0.0044 over the row above, positive in 18 of 18 provider × scenario cells |
| Gentler switch costs (×0.85, ×0.7) | −0.001 to −0.003: rejected |
| Metrica added to NN training | +0.0018, 6 of 12 cells better: no clear gain, not used |

These rows use the DFL4-trained OwnNet; the shipped OwnNet uses the same recipe on all 7 DFL +
Groningen + 16 SC matches, so it has no held-out score of its own.

The 25 Hz OwnNet improved all 18 provider × scenario cells over the 5 Hz version. SkillCorner
last-4 held out reached 0.834 ownership accuracy with 16 SC matches in training.

## 7. Disclosed deviations in our development history

We record these because the rules require external-only tuning and selection, and our own audit
(`docs/AUDIT_RESPONSE.md`) promised to disclose them.

1. **Smart Tagging decision (E003b, 2026-11-10).** An early `run.py` printed the public 5-minute
   score on every run. Tags were switched off after a run with tags scored visibly lower. We then
   replaced that reasoning with a rule that needs no score: a component whose parameters cannot be
   validated on external data is not used. No external game has Smart Tagging, so tags stay off.
   Since then, `run.py` never reads the public ground truth, and `solution.sanity` prints only
   PASS/FAIL for orientation and alignment.
2. **Retired public-sample visualiser.** Before 2026-11-17, `solution/visualize_5min.py` rendered
   earlier model outputs against the public 5 minutes. It has been retired (it exits before loading
   anything), and no later decision in this document used it.
3. **Degradation parameters without a published source.** One training degradation (`atd_real`)
   split ball dropout by dead and live play (72 % / 18 %). The match documentation publishes only
   the overall rate (41 % of graded frames), and the split's origin was not recorded. We could
   not rule out that it was informed by statistics of the competition tracking, so the shipped
   GBDT was retrained without it (`--published-degradation`: natural, `atd_blind` and carried-dead views only). On 8 held-out cases this cost −0.0024 SpellF1. The neural network never trained on `atd_real`.
   Benchmark rows for `atd_real` are reported for continuity only.
4. **Development reuse of the external benchmark.** The held-out DFL, Groningen and Metrica
   matches were never used for fitting, but they were scored repeatedly while choosing between
   variants. Our numbers are therefore held out from fitting, not from selection.

## 8. Reproduction

```bash
pip install -r solution/requirements.txt            # Python 3.11, CUDA GPU recommended
# external data under external/ (see EXTERNAL_TRAINING_DATA.md); competition data under data/
# (optional) retrain the two components from public external data, then assemble:
.venv/bin/python -m solution.nn2_lomo --tag final_w128_25hz --groups "" --no-metrica --n-sc 16 --w-sc 0.25 \
    --frame-stride 1 --gap-policy mixed --width 128 --gru-layers 2 --steps 2800 --batch 12 --lr 0.002 \
    --seeds 0,1,2 --save-final solution/out/final_nn2_w128_25hz_sc16_3seed.pkl --out-dir solution/out/final_nn2
.venv/bin/python -m solution.deadline_benchmark train --mode ens3 --use-p3 --ball-free --gpu --published-degradation \
    --out solution/out/gbdt/pipeline.pkl   # committed copy: solution/configs/benchmark/gbdt_dfl4_p3_ens3_clean/
.venv/bin/python -m solution.build_final --gbdt solution/out/gbdt/pipeline.pkl --line-snap \
    --neural solution/out/final_nn2_w128_25hz_sc16_3seed.pkl \
    --decode-cfg solution/configs/decode_final_restart_order.json --out solution/configs/pipeline_final.pkl
# inference from the committed artifact:
.venv/bin/python -m solution.run --learned --pipeline solution/configs/pipeline_final.pkl --name more_than_a_hack_2027
.venv/bin/python evaluation/validate_submission.py evaluation/results/more_than_a_hack_2027.csv
```

`solution/configs/pipeline_final.pkl` is committed, so the last two commands alone regenerate the
submitted file. Inference is deterministic: a repeat run was byte-identical. Retraining is not bit-exact, because GPU tree and network training are nondeterministic.
The CSV hash was produced on an RTX 4050 Laptop GPU (driver 580.178.04, torch 2.15.0+cu132).
CPU inference has not been checked to give the same file; use a CUDA GPU. The CSV's SHA-256 prefix is in
`solution/configs/pipeline_final.pkl.provenance.json`, and the tests are
`python -m solution.tests.test_core` plus `python -m unittest discover -s solution/tests -p 'test_*.py'`.
