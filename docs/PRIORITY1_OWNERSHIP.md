# Priority 1: ownership sequence stream (2026-11-17)

The live ownership branch predicts `P(H | live)` and a team-symmetric change
probability. It has no dead-ball output. `PipelineV4` keeps its ball-free dead
heads and blends only the conditional H/A probabilities.

## What was already present

The uncommitted `neural_team.py` and `neural_v2.py` implementations and their
cached predictions were recovered before making changes. The v2 model uses
player-set pooling, a dilated residual TCN and a bidirectional GRU. It builds
last-seen/next-seen ball positions and gap timers inside each training crop,
allowing a fresh missing-ball pattern on every iteration. Mixed augmentation
includes 5–30 second bursts. The targets retain native short spells, with extra
loss weight on spells of at most one/two seconds.

The old `solution/out/nn2_dev1.log` reported a 0.7 neural blend improving three
DFL matches' `atd_real` SpellF1 from 0.6004 to 0.6713, WPY burst from 0.4695 to
0.6377 and Groningen `atd_real` from 0.5896 to 0.6626. These are **exploratory
results, not strict held-out evidence**: the associated `oof_G0` GBDT cache has
outer-fold leakage in its stage-two training inputs. The neural component was
trained on four different DFL fixtures plus Metrica 1–3, but no trained weights
or run manifest survived. These results do not justify promoting a submission.

## Changes and guardrails

- A shared `solution/ownership_blend.py` copies the original dead log-probability
  exactly and uses a centered H/A difference plus dyadic quantization to keep
  exact team-swap equivariance. Zero neural weight is an exact no-op.
- Optional `frame_stride=5` reduces CPU training/inference to 5 Hz. Ball
  derivatives use the effective sampling rate, player velocities remain in
  metres/second, and gap-age channels retain the original 25 Hz time units.
  Inference interpolates predictions back to 25 Hz within each contiguous block.
  Existing models retain the previous stride-one behavior.
- `nn2_lomo` accepts an explicit training-source whitelist, validates that the
  requested sources exist, saves trained weights for every group, and records
  arguments, training/held-out fixtures, a source fingerprint and model checksum.
  Public external feature caches are keyed by preprocessing source fingerprint.
- `solution/tests/test_neural.py` verifies exact dead preservation, exact H/A
  swap at both sampling rates, finite inference with no ball observations,
  sampling-rate-correct velocity and long-gap generation. The existing core
  guardrail tests also pass.

The unchanged dead **emissions** are a precise invariant. A joint Viterbi decoder
can still move decoded dead intervals when H/A emissions or transition costs
change; final-mask independence is a separate, stronger property and is not
claimed by the ownership blend alone.

Compaction retains at most 11 observed players per team. Pooling is insensitive
to the order of the retained players, but frames with more than 11 simultaneous
finite tracks are capped using stable input-slot order. Duplicate/excess-track
handling therefore remains a separate hardening opportunity; full raw-slot
permutation invariance is not claimed for that case.

## Strict first-four-DFL ablation

Both candidate models train on WMX, WN1, WOH and WOY only, with native labels.
They use no Metrica, SkillCorner or Groningen training examples and no learned
ball imputer. WPY, WQQ, WR9, Groningen and Metrica 1–3 are excluded from every
fit. This is a held-out comparison of a fixed training split, not seven-fold
LOMO or a benchmark against untouched participant development fixtures.

Both runs use width 64, one bidirectional GRU layer, stride five, seed zero,
1,200 optimizer steps, batch size eight and two CPU threads. The mixed run uses
the existing short/long/missing-not-at-random gap recipe; the short run samples
short gaps only. This ablates the **gap recipe**, not duration alone: missing
fractions and training crop draws may also differ.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m solution.nn2_lomo \
  --tag deadline_mixed --groups '4,5,6,g' \
  --train-keys dfl0,dfl1,dfl2,dfl3 --no-metrica \
  --frame-stride 5 --width 64 --gru-layers 1 --batch 8 --steps 1200 --threads 2 \
  --out-dir solution/out/deadline_neural
```

For the matched short-gap run add `--gap-policy short` and use a different tag
and output directory (`solution/out/deadline_neural_short`). Weights, manifests,
predictions and logs stay in ignored `solution/out/`; the promoted model remains
unchanged. The strict baseline and ownership-only/change-head comparisons are
recorded in the main deadline benchmark documentation when complete.

## Additional provider confirmation

Before reading confirmation scores, the final four IDs in the sorted current
SkillCorner inventory were fixed: **2013725, 2015213, 2016236, 2017461**. Both
first-four-DFL neural models exclude all SkillCorner games from fitting.
`solution/nn2_confirm.py` registers these fixtures and the fixed views
`natural/911`, `atd_real/911`, `atd_real/2027` before scoring. It verifies model
manifests and checksums, then reports only native known-live ownership accuracy,
Brier score and negative log likelihood. It reports no dead F1 or fabricated
SkillCorner dead labels. Match/view means give fixtures equal weight. These
scores are confirmation evidence and must not be used to tune parameters.

The trained mixed checkpoint also passed a public WPY slice check: exact H/A
ownership and switch-probability swap differences were both zero, and all-ball-
missing inference remained finite. Evidence is stored in the ignored
`solution/out/deadline_neural/learned_invariants.json`.

A separate **unpromoted** candidate uses the same fixed mixed recipe and 1,200
steps on all seven DFL games plus Groningen, with no Metrica or SkillCorner.
Training completed in 581 seconds. Its 214,082-parameter checkpoint is
`solution/out/deadline_neural_full/models_all.pkl`; all fitted sources are
declared in `manifest_all.json`. Its fit scope differs from the held-out
comparison, so training-fixture scores from this artifact are not generalization
evidence. No competition inference was run for this candidate.

## Completed confirmation and ownership diagnostic

The predeclared SkillCorner confirmation completed on all four fixtures and
three views. These are match/view means, with no fitting or subsequent parameter
selection from this confirmation.

| Neural gap recipe | View | Conditional team accuracy | Brier (lower is better) | NLL (lower is better) |
|---|---|---:|---:|---:|
| Mixed, including long gaps | Natural | 0.7784 | 0.1699 | 0.6061 |
| Short gaps only | Natural | 0.7760 | 0.1747 | 0.6388 |
| Mixed, including long gaps | ATD-real, two fixed seeds | 0.7769 | 0.1680 | 0.5848 |
| Short gaps only | ATD-real, two fixed seeds | 0.7759 | 0.1716 | 0.6142 |

Calibration improves more than accuracy. Native per-fixture ownership accuracy
ranges from roughly 0.74 to 0.80, showing a substantial cross-provider gap from
the DFL diagnostic. Four fixtures are a limited confirmation sample, and the
three views are repeated measurements of those fixtures. SkillCorner has no
native dead labels, so these rows imply no dead-ball or full spell performance.
Machine-readable provenance and rows are in
`solution/out/deadline_neural_confirm/results.json`.

`solution/nn2_oracle.py` supplies an additional **oracle-dead ownership
diagnostic**. It uses the true external dead mask, resets ownership at each
true-live run, and applies a fixed H/A switch cost of 3.6, with no smoothing,
duration filtering or learned change head. This uses reference labels at
inference and is never a production pipeline or a direct participant ranking.
The cached DFL burst view is Elvern's exact `gap-stress-burst-01` mask.

| First-four-DFL neural gap recipe | DFL WPY/WQQ/WR9 natural SpellF1 mean | DFL WPY/WQQ/WR9 burst SpellF1 mean | Groningen natural SpellF1 |
|---|---:|---:|---:|
| Mixed, including long gaps | 0.7664 | 0.6948 | 0.8356 |
| Short gaps only | 0.7785 | 0.6602 | 0.8415 |

On WPY specifically, mixed gives 0.7673 natural / 0.6688 burst, while short gives
0.7831 / 0.6501. Mixed matches 31 of 124 native spells of at most two seconds
under burst masking, versus 30 for short; across the three DFL matches those
counts are 110/322 versus 99/322. The long-gap recipe improves this diagnostic's
burst performance at a cost on clean tracking. It does not establish that our
full pipeline beats Elvern or Ming. Groningen has no cached burst view in this
bounded diagnostic. Full rows and the explicit oracle protocol are in
`solution/out/deadline_neural_oracle.json`.
