# Priorities 3–4: feature experiments and clean training

Reviewed and implemented on 2026-11-17. These changes do not replace the shipped
`pipeline_v4.pkl` or claim a score improvement. Performance decisions require the
independent external experiment report; invariant tests establish correctness,
not predictive quality.

## Existing Priority 3 work retained

The incoming working tree already contained optional `use_p3` ownership features:

- 45 strictly ball-free opposing centroid-axis motion columns: team motion,
  directional agreement, spread and displacement at two time scales, with
  rolling context. These implement the idea in Elvern's `04.48`.
- 10 fixed Laplacian deblurring and ball-trajectory kink columns. This is a
  Laplacian recipe; it is not a reproduction of COOKED's full SG/MA pipeline.
- A separate ball-free ownership classifier with continuous routing from 2 to
  12 seconds of missing ball, up to 0.65 mixture weight, further attenuated by
  ball confidence. Its constants are experimental external-validation choices,
  not measurements from the competition match.

`use_p3=False` preserves the existing 156-column ownership feature schema. The
optional features append 55 columns. The dead feature builder remains ball-free.
`ball_free_team_cols` selects exactly the 45 centroid-axis columns, excluding all
ball-dependent deblurring columns.

One gap-router correctness bug was fixed: an entire block without a visible
ball had a gap length of zero. It now has the block's length. Leading, trailing
and internal gaps retain their existing treatment.

Optional augmentation now covers 5–30 second hidden-ball bursts. A second
external-training-only recipe adds fixed 21-frame degree-2 Savitzky-Golay
smoothing within finite ball runs, and hide/hold/drift handling within labelled
stoppages. This addresses the frozen-dead-ball shortcut described in MYA S9.
Both recipes are off by default, so they can be ablated individually. Neither
updates player data or labels, fits parameters to inference data, or reads the
competition fixture.

## Priority 4 corrections

`solution/train_v4.py` now exposes:

```python
pipeline, provenance = fit_pipeline(
    raw_train_matches,
    skillcorner_ids=(),
    mode="lgbm",
    use_gpu=False,
    use_p3=False,
    include_sc_stage2=True,
    seed=0,
)
```

The caller supplies the complete training fixture set. All evaluation fixtures
must be excluded, including any auxiliary provider used for a transfer test.
The trainer rejects unregistered source metadata, missing external labels, and
competition Smart Tagging. These checks supplement the data contract; they do
not establish dataset rights on their own.

All scenarios of one fixture belong to one inner fold. For every Stage-2 OOF
prediction, both the imputer and the Stage-1 classifier exclude that fixture's
fold. Training features are recomputed through that fold's imputer. Merely
changing the imputer used to predict the validation rows would leave leakage in
the Stage-1 training features. A separate all-training-data imputer and Stage-1
fit produce the final inference model.

SkillCorner participates in the Stage-2 **ownership** fit by default. Its
`has_real_dead_ball=False` metadata continues to exclude it from both dead heads.
`--no-skillcorner-stage2` disables its Stage-2 contribution for an explicit
ablation; `--sc-s2-weight` scales its ownership supervision. This does not claim
SkillCorner improves scores. `--no-metrica` excludes Metrica from all fits,
including every imputer. Metrica's documented rights caveat remains separate from
its predictive utility.

Seeds are stable SHA-256 functions of the base fixture, scenario and run seed.
The old repeated `external.degrade(..., seed=0)` pattern is removed from this
trainer. Optional long-burst and stoppage-motion scenarios get distinct seeds.

`external.degrade(..., tracking_version="unified_v4")` fragments player tracklets
into spare team-preserving slots and recomputes speeds from the corrupted
positions through `datasets.compute_speed`. It does not reuse clean speeds.
The historical `tracking_version="legacy"` remains the function's default so
older Metrica benchmark calls retain their historical meaning. The **new
trainer** explicitly defaults to `unified_v4`. These are different recipes and
must be identified in comparisons.

Every trained pipeline carries `training_provenance`; the CLI also writes
`<model>.provenance.json`. It records source fixture names, scenario seeds,
inner-fold Stage-1/imputer/OOF groups, feature and augmentation flags, Metrica
tracking version, sampling stride and model family. `--save-oof` optionally
writes external-only Stage-1 probabilities and base features locally for
training-only decoder analysis. OOF files are not test predictions or an
independent score of the Stage-2 model fitted on them.

## Bounded runs and ablations

Example independent-split training on the first four DFL fixtures only:

```bash
MTAH_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -m solution.train_v4 \
  --dfl-games J03WMX J03WN1 J03WOH J03WOY \
  --no-groningen --no-metrica --sc-limit 0 --mode lgbm --cpu \
  --out solution/out/deadline_dfl4_baseline.pkl
```

Keep the same fixture split and seeds when separately adding `--use-p3`,
`--ball-free-fallback`, `--long-bursts` and `--stoppage-augmentation`. To test
SkillCorner Stage-2 handling, include the same `--sc-limit` on both runs and vary
only `--no-skillcorner-stage2`. To test Metrica inclusion, vary only
`--no-metrica` and report that Metrica ceases to be a transfer holdout when it is
included. Repeatedly selecting on WPY/WQQ/WR9 does not make those matches a fresh
test set, even when none is fitted.

`solution/ablate_sc_stage2.py` provides a cheaper, more tightly controlled
Stage-2 ablation after a DFL4 baseline and its saved inner OOF rows exist:

```bash
MTAH_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m solution.ablate_sc_stage2 fit \
  --baseline <baseline.pkl> --oof-dir <baseline-inner-oof-directory> \
  --sc-limit 2 --out solution/out/deadline_sc_stage2.pkl
MTAH_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m solution.ablate_sc_stage2 evaluate \
  --candidate solution/out/deadline_sc_stage2.pkl \
  --out solution/out/deadline_sc_stage2_results.json
```

It retains both dead heads, the final Stage-1 model, and the decoder exactly.
The DFL Stage-2 examples are the original fixture-group OOF rows. Additional
SkillCorner examples are predicted by that final DFL4 Stage-1, which has never
fitted SkillCorner players, balls or labels. Only the ownership Stage-2 model is
refitted. This isolates SkillCorner Stage-2 supervision; it is not an ablation of
adding SkillCorner to the imputer or Stage-1. Its provenance explicitly records
that DFL and SkillCorner examples have different, independently fitted Stage-1
ancestors. It evaluates frozen natural and two independent ATD-real masks on
WPY/WQQ/WR9 plus Groningen and asserts bit-for-bit identical dead probabilities.
Results remain pending until the baseline and this paired run complete.
Optional `evaluate --sc-eval-limit 4` also checks the last four SkillCorner
fixtures, rejecting overlap with the Stage-2 additions. Since SkillCorner has no
valid dead labels, this section reports only conditional ownership accuracy,
balanced accuracy and conditional log-loss on every labelled H/A frame. DFL and
Groningen reports include strict ownership accuracy on all GT-live frames so
predicting dead cannot hide ownership errors.
The candidate manifest predeclares descriptive confirmation margins: at most
0.01 mean conditional-accuracy drop and at most 0.03 worst-fixture drop. Each
fixture first averages its natural and ATD-real views, then fixtures receive
equal weight. These guards do not turn historically reused fixtures into a new
test set or establish statistical significance.

`solution/p3_removal_diagnostics.py` supplies a different, cheaper diagnostic:
full frozen P3 versus its router removed, its centroid-axis columns replaced
with training-only medians, and its deblurring columns replaced with training-only
medians. All heads remain fitted and each external view's expensive feature
extraction is cached. It scores DFL3/Groningen natural, two ATD-real masks and the
matched long-burst stress condition. It verifies unchanged dead probabilities,
uncached versus cached feature-path equality on one case, and cached
decode-versus-predict equality for all full-model views.

```bash
MTAH_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m solution.p3_removal_diagnostics \
  --pipeline <p3-model.pkl> --out solution/out/p3_removal_results.json
```

This is explicitly a **module-removal diagnostic**, not a causal retraining
ablation. The Stage-2 head learned routed P3 inputs; removing the router or
neutralizing feature groups changes that distribution. A score change indicates
reliance or sensitivity of this particular fitted combination, not what a
separately trained baseline-plus-component model would achieve. A baseline/P3
comparison also changes axes, deblurring and routing together and cannot isolate
their individual contributions.

## Validation and historical interpretation

`python -m solution.tests.test_training_hygiene` passes seven focused tests:
fixture-group exclusion of the complete preprocessing pipeline; Stage-2
SkillCorner isolation from dead labels; seed independence; corrupted-track speed
recomputation; exact P3 prefix/swap/ball-free feature invariants; full-block gap
length; and deterministic augmentation that preserves labels and player arrays.
All use synthetic fixtures. No competition data, GT, video, or per-frame output
was opened.

The corrected trainer is a new recipe, not a byte-for-byte reproducer of the old
31-fixture E014 artifact. The old trainer used a globally fitted imputer for its
Stage-2 OOF construction and excluded SkillCorner there. Existing scores from a
different evaluation harness also need their own full preprocessing and
higher-order stacking audit; inner OOF alone cannot establish an outer held-out
claim. Preserve the legacy artifact and label old scores by their actual
protocol instead of silently replacing the historical record.

One additional speed-mask issue was identified and deliberately left outside the
currently running numeric recipe: `datasets.compute_speed` can return a finite
central-difference speed when the current player's position is missing but its
neighbours are finite. Most features gate on the current position; constellation
mean/max-speed summaries currently sum the speed array directly. Hardening this
requires masking such speeds or their summaries and recording a new recipe
version before rerunning comparisons. Do not mix the change into an active
experiment or imply the current unified-speed fix also resolves this separate
masking issue.
