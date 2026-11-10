# Response to the external audit (2026-11-10)

An audit of the pipeline as it stood after E005 raised nine findings. This file records, for each,
whether it was correct, what was changed, and what remains. Verdicts are ours; where the audit
overstated something we say so and say why.

All numbers quoted before this audit (E001–E005, benchmark v1/v2) are **superseded**. They are
kept in `solution/experiments_v1v2_superseded.csv` for the record and must not be quoted.

| # | Finding | Verdict | Action |
|---|---|---|---|
| 1 | Smart Tagging was switched off after seeing the public 5-minute score | **Correct** | Process fixed; decision re-made on a rule that does not need the score; disclosed |
| 2 | Unpublished statistics of the target match were computed and influenced the benchmark | **Correct** | Statistics removed; log corrected; benchmark no longer depends on the idea |
| 3 | Dead-ball simulation and dead-ball cues were circular | **Correct, the most serious** | Ball removed from all dead-ball evidence; benchmark v3 |
| 3b | "Caused 43% D on the real match" | Not supported | See below |
| 4 | Short-spell merging in labels and in the model is circular | Partly correct | Benchmark now scores against labels with and without the rule |
| 5 | Parameters tuned in-sample, in-sample score used as headline | Partly correct | Held-out is now the headline everywhere |
| 6 | Team evidence leaks into the alive-vs-dead decision | Correct as a design flaw | Decoupled variant added and tested |
| 7 | Overlapping tags stack without bound | **Correct** | Capped. Tags remain off |
| 8 | `absorb_short` turns live play into dead ball | **Correct** | Fixed, with tests |

## 1. Selection on the public ground truth (E003b) — correct

What happened: `run.py` printed the public-window score on every run. Two files were produced,
tags on and tags off; the score was visibly lower with tags on; tags were switched off. The log
said the decision rested on an argument about weight sizes, and admitted the score prompted the
second look. An argument found after seeing the answer is not independent of the answer. This was
a selection influenced by the ground truth, which the rules forbid.

What cannot be undone: we have seen that number.

What was changed:
- `run.py` no longer reads the public ground truth at all.
- `solution/sanity.py` replaces it and prints only PASS/FAIL for two questions (H/A orientation,
  frame alignment). No score reaches the screen.
- The tagging decision is now governed by a rule that does not need any score:
  **a component whose parameters cannot be validated on external data is not used.** No external
  game has Smart Tagging, so tags are off. The same rule would have excluded them before any run.
- This episode goes into the technical document as a disclosed deviation.

Still open, for the team to decide: whether to report this to the organisers proactively. We
recommend doing so alongside the two questions already in DATA_CONTRACT.md.

Note on the audit's own evidence: its finding 7 quotes D recall falling "from 84.0% to 35.0%".
Those figures can only come from scoring against the public ground truth. We have not used them
and they should not guide any change.

## 2. Unpublished target-match statistics — correct

`atd.summary()` printed the share of visible ball positions outside the pitch and the share of
tracks with a speed value. Neither is published. Having seen that the ball is sometimes tracked
outside the pitch, we changed the simulation (E004) so the simulated ball also leaves the pitch.
The E004 entry called this "a reasoned assumption, not fitted to any tracking file". The size of
the effect was not fitted, but the idea came from the statistic, so that sentence was misleading.
It is corrected in the log.

Inspecting the provided data is allowed by the rules; shaping our tuning benchmark with it broke
our own contract (section 2: degradation uses published figures only).

What was changed: both statistics removed from `atd.summary()`; it now prints only figures the
organisers publish. Under finding 3 the benchmark no longer rewards any ball-based dead-ball cue,
so nothing tuned today depends on that observation.

## 3. Circular dead-ball simulation — correct

The external games have no ball during stoppages. We invented a ball path for those frames, then
tuned `b_oob` and `b_still` (and trained the E005 classifier) on data containing that invention.
The model was rewarded for recognising our own simulation. The alive/dead F1 of 0.937 and part of
the 0.525 and 0.655 spellF1 came from this and are not evidence of anything.

One correction of detail: the simulated ball was not at exactly 0 m/s (position noise was added
afterwards), but the tuned threshold `v_still = 2.1 m/s` sat just above that noise, so the
substance of the finding stands.

What was changed:
- **No model may use the ball to decide dead vs alive.** Rule-based: `b_oob`, `b_still`, `b_miss`
  removed; dead-ball evidence is mean player speed over 1 s and 6 s. Learned: split into a dead
  model that sees player kinematics only and a team model trained on live frames only, where the
  ball track is real.
- **Benchmark v3** runs three different invented stoppage pictures (ball fetched and placed; ball
  stays where it stopped; ball never tracked). A model that still depends on the invention shows
  a spread across the three; the per-picture scores are logged for every experiment.

Cost: real ball cues for dead ball (a ball tracked outside the pitch is probably informative on
the real match) are given up, because we have no legitimate way to validate them.

### 3b. "This caused D on 43% of graded frames" — not supported
The true dead-ball share of the full match is unpublished; the external games have 32–41%. The
43% may be too high, but there is no legitimate evidence either way, and "possessions ×0.65" is a
separate effect (merged hand-overs, per the oracle analysis), not a dead-ball one.

## 4. Short-spell merging — partly correct

The proxy labels merge opposing-team stints under 2 s with no completed pass (our reading of
"a touch without control is not a change of possession"). The model's `min_spell` was then tuned
against those labels. This is not leakage, but the audit is right that the model was being tuned
to our own reading of the definition: if the real ground truth does not merge such touches, we
would under-segment.

Size of the effect: the rule removes 58 of 385 possessions in game 1, 66 of 376 in game 2, 16 of
410 in game 3.

What was changed: every benchmark case is scored against both label variants, and the tuning
objective is the mean of the two. Both scores are logged separately.

## 5. In-sample tuning — partly correct

Not correct that there was no held-out test: E003 had a leave-one-game-out check (0.489) and E005
was leave-one-game-out for the model and the decoder (0.651). Correct that the `spellF1` column
and the summary tiles showed the in-sample number, and that the E003 parameters re-scored on v2
(0.525) had no held-out figure at all.

What was changed: `experiments.csv` has a `heldout` column, printed first; the tuner always runs
the leave-one-game-out loop; docs and the notebook quote held-out numbers.

## 6. Team evidence leaking into alive-vs-dead — correct as a design flaw

Scores were `+l/2` for the favoured team and `-l/2` for the other, so the best live state scored
`+|l|/2` against the dead state: clear team evidence also counted as "ball alive". That is a
defensible model in open play and wrong at a throw-in, where one player holds the ball. It also
explains why stacked tag priors (finding 7) suppressed dead ball.

What was changed: `decouple=1` scores the favoured team 0 and the other `-|l|`, so only the
dead-ball evidence decides alive vs dead. Both variants were tuned and compared held-out (E007).
The learned model combines proper probabilities and never had this issue.

## 7. Tag stacking — correct
Overlapping tags were summed. Now clipped to one tag's weight. Tags remain off (finding 1).

## 8. `absorb_short` — correct
A short live spell straight after dead ball was overwritten with D. Now it joins a neighbouring
live spell, and is left alone if dead ball is on both sides. Tests added in
`solution/tests/test_core.py`.

## Results after the fixes (benchmark v3, held-out)

| | Before the audit (superseded) | After |
|---|---|---|
| Organisers' baseline ±50 | 0.243 | 0.250 |
| Rule-based | 0.525 in-sample (0.489 held-out on v1) | **0.432** |
| Learned | 0.655 (0.651 held-out) | **0.551** |

The fixes cost about 0.06 for the rule-based model and 0.10 for the learned one. That is the part
of the earlier scores that came from recognising our own stoppage simulation. Both models still
beat the baseline clearly, and the learned model's score now varies by only 0.017 across the
three stoppage pictures. Details: RESEARCH_LOG.md, E006–E008.
