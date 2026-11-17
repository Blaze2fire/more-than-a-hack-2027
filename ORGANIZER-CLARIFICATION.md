# Initial Challenge: controlling organizer clarification

Recorded **2026-11-04** from an organizer email shared by a participant. This
summary records the clarification for the Initial Challenge.

**Precedence:** the organizer explicitly states that this clarification takes
precedence over previous documentation, emails and individual answers. Apply it
where older repository text differs.

## Application to this repository

- Inspecting inputs/video to understand possession is allowed. Comparing predicted
  labels, spells, transitions or output overlays with the provided video is
  prohibited, even without writing manual corrections. Do not use visual output
  review as debugging, validation or parameter selection.
- All training, retraining, fitting, calibration and parameter tuning require
  external data. This includes rule thresholds, decoder/duration costs and
  self-supervised or pseudo-label training: automatically derived labels/features
  from the provided match do not turn it into external training data.
- Provided video, automatic tracking, Smart Tagging and generated data may be
  inference inputs. Fixed inference calculations, deterministic decoding and
  automatic engineering checks are distinct from fitting/tuning.
- Do not upload provided data to AI services. Earlier wording about written
  authorization for hosted LLMs does not provide an exception to this email.
  Keep provided-data processing local; train remote notebooks only on licensed
  external data. External pretrained models may run locally on permitted inputs.
- Automatic checks remain allowed. Retain the public sample's existing restriction
  to frozen-pipeline correctness checks; scores must not guide selection/tuning.
- Intermediate-step files are allowed. Keep storage/sharing consistent with
  confidentiality, licensing and the existing restriction on public redistribution;
  this permission is not a blanket release of the supplied assets.
- External datasets need verified licenses/rights and documented provenance.
  Human-produced data about this same fixture remains prohibited.
- Existing hand-set constants are not proof of prohibited tuning. Record their
  origin; if they need tuning, use external data. A fixed unvalidated heuristic
  is not evidence of external validation or predictive quality.

The requirement to use remote notebooks for training is a **team convention**,
not a requirement stated in this email. It continues to apply to training on
permitted external data. (implementation contract and audit kept in the team's internal notes).
