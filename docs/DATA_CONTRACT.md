# Data contract

What each data source may be used for, so that nothing we submit breaks the competition rules.
Authority, in order: the organisers' clarification email (bottom of `RULES.md`), then `RULES.md`,
then the other docs. If a use is not clearly allowed here, it is not allowed until we ask.

## 1. Sources and permitted use

| Source | Where | Algorithm input | Tune / train / select on it | Notes |
|---|---|---|---|---|
| BAR-ATH tactical tracking (ATD) | `data/tactical/*_atd.*` | yes | **no** | Unsupervised fitting on it is an open question (see 5). Default: no. |
| BAR-ATH Smart Tagging | `data/tactical/*_smart_tagging.xml` | not used | **no** | No external game has tags, so its weights cannot be validated. |
| BAR-ATH homography | `data/*/*_homography.csv` | not used | no | |
| BAR-ATH broadcast tracking | `data/broadcast/` | not used | no | Not frame-aligned; dropped. |
| BAR-ATH video (both) | `data/*.mp4` | not used | **no** | We do not decode it, train on it, or check output against it. |
| Public 5-min ground truth | `evaluation/gt/*_public.csv` | no | **no** | Correctness check only (see 3). |
| Private ground truth, event data or any human-made data about this game | n/a | **never** | **never** | |
| Metrica sample games 1–3 | `external/metrica/` | n/a | **yes** | Other games. The only tuning and model-selection data. |

Published facts about the match that we do use as constants, because the docs state them:
graded blocks (11,185–90,651 and 110,808–190,197), the second-half kickoff at 113,742 with all `D`
before it, home = Barcelona, pitch 105.3 × 68.0 m, ball missing on 41% of graded frames and about
20.5 tracked objects per frame (used only to shape the degradation of external games).

No other constants tied to specific frames of this match.

## 2. External data register

| Dataset | Source | Licence / terms | Labels | Amount |
|---|---|---|---|---|
| Metrica Sports sample data, Games 1, 2, 3 | https://github.com/metrica-sports/sample-data (downloaded 2026-11-10) | No formal licence. README: "be responsible with the use of this data" and acknowledge the source if used publicly. Anonymised. | Corrected tracking and event data produced by Metrica. Our H/A/D labels are derived from them by `solution/external.py` (no hand labelling by us). | 3 matches, about 287 minutes |

How the proxy labels are derived: `D` where the corrected tracking has no ball; otherwise the team
of the most recent on-ball event (pass, carry, shot, recovery, set piece). A stint under 2 s by the
other team with no completed pass, carry or shot, inside one team's possession, is treated as an
uncontrolled touch (CHALLENGE.md: a touch without control is not a change of possession).

The external tracking is clean. `external.degrade()` makes it resemble automatic tracking using
only the published figures above. The external games have no ball during stoppages; whatever the
benchmark puts there is invented, so no model may use the ball to decide dead vs alive, and the
benchmark runs three different invented pictures to expose any remaining dependence.

## 3. What the public 5 minutes are for

Allowed: confirm frames are aligned and H/A are the right way round.
`python -m solution.sanity <csv>` answers those two questions with PASS/FAIL and prints no score.

Not allowed: looking at a public-window score at all while variants are being compared.
`run.py` printed one on every run until 2026-11-10, and the Smart Tagging decision (E003b) was
influenced by it. That was a breach; see `AUDIT_RESPONSE.md` finding 1. Every keep / drop
decision must cite held-out external-benchmark rows from `solution/experiments.csv`.

Two standing rules that follow:
- A component whose parameters cannot be validated on external data is not used.
- No unpublished statistics of the competition match are computed. `atd.summary()` prints only
  figures the organisers publish.

## 4. AI-assistant boundary

The terms forbid uploading the provided data to AI services. Claude Code is a hosted AI, so:

- Scripts that read `data/` or `cache/` print **aggregates only**: scores, counts, percentages,
  summary statistics. Never rows, positions, tag intervals or per-frame labels.
- The assistant does not open files under `data/`, `cache/`, or per-frame CSVs in
  `evaluation/results/` and `evaluation/gt/`.
- No video frames are extracted or viewed.
- External games are public and may be inspected freely.

## 5. Open questions for the organisers

1. Is unsupervised fitting on the provided **tracking** allowed (for example estimating noise
   levels or gap-length statistics from the ATD)? The email forbids fitting on the ground truth
   and the video; the original rules allowed unsupervised learning on the game.
2. Is it acceptable for an AI coding assistant to see aggregate outputs (scores, counts) of local
   runs on the provided data?

Until answered: no for 1, and 2 follows section 4.

## 6. Checklists

Before each experiment:
- [ ] The comparison is made on the external benchmark (`python -m solution.bench`).
- [ ] No new constant comes from looking at BAR-ATH frames or the public ground truth.
- [ ] Any new external dataset is added to section 2 first.

Before submission:
- [ ] `git status` shows no files from `data/`, `cache/`, `external/`, or result CSVs.
- [ ] A clean clone plus the data bundle regenerates the identical CSV (same sha256).
- [ ] `python evaluation/validate_submission.py <team>.csv` passes on the exact upload.
- [ ] The technical document lists every dataset in section 2 and the proxy-label derivation.
- [ ] Dependencies are pinned.
