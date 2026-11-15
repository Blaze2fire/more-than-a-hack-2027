# External training data inventory

Training eligibility rechecked on 2026-11-13. Paths below are relative to the local `mtah-training-data/` directory outside the competition repository. Keep these datasets separate from the provided match and its public pipeline-check sample. The competition requires external data rights, organizer access, and documented label provenance for any training, calibration, or tuning. Approach 04 fits models on Kaggle. [Competition rules](../RULES.md) · [Organizer clarification](../ORGANIZER-CLARIFICATION.md)

| Local path | Scope | Use for this challenge |
| --- | --- | --- |
| `sportec-dfl-v1/` | Seven complete fixtures and 21 checksum-verified XML files | Dense H/A/D at 25 Hz. CC BY 4.0. Used for both approach 04 heads. |
| `wcsf-groningen/` | One anonymized Dutch Eredivisie fixture, two verified position archives | Dense H/A/D at 10 Hz from SciSports manual flags. MIT, confirmed by the dataset owner. Used for frozen-model transfer first, then for both approach 04 heads in the mixture grid. |
| `skillcorner-current/data/matches/` | Twenty A-League fixtures and 81 verified data files, including the index and 20 tracking payloads | Weak H/A at 10 Hz on labeled frames. MIT. Used for approach 04 ownership only. |
| `skillcorner-legacy/data/matches/` | Nine 2019/20 European fixtures and 19 verified JSON files, including the index | Weak H/A at 10 Hz on labeled frames. MIT. Used for approach 04 ownership only. |
| `soccertrack-v2/experiment-128m/` | Five first-half GSR prefixes and corresponding BAS files | 202 sparse side-ownership anchors with player-only observations at 25 Hz. CC BY 4.0. Approach 04 used 132 training anchors. |
| `elastic-annotations/benchmark/` | Six Parquet files across three of the seven Sportec fixtures | Human-checked event timing and reception references. CC BY 4.0. Approach 04 reweighted verified DFL pass-reception neighborhoods. |
| `pathcrf-synced-events/data/sportec_synced/` | Seven Parquet files for the same seven Sportec fixtures | Event timing aligned by ELASTIC. The underlying Sportec release is CC BY 4.0. Reuse terms for the distributed Parquet remain unverified. |
| `metrica-sample-data/data/` | Three sample fixtures with aligned tracking and events. Nine verified data files. | Suitable for local format and adapter work. Competition training rights remain unverified. Events provide sparse candidate anchors. |

H/A/D means home possession, away possession, and dead ball. Approach 04's
live/dead head uses Sportec labels. The other sources in its Kaggle runs
contributed only to ownership. The competition fixture and the public
five-minute correctness sample are outside all training and tuning.

## Sportec Open DFL / IDSSE v1

The official record contains seven 2022/23 fixtures from Germany's top two divisions, `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, and `J03WR9`. Each fixture has match information, event annotations from trained human operators, and 25 Hz player and ball positions from the TRACAB Gen5 multi-camera system. The ball track supplies provider `BallStatus` (0 inactive, 1 active) and `BallPossession` (1 home, 2 away). For challenge-style labels, give inactive frames precedence and preserve unknown or missing flags as unknown. The provider defines possession through the controlling player. Its handling of loose balls and restart boundaries can differ from the challenge definition. The downloaded v1 files contain 1,002,644 ball-status rows, matching the paper's frame count. The raw XML has 10,498 top-level `Event` records. The paper reports 11,137 events and describes timestamp uncertainty. [Versioned Figshare file and license record](https://api.figshare.com/v2/articles/28196177/versions/1) · [Bassek et al. data paper](https://www.nature.com/articles/s41597-025-04505-y.pdf) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The directory holds all seven `DFL_02_01_matchinformation`, seven `DFL_03_02_events_raw`, and seven `DFL_04_03_positions_raw_observed` XML files. Sizes and MD5s match the official v1 record for all 2,627,505,889 bytes. Seven metadata files and four tracking files were linked from checksum-verified local copies. The remaining tracking files combined pinned PySport mirror downloads with resumable ranges from earlier transfers. The raw event files came from the same pinned mirror. The repository's [IDSSE downloader](../implementations/03_semi_markov/benchmark/idsse_files.py) can reproduce the metadata and raw tracking subset. [PySport mirror revision](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c)

The November 13 recount classified every ball-status row: 293,931 H,
259,174 A, and 449,539 D. These are provider reference labels.

## WCSF Groningen / SciSports

The two position archives at commit `1c9ba798f98a9fffb16081a5d3965774d4e5662c`
contain 1,262,231 player rows across 57,374 distinct 10 Hz timestamps.
The dataset owner confirmed on November 6 that this is an anonymized
official Dutch Eredivisie fixture, that the data is MIT-licensed, and
that SciSports manually labeled `AliveFlag` and `PossessionFlag`.
The absence of a license file in the pinned checkout previously led to
an incorrect pending status in this inventory. Keep the fixture
anonymous and cite the source.
[Owner's confirmation](https://github.com/MatKempeGroningen/WCSF_SolutionRoom/issues/5#issuecomment-6681182193) ·
[Pinned archives and schema](https://github.com/MatKempeGroningen/WCSF_SolutionRoom/tree/1c9ba798f98a9fffb16081a5d3965774d4e5662c)

The November 13 recount found 18,249 H, 12,239 A, 26,776 D, and 110
timestamps with unresolved flag conflicts. Give inactive status
precedence, mask the conflicts, reverse one team's coordinate axes,
and reset at halftime. All known labels are available for dense
supervision. The entire match was held out for the first comparison
of frozen models. The subsequent Kaggle grid used it at 10% and 25%
of total training weight for each head. [Benchmark assessment](INDEPENDENT_POSSESSION_BENCHMARK.md) ·
[Mixture grid](../implementations/04_classical_ml/experiments/groningen_mix/README.md#mixture-grid)

## Sportec event derivatives

- **ELASTIC**, commit `e245224aad7450961eca40954ae2bc85e1e6e696`. For `J03WMX`, `J03WN1` and `J03WPY`, the authors corrected the underlying event records and took the median of three human annotations for event and reception frames. The 3,561 distinct events appear in paired `benchmark/unsynced/` and `benchmark/gt/` files, with 1,329, 1,030 and 1,202 rows per fixture in each version. The authors explicitly release the benchmark data under CC BY 4.0 and request citation alongside the original Sportec source. [Pinned ELASTIC benchmark and license](https://github.com/hyunsungkim-ds/elastic/blob/e245224aad7450961eca40954ae2bc85e1e6e696/README.md#license-and-attribution)
- **PathCRF**, commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`. The seven Parquet files contain 12,327 event rows in total. The authors aligned the original human-tagged Sportec events algorithmically with ELASTIC. PathCRF's preprocessing derives possession states from these aligned events. The repository ships an MPL 2.0 license and identifies the underlying DFL data source. The reuse terms for the precomputed Parquet files remain unconfirmed. [Pinned PathCRF preprocessing description](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/README.md#data-preparation-and-preprocessing)

Join these derivatives to the Sportec match IDs and keep all records from each fixture in the same split. The 3,561 paired ELASTIC rows count once. Both derivatives provide event references, which need an explicit conversion before they can serve as frame-level H/A/D labels.

## SkillCorner broadcast samples

- **Current**, commit `4340d274572876239c154c90bc507a9b3250a656`. The pinned `data/matches.json` lists 20 fixtures from one competition edition, the 2024/25 Australian A-League. The README still describes ten. Each match directory has match metadata, dynamic events, phases of play and extrapolated tracking. The 20 tracking JSONL files use Git LFS. Verify both payload byte size and SHA-256 against the pinned Git LFS pointers before treating a fixture as complete. SkillCorner describes frame-level `possession.player_id` and `possession.group` and marks extrapolated player positions. The automatic labels and phases are useful weak supervision. [Pinned match index](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/data/matches.json) · [Tracking and dynamic event formats](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/README.md) · [MIT license](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE)
- **Legacy**, commit `1ed50b775f93075c2a6d00d0d795f13b7b89709e`. The nine fixtures come from the 2019/20 English, French, Spanish, Italian and German leagues. Each match has `match_data.json` and `structured_data.json` with a per-frame possession group and tracked object where available. The 19 JSON files match the pinned Git blobs and contain 604,104 tracking frames. SkillCorner describes automatic labels and tracking with gaps in broadcast coverage. [Pinned legacy formats](https://github.com/SkillCorner/opendata/blob/1ed50b775f93075c2a6d00d0d795f13b7b89709e/README.md) · [MIT license](https://github.com/SkillCorner/opendata/blob/1ed50b775f93075c2a6d00d0d795f13b7b89709e/LICENSE)

Credit SkillCorner and preserve its copyright and license notice. PySport,
the release partner, explicitly describes the public A-League dataset as
MIT-licensed. Check possession groups, missing coverage and period boundaries
before deriving labels. A phase indicates live play where it is present.
Gaps in coverage and missing ownership groups remain unknown. At 10 Hz,
25 Hz challenge-frame thresholds need conversion through seconds.
[PySport dataset terms](https://pysport.org/analytics-cup/editions/analytics-cup/datasets) ·
[Phase specification](https://skillcorner.github.io/opendata/#phases-of-play-file)

The 20 current tracking payloads total 1,823,448,331 bytes and 1,244,814 parsed frames. Of those, 605,766 frames had no possession group. An absent group needs separate handling from a confirmed dead ball. All 61 index, match metadata, event and phase files match the pinned Git blobs.

From the local `mtah-training-data/` directory, use the local helper to recheck the current checkout's payloads or resume any interrupted transfers. The helper checks each file against the size and SHA-256 in the pinned Git LFS pointer.

```powershell
python -B .\skillcorner-current\.payloads\download.py
```

On a fresh machine, fetch the pinned commits from `https://github.com/SkillCorner/opendata.git`. For a match-only checkout, use sparse checkout for `data/matches/`, `data/matches.json`, `README.md`, `LICENSE`, and `.gitattributes` when present. For the current release, hydrate `data/matches/**/*.jsonl` with Git LFS. The legacy release stores tracking directly in Git.

Approach 04 uses 639,048 current and 342,991 legacy H/A labels on timed frames.
Its current-release consensus variant keeps 512,302 frames whose tracking and
unambiguous phase labels agree. A further 99,412 frames have both labels with
different teams. The live/dead model uses DFL supervision. See the
[multi-provider comparison](../implementations/04_classical_ml/docs/MULTI_PROVIDER.md).

## SoccerTrack v2 public subset

The publisher links a public Drive mirror for the v1.2 Japanese university
matches. GSR tracking and BAS annotations have an explicit CC BY 4.0 data
license. The approach 04 subset uses the first 128 MiB of first-half GSR data
for five fixtures and all their BAS annotations. It contains 13,624 tracking
frames and 202 control-event anchors that match the player and side. The
experiment trained on 132 anchors across three fixtures, leaving the other
frames unknown. It used player-only GSR observations. The BAS-derived ball
trajectory would expose label timing.
[Publisher and mirror](https://github.com/AtomScott/SoccerTrack-v2) ·
[Data license](https://github.com/AtomScott/SoccerTrack-v2/blob/a752d294df2a0e816e827b70f284980698f5a50e/LICENSE-DATA) ·
[Exact inputs and checksums](../implementations/04_classical_ml/docs/SOCCERTRACK_INPUTS.md)

The independently acquired raw BePro XML for fixture `128058` has dense
vendor flags. The dataset card lists that XML and describes the release as
CC BY 4.0. The detailed data license names video and GSR, BAS, and MOT
annotations. Coverage of the raw exports remains unclear. The raw XML and
its flags have no role in the recorded Kaggle experiment.
[Dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2) ·
[Raw tracking audit](INDEPENDENT_POSSESSION_BENCHMARK.md#raw-tracking-inspection-of-fixture-128058)

## Local inspection copies

- **Metrica Sports sample data**, commit `e706dd506b360d69d9d123d5b8026e7294b13996`. Games 1 and 2 have paired home/away tracking CSVs and event CSVs. Game 3 uses EPTS tracking text, metadata XML and event JSON. The sample totals 429,923 distinct 25 Hz tracking frames and 7,300 event records. The nine data files, README and event definitions PDF match the pinned Git blobs and the earlier audit's SHA-256 values. The public README invites exploration and requests attribution. It has no explicit license terms for competition training or redistribution. Treat event intervals as sparse evidence for possession. [Pinned source and terms](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/README.md#legal-stuff)

## Access-gated sources

- **PFF FC World Cup**. Gradient offers tracking, events and play-by-play grades for all 64 matches of the 2022 men's World Cup through a request form. The mirror has a manual gate and displays a GPL-3.0 tag. Provider rights, organizer access, and the suitability of event-derived H/A/D labels remain unverified. [Official release](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset) · [OpenSportsLab PFF mirror](https://huggingface.co/datasets/OpenSportsLab/PFF)
- **Hugging Face course datasets**. The four dataset cards state CC BY-NC 4.0 and require access approval. Label coverage, organizer access, and compatibility with the competition's commercial-use terms remain unverified. [PremierLeague24-25](https://huggingface.co/datasets/peggy44/PremierLeague24-25) · [PremierLeague25-26](https://huggingface.co/datasets/peggy44/PremierLeague25-26) · [ChampionsLeague25-26](https://huggingface.co/datasets/peggy44/ChampionsLeague25-26) · [RealMadrid24-25](https://huggingface.co/datasets/peggy44/RealMadrid24-25)
- **SoccerTrack v2 on Hugging Face**. The complete canonical v1.2 files remain
  gated. The public mirror supplied the subset documented above.
  [Dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2)

PFF and the four course collections remain access pending. The inventory covers
their public metadata only.

### Additional leads for independent confirmation

- **SoccerNet FOOTPASS and SoccerNet-GAR** provide actor-linked action
  annotations. The newer GAR action-spotting release includes full-match
  PFF-derived tracking, with point action labels. The releases require
  access agreements, and GAR specifies research/noncommercial use.
  These are event-supervision leads pending suitable rights.
  [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) ·
  [GAR data and terms](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting)
- **Google Research Football** can generate synthetic motion with ball
  ownership and game mode. The separate 3,000-match CSV export has
  `ON_BALL`, with no explicit stoppage field or verified data license.
  [Simulator observations](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations) ·
  [Third-party export](https://github.com/Dtradke/grf_tracking_data#schema-details)
- **TacSIm** publishes one-team and ball motion in short clips. The task
  omits stoppages and explicit dataset reuse terms remain unverified.
  [Author's release](https://github.com/wenpengpap/TacSIm#dataset)
- **Ball3D** offers ball-out intervals and touch labels under noncommercial
  annotation terms. **DEBS 2013** is a sensor-tracked seven-a-side match
  with referee interruptions and possession statistics. Dense H/A/D
  coverage and competition reuse rights remain unverified.
  [Ball3D terms](https://github.com/lukaszgrad/ball3d/blob/main/DATA.md) ·
  [DEBS schema](https://debs.org/grand-challenges/2013/)
- **Wyscout and StatsBomb Open Data** provide events and, in selected
  StatsBomb matches, event-time snapshots. Both lack the continuous
  trajectory inputs used by approach 04. Wyscout Events is CC BY 4.0.
  StatsBomb's agreement restricts commercial use and third-party transfer.
  [Wyscout Events](https://figshare.com/articles/dataset/Events/7770599) ·
  [StatsBomb terms](https://github.com/hudl/open-data/blob/master/LICENSE.pdf)

- **Driblab Open Data** publishes ten European broadcast tracking matches at
  10 Hz. Its documented frame schema has player and ball trajectories and a
  camera polygon. It does not document possession or live/dead annotations.
  This is a lead for testing how often the ball is missing in broadcast
  tracking, or for adding independently sourced labels. The README does not
  state a data license. [Provider schema](https://github.com/driblab/open-data/blob/main/README.md)
- **DataBallPy's documented open games** comprise the seven DFL fixtures
  already used in approach 04 and one Metrica sample. Treat the package as
  a parser rather than a source of independent fixtures. Its API lists a
  Tracab option, but the documented open-game inventory identifies no
  separate labeled release under that name. [Open-game inventory](https://databallpy.readthedocs.io/en/main/getting_started/loading_in_a_game_page.html#open-sourced-data)
- **DFB match-phase study.** Renner et al. used 203 DFB-provided fixtures
  with 25 Hz TRACAB tracking and Sportec event data. The paper links code,
  and I found no public download for the raw fixtures in its data section.
  Its rule-based phase taxonomy groups stoppages and unclassified play
  together, so the published phase definitions alone cannot supply the
  challenge's dead-ball label. [Data section](https://arxiv.org/html/2610.11571v1#S2.SS1)
- **SoccerMon** supplies wearable GPS and athlete monitoring reports from
  two Toppserien teams. The release describes player trajectories and
  athlete reports, with no documented synchronized ball trajectory and
  dense H/A/D labels for this benchmark. [Dataset description](https://www.nature.com/articles/s41597-024-03386-x)
