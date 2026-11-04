# Phase 2B — Dataset Collection & Preprocessing Report

Date: 2026-11-05
Status: Phase 2B.3 quality/alignment repair completed; full preprocessing is not yet safe

## Scope and safeguards

This phase implements dataset inspection, provenance manifests, subject-independent splits, pose extraction, normalization, causal temporal windows, and PyTorch loading. It does not train a fall model. Original files are opened read-only and are never rewritten. Generated NPZ files belong outside Git. Event boundaries and subject IDs remain blank or unresolved unless the source explicitly supplies them.

GMDCSA-24 is enforced as a locked blind test: `PoseExtractor` and window generation reject locked/blind records. Only the public metadata below was inspected. No GMDCSA media was downloaded, previewed, decoded, or used for a development decision.

## Availability and annotation audit

### CAUCAFall v5

- Official record: https://data.mendeley.com/datasets/7w7fccy7ky/5, DOI `10.17632/7w7fccy7ky.5`, published 2025-03-18, CC BY 4.0.
- Published structure: 10 subject directories; each has 10 activity directories (five staged fall directions and five ADLs); each activity directory contains AVI video, PNG frames, and per-frame TXT segmentation labels.
- Published capture metadata: HIKVISION IR camera, nominal 23 FPS and 1080×960.
- Published labels distinguish fall/no-fall frames, but the description does not define the losing-balance event onset required by the event evaluator.
- Pilot status: Mendeley's official anonymous public-file endpoint was used for one source video and its TXT annotations. A one-byte range request measured the official Download All archive as 8,326,349,365 bytes before transfer; 158 GiB was available. The full archive was not retained because the official per-file API allowed a minimal pilot download. The generic API endpoint returned HTTP 401 and was not bypassed.
- Adapter policy: exact top-level directory names become subject IDs; activity labels require an explicit researcher-reviewed JSON map. Unknown folder names fail instead of being guessed.

Manual fallback if the official automatic endpoint returns 401: open the official v5 record above in a browser, use its **Download All** control, sign in only through Mendeley's normal account flow if requested, review the CC BY 4.0 terms, and save the archive under `~/datasets/RAC-Fall/CAUCAFall/`. Do not copy signed URLs, reuse another person's session, or bypass the login. Before extraction, verify the downloaded file is 8,326,349,365 bytes and record its checksum.

### UR Fall Detection (URFD)

- Official record: https://fenix.ur.edu.pl/~mkepski/ds/uf.html.
- Public inventory: 70 sequences—30 fall and 40 ADL. Falls expose RGB PNG ZIPs for cameras 0 and 1; ADL exposes camera 0. Synchronization CSV rows provide frame number and milliseconds from sequence start.
- License on the official page: CC BY-NC-SA 4.0, non-commercial academic use; commercial use requires contacting the dataset owner.
- The official download page does not map sequence IDs to subject IDs. Secondary dataset literature reports five participants, but that does not supply a trustworthy per-sequence mapping. The adapter records `subject_id=unknown` and the validator forbids assigning such rows to train/val/test splits.
- The public posture feature CSV uses `-1` not lying, `0` transitional/falling, and `1` lying, but those depth-derived posture labels are not treated as RGB event-onset annotations here.
- Local status: no URFD ZIP, PNG, or sync CSV is present; no real file count/checksum/decoding result is claimed.

### GMDCSA-24 v2.1 — locked blind test

- Official record: https://zenodo.org/records/13354453, DOI `10.5281/zenodo.13354453`, CC BY 4.0.
- Zenodo API metadata: one 1,107,545,615-byte ZIP, MD5 `3d36f2c5c1a666b99639e4e9fd843efb`. Version 2.1 says CSV files contain class names and class start/end times.
- Dataset paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/ reports 160 clips (81 fall, 79 ADL), four subjects, three home environments, MP4/H264 at 30 FPS, and per-subject ADL/Fall CSV files.
- Local status: not downloaded. The locked metadata record is `data/manifests/GMDCSA24_BLIND_METADATA.json`. No training, tuning, threshold selection, pose extraction, or clip inspection is allowed in Phase 2B.
- Rights caveat: CC BY covers copyright licensing; participant likeness/consent and redistribution handling still require project-level review before use or publication.

## Implemented data contracts

The CSV manifest contains dataset/version, verified subject provenance, video and sequence IDs, source type/path, action and binary label, optional label boundaries, annotation provenance, timestamp provenance and range, FPS, resolution, frame count, split, source checksum, license, and blind-lock status.

Validation rejects:

- empty or malformed media metadata;
- absolute or escaping source paths;
- duplicate videos;
- unknown subjects assigned to a split;
- the same subject or sequence in multiple splits;
- blind records outside `blind_test`;
- blind records entering pose extraction or window generation.

CAUCAFall split assignment sorts real source subject IDs, shuffles them with a recorded Python `random.Random(seed)`, and requires exactly 10 subjects before producing 6 train, 2 validation, and 2 internal-test subjects. No concrete assignment is published until the local source directory names are verified.

## Pose and temporal formats

Each per-video compressed NPZ stores:

- every decoded frame index and source-derived timestamp;
- frame offsets so frames with zero detections remain explicit;
- tracking ID (`-1` means unavailable), 17×2 pixel coordinates, 17 confidences, and 17 validity flags per detection;
- missing/low-confidence coordinates as `NaN` plus `valid=false`;
- source identity, split, label provenance, image size, inference device, and `imgsz`.

Normalization centers on visible hips, then shoulders, then visible-point mean. Scale uses torso length bounded by 25% of visible body extent, or visible-point extent when torso landmarks are missing. Degenerate/missing poses become zero-valued features with a false mask, never an implicit safe pose.

Temporal windows are generated within one video from one causal target-person trajectory. Source track IDs may change only through annotation-free geometric continuity; anonymous detections cannot replace a previously tracked identity. They inherit the source subject/video split. A second leakage check rejects subject/sequence reuse or overlapping intervals across different splits. Windows contain 51 features per frame (34 normalized coordinates + 17 confidence values), a separate 17-point validity mask, timestamps, video, subject, source track IDs, causal target ID, and split.

Each window also stores the fraction of frames containing its selected track. The default 0.5 minimum rejects mostly-empty windows from short or fragmented track IDs while retaining explicit keypoint validity masks inside accepted windows.

Window schema v3 separates two targets. `ground_state_labels` (and its compatibility alias `labels`) is the verified frame-level ground state at the causal window endpoint: `0` means not on the ground due to a fall, `1` means on the ground due to a fall, and `-1` means unknown. It never reads a later frame. `event_labels` describes a true fall-event interval only when explicit onset/end annotations exist; otherwise it stays `-1`. A class-1 ground-state transition is not promoted into an event onset. PyTorch loaders train/select on ground-state labels and exclude `-1` by default, while retaining unknown samples for explicit QA. The NPZ records the target-selection method and exact configuration.

## CLI

```bash
# CAUCAFall manifest (the JSON map must use exact, locally verified activity directory names)
racfall manifest-caucafall \
  --root /path/outside/repo/CAUCAFall \
  --label-map /path/reviewed_caucafall_labels.json \
  --output /path/outside/repo/manifests/caucafall.csv

racfall split-caucafall \
  --manifest /path/outside/repo/manifests/caucafall.csv \
  --output /path/outside/repo/manifests/caucafall_split.csv \
  --assignment-output /path/outside/repo/manifests/caucafall_split.json \
  --seed 20261104

# Required pilot before full extraction
racfall extract-poses \
  --manifest /path/outside/repo/manifests/caucafall_split.csv \
  --dataset-root /path/outside/repo/CAUCAFall \
  --output-dir /path/outside/repo/poses/caucafall-pilot \
  --device mps --imgsz 416 --limit 1

racfall align-caucafall-annotations \
  --manifest /path/outside/repo/manifests/caucafall-pilot.csv \
  --dataset-root /path/outside/repo/CAUCAFall \
  --reference-png-dir /path/outside/repo/alignment/official_png \
  --pose-dir /path/outside/repo/poses/caucafall-pilot \
  --output-dir /path/outside/repo/alignment/verified --limit 1

# Only after pilot inspection succeeds
racfall extract-poses --manifest ... --dataset-root ... --output-dir ... --device mps --imgsz 416

racfall build-pose-sequences \
  --manifest /path/outside/repo/manifests/caucafall_split.csv \
  --pose-dir /path/outside/repo/poses/caucafall \
  --alignment-dir /path/outside/repo/alignment/verified \
  --output /path/outside/repo/windows/caucafall_w30_s15.npz \
  --window-frames 30 --stride-frames 15 --min-track-coverage 0.5
```

URFD uses `racfall manifest-urfd --root ... --output ...`; ZIP archives can remain unextracted. Subject-independent development use remains blocked until a source-authoritative sequence-to-subject mapping is obtained.

## Verification and measured results

- Automated tests after Phase 2B.2: 25 passed in 0.50 seconds.
- Synthetic fixture coverage: manifest round-trip, URFD ZIP decoding and timestamp-derived FPS, 17-keypoint validity masking, compact NPZ output, normalization, temporal windows, leakage rejection, blind-test rejection, PyTorch Dataset/DataLoader, off-by-one rejection, class transitions, missing annotation indices, terminal-extra quarantine, causal endpoint labels, unknown preservation, and ground-state/event-label separation.
- No accuracy or model-training result exists in Phase 2B.

### Real CAUCAFall v5 one-video pilot

Source and storage:

- Official source only: Mendeley v5 public file `Subject.1/Fall forward/FallForwardS1.avi`.
- External path: `~/datasets/RAC-Fall/CAUCAFall/`; nothing under this path is tracked by Git.
- AVI size: 1,193,966 bytes. SHA-256 `fcd0d92864c70999616e22af619b0bbde020496ce94da2b4c5f24c3efb149bdf`, exactly matching Mendeley metadata.
- Downloaded annotations: 191 frame TXT files (7,258 bytes total) and `classes.txt`. Six official representative PNGs were downloaded for alignment QA only: indices 1, 95, 131, 132, 190, and 191. Original files were not modified.

Decode and annotation integrity:

- OpenCV decoded 190 frames at 20 FPS, 720×480, duration 9.5 s.
- This differs from the dataset page's nominal 23 FPS and 1080×960; container metadata is authoritative for this manifest row.
- The official folder contains 191 PNG/TXT frame pairs, but both container metadata and complete OpenCV decoding report 190 frames. Decoded indices are 0–189 at timestamps 0.00–9.45 s; source PNG/TXT names are one-based `cfs1-00001`–`cfs1-00191`.
- All 191 TXT files contain exactly one five-field normalized YOLO row. No malformed rows, unknown class IDs, or values outside `[0,1]` were found.
- `classes.txt` maps class 0 to `nofall` and class 1 to `fall`; source counts are 131 and 60 frame annotations respectively. In the dataset's original semantics, class 1 records a person on the ground due to a fall, not necessarily fall initiation. These frame classes were not converted into losing-balance onset/end timestamps. Manifest event timestamps remain empty.

### Phase 2B.1 alignment findings

Confirmed findings:

- Official PNGs 95, 131, 132, and 190 each have their lowest full-frame pixel error against decoded AVI frames 95, 131, 132, and 190 respectively. Thus there is no supported global ±1 shift through the middle, the class transition, or the final decoded frame.
- Official PNG 191 also best matches decoded frame 190. Official PNGs 190 and 191 are distinct but near-identical (mean absolute pixel difference 0.011895 on the 0–255 scale). Both therefore represent the final decoded image content; annotation 191 has no distinct decoded AVI frame.
- The opening is visually static: PNG 1 best matches decoded frame 3 by pixel error, so pixels alone cannot uniquely distinguish the first few frames. This ambiguity does not override the exact middle/transition/end evidence.
- TXT class 0 occupies source indices 1–131; class 1 starts at 132 and continues through 191. The accepted mapping preserves source indices 1–190 without a shift, yielding 131 class-0 and 59 class-1 decoded-frame labels. Source annotation 191 is explicitly quarantined and remains recorded in provenance; it is not silently dropped or shifted.
- Pose-to-official-box correspondence for representative decoded frames 1, 95, 131, 132, and 190 placed respectively 100%, 100%, 100%, 88.24%, and 100% of valid keypoints inside the official normalized box. The deterministic acceptance floor is 80%.
- The provenance JSON records source checksum, video metadata, timestamp formula, every annotation filename, class semantics, sampled PNG checksums/errors, bounding-box correspondence, mapping rule, label vector, thresholds, and quarantined source indices. If any required representative frame shifts, any required box fails, the terminal pair is not a near-duplicate, indices are missing, or the count relationship differs, the entire sequence is marked quarantined and all frame labels become `-1`.

Assumption/inference, not confirmed source metadata:

- The most plausible explanation is that the source frame/annotation export included an additional terminal still (for example, an inclusive endpoint), while the AVI contains 190 timed frames. The exact export tool or cause is not documented by the inspected official files, so the pipeline does not encode that explanation—only the measured terminal-extra relationship.

Pose extraction (`YOLO26n-Pose`, MPS, `imgsz=416`, ByteTrack):

- 190 frames processed; 188 pose detections; two frames with no detected pose.
- One stable tracking ID (`1`) covers all 188 detections.
- Phase 2B.1 rerun extraction time 2.5424 s; measured pipeline throughput 74.731 FPS; mean model-reported inference latency 3.511 ms.
- Whole rerun CLI wall time 3.39 s. Maximum resident set size 635,863,040 bytes; peak memory footprint 475,121,128 bytes; zero swaps.
- 2,818 of 3,196 keypoints are valid at the 0.25 threshold (88.17%). Mean valid confidence 0.8821; minimum 0.2512; maximum 0.9988.
- Timestamps are strictly increasing from 0.00 to 9.45 s. Invalid coordinates are `NaN` and have `valid=false`.
- Compressed rerun pose output: 35,899 bytes.

Normalization and temporal windows:

- The first real run exposed a projected-torso scale collapse with maximum absolute normalized coordinate 18.32. Normalization now bounds torso scale by 25% of visible body extent; the rebuilt maximum is 3.57 and all features remain finite.
- Generated 11 causal windows of 30 frames with stride 15 for the one stable track. Feature shape is `(11, 30, 51)` and validity-mask shape is `(11, 30, 17)`.
- Missing normalized coordinates are zero only where the explicit mask is false. Window timestamps are strictly increasing.
- The earlier pre-alignment run correctly left all 11 targets unknown. After verified Phase 2B.1 alignment, causal window endpoints are source frames 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, and 180. Ground-state targets are seven class-0 and four class-1 windows: 11 valid, 0 invalid. No future-frame lookup or overlap across sequences is used.
- All 11 separate event targets remain `-1`, because no source-verified fall onset/end exists. The class transition at source frame 132 is not declared to be a fall-event onset.
- Final rerun alignment audit wall time was 0.97 s; maximum resident set size was 302,104,576 bytes; zero swaps. Its JSON output is 9,385 bytes.
- Rebuilt window generation wall time was 0.10 s; maximum resident set size was 38,715,392 bytes; zero swaps. Compressed schema-v2 output is 35,548 bytes.

Synthetic test speed is deliberately not reported as dataset processing performance.

## Phase 2B.2 multi-subject validation

### Scope and official downloads

Five additional activities were selected to span five new subjects, three fall types, and two ADLs: Subject.2/Fall backwards, Subject.3/Fall left, Subject.4/Sit down, Subject.5/Walk, and Subject.6/Fall sitting. Only their five AVIs, TXT annotations, `classes.txt`, and 27 representative PNGs were downloaded through Mendeley's official anonymous `public-api`/`public-files` endpoints. The generic authenticated API returned HTTP 401 and was not bypassed. All 979 AVI/TXT files and all representative PNGs were checked against official per-file size and SHA-256 metadata. The full 8.33 GB archive was not downloaded.

The audit exposed two official source variants now covered by regression tests: most later frame names have no dash before the five-digit index (for example `cas200001.txt`), and ADL `classes.txt` contains only `nofall` while fall folders contain `nofall` and `fall`.

### Per-sequence results

| Subject / activity | AVI frames | TXT frames | Source class counts | Alignment decision | Confirmed reason |
| --- | ---: | ---: | --- | --- | --- |
| Subject.1 / Fall forward | 190 | 191 | 131 class 0, 60 class 1 | verified; TXT 191 quarantined | PNG/TXT 1–190 align; terminal PNG 191 best-matches AVI frame 190 |
| Subject.2 / Fall backwards | 119 | 121 | 63 class 0, 58 class 1 | sequence quarantined | middle/transition align, but PNG 119 best-matches AVI 105 and PNGs 120–121 best-match AVI 104; terminal relationship is unresolved |
| Subject.3 / Fall left | 171 | 172 | 102 class 0, 70 class 1 | sequence quarantined | pixel indices align, but transition-frame pose lies completely outside the official subject box; the detector selected a different image region |
| Subject.4 / Sit down | 199 | 200 | 200 class 0 | sequence quarantined | pixel indices align, but no pose was detected at representative frame 100, so bounding-box correspondence is unverified |
| Subject.5 / Walk | 226 | 227 | 227 class 0 | verified; TXT 227 quarantined | required frames align and terminal PNG 227 best-matches AVI 226 |
| Subject.6 / Fall sitting | 248 | 249 | 107 class 0, 142 class 1 | verified; TXT 249 quarantined | required frames align and terminal PNG 249 best-matches AVI 248 |

The terminal-still rule now accepts one or more extras only when every extra PNG best-matches the final decoded frame, all required middle/transition/final samples retain their own index, required pose boxes correspond, and the maximum consecutive terminal-PNG mean absolute difference is at most one intensity level on the 0–255 scale. This measured bound accepted terminal MAEs 0.0119–0.5713 while still rejecting Subject.2 for its shifted final content. Failed sequences keep every ground-state target at `-1`; labels are neither shifted nor partially rescued.

### Pose and preprocessing measurements

All six videos used YOLO26n-Pose, MPS, `imgsz=416`, confidence 0.25, and ByteTrack:

- 1,153 frames, 1,163 pose detections, and 77 frames with no detection (6.68%).
- Per-video missing-frame rates: 1.05%, 5.88%, 24.56%, 3.52%, 0.00%, and 7.66% for Subjects 1–6 respectively.
- 17,310 of 19,771 detected keypoints were valid; 2,461 were missing/low-confidence (12.45%). Subject.3 had the highest invalid-keypoint rate at 28.35%.
- Summed extraction time was 9.6223 s, equivalent to 119.826 processed frames/s across the six runs. Whole CLI wall time was 10.73 s, mean model-reported inference latency was 3.459 ms, maximum RSS was 641,908,736 bytes, and swaps were zero.
- Six compressed pose files total 220,595 bytes. Multiple/fragmented tracks occurred in Subjects 2, 3, and 6; untracked detections remain explicit as track ID `-1`.

The original window builder generated full-video windows even for tracks visible in only 3–6 frames. Phase 2B.2 adds a stored `track_coverage` value and a configurable `--min-track-coverage`, default 0.5. With 30-frame windows and stride 15, 79 mostly-empty candidates were rejected. The remaining 65 windows contain 41 valid ground-state targets (31 class 0, 10 class 1) and 24 unknown targets from quarantined sequences. All 39 fall-video event targets remain `-1`; the 26 ADL event targets are `0`. Output is 184,441 bytes, all features are finite, maximum absolute normalized coordinate is 3.973, wall time is 0.32 s, and maximum RSS is 41,271,296 bytes.

### Subject-independent split verification

Official folder metadata contains exactly 10 source-named subjects, each with the same five fall and five ADL activity folders. Split assignment now operates on sorted unique subject IDs, independent of input row order, then uses recorded `random.Random(seed)` shuffling. For seed `20261104`:

- train: Subjects 1, 2, 4, 6, 7, and 10 — 60 sequences, 30 fall / 30 ADL;
- validation: Subjects 8 and 9 — 20 sequences, 10 fall / 10 ADL;
- internal test: Subjects 3 and 5 — 20 sequences, 10 fall / 10 ADL.

Repeated and reversed-input runs produce the same assignment. Manifest and window validators still reject subject reuse, sequence reuse, or overlapping windows across splits. This verifies the metadata-level 6/2/2 plan; it does not authorize use of unprocessed test-subject content for development decisions.

### Full-source inventory and disk plan

Metadata-only enumeration of all 100 official activity folders found 40,200 files totaling 8,318,185,768 bytes: 100 AVI files (135,044,978 bytes), 20,001 PNGs (8,182,379,732 bytes), and 20,099 TXT files including class files (761,058 bytes). Ninety-five folders have equal PNG/indexed-TXT counts. Five folders have one fewer indexed TXT than PNG: Subject.8/Pick up object, Subject.8/Hop, Subject.5/Kneel, Subject.6/Walk, and Subject.9/Walk. These are known quarantine candidates until sequence-level evidence is reviewed.

The official Download All archive is 8,326,349,365 bytes. Keeping both archive and extracted files requires 16,644,535,133 bytes (15.50 GiB) before generated outputs. The six-video measurements project roughly 3.8 MB of compressed pose data for 20,001 frames and about 3.4 MB for a one-track baseline of 1,186 windows, but tracking multiplicity and quarantine QA will change those values. Reserve at least 1 GiB for generated manifests, provenance, pose files, windows, temporary atomic writes, and logs rather than relying on the small projection.

If later approved, full processing should proceed one subject at a time and one video per worker on MPS: freeze official metadata/checksums; download to `.part`; verify size/SHA-256 before atomic rename; build and validate the manifest; extract one pose NPZ per video; run per-sequence representative alignment; quarantine failures; then generate windows only after the subject split is frozen. A resume ledger must key every artifact by source SHA-256, model name, MPS/imgsz/confidence/tracker settings, schema version, and output SHA-256. Existing valid artifacts may be skipped; mismatched or partial artifacts must be recomputed. Subject batches prevent cross-split writes, and final integrity checks must reconcile official inventory, decoded frames, annotation indices, pose offsets, alignment status, window coverage, and leakage validation.

### True-event annotation protocol

`docs/FALL_EVENT_ANNOTATION_PROTOCOL.md` defines separate manual landmarks for fall initiation, ground contact, recovery start, and recovery end, including unknown reason codes, two independent reviewers, adjudication, immutable provenance, temporal ordering, and test-subject sealing. No timestamps were created in this phase, and no source ground-state transition was treated as event initiation.

## Safety decision and unresolved gates

Full preprocessing is **not safe to start yet**. Only two of the five new sequences passed the combined alignment/pose correspondence gate; three were quarantined. Metadata also identifies five full-dataset folders with missing indexed TXT files. Before requesting the full archive, the project must:

1. decide whether alignment and pose-quality quarantine should be separate gates while keeping both fail-closed for training;
2. investigate Subject.2's late-sequence AVI/PNG divergence without shifting annotations;
3. define a reviewed policy for detector failures/false detections at representative transition frames;
4. implement and test the resume ledger and atomic artifact validation described above;
5. review participant likeness/consent and redistribution handling under CC BY 4.0;
6. approve the event-annotation protocol before creating any true-event timestamps.

Phase 3 training is intentionally out of scope and must not start until these gates are resolved.

## Phase 2B.3 data quality and alignment repair

### Confirmed findings

**Subject 2 temporal divergence.** All 121 official PNGs were compared with all 119 decoded AVI frames using a complete 121×119 matrix of 64×48 grayscale mean absolute errors. Decoded timestamps are monotonic from 0.00 to 5.90 seconds in 0.05-second steps. Official PNGs 91 and 92 are byte-distinct files with identical decoded pixels. A monotonic one-to-one source-skip audit has two exactly tied best solutions: skip `[91, 118]` or `[92, 118]`, each with mean cost `2.5633808164`. Other second-skip choices are nearly tied, while unconstrained per-frame best matches contain three monotonic violations. This proves one ambiguous duplicate but does not reliably identify the second extra source frame. No mapping or label shift was applied; Subject 2 remains quarantined.

**Subject 3 target selection.** The previous transition detections at frames 102–103 had anonymous track ID `-1`, pose center near `(168, 227)`, and corresponded to the monitor/laptop region rather than the annotated person near x=241–352. Official boxes were used only to score this audit, never as selector input. The repaired selector uses current/past pose geometry, stable tracked identity, confidence/validity quality, and conservative long-gap reacquisition. Once a non-negative identity exists, an anonymous detection cannot take over. The false transition target is now missing rather than reassigned. Only 72/171 frames form a supported causal target trajectory and required frames 102, 103, and 171 have no accepted target, so the sequence remains quarantined.

**Subject 4 missing midpoint.** Visual review of frames 93–103 confirms that the person remains in the scene, bent over and partly occluded by the chair; this is not a genuinely empty frame. At detector confidence 0.10, poses appeared only on frames 93, 95, 102, and 103; confidence 0.05 additionally recovered 94 and 99–101 but still failed on 96–98. Lowering the production threshold therefore does not repair the interval and would accept weaker detections. With the unchanged extraction threshold, 12/199 target frames are missing and representative frame 100 has no pose. Subject 4 remains quarantined.

**Target and pose quality policy.** Selection has no annotation arguments. It rejects poses with fewer than five finite valid keypoints or mean valid confidence below 0.25, prevents anonymous takeover, requires center/scale continuity across track-ID changes, and uses stricter geometry after gaps longer than 20 frames. Missing frames and keypoints remain explicit; normalization now also requires finite positive confidence. Official boxes remain an offline audit score only. These rules are fail-closed and were not relaxed to increase accepted samples.

**Six-sequence decisions after repair.** Subjects 1/Fall forward, 5/Walk, and 6/Fall sitting remain verified with their final unmatched TXT explicitly quarantined. Subject 2 remains quarantined for ambiguous temporal divergence. Subject 3 remains quarantined because correct target support is absent at the class transition/final representative frame. Subject 4 remains quarantined because midpoint pose correspondence cannot be established. No sequence was released without documented evidence, and all labels for quarantined sequences remain `-1`.

### Before/after quality metrics

The prior causal audit selected 1,061 target frames across the six videos. The repaired fail-closed selector retains 984/1,153 frames and marks 169 (14.66%) missing: Subjects 1–6 retain 187/190, 112/119, 72/171, 187/199, 226/226, and 200/248 respectively. The reduction is intentional: anonymous/unsupported detections are no longer treated as the target. Across retained poses, 14,739/16,728 keypoints are valid and 1,989 (11.89%) are explicitly missing. This denominator differs from the earlier all-detection metric (2,461/19,771 missing, 12.45%), so the percentages are not an accuracy comparison.

At 30 frames and stride 15, the unchanged 50% coverage guard produces 61 windows: 37 valid ground-state targets (27 class 0, 10 class 1) and 24 unknown targets. It rejects seven candidate windows. Sensitivity checks produced 62 windows at 25% coverage (38 valid, 24 unknown; six rejected) and 55 at 75% (37 valid, 18 unknown; 13 rejected). The 50% default remains because every accepted window must contain a target in at least half its frames; it was not chosen or relaxed to maximize sample count. The final 50% output is 173,290 bytes. A repeat build was byte-for-byte identical with SHA-256 `aa6008480b3ad406278b20b32e92a7919d088c28e575fd24158dfd283a885b4e`.

The six-video pose extraction was not rerun because Phase 2B.3 changes only downstream selection/QA and the immutable pose NPZ inputs are unchanged. The measured extraction totals therefore remain: 1,153 frames in 9.6223 seconds of summed pipeline time (`119.826` frames/s). The end-to-end CLI wall time was 10.73 seconds (`107.456` frames/s); the 1.1077-second difference (10.32% of wall time) is process/model startup, decode/serialization, and command overhead outside the per-video extraction timers. Mean model-reported inference latency remains 3.459 ms and is not presented as end-to-end throughput.

The final alignment rerun processed all six sequences in 5.25 seconds wall time, with maximum RSS 457,900,032 bytes, peak memory footprint 247,694,200 bytes, and zero swaps. The final 50% window build took 0.29 seconds with maximum RSS 41,074,688 bytes. No fall-detection accuracy exists because no model was trained.

### Split contamination control

Subjects 3 and 5 were already inspected during the Phase 2B.2/2B.3 development audits. The previously published seed-`20261104` assignment placed both in the internal test set; that assignment can no longer represent an untouched test. `configs/caucafall_development_exposure.json` records this history, and split generation now raises if either subject would be assigned to `test`. No replacement seed or test subjects were selected from these audit results. A replacement 6/2/2 protocol must be frozen prospectively before full preprocessing.

### Automated verification

The final suite has 29 passing tests in 0.93 seconds (repeat runs observed 0.73–0.93 seconds). New regressions cover duplicate-frame alignment ambiguity, anonymous false-region takeover, conservative long-gap reacquisition, pose-quality/missing-keypoint accounting, and exposed-subject test protection, in addition to the existing off-by-one, class-transition, causal-label, leakage, and blind-test guards.

### Confirmed conclusions versus assumptions

Confirmed: Subject 2 has one exact duplicate PNG pair and no unique monotonic two-skip solution; Subject 3's prior transition selection was the wrong image region; Subject 4's person is visually present while pose inference fails through part of the midpoint interval; repaired outputs are deterministic; three sequences pass and three remain quarantined.

Not confirmed: the CAUCAFall export process that created extra/duplicate frames, the correct second Subject 2 source skip, and any true fall initiation/contact/recovery timestamps. None is encoded as a label or boundary.

### Remaining blockers and full-processing decision

Full preprocessing remains **unsafe**. The blocking items are the unresolved Subject 2 temporal mapping, unsupported transition/midpoint poses in Subjects 3 and 4, five metadata-audited folders with PNG/TXT count discrepancies, the contaminated former internal-test assignment, and the not-yet-implemented atomic resume ledger/integrity validator. The full-data plan remains subject-batched, one MPS worker at a time, with `.part` downloads, official size/SHA-256 verification, source- and configuration-keyed resume entries, atomic artifact rename, per-sequence quarantine, and final inventory/offset/alignment/window/leakage reconciliation. No full archive download, GMDCSA access, training, or event timestamp invention occurred.
