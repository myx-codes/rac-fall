# Phase 2B — Dataset Collection & Preprocessing Report

Date: 2026-11-05
Status: Phase 2B.4 canonical-source pilot passed; controlled full preprocessing awaits download approval

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

## Phase 2B.4 canonical dataset source and controlled preprocessing

This section supersedes the Phase 2B.3 full-processing decision while retaining its historical AVI findings. No full archive was downloaded, no model was trained, and GMDCSA-24 remained locked.

### Canonical-source decision

Official PNG ordinal is now the canonical frame order for CAUCAFall ground-state training, and the TXT carrying the same embedded one-based index is its only frame label. AVI remains useful for video review and its container FPS can be recorded as a timing-rate proxy, but PNG and AVI indices are never assumed to correspond. Canonical timestamps therefore use `PNG ordinal / secondary AVI FPS` and explicitly do not claim source capture timestamps or AVI alignment.

The audit requires PNG indices to begin at one, remain contiguous, decode at one resolution and use a single filename prefix. TXT rows retain the original class semantics: class 0 is not on the ground due to a fall; class 1 is on the ground due to a fall, not fall initiation. A missing same-index TXT produces `ground_state_label=-1` and `label_valid=false`. It is never filled from a neighbor. An orphan TXT, PNG gap, prefix conflict, decode error or resolution change quarantines the sequence. Event labels remain `-1` unless a separate reviewed event protocol supplies boundaries.

The real pilot exposed an official filename variant: source index 91 is named `cas200091 - copia.png`. The strict first pass quarantined it as a missing PNG/orphan TXT. The parser now accepts only this documented optional ` - copia` suffix while preserving embedded index 91; a regression test covers the case.

### Real Subject 2 canonical pilot

Inputs were the already-downloaded 121 official PNGs and 121 official TXT annotations; no new dataset media was downloaded. The canonical PNG tree SHA-256 is `e03c5d78a39deea277ef6b6e7f3cf30a0e4a83f2d48362875839d03007c5d5ee`.

- 121 PNG indices and 121 TXT indices pair exactly from 1 through 121.
- All PNGs decode at 720×480 and retain monotonic ordinal order.
- Ground-state distribution is 63 class-0 and 58 class-1 frames; unknown labels: 0.
- Event labels known: 0. No onset, ground-contact, recovery or end time was inferred.
- The secondary AVI rate is 20 FPS, yielding a canonical ordinal timing proxy from 0.00 through 6.00 seconds. This does not resolve or need the AVI's 119-frame divergence.
- Final canonical audit time was 0.68 seconds; maximum RSS was 78,086,144 bytes. The provenance JSON SHA-256 is `4d822049c7f24812047a74d09eadf3158f04be1bf33014d38332cb8412a783ed`.

This sequence was quarantined under AVI-primary alignment in Phase 2B.3, but it is verified under the new PNG-primary contract because every canonical image has an independently paired same-index label. No AVI label shift was applied.

### Pose quality and inference-only diagnostics

The canonical pilot used YOLO26n-Pose, MPS, `imgsz=416`, detector confidence 0.25, keypoint confidence 0.25 and ByteTrack. It produced 112 detections/causal target frames from 121 frames and marked nine frames missing (7.44%). Of 1,904 possible selected-pose keypoints, 1,611 are valid and 293 are invalid (15.39%). The sequence passes the frozen 50% pose-coverage gate.

Final extraction took 2.7986 seconds (`43.236` FPS). The checksum/audit/quality batch wall timer was 3.617532 seconds (`33.448` FPS), while the outer process wall time was 3.90 seconds. Maximum RSS was 608,256,000 bytes and peak memory footprint was 454,854,144 bytes. Pose and quality artifacts total 23,977 bytes; the observed filesystem free-space change was 212,992 bytes including filesystem allocation and ledger/summary writes.

The nine baseline-missing frames were evaluated without annotation boxes. Independent full-frame prediction recovered 5/9 at confidence 0.25, 5/9 at 0.10 and 7/9 at 0.05. Four annotation-free 60%-size overlapping tiles found a usable pose candidate on 9/9 at confidence 0.25. Diagnostic inference took 1.60056 seconds; outer wall time was 2.33 seconds with maximum RSS 610,992,128 bytes. These candidates are not used for training: tile coordinate merging, duplicate suppression and causal identity continuity are not yet frozen or validated. Production extraction thresholds were not changed.

### Quality gates, deterministic windows and resume

Each sequence receives a quality JSON containing canonical status, class/invalid-label counts, pose coverage, missing-pose rate, keypoint validity, selector configuration and quarantine reasons. Window generation accepts `--quality-dir`, skips non-eligible sequences, preserves unknown endpoint labels, and refuses internal-test windows without explicit authorization.

The pilot generated seven 30-frame windows at stride 15 and 50% coverage: three class-0, four class-1 and zero unknown ground-state endpoints. All seven event targets remain unknown. No window was rejected for coverage or sequence quality. The quality-enforced output is 24,574 bytes and repeated builds are byte-identical with SHA-256 `e4a3d4d09c881edd595e9460793c5973fb15cd0f78a0265d6d0dc32fb4f5c2a7`.

Controlled batches require a manifest source checksum and verify it before inference. Pose and quality files are written through `.partial` paths and atomic rename. The JSON ledger signature covers the complete manifest record, model, device, image size, both confidence thresholds, quality threshold and schema versions. Resume requires matching signature plus matching pose/quality SHA-256; otherwise the sequence is recomputed or fails on source mismatch. An unchanged real-pilot rerun reported one `resumed_verified`, performed zero inference, consumed zero measured disk bytes and reports no synthetic throughput value.

### Frozen subject-independent split

The earlier seed split is retired. Subjects 3 and 5 were exposed during development and now belong only to development roles. Media for Subjects 9 and 10 has not been downloaded, decoded, previewed or used for tuning; only public metadata/file inventory was inspected. The split was frozen before full preprocessing in `configs/caucafall_split_phase2b4.json`:

- train: Subjects 1, 2, 4, 5, 6, 8 — 60 sequences, 30 fall / 30 ADL by official activity inventory;
- validation: Subjects 3, 7 — 20 sequences, 10 fall / 10 ADL;
- sealed internal test: Subjects 9, 10 — 20 sequences, 10 fall / 10 ADL.

The assignment is fixed rather than selected from pose quality or model results. Batch preprocessing skips `test` by default, and test window generation fails unless explicitly authorized for the frozen evaluation. GMDCSA-24 remains the separate locked external blind test.

### Full-dataset workflow and disk gate

The implemented workflow is: official download and checksum verification; canonical manifest; frozen split; per-sequence PNG/TXT audit; subject-batched MPS extraction; quality/quarantine reports; quality-enforced windows; and final ledger/inventory reconciliation. `docs/PHASE_2B4_WORKFLOW.md` contains commands and acceptance checks.

The official archive is still 8,326,349,365 bytes. Archive plus extracted source requires approximately 15.50 GiB before generated data; current measured free space during the pilot was about 168.3 GB, so capacity is sufficient. The download remains blocked on explicit user approval.

### Automated verification

The suite now has 35 passing tests in 0.89 seconds. New coverage includes missing-TXT invalid masks, canonical causal labels, the canonical adapter's PNG-count/AVI-timing separation, the official `copia` filename, orphan-TXT quarantine, frozen Subjects 9/10, checksum and audit-provenance invalidation, atomic/resumable batch behavior, annotation-free recovery diagnostics, quality provenance, and internal-test sealing. Compile, CLI help and diff checks also pass.

### Readiness and remaining blockers

The code and pilot evidence are **ready for controlled train/validation preprocessing after explicit approval to download the official archive**. They are not authorization to begin Phase 3 training.

Remaining blockers are:

1. the 8.33 GB archive is not downloaded and requires explicit approval;
2. full-data canonical audits and real batch distributions cannot be reported until the archive is present;
3. five metadata-audited folders with fewer TXT than PNG will contain explicit invalid labels and must be reviewed through generated provenance, not repaired;
4. overlapping-tile recovery is diagnostic only until coordinate/identity merging is validated without annotation boxes;
5. PNGs have no independent source timestamps, so AVI FPS remains a timing-rate proxy rather than index alignment;
6. participant likeness/consent and redistribution handling still require project review;
7. Subjects 9/10 must remain sealed until the frozen internal evaluation.

## Phase 2B.5 controlled train/validation preparation

Status on 2026-11-06: **stopped at the canonical alignment gate; no pose preprocessing or training was run**.

### Official download and integrity

Only the official CAUCAFall v5 Mendeley `public-files` endpoints were used. The approved preflight inventory contained 30,820 PNG/TXT files totaling 6,137,241,092 bytes. Of these, 1,166 files were already present and verified, 219 could be reused by exact official size and SHA-256, and 29,435 files totaling 6,072,582,576 bytes required network transfer. Downloading was resumable, limited to Subjects 1–8, used bounded retries, processed subjects sequentially, and retained `.partial` files until verification and atomic rename.

The completed global reconciliation found exactly 30,820/30,820 files and 6,137,241,092/6,137,241,092 bytes at their final paths with official size and SHA-256 matches; no `.partial` files remained. The resumed run took 3,097.888985 seconds. Its ledger recorded 29,122 newly downloaded files/5,994,253,069 bytes, while files completed before the controlled restart were recognized as already verified. Final free space was 161,310,478,336 bytes at download completion, well above the 2 GiB reserve (later filesystem activity may change this value). Raw and generated data remain outside Git. Original files were not modified.

| Subject | Official files | Official bytes | Resumed-run elapsed (s) |
|---|---:|---:|---:|
| 1 | 3,318 | 695,911,387 | 286.047671 |
| 2 | 2,952 | 622,197,881 | 323.079101 |
| 3 | 3,850 | 949,381,739 | 364.418247 |
| 4 | 3,738 | 850,519,259 | 406.614626 |
| 5 | 4,423 | 895,464,345 | 376.295983 |
| 6 | 4,373 | 678,937,706 | 428.496962 |
| 7 | 3,906 | 838,358,421 | 484.931180 |
| 8 | 4,260 | 606,470,354 | 427.995597 |

### Frozen split and exposure

The manifest contains 80 sequences and 15,371 canonical PNG frames. The frozen train/validation-only assignment is 60 train sequences (Subjects 1, 2, 4, 5, 6, 8; 30 fall activities and 30 ADL activities) and 20 validation sequences (Subjects 3, 7; 10 fall activities and 10 ADL activities). Subjects 9 and 10 were not downloaded or included in the manifest. GMDCSA-24 was not accessed.

Subjects 3 and 5 were already exposed during Phase 2B.2/2B.3 development auditing. Subject 3 is therefore a development validation subject, not an untouched test subject; Subject 5 remains in training. The sealed internal test remains Subjects 9 and 10.

### Canonical annotation gate and stop decision

The all-sequence PNG/TXT audit produced 75 `verified`, three `verified_with_invalid_labels`, and two `quarantined` sequences. Across all source sequences there are 14,970 valid ground-state labels and 401 unknown labels after fail-closed quarantine. Confirmed findings are:

- `Subject.1/Sit down`: both PNG and TXT index 148 are absent between paired indices 147 and 149. The official files that do exist match their published checksums. All 188 frames in this sequence are quarantined as unknown.
- `Subject.5/Kneel`: paired indices 1–209 are followed by PNG index 236; PNG indices 210–235 are absent and TXT index 236 is absent. All 210 frames are quarantined as unknown.
- `Subject.6/Walk`: TXT index 260 is absent; the one corresponding frame label remains unknown while the sequence is otherwise verified.
- `Subject.8/Hop`: canonical TXT index 96 is absent; the one corresponding frame label remains unknown. A differently named annotation is not silently substituted.
- `Subject.8/Pick up object`: canonical TXT index 90 is absent; the one corresponding frame label remains unknown. A differently named annotation is not silently substituted.

| Subject | Sequence status | Frames | Valid labels | Unknown labels | Ground state 0 | Ground state 1 |
|---|---|---:|---:|---:|---:|---:|
| 1 | 9 verified, 1 quarantined | 1,654 | 1,466 | 188 | 1,151 | 315 |
| 2 | 10 verified | 1,471 | 1,471 | 0 | 1,061 | 410 |
| 3 | 10 verified | 1,920 | 1,920 | 0 | 1,360 | 560 |
| 4 | 10 verified | 1,864 | 1,864 | 0 | 1,253 | 611 |
| 5 | 9 verified, 1 quarantined | 2,207 | 1,997 | 210 | 1,209 | 788 |
| 6 | 9 verified, 1 verified-with-invalid | 2,182 | 2,181 | 1 | 1,425 | 756 |
| 7 | 10 verified | 1,948 | 1,948 | 0 | 1,245 | 703 |
| 8 | 8 verified, 2 verified-with-invalid | 2,125 | 2,123 | 2 | 1,419 | 704 |

This is an official-inventory alignment problem, not a download corruption: the complete local inventory matches official sizes and SHA-256 values. No index shift, relabeling, timestamp invention, or repair was applied. Because the user required stopping on critical data integrity or alignment failures, MPS pose extraction, per-subject pose metrics, temporal windows, DataLoader readiness measurements, and Phase 3 training readiness were not produced. Proceeding would require an explicit protocol decision to exclude the two quarantined sequences and continue with the 78 remaining sequences; it must not reinterpret their missing indices.

### Implementation and automated verification

The CLI now supports official inventory planning/downloading, PNG-only canonical manifests with an explicit frozen ordinal timing proxy, and the frozen train/validation split while sealed Subjects 9/10 remain absent. The downloader rejects non-official URLs, verifies official size/SHA-256, resumes `.partial` transfers, bounds retries, reuses only checksum-identical local files, checks disk capacity, and fails rather than overwriting a mismatched target. Canonical source hashes exclude secondary AVI data but include the immutable PNG/TXT source.

The suite has 39 passing tests. Added regression coverage includes official URL/inventory validation, exact checksum reuse, target mismatch refusal, PNG-only timing provenance, canonical hash scope, frozen train/validation isolation, and fail-closed quarantine when both PNG and TXT share an otherwise-hidden index gap.

## Phase 2B.5 approved continuation: controlled pose preprocessing

Status on 2026-11-06: **controlled preprocessing completed; ready to configure Phase 3 training, but no model training has started**.

The user explicitly approved continuing with the 78 audit-eligible sequences. `Subject.1/Sit down` and `Subject.5/Kneel` remained excluded, and their original files and audit SHA-256 values were preserved. The run processed 58 train and 20 validation sequences (14,973 frames) sequentially by subject using `yolo26n-pose.pt` (`eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`), Apple MPS, `imgsz=416`, detector confidence 0.25, keypoint confidence 0.25, and ByteTrack. Subjects 9/10 and GMDCSA-24 remained absent.

### Pose and sequence quality results

All 78 pose/quality artifacts completed without source, audit, decode, or checksum failure. The fixed 50% causal-target coverage gate accepted 64 sequences and quarantined 14; the threshold was not relaxed. The accepted sequences contain 12,206 frames, with a causal target on 10,759 frames (88.1451745% accepted-sequence coverage). Selected accepted poses contain 163,896 valid and 19,007 invalid keypoints.

Pose-quality quarantines were:

- Subject 1: `Fall right`, `Kneel`;
- Subject 3: `Fall backwards`, `Fall forward`, `Fall left`, `Fall sitting`;
- Subject 6: `Fall backwards`, `Fall right`;
- Subject 7: `Fall left`;
- Subject 8: `Fall backwards`, `Fall forward`, `Fall left`, `Fall right`, `Pick up object`.

Every quarantine reason is causal target pose coverage below 50%; no annotation box was used for inference and no recovery, index shifting, or threshold tuning was applied. The known missing TXT labels remained `-1`: Subject 6 `Walk` index 260, Subject 8 `Hop` index 96, and Subject 8 `Pick up object` index 90. They remain represented by explicit label-validity masks. None happened to fall on the endpoint of an accepted causal window, so the final window set has zero unknown endpoint labels.

| Subject | Split | Accepted / processed sequences | Frames | Labels -1 / 0 / 1 | All-sequence pose coverage | Valid / invalid windows | Window labels 0 / 1 | Wall FPS | Wall seconds | Peak RSS bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | train | 7 / 9 | 1,466 | 0 / 1,151 / 315 | 75.9891% | 59 / 0 | 47 / 12 | 67.684 | 21.659506 | 609,763,328 |
| 2 | train | 10 / 10 | 1,471 | 0 / 1,061 / 410 | 94.2896% | 80 / 0 | 57 / 23 | 65.197 | 22.562454 | 618,184,704 |
| 3 | val | 6 / 10 | 1,920 | 0 / 1,360 / 560 | 65.2083% | 61 / 0 | 51 / 10 | 69.812 | 27.502292 | 615,743,488 |
| 4 | train | 10 / 10 | 1,864 | 0 / 1,253 / 611 | 82.3498% | 96 / 0 | 66 / 30 | 69.937 | 26.652721 | 619,937,792 |
| 5 | train | 9 / 9 | 1,997 | 0 / 1,209 / 788 | 89.7346% | 114 / 0 | 69 / 45 | 68.912 | 28.978948 | 617,840,640 |
| 6 | train | 8 / 10 | 2,182 | 1 / 1,425 / 756 | 76.3520% | 90 / 0 | 71 / 19 | 70.591 | 30.910321 | 624,852,992 |
| 7 | val | 9 / 10 | 1,948 | 0 / 1,245 / 703 | 88.5524% | 100 / 0 | 65 / 35 | 68.685 | 28.361356 | 627,261,440 |
| 8 | train | 5 / 10 | 2,125 | 2 / 1,419 / 704 | 55.7647% | 55 / 0 | 49 / 6 | 74.571 | 28.496255 | 632,127,488 |

The subject batch wall total was 215.123853 seconds for 14,973 frames (69.602 FPS). Summed extraction time was 201.9797 seconds (74.131 FPS); frame-weighted model-reported inference latency was 3.483 ms. Maximum measured subject-process RSS was 632,127,488 bytes. These are preprocessing measurements, not fall-detection accuracy or real-time safety claims.

### Causal windows and split isolation

The frozen 30-frame, stride-15 generator evaluated 884 candidate windows. Sequence quarantine removed 165 candidates and the unchanged 50% per-window track gate removed 64 more, leaving exactly 655 valid windows and zero invalid endpoint-label windows:

- train: 494 windows — 359 ground-state 0 and 135 ground-state 1;
- validation: 161 windows — 116 ground-state 0 and 45 ground-state 1;
- total: 475 ground-state 0 and 180 ground-state 1.

Ground-state labels remain separate from event labels. Because no true event boundaries exist, all 258 fall-activity windows retain event label `-1`; the 397 ADL windows carry event label 0. No onset or end timestamp was inferred. The window artifact stores normalized `(x, y, confidence)` features, 17-keypoint validity masks, timestamps, causal track coverage, and source tracking IDs. Its shape is `(655, 30, 51)` with validity shape `(655, 30, 17)`.

The PyTorch interface loaded 494 train and 161 validation samples and produced `(32, 30, 51)` test batches for each split. Subject, sequence, and overlapping-window leakage validation passed. Train subjects are 1, 2, 4, 5, 6, 8; validation subjects are 3 and 7; Subjects 9 and 10 are absent.

### Integrity, reproducibility and readiness

A resume-only run checksum-verified all 78 pose and quality artifacts (`resumed_verified=78`, `failed=0`) and performed zero inference. Independent window generation produced byte-identical 1,852,907-byte NPZ files with SHA-256 `1c7b461b281e401a636165b9c6be8c291b27f4ea1430fc3ec452df15cc2cf364`; all 22 arrays also matched. No `.partial` artifacts remained. Generated data stays outside Git.

The automated suite has 39 passing tests in 0.67 seconds. It now also verifies per-subject window accounting. Compile and diff checks pass.

Phase 3 may use only the 494 train windows for fitting and the 161 validation windows for development evaluation. The 14 pose-quality quarantines and two source-alignment quarantines remain excluded. The dataset is ready for training configuration and baseline design, subject to explicit approval to start Phase 3; it is not evidence of fall-model accuracy, and sealed Subjects 9/10 and GMDCSA-24 must remain untouched.
