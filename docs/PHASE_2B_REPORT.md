# Phase 2B — Dataset Collection & Preprocessing Report

Date: 2026-11-04
Status: Phase 2B.1 annotation alignment validated on one real CAUCAFall v5 video; full preprocessing stopped pending review

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

Temporal windows are generated within one video and one track only. They inherit the source subject/video split. A second leakage check rejects subject/sequence reuse or overlapping intervals across different splits. Windows contain 51 features per frame (34 normalized coordinates + 17 confidence values), a separate 17-point validity mask, timestamps, video, subject, track, and split.

Window schema v2 separates two targets. `ground_state_labels` (and its compatibility alias `labels`) is the verified frame-level ground state at the causal window endpoint: `0` means not on the ground due to a fall, `1` means on the ground due to a fall, and `-1` means unknown. It never reads a later frame. `event_labels` describes a true fall-event interval only when explicit onset/end annotations exist; otherwise it stays `-1`. A class-1 ground-state transition is not promoted into an event onset. PyTorch loaders train/select on ground-state labels and exclude `-1` by default, while retaining unknown samples for explicit QA.

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
  --window-frames 30 --stride-frames 15
```

URFD uses `racfall manifest-urfd --root ... --output ...`; ZIP archives can remain unextracted. Subject-independent development use remains blocked until a source-authoritative sequence-to-subject mapping is obtained.

## Verification and measured results

- Automated tests after Phase 2B.1: 23 passed in 0.53 seconds.
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

## Unresolved gate before full preprocessing

1. Determine whether the terminal-extra pattern and nominal-vs-container FPS/resolution discrepancy recur by auditing additional development videos before generalizing this mapping; any sequence without its own required evidence remains quarantined.
2. Review participant likeness/consent and redistribution handling under CC BY 4.0.
3. Obtain the remaining official files outside Git only after approval, verify the complete 10-subject/100-activity inventory and checksums, then generate the frozen 6/2/2 assignment.
4. Define and review true event onset/end annotations on development subjects only; do not derive them automatically from ground-state class transitions.
5. Re-run a small multi-activity development pilot before authorizing full pose extraction.

Phase 3 training is intentionally out of scope and must not start until these gates are resolved.
