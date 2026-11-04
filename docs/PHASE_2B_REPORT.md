# Phase 2B — Dataset Collection & Preprocessing Report

Date: 2026-11-04
Status: pipeline implemented, synthetic-tested, and validated on one real CAUCAFall v5 video; full preprocessing stopped pending review

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

Temporal windows are generated within one video and one track only. They inherit the source subject/video split. A second leakage check rejects subject/sequence reuse or overlapping intervals across different splits. Windows contain 51 features per frame (34 normalized coordinates + 17 confidence values), a separate 17-point validity mask, timestamps, label, video, subject, track, and split. Pure ADL windows receive label `0`. A fall-source window receives `1` only when its causal endpoint lies inside an explicit source/researcher-verified event interval; fall clips without an interval receive `-1` (unknown), not an invented target. PyTorch loaders exclude unknown targets by default but can expose them for annotation/QA.

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

# Only after pilot inspection succeeds
racfall extract-poses --manifest ... --dataset-root ... --output-dir ... --device mps --imgsz 416

racfall build-pose-sequences \
  --manifest /path/outside/repo/manifests/caucafall_split.csv \
  --pose-dir /path/outside/repo/poses/caucafall \
  --output /path/outside/repo/windows/caucafall_w30_s15.npz \
  --window-frames 30 --stride-frames 15
```

URFD uses `racfall manifest-urfd --root ... --output ...`; ZIP archives can remain unextracted. Subject-independent development use remains blocked until a source-authoritative sequence-to-subject mapping is obtained.

## Verification and measured results

- Automated tests after the real-pilot normalization fix: 20 passed in 0.75 seconds.
- Synthetic fixture coverage: manifest round-trip, URFD ZIP decoding and timestamp-derived FPS, 17-keypoint validity masking, compact NPZ output, normalization, temporal windows, leakage rejection, blind-test rejection, PyTorch Dataset, and DataLoader batching.
- No accuracy or model-training result exists in Phase 2B.

### Real CAUCAFall v5 one-video pilot

Source and storage:

- Official source only: Mendeley v5 public file `Subject.1/Fall forward/FallForwardS1.avi`.
- External path: `~/datasets/RAC-Fall/CAUCAFall/`; nothing under this path is tracked by Git.
- AVI size: 1,193,966 bytes. SHA-256 `fcd0d92864c70999616e22af619b0bbde020496ce94da2b4c5f24c3efb149bdf`, exactly matching Mendeley metadata.
- Downloaded annotations: 191 frame TXT files (7,258 bytes total) and `classes.txt`. PNG copies were not downloaded. Original files were not modified.

Decode and annotation integrity:

- OpenCV decoded 190 frames at 20 FPS, 720×480, duration 9.5 s.
- This differs from the dataset page's nominal 23 FPS and 1080×960; container metadata is authoritative for this manifest row.
- The official folder contains 191 PNG/TXT frame pairs, but the AVI decodes to 190 frames. The extra annotation is `cfs1-00191.txt`; this unresolved off-by-one prevents blindly aligning every annotation to an AVI frame.
- All 191 TXT files contain exactly one five-field normalized YOLO row. No malformed rows, unknown class IDs, or values outside `[0,1]` were found.
- `classes.txt` maps class 0 to `nofall` and class 1 to `fall`; counts are 131 and 60 frame annotations respectively. These posture/segmentation classes were not converted into losing-balance onset/end timestamps. Manifest event timestamps remain empty.

Pose extraction (`YOLO26n-Pose`, MPS, `imgsz=416`, ByteTrack):

- 190 frames processed; 188 pose detections; two frames with no detected pose.
- One stable tracking ID (`1`) covers all 188 detections.
- Extraction time 2.6827 s; measured pipeline throughput 70.823 FPS; mean model-reported inference latency 3.485 ms.
- Whole CLI wall time 3.81 s. Maximum resident set size 634,486,784 bytes; peak memory footprint 466,126,264 bytes; zero swaps.
- 2,818 of 3,196 keypoints are valid at the 0.25 threshold (88.17%). Mean valid confidence 0.8821; minimum 0.2512; maximum 0.9988.
- Timestamps are strictly increasing from 0.00 to 9.45 s. Invalid coordinates are `NaN` and have `valid=false`.
- Compressed pose output: 35,876 bytes.

Normalization and temporal windows:

- The first real run exposed a projected-torso scale collapse with maximum absolute normalized coordinate 18.32. Normalization now bounds torso scale by 25% of visible body extent; the rebuilt maximum is 3.57 and all features remain finite.
- Generated 11 causal windows of 30 frames with stride 15 for the one stable track. Feature shape is `(11, 30, 51)` and validity-mask shape is `(11, 30, 17)`.
- Missing normalized coordinates are zero only where the explicit mask is false. Window timestamps are strictly increasing.
- All 11 targets are `-1` (unknown), because no source-verified fall onset/end exists. Default training loaders therefore exclude them; an explicit QA loader produced a `(4, 30, 51)` batch without training.
- Window generation wall time 0.09 s; maximum resident set size 35,487,744 bytes; zero swaps. Compressed window output: 34,604 bytes.

Synthetic test speed is deliberately not reported as dataset processing performance.

## Unresolved gate before full preprocessing

1. Resolve or explicitly document the 191 annotation rows versus 190 decoded AVI frames before frame-level annotation alignment.
2. Decide whether the nominal-vs-container FPS/resolution discrepancy is systematic by auditing additional development videos, without preprocessing the full dataset yet.
3. Review participant likeness/consent and redistribution handling under CC BY 4.0.
4. Obtain the remaining official files outside Git, verify the complete 10-subject/100-activity inventory and checksums, then generate the frozen 6/2/2 assignment.
5. Define and review event onset annotations on development subjects only; do not derive them automatically from `fall` posture labels.
6. Re-run a small multi-activity development pilot before authorizing full pose extraction.

Phase 3 training is intentionally out of scope and must not start until these gates are resolved.
