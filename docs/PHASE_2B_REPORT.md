# Phase 2B — Dataset Collection & Preprocessing Report

Date: 2026-11-04
Status: pipeline implemented and synthetic-tested; real CAUCAFall pilot blocked on approved local data access

## Scope and safeguards

This phase implements dataset inspection, provenance manifests, subject-independent splits, pose extraction, normalization, causal temporal windows, and PyTorch loading. It does not train a fall model. Original files are opened read-only and are never rewritten. Generated NPZ files belong outside Git. Event boundaries and subject IDs remain blank or unresolved unless the source explicitly supplies them.

GMDCSA-24 is enforced as a locked blind test: `PoseExtractor` and window generation reject locked/blind records. Only the public metadata below was inspected. No GMDCSA media was downloaded, previewed, decoded, or used for a development decision.

## Availability and annotation audit

### CAUCAFall v5

- Official record: https://data.mendeley.com/datasets/7w7fccy7ky/5, DOI `10.17632/7w7fccy7ky.5`, published 2025-03-18, CC BY 4.0.
- Published structure: 10 subject directories; each has 10 activity directories (five staged fall directions and five ADLs); each activity directory contains AVI video, PNG frames, and per-frame TXT segmentation labels.
- Published capture metadata: HIKVISION IR camera, nominal 23 FPS and 1080×960.
- Published labels distinguish fall/no-fall frames, but the description does not define the losing-balance event onset required by the event evaluator.
- Local status: no archive or extracted files are present. The Mendeley page is reachable, but its file API returned HTTP 401 without credentials. Archive size, local clip count, checksums, exact directory names, actual container FPS/resolution, and annotation completeness are therefore unresolved.
- Adapter policy: exact top-level directory names become subject IDs; activity labels require an explicit researcher-reviewed JSON map. Unknown folder names fail instead of being guessed.

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

Normalization centers on visible hips, then shoulders, then visible-point mean. Scale uses torso length when available and visible-point extent otherwise. Degenerate/missing poses become zero-valued features with a false mask, never an implicit safe pose.

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

- Automated tests: 20 passed; repeated local Python 3.11 runs completed in 0.55–0.83 seconds.
- Synthetic fixture coverage: manifest round-trip, URFD ZIP decoding and timestamp-derived FPS, 17-keypoint validity masking, compact NPZ output, normalization, temporal windows, leakage rejection, blind-test rejection, PyTorch Dataset, and DataLoader batching.
- Real CAUCAFall count: not measured (dataset absent).
- Real CAUCAFall preprocessing errors: not measured (dataset absent).
- Real CAUCAFall MPS extraction speed: not measured (dataset absent).
- No accuracy, event boundaries, or model-training result exists in Phase 2B.

Synthetic test speed is deliberately not reported as dataset processing performance.

## Unresolved gate before full preprocessing

1. Obtain approval and sufficient disk space for CAUCAFall v5 outside Git.
2. Verify archive checksum, extracted 10-subject/100-activity layout, exact activity directory names, and rights/likeness handling.
3. Review the exact activity-to-label map and generate the frozen 6/2/2 assignment.
4. Run `extract-poses --limit 1`; record real frame count, decode errors, pose/tracking coverage, MPS FPS, and inference latency.
5. Inspect annotations on development subjects to define event onset without using the internal test or GMDCSA blind set.
6. Proceed to full CAUCAFall preprocessing only after the pilot passes.

Phase 3 training is intentionally out of scope and must not start until these gates are resolved.
