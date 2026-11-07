# RAC-Fall Phase 4B: Scene Context Integration

**Audit date:** 2026-11-08
**Outcome:** stopped after the preregistered training-only feasibility audit
**Task represented by the frozen model:** causal ground-state classification, not fall-event detection

## Executive result

The local scene-object detector did **not** pass the preregistered feasibility gate. On the fixed training-only manual sample it achieved object precision 1.00 but recall 0.45, below the frozen minimum of 0.60. The automatic gates passed, but manual verification found 11 missed chairs among 20 reviewed frames. Therefore scene context was not connected to the Phase 4A verifier, validation subjects were not accessed for Phase 4B, and no context-aware validation metrics are reported.

This is a negative result, not evidence that scene context cannot help. It shows that this detector/configuration is not reliable enough to condition a safety-relevant decision under the declared protocol.

## Frozen inputs and safeguards

- The Phase 3A Pose-TCN checkpoint and Phase 4A verifier were not modified, retrained or invoked for tuning.
- The audit used only training subjects 2, 4, 5, 6 and 8. Validation subjects 3 and 7 and sealed subjects 9 and 10 were not accessed. GMDCSA-24 remained sealed.
- Sampling, thresholds and the prospective ablations were frozen in `configs/phase4b_scene_context.json` before inference. Configuration SHA-256: `20f3cb6167141272fc8c9ec96351b7b0c6f8b0eba6019042f555b54690f24f99`.
- Person regions came from the existing inference-only pose output. Annotation boxes and labels were not supplied to inference.
- RGB was decoded locally. The audit stores metadata, hashes, detections and relationships only. Temporary contact sheets used for manual review were deleted; no raw or derived RGB is retained by the pipeline by default.
- Missing targets and objects remain explicit. Missing or ambiguous context never means NORMAL, and object presence alone is never treated as proof of a safe activity.

## Detector and license audit

The detector is the official Ultralytics `yolo26n.pt` COCO-pretrained detection checkpoint, restricted at inference to COCO class IDs 56 (chair), 57 (couch) and 59 (bed). The 5.3 MB weight file was stored outside Git and verified before use:

`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

The installed Ultralytics package reports AGPL-3.0; deployment or commercial use requires a separate license review. OpenCV reports Apache-2.0. The official CAUCAFall v5 Mendeley record reports CC BY 4.0. Sources checked on the audit date: [Ultralytics detection documentation](https://docs.ultralytics.com/tasks/detect/), [Ultralytics licensing documentation](https://www.ultralytics.com/license), and [CAUCAFall v5 on Mendeley Data](https://data.mendeley.com/datasets/7w7fccy7ky/5). These checks establish the research audit basis but are not legal advice.

## Preregistered feasibility protocol

The fixed audit selected `Sit down` and `Fall sitting` for each of five training subjects: 10 sequences, eight evenly spaced frames per sequence, and two preselected manual-review frames per sequence. The automatic requirements were target-pose availability at least 0.75 and persistent context in at least 0.50 of sequences. Manual requirements were object precision at least 0.80, recall at least 0.60 and relationship plausibility at least 0.80.

The detector returned chair/sofa/bed confidence and bounding boxes. For each frame the pipeline records causal person-object IoU, normalized edge proximity, center distance and relative vertical position. Temporal persistence is computed within each sequence only. Occlusion, no target, no object and ambiguous manual observations are represented explicitly rather than imputed.

## Measured results

### Automatic audit

| Measure | Result |
|---|---:|
| Training sequences | 10 |
| Sampled frames | 80 |
| Target-pose availability | 0.9000 |
| Frames with a scene object detection | 0.4625 |
| Frames with a person-object relationship | 0.4125 |
| Sequences with persistent context | 0.5000 |
| Total detector inference time | 1.5755 s |
| Mean latency including first-load cost | 19.69 ms/frame |
| Median latency | 10.25 ms/frame |
| p95 latency | 14.28 ms/frame |
| Maximum/first-load latency | 725.03 ms |
| Warm mean, excluding first inference | 10.77 ms/frame |
| Warm p95, excluding first inference | 14.23 ms/frame |

Only chairs were detected in this sample (39 detections); no sofa or bed detections were produced. Persistent context was present in 5/10 sequences. At the frame level, detections were not a clean activity separator: at least one chair was detected in 21/40 `Fall sitting` frames and 16/40 `Sit down` frames.

### Manual quality audit

| Measure | Result | Gate |
|---|---:|---:|
| Reviewed frames | 20 | fixed |
| True-positive object observations | 9 | — |
| False-positive object observations | 0 | — |
| False-negative object observations | 11 | — |
| Object precision | 1.0000 | >= 0.80 |
| Object recall | 0.4500 | **>= 0.60, failed** |
| Relationship plausibility when available | 1.0000 | >= 0.80 |
| Occluded/ambiguous frames | 2 | reported |

The dominant failure is missed chair context, including partially visible or challenging views. High precision does not compensate for low recall because absence of a detection cannot safely mean absence of a support object. The available spatial relationships appeared plausible, but they were available too selectively.

## Integrity and reproducibility

External audit artifacts remain outside Git under `~/datasets/RAC-Fall/CAUCAFall/phase4b/training-scene-audit/`:

| Artifact | SHA-256 |
|---|---|
| `automatic-audit.json` | `f5c9f42d06d39082b3865b67bba205c049304e5cce35466cb8220a7a4c98d7f1` |
| `manual-audit.csv` | `dbfcefd079e7576fa8843d67e59ac8b2692f81a27d0467437935c0afa6b3e0b5` |
| `manual-validation.json` | `86849b79dbf4820d71e269716a66bee0c8d57240a153551274619eb1ca82526e` |

The CLI validates detector SHA-256, rejects validation/sealed subjects in the training-only audit, writes atomically, and produces a fixed manual-review template. The manual validator rejects incomplete reviews and non-training subjects.

## Stop decision and unreported metrics

The user required stopping if reliable context could not be extracted. Because manual recall failed, the controlled Phase 4B prototype was not enabled and the frozen validation set was not evaluated. Consequently FP, FN, recall, Macro-F1, abstention, coverage, strict abstention-as-error metrics and added end-to-end verifier latency do not exist for a context-aware variant. Reporting values for them would fabricate an experiment that was intentionally not run.

No learned verifier was trained. If a future learned verifier is approved, its prerequisite remains subject-grouped out-of-fold TCN predictions generated only from training subjects; in-sample TCN predictions and validation labels are prohibited.

## Remaining blockers and next experiment

1. Establish a new, preregistered training-only audit on a disjoint training sample before changing detector thresholds, input resolution or model family. Reusing these 20 reviewed frames for selection would overfit the feasibility check.
2. Compare a license-compatible detector or segmentation model and evaluate chair/sofa/bed recall under occlusion while retaining explicit unknown context.
3. Require the replacement to pass object recall, precision, relationship plausibility and persistence gates before enabling the already-preregistered context ablations.
4. Keep context conservative: it may delay confirmation by at most one window, never force NORMAL, and must fall back to the context-free Phase 4A path when unavailable.

Phase 4B therefore stops at the training-only feasibility boundary. The frozen TCN, Phase 4A verifier, validation split, sealed subjects and external blind dataset remain unchanged.
