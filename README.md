# RAC-Fall — research starter v0.1 (2026-11-04)

**Risk-Adaptive Context-Aware Fall Verification**. This is a reproducible research **starter**, not a trained fall detector or medical/safety product.

**Status:** Research proposal, dataset protocols, reference event-level evaluator, pretrained pose smoke test, and a live tracked pose visualizer are implemented. Fall detection, TCN, scene model, learned RACF, and full experiments are **not implemented**. No fall-detection accuracy claims.

## How to run (macOS Apple Silicon / Python 3.11+)

```bash
cd RAC-Fall
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
racfall env-check
racfall eval-events --manifest tests/fixtures/manifest.csv --ground-truth tests/fixtures/gt_events.csv --alerts tests/fixtures/alerts.csv
```

### Optional pretrained pose-only feasibility (downloads official weights on first use)

```bash
python -m pip install -e '.[vision,dev]'
racfall env-check
racfall pose-smoke --source /path/to/consented_sample.mp4 --max-frames 50 --device auto
# webcam (may need camera permissions):
racfall pose-smoke --source 0 --max-frames 50 --device auto
```

YOLO pretrained pose inference only. **No fall detection decisions** are made by this test. Neither raw frames nor keypoints are saved to disk in default mode. Test model throughput and MPS availability on the user's Mac; a Linux runner cannot validate Apple MPS. Model download requires internet on first run.

### Real-time pose visualization (Phase 2A)

```bash
python -m pip install -e '.[vision,dev]'
racfall pose-live --source 0 --imgsz 416
```

`pose-live` opens webcam index 0 with OpenCV's macOS AVFoundation backend and runs `yolo26n-pose.pt` on Apple MPS by default. It draws all 17 COCO keypoints, skeleton connections, persistent ByteTrack IDs, rolling display FPS, model-reported inference latency, and mean keypoint confidence for each pose. Press **P** to pause/resume and **Q** to quit. Use `--device cpu` only when MPS is unavailable.

The live command processes frames only in memory: it does not save raw frames, screenshots, video, keypoints, or tracking data. Camera-open/read failures produce an error, and the camera plus OpenCV windows are released during normal exit and errors. The summary printed after quitting contains aggregate runtime measurements only. This remains pose visualization—not fall detection.

### Dataset preprocessing (Phase 2B)

Phase 2B adds strict CAUCAFall-video and URFD-RGB-sequence adapters, provenance manifests, reproducible subject splits, tracked pose extraction, normalization, causal windows, and PyTorch loaders. Raw datasets and generated `.npz` files must remain outside Git. GMDCSA-24 is locked against pose extraction and development use.

Phase 2B.5 downloaded and checksum-verified the approved Subjects 1–8 official PNG/TXT inventory. The full canonical audit then found two official filename-index discontinuities (`Subject.1/Sit down` and `Subject.5/Kneel`), so controlled pose preprocessing stopped before inference. Subjects 9 and 10 remain sealed and absent; Subjects 3 and 5 remain documented as development-exposed. See the Phase 2B report before starting Phase 3.

```bash
racfall download-caucafall-canonical --folders-json /external/official-folders.json --metadata-dir /external/official-metadata --dataset-root /external/CAUCAFall --reuse-root /external --report-dir /external/download-report --plan-only
racfall manifest-caucafall-png --root /external/CAUCAFall --label-map configs/caucafall_activity_labels.json --output /external/manifests/canonical.csv --timing-fps 20
racfall split-caucafall-trainval-frozen --manifest /external/manifests/canonical.csv --output /external/manifests/canonical-split.csv --assignment-output /external/manifests/frozen-split.json
racfall audit-caucafall-canonical --manifest /external/manifests/canonical-split.csv --dataset-root /external/CAUCAFall --output-dir /external/canonical-audit
racfall preprocess-caucafall-batch --manifest /external/manifests/train-val-batch.csv --dataset-root /external/CAUCAFall --canonical-audit-dir /external/canonical-audit --output-dir /external/processed --device mps --imgsz 416
racfall build-pose-sequences --manifest /external/manifests/train-val-batch.csv --pose-dir /external/processed/poses --alignment-dir /external/canonical-audit --quality-dir /external/processed/quality --output /external/windows/train-val.npz
```

CAUCAFall frame labels are causal ground-state classification (`1` means on the ground due to a fall), not fall-event onset/end annotations. Missing TXT labels remain `-1` with an invalid mask. Annotation boxes are excluded from inference. Sequence quality reports quarantine unresolved sources/poses before window generation.

See `docs/PHASE_2B_REPORT.md` for the audited source facts, full CLI workflow, schemas, blind-test policy, and unresolved real-data pilot gate. No Phase 3 model training is implemented.

## Research documents

- `docs/RESEARCH_PROPOSAL.md` — falsifiable hypotheses and target metrics.
- `docs/LITERATURE_REVIEW.md` — verified evidence and novelty limitations.
- `docs/DATASET_PROTOCOL.md` — sources, rights, splits and leakage controls.
- `docs/EVALUATION_PROTOCOL.md` — event matching and false alerts/hour.
- `docs/ARCHITECTURE.md` — planned data flow and interfaces.
- `docs/ROADMAP.md` — implementation gates and sprint plan.
- `docs/MASTER_PLAN_V1.md` — user-provided original master plan.
- `docs/SPRINT_01_REPORT.md` — implemented vs not validated.
- `docs/PHASE_2B_REPORT.md` — dataset audit, preprocessing implementation and pilot gate.
- `docs/PHASE_2B4_WORKFLOW.md` — canonical PNG/TXT, frozen split, batch/resume and integrity workflow.
- `docs/FALL_EVENT_ANNOTATION_PROTOCOL.md` — proposed human-review protocol for initiation, contact and recovery landmarks.

## Evaluation CSV contract

`--manifest`: `video_id,dataset,subject_id,split,activity,duration_s,has_fall` (unique video_id; positive duration; has_fall `0` or `1`).

`--ground-truth`: `video_id,event_id,onset_s,end_s`; onset and end are measured in **seconds from video start**, with onset defined as the start of the losing-balance fall event, not the first static frame labeled "fall". Manually verify any inferred onset annotations.

`--alerts`: `video_id,alert_s` (one row per emitted alarm; timestamps seconds).

By default, alerts within 10 s of the **last retained alert** for the same clip are collapsed as a declared event-refractory policy. Each true event can match at most one retained alert within `[onset, onset + 5]` s. Outputs: event recall, precision/F1, median delay, and **false alarms per hour on exclusively ADL (has_fall=0) clips**. The latter is separate from all unmatched alerts. No automated future-frame rescue is allowed.

**This evaluator is a reference research utility, not a clinical safety validator.**

## Privacy & rights

Default pose smoke run does **not** save RGB frames or pose trajectories. Research datasets/subjects may have separate image/likeness rights: do not redistribute data or publish identifiable demo frames without confirming permissions. Dataset licenses and the Ultralytics model/software license require separate checks before commercial redistribution.
