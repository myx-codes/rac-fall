# RAC-Fall — research starter v0.1 (2026-11-04)

**Risk-Adaptive Context-Aware Fall Verification**. This is a reproducible research **starter**, not a trained fall detector or medical/safety product.

**Status:** Research proposal, dataset protocols, reference event-level evaluator and optional pretrained pose smoke test are implemented. TCN, scene model, learned RACF, webcam application and full experiments are **not implemented**. No empirical performance claims.

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

## Research documents

- `docs/RESEARCH_PROPOSAL.md` — falsifiable hypotheses and target metrics.
- `docs/LITERATURE_REVIEW.md` — verified evidence and novelty limitations.
- `docs/DATASET_PROTOCOL.md` — sources, rights, splits and leakage controls.
- `docs/EVALUATION_PROTOCOL.md` — event matching and false alerts/hour.
- `docs/ARCHITECTURE.md` — planned data flow and interfaces.
- `docs/ROADMAP.md` — implementation gates and sprint plan.
- `docs/MASTER_PLAN_V1.md` — user-provided original master plan.
- `docs/SPRINT_01_REPORT.md` — implemented vs not validated.

## Evaluation CSV contract

`--manifest`: `video_id,dataset,subject_id,split,activity,duration_s,has_fall` (unique video_id; positive duration; has_fall `0` or `1`).

`--ground-truth`: `video_id,event_id,onset_s,end_s`; onset and end are measured in **seconds from video start**, with onset defined as the start of the losing-balance fall event, not the first static frame labeled "fall". Manually verify any inferred onset annotations.

`--alerts`: `video_id,alert_s` (one row per emitted alarm; timestamps seconds).

By default, alerts within 10 s of the **last retained alert** for the same clip are collapsed as a declared event-refractory policy. Each true event can match at most one retained alert within `[onset, onset + 5]` s. Outputs: event recall, precision/F1, median delay, and **false alarms per hour on exclusively ADL (has_fall=0) clips**. The latter is separate from all unmatched alerts. No automated future-frame rescue is allowed.

**This evaluator is a reference research utility, not a clinical safety validator.**

## Privacy & rights

Default pose smoke run does **not** save RGB frames or pose trajectories. Research datasets/subjects may have separate image/likeness rights: do not redistribute data or publish identifiable demo frames without confirming permissions. Dataset licenses and the Ultralytics model/software license require separate checks before commercial redistribution.
