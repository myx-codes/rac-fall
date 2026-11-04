# Dataset protocol and audit checklist v1.1

## Three sources — availability verified at metadata level only

| Dataset | Use | Verified facts | Unresolved before training |
|---|---|---|---|
| CAUCAFall v5 (2025-03-18), doi:10.17632/7w7fccy7ky.5 | Source development / subject-wise split | 10 subjects, five staged falls + five ADL classes, CC BY 4.0. Per-frame labels, but onset-of-fall not directly supplied by the published description. https://data.mendeley.com/datasets/compare/7w7fccy7ky | Verify v5 download availability, structure/checksums, subject IDs, frame rates, complete event-onset annotations. |
| UR Fall Detection | Supplementary cross-dataset development only | 30 fall + 40 ADL sequences, Kinect RGB/depth and accelerometers. https://fenix.ur.edu.pl/~mkepski/ds/uf.html | Verify RGB video reconstruction, license/redistribution, subject IDs and annotation timestamps. |
| GMDCSA-24 v2.1 (2024-08-21), doi:10.5281/zenodo.13354453 | **Sealed final blind test** | 81 falls + 79 ADL, 4 subjects, three homes; ZIP 1,107,545,615 bytes; Zenodo metadata says CC BY 4.0; v2.1 updated CSV class/timing. https://zenodo.org/records/13354453 | Confirm local archive/checksum only at the blind-test stage and review consent/image rights. No preview, pose extraction, training or tuning in Phase 2B. |

The repo includes **no dataset binaries, downloadable archives or identity-bearing video**. Download manually under verified permissions. The word *blind* means do not preview clips for error-driven tuning, choose thresholds or retrain based on final test. Metadata-only prechecks may be done by a separate custodian.

## CAUCAFall internal split
Start with 6 train, 2 validation, 2 internal test **by subject**, not by frame/clip. Choose subject IDs once using a documented, reproducible seed; retain exact manifest. If class balance is poor, revise split with train/validation criteria before looking at internal test. Do not use random frame split.

## Standard manifest
CSV columns: `video_id,dataset,subject_id,split,activity,duration_s,has_fall`. Annotations: `video_id,event_id,onset_s,end_s`. Onset is manually inspected transition time. Keep `camera_id`, `scene_id`, `fps`, `source_sha256`, `licence`, and `consent` in restricted provenance metadata when verified (not required for initial evaluator).

## Anti-leakage protocol
- No same subject across train/validation/internal-test within CAUCAFall.
- Keep augmented clips, frames and camera views from one source event together.
- Freeze class mappings and model thresholds using source development only; URFD may be used for external **development**.
- GMDCSA-24 remains one-time test only; publishing a benchmark score does not justify iterative edits without a new, untouched test.
- Store raw videos separately, never in Git. Do not copy video snippets into test logs.

## Extended negative exposure
Target 10 h initially, 30 h independently recorded or permission-cleared pure ADL footage, varied cameras/lighting/subjects. Include sitting, lying down intentionally, picking up an object, kneeling, exercise, occlusion, multi-person transitions and empty-camera episodes. Record exactly what was tested. Compute alarm rates with honest confidence intervals: a short ADL exposure cannot substantiate <=0.1 false alarms/hour.

## Pending data audit (must remain pending until local download)
- [ ] Verify each source's accessibility and licensing, including human likeness/face permissions.
- [ ] Download pinned version and record SHA256, #clips, #subjects, FPS and duration.
- [ ] Validate start/end annotation convention with sample cases from development sets only.
- [ ] Derive and lock exact subject split IDs.
- [ ] Document camera/scene overlap and available negative hours.
