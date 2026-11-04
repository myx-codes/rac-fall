# Planned RAC-Fall system architecture v1.1

```text
Consented webcam/MP4 frames  (local, memory-only)
       |
   Person / Pose: YOLO26n-pose + tracking
       | (17 keypoints, confidence, track ID)
       +----> Object context + scene relationships (chair/bed/sofa proximity)
       |                                      |
   Normalized trajectory -> causal TCN -> p(fall_t), motion, pose quality
       |                                      |
       +--------------> RACF risk verifier <--+
                                |
                 calibrated risk + uncertainty
                                |
            NORMAL / UNCERTAIN / FALL ALERT
                                |
        minimal local event timestamps & statistics
```

## Interfaces to implement after data audit
- `FrameRecord`: `video_id, frame_index, timestamp_s, image_in_memory`; raw frame is **not persisted** by default.
- `PoseTrack`: `track_id, timestamp_s, bbox, 17*(x_norm,y_norm,confidence)`; when person untrackable, use quality/missingness flags; no assumption that missing is safe.
- `MotionWindow`: causal past `T` frames, normalized by bbox/body scale and timestamp; no looking ahead.
- `ContextFeatures`: object category, detection quality, normalized overlap/proximity and coarse floor-contact proxies, **not metric 3D height**.
- `VerifierInput`: causal TCN logits, motion risk, object relations, pose quality, missingness, time-since-candidate.
- `Decision`: `state, time_s, confidence, alert_id, model_version, calibration_version`. Use deterministic debouncing and re-alert rules.

## Training and comparison
M0 posture rules; M1 TCN; M2 TCN+fixed; M3 TCN+context; M4 risk-adaptive context and uncertainty. Freeze validation split, calibration, thresholds and onset/time matching before external test. Avoid claiming learned verifier exists until weights and train logs are produced.

## Security and privacy constraints
Never write RGB/video/audio/faces to disk by default; no network streaming. Pose keypoints still contain identifying movement characteristics, so offer separate opt-in retention controls. Explicit settings needed for capture permission, retention windows and consent. Test data isolation and deletion. No clinical/emergency deployment without independent evaluation.

## Local build strategy
First research docs/protocol + one reference evaluator; then optional pose smoke; then dataset manifest; then simple M0; then TCN; then context; then M4; finally FastAPI/React. `src/racfall/pose_smoke.py` is feasibility only, not a fall classifier.
