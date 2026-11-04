# RAC-Fall — Product requirements v0.1

## User and goal
Local single-machine research demonstrator for verifying suspected falls, primarily for a researcher testing consumer webcam / MP4 input. Not a clinical monitoring product.

## Key jobs
1. Load user-consented local video or webcam and reliably detect/track person poses.
2. Model motion history and differentiate falling from sitting, bending, kneeling and voluntary lying down.
3. Use scene relationships and prediction uncertainty to issue NORMAL, UNCERTAIN or FALL ALERT.
4. Measure false alerts per hour, event recall, onset-to-alert delay, calibration and resources under a frozen benchmark protocol.
5. Avoid retaining raw person imagery by default and document limitations of skeleton-only privacy.

## Functional requirements
- Video input: webcam index / local files; no cloud requirement after initial model weight installation.
- Multi-person tracking: separate track IDs, no identity recognition; uncertain if tracking breaks.
- Context detection: support chair, bed, sofa relationships only with documented confidence and camera-geometry limitations.
- Risk-adaptive verification: learnable/calibrated candidate, with independently measured ablations.
- Local events dashboard: event review and metadata only, with explicit retention settings.
- Research export: per-clip metrics, merged event metrics, parameter hashes, no raw footage.

## Non-functional
- MacBook Air Apple Silicon with MPS if supported; target >=10 FPS at 720p (to be tested).
- Inference fully local; default no raw-frame writes; no face recognition.
- Reproducible subject-independent tests, blind external set, no hidden peeking or changing operating points after test.
- Safety warning prominently visible: experimental prototype; not emergency response.

## Definition of done
The prototype runs live on the user's Mac with independently measured event-level performance, uncertainty/failure-case report, data rights documented and reproducible tests passing. Numerical targets are aspirational, not preclaimed.
