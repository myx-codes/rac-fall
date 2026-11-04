# Event-level evaluation protocol v1.1 (predefined before tuning)

## Basic units
- One **fall event** is a real transition to an uncontrolled loss-of-balance condition, not every frame in 'fall' posture.
- Each ground-truth event: `video_id,event_id,onset_s,end_s`, seconds from clip start. The onset definition must be consistent across datasets and is not automatically available from all frame labels.
- Each system **alert**: `video_id,alert_s` emitted strictly causally. A low-priority `UNCERTAIN` display is not counted as an emergency alert unless a fixed notification policy says so.

## Detection and matching (reference evaluator implementation)
1. Sort alerts within each video; impose a **10-second refractory period** from the last retained alarm (`cooldown_s=10`). This applies equally to all models and may be changed ONLY based on development sets before locking evaluation.
2. For each GT fall event in onset order, match the first unmatched alert in `[onset_s, onset_s + 5]` seconds (inclusive). One alert cannot match two falls; additional retained alarms are unmatched.
3. TP = matched GT; FN = unmatched GT; FP = unmatched retained alerts across videos. Recall = TP/(TP+FN); precision = TP/(TP+FP); F1 as harmonic mean. If no events or denominator zero, report `null`, not an invented perfect score.
4. Alert delay = alert time minus annotated onset (for matched events). Median of matched delays, in seconds.
5. **FA/hour on pure ADL** = count of retained alerts on clips marked `has_fall=0`, divided by the sum of those clips' durations in hours. This is intentionally different from total FP and only valid for *known negative exposure*. If no negative exposure, report `null`.

The reference code in `src/racfall/evaluation/events.py` checks event bounds and subject manifests. An alert prior to onset does not count as a true positive under this policy; separate pre-fall prediction research would need another policy.

## Operating-point comparison
- Select thresholds only on development data. For target recall ≥90%, optimize FP/hour subject to the constraint. Also report FA/hour versus recall Pareto curve and delay; a method cannot be declared superior merely by suppressing alarms.
- Compare M0, M1, M2, M3, M4 with identical event-matching/refractory policy. Report operating points at matched recall and at matched ADL FA/h. Include calibration (ECE/Brier) for event probability outputs and uncertainty coverage/risk curves.
- For subject-level paired comparisons, bootstrap **subjects or sessions**, not independent frames. Report confidence intervals with protocol and resampling seed; 4-person external set provides wide uncertainty.
- Timing: compare live causal end-to-end capture-to-alert latency to offline compute time; include FPS, p50/p95 inference latency and power/memory where measurable. Don't assert online causality based solely on offline sliding-window evaluation.

## Validation gates
No dataset -> no benchmark; no true onset labels -> no valid onset-to-alarm delay; too little ADL exposure -> no confident low FA/h claim; no independent subjects -> insufficient generalization claim.
