# Phase 3B frozen evaluation protocols

These protocols separate the currently measurable **ground-state endpoint classification** task from a future **fall-event** task. Results from one must never be reported as results for the other.

## Ground-state baseline protocol

The registered M1 baseline is `phase3a-causal-tcn-seed29-epoch6`, threshold 0.5. Its checkpoint, Phase 3A configuration and `racfall.windows.v3` preprocessing artifact must pass the SHA-256 values in `configs/phase3b_frozen_baseline.json` before evaluation. A mismatch is a hard failure. No weights, input labels, normalization, threshold or missing-value policy may change under this baseline ID.

The unit of prediction is a 30-frame causal window. The target is the verified CAUCAFall ground-state label at the final input frame: class 1 means on the ground due to a fall; it does not mean that a fall began within the window. Report accuracy, balanced accuracy, macro-F1, both class recalls, AUPRC and `[[TN, FP], [FN, TP]]` at the frozen threshold. Also report each subject and each sequence, all error windows, merged overlapping error clusters, pose quality and track continuity. Never treat overlapping windows as independent events or bootstrap them independently. Development uncertainty uses subject/session blocks; with two validation subjects it is descriptive.

Future ground-state comparisons must use the same eligible windows, labels, split, preprocessing and threshold-selection budget. Model-selection decisions use training/validation only. Compare paired per-window outputs descriptively, but draw uncertainty at the subject/session level. Report calibration and selective risk/coverage when models emit uncertainty. Subjects 9/10 remain unopened until a separately approved, frozen test run; GMDCSA-24 remains locked.

## Separate fall-event annotation protocol

Event annotation is a new immutable human-reviewed layer; it is never inferred from a source class transition. For every verified fall sequence, two independent reviewers mark zero-based frames for:

- `fall_initiation`: first visible involuntary loss of balance that continues into the fall;
- `ground_contact`: first fall-related contact by a normally non-supporting body region;
- `recovery_start`: first intentional attempt to leave the post-fall state;
- `recovery_end`: first stable non-ground posture.

Every landmark can be `-1` with an explicit unknown reason. Reviewers work from verified AVI/source-frame alignment, record contacted body region and confidence, and preserve independent rows. A third reviewer adjudicates known/unknown disagreement or disagreement over two frames. Temporal ordering, bounds, alignment checksum and media checksum are validated. The detailed immutable CSV contract remains in `FALL_EVENT_ANNOTATION_PROTOCOL.md`.

## Future event evaluation protocol

Before any sealed evaluation, freeze the causal candidate threshold, verifier policy, alert threshold, refractory period, matching interval, uncertainty action and all calibration parameters. An alert is a timestamped system decision, not every positive window.

Report event recall/precision/F1 after one-to-one matching, unmatched events, unmatched alerts, false alarms per hour on verified negative exposure, and alert delay from `fall_initiation`. Also report delay from `ground_contact` as a separate secondary quantity: negative values indicate pre-contact alerts and must not be clipped. Recovery behavior is evaluated separately as alert persistence/reset delay after `recovery_start` and `recovery_end`; it cannot redefine event recall. Unknown landmarks are excluded only from the metric that requires that landmark, with the exclusion count reported.

The reference matching policy currently uses a 10-second refractory period and matches the first unmatched alert in `[fall_initiation, fall_initiation + 5 s]`. Any alternative or pre-fall-warning analysis is a separate prespecified endpoint. Subject/session block bootstrap is required. Ground-state metrics may accompany event results for diagnosis, but may not substitute for them. No event performance exists until the separate annotations are completed.

## Prospective Phase 4 ablations

All ablations retain the frozen Pose-TCN input and operate causally:

1. **M1 frozen TCN:** probability threshold only.
2. **M2 fixed verification:** identical candidate threshold and one globally fixed confirmation duration/threshold.
3. **M3 context:** M2 plus causally computed chair/bed/sofa proximity and coarse support/floor relations; no annotation boxes at inference.
4. **M4-quality:** M3 plus confidence, missing-keypoint rate, track coverage/switches, probability stability and calibrated abstention.
5. **M4-RACF:** the same inputs with a bounded monotonic risk-conditioned confirmation policy.

Compare M2–M4 at matched event recall and at matched ADL false-alarm budgets. Also report delay, calibration, selective coverage/risk, latency, state occupancy and the number of `UNCERTAIN` outcomes. Fit/calibrate only on development data and freeze all choices before sealed tests. Report the simpler fixed verifier as the preferred result if it is not credibly worse than RACF.
