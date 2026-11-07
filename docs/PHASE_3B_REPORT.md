# RAC-Fall Phase 3B: Baseline Freezing and Failure Analysis

Date: 2026-11-08
Status: completed on Subjects 3/7 validation data only; no retraining and no event-level evaluation

## Frozen baseline and scope

The Phase 3A causal TCN is registered as `phase3a-causal-tcn-seed29-epoch6`: seed 29, epoch 6, threshold 0.5, 79,361 parameters and validation BCE 0.13564294. The checked-in registry is `configs/phase3b_frozen_baseline.json`. Analysis verified all three immutable inputs before inference:

| Artifact | SHA-256 |
|---|---|
| Selected checkpoint | `468ccf4132e988df5406034b6b306a882c9e92d6e0c16c4853c97337ad0bc72b` |
| Phase 3A configuration | `d965b2b9a1ca636b694a2946e0bf757725ee3cb797e80fd334425ef7c6ee6008` |
| Phase 2B.5 windows | `1c7b461b281e401a636165b9c6be8c291b27f4ea1430fc3ec452df15cc2cf364` |

The original checkpoint was read but not rewritten. Only the 161 validation windows from Subjects 3 and 7 were evaluated. Subjects 9/10 and GMDCSA-24 were not accessed. Predictions, reports, registry verification and 15 SVG probability plots are outside Git under `~/datasets/RAC-Fall/CAUCAFall/phase3b/failure-analysis/`.

This remains ground-state endpoint classification. Class 1 means on the ground due to a fall at the final window frame. Nothing in Phase 3B measures fall initiation, ground contact, an alert, detection delay or false alarms/hour.

## Reproduced frozen performance

The frozen checkpoint exactly reproduced Phase 3A: accuracy 0.956522, balanced accuracy 0.963027, macro-F1 0.947739, class-0 recall 0.948276, class-1 recall 0.977778, AUPRC 0.989613 and confusion matrix `[[110, 6], [1, 44]]`. Subject 3 had five false positives and no false negatives; Subject 7 had one false positive and one false negative.

## Complete error audit

There are seven erroneous windows in six non-overlapping/overlapping clusters:

| Subject / sequence | Frames | Truth → prediction | p(class 1) | Missing keypoints | Full missing frames | Track coverage |
|---|---:|---:|---:|---:|---:|---:|
| S3 / Sit down | 45–75 | 0 → 1 | 0.7384 | 49.4% | 12 | 60.0% |
| S3 / Sit down | 75–105 | 0 → 1 | 0.7611 | 46.5% | 12 | 60.0% |
| S3 / Sit down | 105–135 | 0 → 1 | 0.5954 | 44.3% | 10 | 66.7% |
| S3 / Sit down | 195–225 | 0 → 1 | 0.5371 | 28.2% | 5 | 83.3% |
| S3 / Sit down | 210–240 | 0 → 1 | 0.8320 | 51.2% | 12 | 60.0% |
| S7 / Fall sitting | 15–45 | 0 → 1 | 0.6283 | 12.2% | 0 | 100% |
| S7 / Fall forward | 15–45 | 1 → 0 | 0.0501 | 9.4% | 0 | 100% |

The two Sit-down windows at 195–225 and 210–240 overlap and form one cluster; the other errors are distinct at the window-overlap level. The CSV retains all 161 rows so correct neighbors and skipped candidate endpoints remain visible.

## Sit-down trajectory findings

Subject 3 / Sit down accounts for 5/6 false positives and 5/14 sequence windows. Its false-positive windows have median missing-keypoint rate 46.5%, 12 fully missing frames, 60% track coverage, one track switch and zero normalized endpoint velocity. Correct windows from the same sequence have medians of 22.7% missing keypoints, four fully missing frames, 86.7% coverage, one switch and endpoint velocity 0.0656. Mean visible confidence is also lower for the false positives (mean 0.7771 versus 0.8043).

Across all validation errors, median missingness is 44.3% versus 13.1% for correct predictions, full missing frames are 10 versus 0, track coverage is 66.7% versus 100%, and track switches are one versus zero. Incorrect windows have slightly higher median mean motion (0.1203 versus 0.0991 normalized units/frame), but lower endpoint motion (0 versus 0.0605). These descriptive comparisons support two interacting failure mechanisms:

1. pose/track loss leaves long zero-filled intervals and weak endpoint evidence;
2. a stationary seated/ground-adjacent posture is then confused with the learned ground state.

This is an association on one development subject, not proof of causation. The clean Subject 7 Sit-down sequence has 0/11 errors, full track coverage and no fully missing frames, reinforcing the quality/domain interaction. The S7 Fall-sitting FP is different: input quality is high and the next window changes to source class 1, so the model produces a class-1 score before the ground-state endpoint label changes. That is a classification FP, not evidence of event detection. The sole FN is the first positive Fall-forward endpoint; quality is high, but the score remains 0.0501 before rising to 0.9983 in the next overlapping window. This is a temporal transition/representation failure rather than pose dropout.

## Inference consistency and sensitivity

Twenty repeated synchronized MPS passes over all 161 windows produced zero decision disagreements and zero maximum probability difference from the first pass. The reported maximum probability standard deviation was `2.98e-7`, a floating-point variance-computation residual. Total repeated inference time was 0.261595 s, or 13.0797 ms per 161-window pass; this excludes pose extraction.

Fixed, prespecified sensitivity checks were diagnostic only and did not change the checkpoint or threshold:

| Perturbation | Accuracy | Macro-F1 | Decision flips | Mean / max probability change |
|---|---:|---:|---:|---:|
| valid-coordinate noise, sigma 0.01 | 0.9565 | 0.9477 | 0 | 0.00016 / 0.00327 |
| 10% valid-keypoint dropout | 0.9441 | 0.9336 | 2 | 0.01541 / 0.21305 |
| final three pose frames missing | 0.9503 | 0.9399 | 1 | 0.06760 / 0.32324 |

The baseline is stable to small coordinate noise but materially more sensitive to missing evidence, consistent with the observed Sit-down failures. These are validation-only stress tests, not estimates of deployment corruption or reasons to retune Phase 3A.

## Evaluation and annotation protocols

`docs/PHASE_3B_PROTOCOLS.md` freezes the future ground-state comparison rules and separately defines event annotation/evaluation. True fall initiation, ground contact, recovery start and recovery end require independent human review and adjudication; they are never copied from ground-state transitions. Future event reports must include one-to-one event recall/precision/F1, initiation-to-alert delay, separately reported ground-contact delay, recovery/reset behavior and false alarms/hour on verified negative exposure. No such annotations or results were created here.

## Concrete Phase 4 RACF design

Keep the frozen TCN as M1 and expose its causal probability, logit slope and recent stability. Add a small causal verifier input containing motion magnitude, pose confidence, missing-keypoint rate, fully missing frames, track coverage/switches and—only after an inference-safe context extractor exists—chair/bed/sofa proximity and coarse support relations. Annotation bounding boxes never enter inference.

Use deterministic states `NORMAL → CANDIDATE → ALERT → COOLDOWN`, with `UNCERTAIN` as an explicit quality outcome rather than NORMAL. A calibrated candidate score starts verification. A bounded monotonic policy may shorten confirmation for high-risk, high-quality trajectories and lengthen it for ambiguous sit/kneel motions; quality below a frozen floor yields UNCERTAIN. The adaptive policy cannot use future frames, cannot suppress a candidate merely because pose is missing, and must expose its reason/features in the error report.

Evaluate M1, fixed verification M2, context M3, quality/abstention M4-quality and full risk-adaptive M4-RACF under the prospective ablations in `PHASE_3B_PROTOCOLS.md`. The primary comparison is fixed versus adaptive verification at matched event recall and matched ADL false-alarm budgets, with delay, calibration, selective risk/coverage and compute cost reported together. If the fixed verifier is not credibly worse, retain the simpler design. Training and event evaluation must wait for separate verified event annotations; sealed data remains untouched.

## Reproducibility and tests

The command is:

```bash
racfall analyze-phase3b \
  --windows ~/datasets/RAC-Fall/CAUCAFall/phase2b5/windows/trainval.npz \
  --checkpoint ~/datasets/RAC-Fall/CAUCAFall/phase3a/pose-tcn-baseline/checkpoints/causal_tcn-seed-29.pt \
  --output-dir ~/datasets/RAC-Fall/CAUCAFall/phase3b/failure-analysis \
  --device mps
```

The external `analysis.json` SHA-256 from this run is `9de0d6b485be729c7e875fa3b5436ed004ad81c0234ec164402f9e9e68cd7893`; the external frozen registration SHA-256 is `846fed84cfb975d043190b25de1036d4310d4dba3137267e2e37589177b638e1`. The complete output directory is 144 KiB and has no `.partial` files. All 49 automated tests pass. They cover diagnostics, overlapping error clusters, registry integrity metadata, activity decoding and fail-closed validation subject scope; source compilation and `git diff --check` also pass.
