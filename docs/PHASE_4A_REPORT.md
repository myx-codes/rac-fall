# RAC-Fall Phase 4A: Quality-Aware Risk-Adaptive Verification

Date: 2026-11-08
Status: completed once under the preregistered validation-only protocol; no model training

## Scope and immutable inputs

Phase 4A applies deterministic causal verification to the frozen Phase 3A Pose-TCN. It does not retrain or modify the TCN and does not fit another classifier. The run verified seed 29, epoch 6 and checkpoint SHA-256 `468ccf4132e988df5406034b6b306a882c9e92d6e0c16c4853c97337ad0bc72b`. It also verified the Phase 2B.5 window SHA-256 `1c7b461b281e401a636165b9c6be8c291b27f4ea1430fc3ec452df15cc2cf364` and Phase 3A configuration SHA-256 `d965b2b9a1ca636b694a2946e0bf757725ee3cb797e80fd334425ef7c6ee6008`.

Only 161 validation windows from Subjects 3 and 7 were loaded. No raw images, annotation boxes, Subjects 9/10 or GMDCSA-24 were accessed. The output remains ground-state endpoint verification. Confirmation delay is measured in 15-frame causal windows from a verified positive ground-state endpoint; it is not fall initiation delay, alert delay or an event-level result.

## Preregistered design

`configs/phase4a_quality_verifier.json` was written before the controlled run and has SHA-256 `ea49f1616d6f6646bc2c77ef0508855b3b22a7ea62450e83eca753c652c62e5f`. The results below are from the first run; no threshold was changed after observing them.

Every window uses only inference-available values: frozen TCN probability and causal one-step trend, normalized pose velocity, mean visible-keypoint confidence, missing-keypoint fraction, fully missing frames, track coverage and source-track continuity. Ground-state labels are used only after inference for evaluation.

The deterministic states are `NORMAL`, `CANDIDATE`, `UNCERTAIN`, `ALERT` and `COOLDOWN`. A sequence gap resets causal verification. An active alert persists while probability is at least 0.35; release enters a two-window cooldown. Regression tests cover the transition sequence and cooldown even though no validation timeline happened to exercise `COOLDOWN`.

The preregistered low-quality conditions are coverage below 0.75, missingness above 0.40, confidence below 0.65, more than seven fully missing frames, or more than two track switches. Low quality is never converted to NORMAL by any verification variant. The frozen TCN remains an unchanged comparison and therefore has no abstention mechanism.

Four variants were evaluated:

1. **Frozen TCN:** direct probability threshold 0.5.
2. **Fixed verification:** two consecutive positive windows; low-quality negative evidence becomes UNCERTAIN.
3. **Quality-aware verification:** two consecutive positive windows and every low-quality window becomes UNCERTAIN.
4. **Risk-adaptive verification:** low-quality windows become UNCERTAIN; a fixed weighted causal risk score maps to one, two or three confirmation windows.

## All-sample outcomes

`ALERT` is the positive decision. `NORMAL`, `CANDIDATE`, `COOLDOWN` and `UNCERTAIN` are not positive; consequently every uncertain positive remains an FN in all-sample recall. FP and FN below therefore cover all 161 windows rather than only retained samples.

| Variant | FP | FN | Precision | Recall | Macro-F1 | UNCERTAIN | Coverage | Mean ground-state confirmation delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen TCN | 6 | 1 | 0.8800 | 0.9778 | **0.9477** | 0.00% | 100.00% | 0.2 windows |
| Fixed verification | 1 | 6 | 0.9750 | 0.8667 | 0.9441 | 6.21% | 93.79% | 1.0 windows |
| Quality-aware verification | 0 | 11 | 1.0000 | 0.7556 | 0.9077 | 11.18% | 88.82% | 1.6 windows |
| Risk-adaptive verification | 0 | 8 | 1.0000 | 0.8222 | 0.9346 | 11.18% | 88.82% | 1.2 windows |

All five observed positive ground-state runs eventually reached ALERT in every variant, but per-window recall falls because CANDIDATE and UNCERTAIN windows are not counted as positives. The longest quality/risk-aware confirmation delay was four windows. This is a ground-state diagnostic over overlapping windows, not evidence that an actual fall would be detected four windows late.

## Abstention accounting

Quality-aware and risk-adaptive verification each produced 18 uncertain windows: 14 label-0 and four label-1. To prevent selective reporting, every UNCERTAIN was also treated as an error against its known validation class in a strict secondary analysis.

| Variant | Strict FP-equivalent | Strict FN-equivalent | Strict Macro-F1 | Conditional non-abstained Macro-F1 | Conditional recall | Conditional coverage |
|---|---:|---:|---:|---:|---:|---:|
| Frozen TCN | 6 | 1 | 0.9477 | 0.9477 | 0.9778 | 100.00% |
| Fixed verification | 11 | 6 | 0.8731 | 0.9427 | 0.8667 | 93.79% |
| Quality-aware verification | 14 | 11 | 0.8110 | 0.9367 | 0.8293 | 88.82% |
| Risk-adaptive verification | 14 | 8 | 0.8367 | 0.9647 | 0.9024 | 88.82% |

Risk-adaptive conditional Macro-F1 is higher than the baseline, but that comparison excludes 11.18% of windows and is therefore not the primary conclusion. Its strict result is lower than the baseline. The uncertainty state is useful for exposing unreliable evidence, but the current gate does not yet demonstrate an overall classification improvement.

## Targeted failure analysis

### Subject 3 / Sit down

The frozen model produces five false-positive ALERT windows among 14. Fixed verification reduces them to one, with four CANDIDATE windows and three uncertain windows. Both quality-aware variants produce zero alerts, but they mark seven of 14 windows UNCERTAIN and leave one as CANDIDATE. Their apparent 14/14 non-alert outcome therefore includes only 50% coverage; strict accounting treats the seven abstentions as errors rather than hidden true negatives.

The result confirms that the Phase 3B quality features identify the problematic windows, but the gate is broad: it also abstains on other low-quality windows. It has not learned that the activity is Sit down and receives no action label or ground-truth geometry at inference.

### Subject 7 / Fall forward

The original frames 15–45 FN remains NORMAL for all variants because the frozen TCN probability is 0.0501 and input quality is high; quality gating cannot repair a confident temporal model error. Frozen TCN then alerts on seven of the remaining positive windows. Fixed verification alerts on six, quality-aware on four, and risk-adaptive on six. Risk-adaptive verification therefore adds one further FN relative to the baseline in this sequence and marks one positive window uncertain because confidence falls below the preregistered gate.

This separates two failure types: quality-aware logic can intercept unreliable Sit-down evidence, but it cannot correct a high-quality, low-probability Fall-forward transition without changing the frozen temporal representation or introducing verified event supervision.

## Trade-off conclusion

Quality-aware verification improves the **false-positive side** of the validation trade-off but does not improve the complete trade-off under strict accounting. Risk-adaptive verification is better than fixed-duration quality-aware verification at the same 11.18% abstention rate: it improves all-sample recall from 0.7556 to 0.8222, Macro-F1 from 0.9077 to 0.9346 and mean confirmation delay from 1.6 to 1.2 windows while retaining zero FP. However, compared with the frozen TCN it exchanges six fewer FP for seven additional FN and 18 abstentions. The fixed verifier is the closest overall comparator: one FP, six FN, 6.21% uncertainty and Macro-F1 0.9441, but its recall loss is still material.

Therefore Phase 4A does **not** establish that quality-aware or risk-adaptive verification is preferable. It establishes a reproducible operating trade-off and an explicit uncertainty path. Further threshold adjustment against these same seven known errors would be validation overfitting and was not performed.

## Future learned verifier guardrail

No learned verifier was used. If one is pursued, its training inputs must first be built from subject-grouped out-of-fold predictions on training Subjects 1, 2, 4, 5, 6 and 8. Each fold trains its base predictor without one complete subject and generates causal predictions only for that held-out training subject. The verifier may fit only on the concatenated out-of-fold training predictions. Subjects 3/7 remain model-selection/calibration data and cannot supply verifier training rows; Subjects 9/10 and GMDCSA-24 remain sealed. The fold assignment, seeds, features and calibration method must be frozen before producing validation predictions.

## Reproducibility

The implemented command is:

```bash
racfall verify-phase4a \
  --windows ~/datasets/RAC-Fall/CAUCAFall/phase2b5/windows/trainval.npz \
  --checkpoint ~/datasets/RAC-Fall/CAUCAFall/phase3a/pose-tcn-baseline/checkpoints/causal_tcn-seed-29.pt \
  --output-dir ~/datasets/RAC-Fall/CAUCAFall/phase4a/quality-verification \
  --device mps
```

Generated `results.json` and the 161-row per-window state/feature CSV are outside Git. The first-run results SHA-256 is `dc689363db1ba004eca11abe5869455ade89b3d9ca79fb393e3abf142cd58494`; no `.partial` files remain. Automated regression tests cover tracking loss, missing keypoints, low-quality abstention, probability fluctuation, fixed confirmation, risk-adaptive confirmation, cooldown and strict/conditional metrics.
