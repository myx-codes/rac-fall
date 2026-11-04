# RAC-Fall: Research proposal v1.1
Date: 2026-11-04 | Stage: preregistration draft | Not tested

## Problem
When a fall alarm is emitted on the basis of a single posture or fixed-confidence threshold, benign transitions (sitting, kneeling, lying on a sofa) can cause repeated false alarms. Increasing the confirmation threshold can miss real falls. A new 2026 paper already investigates causal RGB–pose TCN, reliability gating and alarm verification (Xiao et al., 2026); combining those blocks alone is **not novel**. Our prospective contribution is a **predefined, risk-conditioned verification policy**, evaluated under *matched event recall / false-alarm budget*, with quality-based abstention and audited operational costs.

## RQs and falsifiable hypotheses
- RQ1: At matched event-level recall, can risk-conditioned verification reduce false alarms/hour relative to fixed verification? H1: RACF decreases ADL false alarms/hour at an operating point with event recall >= 90% (target, not guarantee). If no improvement or large confidence intervals, H1 is not supported.
- RQ2: Are object relations actually useful beyond pose and motion? H2: Context features reduce confusing ADL alarms with no statistically credible fall-recall loss. Compare M3 vs M1 and report camera-domain shift.
- RQ3: Can uncertainty-aware abstention improve calibration and limit unsafe silent failures? H3: Selective risk/coverage curves and missed-fall analysis improve over single-state decisions. 'Uncertain' does **not** count as an emergency alert unless a separately prespecified intervention policy sends one.
- RQ4: How much accuracy is lost by minimal-retention local inference relative to RGB-assisted inference? H4: Skeleton-only vs RGB-assisted performance, privacy risks and compute budgets are explicitly compared; no guarantee skeleton-only is anonymous.

## Predefined models
M0 posture heuristic; M1 pose TCN; M2 pose TCN + fixed verification; M3 pose TCN + static context; M4 M3 + **risk-adaptive** verifier and quality abstention. Training folds, thresholds, refractory periods, alert matching and architecture are locked before testing on GMDCSA-24.

## Mechanism concept (to be trained/ablated)
At each causal time t, compute fall posterior p_t, motion risk r_t, scene context c_t, and quality q_t from only frames up to t. A calibrator learns the probability that a suspected event is a true fall. A bounded, monotonic mapping from risk to confirmation time/threshold may allow short verification for high-risk, high-quality sequences while requesting additional evidence for ambiguous benign motions. Very low quality yields UNCERTAIN, not implicit NORMAL. Verify whether simpler fixed threshold dominates before asserting adaptive benefits.

## Success criteria (aspirational, all measured together)
- Event recall >= 90%; false alarms <= 0.1/hour on representative ADL exposure; median alarm delay <= 2 s; live end-to-end >= 10 FPS at 720p on an actual MacBook Air M5.
- Main comparison: improvement in FA/hour **at matched recall**, or equivalently recall at a fixed false-alarm budget, with bootstrap CIs clustered by subject where feasible.
- Report performance at real test operating points even when targets fail, plus errors due to pose dropout, occlusion and scene context mistakes.

## Non-goals
No claim of emergency-grade clinical reliability, predictive fall prevention, identification of a person, face recognition, or anonymization merely by removing RGB.

## Source starting points
- Xiao et al., 2026: https://doi.org/10.3390/info17111077
- CAUCAFall: https://pmc.ncbi.nlm.nih.gov/articles/PMC9508401/
- GMDCSA-24: https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/
- 2026 systematic review: https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70115
