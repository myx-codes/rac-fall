# 16-week gate-based roadmap
Based on the user-provided master plan (`MASTER_PLAN_V1.md`), starting 2026-11-05; dates are provisional and depend on data rights/quality.

| Weeks | Task | Completion gate |
|---|---|---|
| 1–2 (Nov 5–18) | Research, legal dataset audit, subject split, evaluation specification | All datasets inspected locally; manifest checked; event annotation definition locked |
| 3–4 (Nov 19–Dec 2) | Webcam/MP4 pose and tracking, M0 posture baseline | On Mac M5 tested with measured FPS, end-to-end demo, reproducible M0 metrics |
| 5–6 (Dec 3–16) | Pose normalization, causal TCN | M1 compared with M0 under same subject split |
| 7–9 (Dec 17–Jan 6) | Context and calibrated RACF | M2/M3/M4 ablations; matched recall trade-offs |
| 10–11 (Jan 7–20) | Privacy, uncertainty, retention auditing | No raw-video persistence in default mode verified; quality-abstention tests |
| 12–14 (Jan 21–Feb 10) | Locked cross-dataset evaluation & stress tests | Blind GMDCSA once; grouped CIs, long negative ADL test |
| 15–16 (Feb 11–24) | Real-time app/optimization/docs | Working locally evaluated prototype; GitHub, report, demo with consent |

## Sprint 01 deliverables in this starter
- Research hypothesis and prior-art matrix — **drafted**.
- Dataset protocol — **metadata-verified, actual downloads pending**.
- Event evaluator + tests — **implemented and synthetic-tested**.
- Pose smoke + MPS checker — **implemented, requires Mac validation**.
- Dashboard and trained fall recognition — **deliberately deferred**.

## Explicit change control
Before each stage: write specification + acceptance tests; Codex implements only that stage; run experiments; peer review logs/data leaks; commit. No evaluation target may be edited post-blind-test to make results look stronger.
