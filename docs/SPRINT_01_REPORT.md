# Sprint 01 — initial package status (2026-11-04)

## Finished in the starter package
- Source-based research and dataset license/audit checklist.
- Clear novelty boundary against Xiao et al. (2026).
- Event-level, ADL-exposure evaluation protocol and reproducible Python implementation.
- Unit tests with synthetic examples (not benchmark results).
- Optional YOLO26n pose inference smoke-test command and system/MPS diagnostics.

## Not finished / cannot be claimed
- Datasets not downloaded; individual media terms not yet verified.
- CAUCAFall subjects not yet assigned to exact reproducible split.
- No trained baseline, no live alert engine, no real fall metrics.
- No Mac M5 FPS/latency measurement; runner is not the user's machine.
- GMDCSA test not opened or evaluated, preserving blind status.

## Immediate next step on Mac
Install project and run unit tests and `racfall env-check`; then run one consented short MP4 with `racfall pose-smoke`. Record OS/torch/MPS/model load/actual FPS before building M0. Keep 2+ weeks for dataset/onset audit as per original plan.
