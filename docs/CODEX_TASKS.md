# Codex tasks — strictly incremental

## TASK-001: Mac feasibility, dataset audit infrastructure

**Scope:** Extend the existing repository; do not replace documentation, change benchmark matching policy, build dashboard, or train an unvalidated fall model.

1. Inspect project, run `pytest -q`, `racfall env-check` and report actual Mac M5, Python, torch, MPS details.
2. Fix only verified Mac install/runtime problems. Run `racfall pose-smoke` on a short consented sample MP4 (or webcam when explicitly permitted). Report FPS, model load status, resolution, p50/p95 frame timing if feasible.
3. Implement a dataset manifest auditor accepting local source files and validating clip IDs, subject IDs, duration, FPS, duplicate clips, SHA256 and basic license metadata. **Never auto-download** unverified user-sensitive footage; never commit media or paths with personal details.
4. Add unit tests for audit cases. Keep GMDCSA-24 blind: do not preview labels/clips or tune on it.
5. Output a concise `docs/TASK_001_REPORT.md` with executed commands, actual pass/fail logs, open risks and next gate. Commit only code/docs/tests; no video or model weights.

**Acceptance:** tests pass; no private data persisted; MPS issue and first-model inference are evidenced, not assumed; source audit manifest has explicit unresolved fields. If no sample is available, report blocked instead of inventing timings.

## TASK-002 (not yet authorized)
Only after dataset audit and onset definitions are approved: implement M0 posture rules with testable feature extraction and event metrics.
