# RAC-Fall Phase 4C: Fall-Event Annotation and Evaluation Foundation

**Date:** 2026-11-08
**Status:** training-only pilot infrastructure complete; human landmark reviews pending
**Model training or validation evaluation:** none

## Outcome

Phase 4C adds a local frame-by-frame annotation, independent review, adjudication-support and event-evaluation foundation. The pilot queue covers three fall types and three ADLs from training subjects only. No event boundary was inferred from CAUCAFall ground-state labels, and no reviewer boundary was fabricated during implementation.

The pilot is structurally ready but annotation coverage is currently **0/6 reviewed sequences** and **0/30 exact landmarks**. Completing it requires two independent human reviewers and adjudication. Consequently no real event recall, delay or false-alarm result is reported.

## Frozen model and data isolation

- Frozen checkpoint `phase3a-causal-tcn-seed29-epoch6` was not loaded, rewritten or retrained. Its file SHA-256 remains `468ccf4132e988df5406034b6b306a882c9e92d6e0c16c4853c97337ad0bc72b`, matching the registry.
- Phase 4A and Phase 4B modules were not modified by the event-annotation workflow.
- The queue contains only Subjects 2, 4 and 6, all in the frozen training split.
- Subjects 3/7, Subjects 9/10 and GMDCSA-24 were not accessed.
- RGB is decoded locally and displayed in memory. The tool does not save screenshots, frames or rendered video.
- Annotation CSVs, queue metadata and validation output remain outside Git.

## Annotation contract

Each sequence has five separately reviewed landmarks:

1. `fall_initiation`: first visible involuntary loss of balance that continues into the fall.
2. `ground_contact`: first fall-related contact by a normally non-supporting body region.
3. `settled_ground_state`: first stable post-contact frame after major fall-related motion ceases.
4. `recovery_start`: first intentional attempt to leave the post-fall state.
5. `recovery_end`: first stable non-ground posture.

Every landmark uses one of `exact`, `unknown`, `ambiguous`, or `not_applicable`. Only `exact` may contain a zero-based frame index; all other statuses require frame `-1` and unknown/ambiguous decisions require a reason. The row also stores event validity, contact region, confidence, reviewer, review round, creation time, source and canonical-audit checksums, tool version and provenance.

The tool intentionally does not load or display source ground-state labels. In particular, a `0 → 1` ground-state transition is never proposed as fall initiation.

## Timing provenance and uncertainty

Timestamps are derived, never typed:

`timestamp_s = zero_based_frame / fps_proxy`

The current canonical pilot uses the frozen 20 FPS development proxy. Its half-frame quantization uncertainty is ±0.025 s. The systematic relationship between PNG ordinals and AVI timing is explicitly recorded as unknown; the pipeline does not assume equivalent indices and does not invent a numeric systematic bound.

## Training-only pilot inventory

| Subject | Sequence | Type | Frames |
|---|---|---|---:|
| Subject 2 | Fall forward | fall | 107 |
| Subject 4 | Fall backwards | fall | 193 |
| Subject 6 | Fall sitting | fall | 249 |
| Subject 2 | Sit down | ADL | 113 |
| Subject 4 | Walk | ADL | 241 |
| Subject 6 | Kneel | ADL | 190 |
| **Total** | **6 sequences** | **3 fall / 3 ADL** | **1,093** |

All six canonical audits were `verified`, none was quarantined, and every official PNG inventory was contiguous from source index 1 through its declared frame count.

Current coverage:

| Measure | Count |
|---|---:|
| Pilot sequences prepared | 6 |
| Sequences independently reviewed | 0 |
| Required independent review decisions | 12 |
| Landmark slots | 30 per reviewer |
| Exact landmarks currently accepted | 0 |
| Pending/unknown landmarks in reviewer-1 template | 30 |
| Adjudicated events available for evaluation | 0 |

This zero boundary coverage is a real limitation, not an annotation failure hidden by automatic estimates.

## Review and agreement support

`validate-event-annotations` checks schema, training-only scope, checksums, frame bounds, temporal ordering, exact/non-exact encoding, confidence, unknown reasons and reviewer provenance. It derives timestamps together with quantization and systematic timing uncertainty.

`compare-event-reviews` requires distinct, completed reviewers and reports:

- raw event-validity agreement and Cohen's kappa;
- exact-frame agreement;
- agreement within the frozen two-frame tolerance;
- known/unknown or exact/ambiguous disagreements;
- a per-sequence adjudication-required decision.

Event-validity disagreement, any known/unknown disagreement, or an exact-frame difference greater than two frames requires adjudication. Independent rows remain separate; the comparison does not overwrite either review.

## Event evaluator

The evaluator retains the frozen one-to-one causal policy: per-video alerts are sorted, alerts inside the 10 s refractory period are suppressed relative to the last retained alert, and the first unmatched retained alert in `[fall_initiation, fall_initiation + 5 s]` matches one event. An alert cannot match two events.

Reported outputs include event TP/FN/unmatched alerts, recall, precision, F1, initiation delay, separate ground-contact delay, duplicate-alert count, ADL exposure hours and false alarms per ADL hour. Negative ground-contact delay is preserved for a pre-contact alert. Late alerts do not rescue a missed event. Unknown or ambiguous initiation boundaries are excluded explicitly with counts; they are never converted into ground truth.

The event reader accepts adjudicated rows by default. Therefore pending or merely independent pilot annotations cannot accidentally become evaluation ground truth.

## Reproducibility

Configuration SHA-256:

`b30eb741c4f096043ff0bc3dbb777c8538657e6b3a918956959d260698f153e8`

External pilot artifacts under `~/datasets/RAC-Fall/CAUCAFall/phase4c/event-annotation-pilot/`:

| Artifact | SHA-256 |
|---|---|
| `pilot-queue.json` | `7b3e320b9ac6784b81257621ff5c0b3f41b4357efcb234264eef699fc719bf9c` |
| `annotations-reviewer-1.csv` | `c7058ad0117f13de2ac2c571a670eb5911eb2619879b0d21bded50750f98c9c8` |
| `pilot-validation.json` | `8689b4d54b6b37e63cf89351aea7f4bd0aecfb1c64045fa268508bec3317d19b` |

The artifacts total 16 KiB and contain metadata/decisions only, not RGB.

## Verification and remaining limitations

Synthetic tests cover one-to-one matching, duplicate suppression, late detection, missed falls, negative ground-contact delay, ambiguous onset exclusion, false alerts and ADL exposure, landmark ordering, forbidden subjects, pending reviews and IAA adjudication triggers.

The complete repository suite passed: **71 tests in 0.54 s**. Source compilation and `git diff --check` also passed.

Remaining limitations:

1. Two independent human reviews and adjudication have not been performed, so no event annotations are released.
2. The 20 FPS timing source is a proxy. Real-time delay claims require verified per-sequence timing or a defensible PNG-to-clock mapping.
3. Six sequences are sufficient to exercise the workflow, not to establish landmark prevalence, annotation reliability or model performance.
4. No real system alerts were evaluated. Event recall, detection delay and FA/hour remain unmeasured.
5. Annotation coverage must be expanded across the training subjects before any RACF development uses these landmarks; validation and sealed subjects remain unavailable for policy tuning.

Phase 4C stops at the requested training-only pilot and evaluation-tool verification boundary. No classifier was trained, and no validation or sealed data was opened.
