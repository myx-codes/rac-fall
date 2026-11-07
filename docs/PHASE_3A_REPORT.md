# RAC-Fall Phase 3A: Pose-TCN Baseline Training

Date: 2026-11-07
Status: completed on the Phase 2B.5 development split; no sealed-test evaluation

## Scope and label semantics

Phase 3A trains a lightweight causal temporal classifier on the verified Phase 2B.5 artifact (`SHA-256 1c7b461b281e401a636165b9c6be8c291b27f4ea1430fc3ec452df15cc2cf364`). It contains 494 train windows from Subjects 1, 2, 4, 5, 6 and 8, plus 161 validation windows from Subjects 3 and 7. Every window has 30 causal frames. Train labels are 359 class 0 and 135 class 1; validation labels are 116 class 0 and 45 class 1.

Class 1 means the person is on the ground due to a fall at the causal window endpoint. It is not a fall-event onset label. The results below are validation ground-state classification results, not fall-event detection accuracy, alert recall, or clinical performance.

Subjects 9/10 and GMDCSA-24 were not loaded, evaluated, or used for model selection. Validation was used for early stopping and checkpoint selection. Consequently these numbers are development results and are not an unbiased test estimate.

## Models and training protocol

The causal residual TCN receives 68 channels per frame: 51 normalized pose/confidence features and 17 explicit validity-mask channels. It uses three residual blocks, kernel size 3, 64 channels, dilations 1/2/4, ReLU, dropout 0.2, left-only padding and an endpoint logit. It has 79,361 parameters and a 29-frame receptive field over the 30-frame input. A unit test verifies that changing future timesteps cannot change earlier logits.

The static-pose baseline is a 69-parameter linear classifier over only the endpoint's 51 pose/confidence and 17 mask values. The majority baseline always predicts train-majority class 0.

Both learned models used AdamW (`lr=0.001`, weight decay `0.0001`), weighted binary cross entropy (`pos_weight=359/135=2.6592592593`), batch size 32, decision threshold 0.5, up to 100 epochs, validation-loss early stopping with patience 15 and minimum delta `0.0001`, and seeds 17, 29 and 43. The frozen configuration is in `configs/phase3a_pose_tcn.json`.

## Window-level validation results

All learned-model values in this table are mean ± population standard deviation across the three seeds.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Recall class 0 | Recall class 1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Majority class 0 | 0.7205 | 0.5000 | 0.4188 | 1.0000 | 0.0000 | 0.2795 |
| Static endpoint pose | 0.8820 ± 0.0051 | 0.9113 ± 0.0035 | 0.8670 ± 0.0052 | 0.8448 ± 0.0070 | 0.9778 ± 0.0000 | 0.9675 ± 0.0025 |
| Causal residual TCN | **0.9607 ± 0.0029** | **0.9591 ± 0.0039** | **0.9519 ± 0.0030** | **0.9626 ± 0.0108** | 0.9556 ± 0.0181 | **0.9873 ± 0.0017** |

The majority confusion matrix is `[[116, 0], [45, 0]]` in `[[TN, FP], [FN, TP]]` order.

### Causal TCN runs

| Seed | Epochs / best epoch | Best validation BCE | Accuracy | Balanced accuracy | Macro-F1 | Recall 0 / 1 | AUPRC | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 18 / 3 | 0.156042 | 0.9627 | 0.9605 | 0.9543 | 0.9655 / 0.9556 | 0.9855 | `[[112,4],[2,43]]` |
| 29 | 21 / 6 | **0.135643** | 0.9565 | 0.9630 | 0.9477 | 0.9483 / 0.9778 | 0.9896 | `[[110,6],[1,44]]` |
| 43 | 19 / 4 | 0.141548 | 0.9627 | 0.9537 | 0.9537 | 0.9741 / 0.9333 | 0.9866 | `[[113,3],[3,42]]` |

If one development checkpoint must be carried forward, seed 29 is selected solely by the prespecified minimum validation BCE. Its checkpoint SHA-256 is `468ccf4132e988df5406034b6b306a882c9e92d6e0c16c4853c97337ad0bc72b`. This does not make its validation metrics a test result.

### Static-pose runs

| Seed | Epochs / best epoch | Best validation BCE | Accuracy | Balanced accuracy | Macro-F1 | Recall 0 / 1 | AUPRC | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 100 / 97 | 0.286182 | 0.8820 | 0.9113 | 0.8670 | 0.8448 / 0.9778 | 0.9657 | `[[98,18],[1,44]]` |
| 29 | 100 / 99 | **0.275584** | 0.8882 | 0.9156 | 0.8734 | 0.8534 / 0.9778 | 0.9657 | `[[99,17],[1,44]]` |
| 43 | 100 / 93 | 0.280485 | 0.8758 | 0.9070 | 0.8607 | 0.8362 / 0.9778 | 0.9711 | `[[97,19],[1,44]]` |

The static baseline did not satisfy the patience rule before epoch 100 because validation loss continued to improve by at least the configured minimum delta.

## Subject-aware validation

Overlapping windows are not treated as independent for uncertainty. Metrics are reported separately for each validation subject and a 2,000-sample subject-block bootstrap samples whole subjects, never individual windows. With only Subjects 3 and 7, the intervals are descriptive and low-resolution; they must not be interpreted as population confidence intervals.

Mean TCN results across seeds:

| Subject | Windows | Accuracy | Balanced accuracy | Macro-F1 | Recall 0 | Recall 1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Subject 3 | 61 | 0.9344 | 0.9474 | 0.8943 | 0.9281 | 0.9667 | 0.9782 |
| Subject 7 | 100 | 0.9767 | 0.9711 | 0.9742 | 0.9897 | 0.9524 | 0.9905 |

For selected TCN seed 29, the subject-macro balanced accuracy is 0.9645 and subject-macro F1 is 0.9261. Its subject-block bootstrap ranges are accuracy 0.9180–0.9800, balanced accuracy 0.9510–0.9780, macro-F1 0.8742–0.9780, and AUPRC 0.9667–0.9956. These ranges largely reflect the two individual subjects rather than a broad subject population.

## Training and inference performance

Training ran on Apple MPS with PyTorch 2.15.0 and Python 3.11.18.

| Model / seed | Training seconds | Median batch-1 latency (ms/window) | Median batch-32 latency (ms/window) |
|---|---:|---:|---:|
| TCN / 17 | 96.734 | 0.3794 | 0.01459 |
| TCN / 29 | 118.368 | 0.3686 | 0.01468 |
| TCN / 43 | 105.578 | 0.3395 | 0.01470 |
| Static / 17 | 540.639 | 0.3056 | 0.00639 |
| Static / 29 | 544.641 | 0.4861 | 0.00640 |
| Static / 43 | 559.357 | 0.4455 | 0.00635 |

TCN training totaled 320.6799 seconds; static training totaled 1,644.6379 seconds because all static runs reached 100 epochs. The complete experiment, including evaluation, subject bootstrap and synchronized latency benchmarks, took 1,984.8238 seconds (33 minutes 4.824 seconds). Peak process RSS was 775,880,704 bytes. Latency used 10 warmups plus 100 synchronized MPS measurements per batch size and excludes pose extraction.

## Failure analysis

TCN seeds produced 6, 7 and 6 validation mistakes. Across runs these were 13 false-positive occurrences and six false-negative occurrences. Four windows failed in all three seeds:

- Subject 3 `Sit down`, frames 45–75, 105–135 and 210–240: class 0 predicted as class 1;
- Subject 7 `Fall forward`, frames 15–45: class 1 predicted as class 0.

The largest recurring cluster was Subject 3 `Sit down` (11 error occurrences across three runs). Other errors appeared in Subject 7 `Fall forward`, `Fall sitting`, `Fall right`, and Subject 3 `Fall right`. Because adjacent windows share frames, counts are not independent events and are not fall alarms.

The static baseline produced 18–20 errors per seed, dominated by false positives in `Sit down`, `Kneel` and `Pick up object`. Its higher class-1 recall but lower class-0 recall shows that endpoint posture alone confuses several ADL ground-adjacent poses; temporal context materially reduces these false positives on this validation split.

## Reproducibility and artifacts

The external experiment directory contains six checkpoints, six JSON training curves, six failure manifests, the exact configuration and `results.json`. No `.partial` files remain. The final results JSON is 30,645 bytes with SHA-256 `60fc36477f902248469a76f43898b3603419bdd98eeb72f6d2adaad9086e8956`. Checkpoints include the dataset SHA-256, seed, best epoch, validation loss, full model state and label semantics; the selected checkpoint was reloaded on CPU and passed a forward-shape check.

The automated suite has 43 passing tests, including causality, endpoint-only static behavior, validity-mask concatenation, metric/confusion-matrix correctness, tie-aware AUPRC and subject-block bootstrap behavior.

## Conclusion

The causal TCN outperforms both baselines on the two-subject development validation set, especially by reducing static-pose false positives. Phase 3A establishes a usable baseline and selected development checkpoint. It does not establish fall-event detection performance or generalization to sealed subjects. Subjects 9/10 and GMDCSA-24 must remain sealed until their prospectively defined evaluation phase.
