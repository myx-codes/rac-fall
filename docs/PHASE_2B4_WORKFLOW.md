# Phase 2B.4 Controlled CAUCAFall Workflow

This workflow uses official PNG files as the canonical frame order and same-index TXT files as ground-state labels. AVI is secondary: its container FPS may provide a timing-rate proxy, but AVI frame indices are never aligned to or substituted for PNG indices.

## Frozen split

The split is frozen in `configs/caucafall_split_phase2b4.json` before full preprocessing:

- train: Subjects 1, 2, 4, 5, 6, 8;
- validation: Subjects 3, 7;
- sealed internal test: Subjects 9, 10.

Subjects 3 and 5 were exposed during development; they are not test subjects. Subjects 9 and 10 had metadata-only inspection and no media exposure at freeze. Internal-test preprocessing and window generation require explicit flags. GMDCSA-24 remains a separate locked external blind test.

## Full-data commands

Do not run the download step until the user explicitly approves the official 8,326,349,365-byte Mendeley archive. Store archive, extraction and generated artifacts outside Git.

```bash
racfall manifest-caucafall-png \
  --root ~/datasets/RAC-Fall/CAUCAFall/CAUCAFall \
  --label-map configs/caucafall_activity_labels.json \
  --output ~/datasets/RAC-Fall/CAUCAFall/phase2b4/manifests/canonical.csv

racfall split-caucafall-frozen \
  --manifest ~/datasets/RAC-Fall/CAUCAFall/phase2b4/manifests/canonical.csv \
  --output ~/datasets/RAC-Fall/CAUCAFall/phase2b4/manifests/canonical-split.csv \
  --assignment-output ~/datasets/RAC-Fall/CAUCAFall/phase2b4/manifests/frozen-split.json

racfall audit-caucafall-canonical \
  --manifest ~/datasets/RAC-Fall/CAUCAFall/phase2b4/manifests/canonical-split.csv \
  --dataset-root ~/datasets/RAC-Fall/CAUCAFall/CAUCAFall \
  --output-dir ~/datasets/RAC-Fall/CAUCAFall/phase2b4/canonical-audit

# Run train/validation subject batches first. Test records remain sealed unless
# --allow-internal-test is explicitly supplied for the frozen evaluation.
racfall preprocess-caucafall-batch \
  --manifest /external/train-or-validation-batch.csv \
  --dataset-root ~/datasets/RAC-Fall/CAUCAFall/CAUCAFall \
  --canonical-audit-dir ~/datasets/RAC-Fall/CAUCAFall/phase2b4/canonical-audit \
  --output-dir ~/datasets/RAC-Fall/CAUCAFall/phase2b4/processed \
  --device mps --imgsz 416 \
  --detector-confidence 0.25 --keypoint-confidence 0.25 \
  --min-pose-coverage 0.5

racfall build-pose-sequences \
  --manifest /external/train-or-validation-batch.csv \
  --pose-dir ~/datasets/RAC-Fall/CAUCAFall/phase2b4/processed/poses \
  --alignment-dir ~/datasets/RAC-Fall/CAUCAFall/phase2b4/canonical-audit \
  --quality-dir ~/datasets/RAC-Fall/CAUCAFall/phase2b4/processed/quality \
  --output ~/datasets/RAC-Fall/CAUCAFall/phase2b4/windows/train-val.npz \
  --window-frames 30 --stride-frames 15 --min-track-coverage 0.5
```

## Fail-closed rules

- PNG indices must start at one, remain contiguous and have one source prefix. The documented official ` - copia.png` filename variant retains its embedded index.
- A same-index TXT supplies a ground-state label. Missing TXT becomes label `-1` with `label_valid=false`; it is never copied from a neighbor. TXT without a PNG, PNG gaps, prefix conflicts, decode failures or resolution changes quarantine the sequence.
- Class 1 means on the ground due to a fall. All event labels stay `-1` unless separately reviewed event boundaries exist.
- Pose inference never receives annotation bounding boxes. Sequence quality requires at least 50% causal target coverage by the frozen selector. Quarantined sequences are skipped by window generation when `--quality-dir` is supplied.
- Every source tree is checksum-verified against its manifest before inference. The resume ledger keys source, model and preprocessing configuration; existing pose and quality artifacts must also match their recorded SHA-256 values.
- Pose/quality writes and ledger updates use temporary files plus atomic rename. Batch summaries report class counts, valid/invalid labels, sequence status, missing target rate, throughput, artifact bytes and filesystem usage.

## Resume and integrity review

Before accepting a batch, confirm that `failed=0`, inspect every quarantine reason, reconcile requested/processed/resumed/sealed counts, and verify that the ledger contains one completed entry per allowed sequence. Re-running an unchanged batch must report `resumed_verified` and perform no inference. Any source or configuration change produces a new signature and recomputation; a checksum mismatch fails rather than overwriting provenance.
