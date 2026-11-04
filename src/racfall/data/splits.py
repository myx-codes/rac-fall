"""Reproducible subject-independent split assignment."""
from __future__ import annotations

import random
from typing import Iterable

from .manifest import ManifestRecord, validate_manifest, with_split


def assign_subject_splits(
    records: Iterable[ManifestRecord],
    *,
    seed: int = 20261104,
    train_subjects: int = 6,
    val_subjects: int = 2,
    test_subjects: int = 2,
) -> tuple[list[ManifestRecord], dict]:
    rows = list(records)
    if any(row.locked_blind for row in rows):
        raise ValueError("Locked blind records cannot be assigned development splits")
    datasets = {row.dataset for row in rows}
    if datasets != {"CAUCAFall"}:
        raise ValueError("The 6/2/2 subject split is defined only for CAUCAFall")
    subjects = sorted({row.subject_id for row in rows})
    if any(subject.lower() in {"unknown", "unverified"} for subject in subjects):
        raise ValueError("Verified subject IDs are required")
    expected = train_subjects + val_subjects + test_subjects
    if len(subjects) != expected:
        raise ValueError(f"Expected exactly {expected} subjects, found {len(subjects)}")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    assignment = {
        "train": sorted(subjects[:train_subjects]),
        "val": sorted(subjects[train_subjects:train_subjects + val_subjects]),
        "test": sorted(subjects[train_subjects + val_subjects:]),
    }
    by_subject = {subject: split for split, members in assignment.items() for subject in members}
    output = [with_split(row, by_subject[row.subject_id]) for row in rows]
    validate_manifest(output)
    return output, {"seed": seed, "algorithm": "sorted subjects + Python random.Random.shuffle", **assignment}
