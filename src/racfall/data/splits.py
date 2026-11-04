"""Reproducible subject-independent split assignment."""
from __future__ import annotations

import random
from typing import Iterable

from .manifest import ManifestRecord, validate_manifest, with_split


# These subjects were inspected during the Phase 2B.2/2B.3 development audit.
# They are no longer eligible to represent an untouched internal test set.
DEVELOPMENT_EXPOSED_SUBJECTS = frozenset({"Subject.3", "Subject.5"})


def assign_subject_ids(
    subject_ids: Iterable[str],
    *,
    seed: int = 20261104,
    train_subjects: int = 6,
    val_subjects: int = 2,
    test_subjects: int = 2,
    exposed_subjects: Iterable[str] = DEVELOPMENT_EXPOSED_SUBJECTS,
) -> dict:
    """Assign verified subject IDs independently of media availability or row order."""
    supplied = list(subject_ids)
    subjects = sorted(set(supplied))
    if len(subjects) != len(supplied):
        raise ValueError("Subject IDs must be unique")
    if any(subject.lower() in {"unknown", "unverified"} for subject in subjects):
        raise ValueError("Verified subject IDs are required")
    expected = train_subjects + val_subjects + test_subjects
    if len(subjects) != expected:
        raise ValueError(f"Expected exactly {expected} subjects, found {len(subjects)}")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    assignment = {
        "seed": seed,
        "algorithm": "sorted unique subject IDs + Python random.Random.shuffle",
        "train": sorted(subjects[:train_subjects]),
        "val": sorted(subjects[train_subjects:train_subjects + val_subjects]),
        "test": sorted(subjects[train_subjects + val_subjects:]),
    }
    exposed_test = sorted(set(assignment["test"]) & set(exposed_subjects))
    if exposed_test:
        raise ValueError(
            "Development-exposed subjects cannot be treated as untouched test data: "
            f"{exposed_test}. Freeze a replacement protocol before preprocessing; "
            "do not choose one from audit performance."
        )
    return assignment


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
    assignment = assign_subject_ids(
        subjects, seed=seed, train_subjects=train_subjects,
        val_subjects=val_subjects, test_subjects=test_subjects,
    )
    by_subject = {
        subject: split for split in ("train", "val", "test")
        for subject in assignment[split]
    }
    output = [with_split(row, by_subject[row.subject_id]) for row in rows]
    validate_manifest(output)
    distribution = {
        split: {
            "records": sum(row.split == split for row in output),
            "class_counts": {
                label: sum(row.split == split and row.label == label for row in output)
                for label in ("non_fall", "fall", "unknown")
            },
        }
        for split in ("train", "val", "test")
    }
    return output, {**assignment, "distribution": distribution}
