"""Reproducible subject-independent split assignment."""
from __future__ import annotations

import random
from typing import Iterable

from .manifest import ManifestRecord, validate_manifest, with_split


# These subjects were inspected during the Phase 2B.2/2B.3 development audit.
# They are no longer eligible to represent an untouched internal test set.
DEVELOPMENT_EXPOSED_SUBJECTS = frozenset({"Subject.3", "Subject.5"})
FROZEN_2B4_SPLIT = {
    "train": ("Subject.1", "Subject.2", "Subject.4", "Subject.5", "Subject.6", "Subject.8"),
    "val": ("Subject.3", "Subject.7"),
    "test": ("Subject.9", "Subject.10"),
}


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


def assign_frozen_2b4_splits(
    records: Iterable[ManifestRecord],
) -> tuple[list[ManifestRecord], dict]:
    """Apply the prospectively frozen Phase 2B.4 split with Subjects 9/10 blind."""
    rows = list(records)
    if any(row.locked_blind for row in rows):
        raise ValueError("Locked external blind records cannot enter CAUCAFall splits")
    if {row.dataset for row in rows} != {"CAUCAFall"}:
        raise ValueError("The frozen split is defined only for CAUCAFall")
    expected = {subject for split in FROZEN_2B4_SPLIT.values() for subject in split}
    observed = {row.subject_id for row in rows}
    if observed != expected:
        raise ValueError(
            f"Frozen Phase 2B.4 split requires subjects {sorted(expected)}, found {sorted(observed)}"
        )
    by_subject = {
        subject: split for split in ("train", "val", "test")
        for subject in FROZEN_2B4_SPLIT[split]
    }
    output = [with_split(row, by_subject[row.subject_id]) for row in rows]
    validate_manifest(output)
    distribution = {
        split: {
            "subjects": list(FROZEN_2B4_SPLIT[split]),
            "records": sum(row.split == split for row in output),
            "class_counts": {
                label: sum(row.split == split and row.label == label for row in output)
                for label in ("non_fall", "fall", "unknown")
            },
        }
        for split in ("train", "val", "test")
    }
    assignment = {
        "protocol": "RAC-Fall Phase 2B.4 frozen subject split",
        "frozen_date": "2026-11-05",
        "selection_basis": (
            "Subjects 9 and 10 had no media exposure; remaining roles were frozen "
            "without using sequence quality or model performance"
        ),
        "development_exposed_subjects": sorted(DEVELOPMENT_EXPOSED_SUBJECTS),
        "internal_blind_test_subjects": list(FROZEN_2B4_SPLIT["test"]),
        **{split: list(FROZEN_2B4_SPLIT[split]) for split in ("train", "val", "test")},
        "distribution": distribution,
    }
    return output, assignment


def assign_frozen_2b5_trainval_splits(
    records: Iterable[ManifestRecord],
) -> tuple[list[ManifestRecord], dict]:
    """Apply the frozen split only to downloaded Subjects 1-8; keep 9/10 absent."""
    rows = list(records)
    if any(row.locked_blind for row in rows):
        raise ValueError("Locked external blind records cannot enter CAUCAFall splits")
    if {row.dataset for row in rows} != {"CAUCAFall"}:
        raise ValueError("The frozen split is defined only for CAUCAFall")
    trainval = set(FROZEN_2B4_SPLIT["train"]) | set(FROZEN_2B4_SPLIT["val"])
    observed = {row.subject_id for row in rows}
    if observed != trainval:
        raise ValueError(
            f"Phase 2B.5 requires exactly downloaded train/validation subjects "
            f"{sorted(trainval)}, found {sorted(observed)}"
        )
    by_subject = {
        subject: split for split in ("train", "val")
        for subject in FROZEN_2B4_SPLIT[split]
    }
    output = [with_split(row, by_subject[row.subject_id]) for row in rows]
    validate_manifest(output)
    distribution = {
        split: {
            "subjects": list(FROZEN_2B4_SPLIT[split]),
            "records": sum(row.split == split for row in output),
            "class_counts": {
                label: sum(row.split == split and row.label == label for row in output)
                for label in ("non_fall", "fall", "unknown")
            },
        }
        for split in ("train", "val")
    }
    return output, {
        "protocol": "RAC-Fall Phase 2B.5 frozen train/validation preparation",
        "frozen_date": "2026-11-05",
        "development_exposed_subjects": sorted(DEVELOPMENT_EXPOSED_SUBJECTS),
        "sealed_absent_subjects": list(FROZEN_2B4_SPLIT["test"]),
        "train": list(FROZEN_2B4_SPLIT["train"]),
        "val": list(FROZEN_2B4_SPLIT["val"]),
        "test": list(FROZEN_2B4_SPLIT["test"]),
        "distribution": distribution,
    }
