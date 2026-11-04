"""Strict dataset manifest schema with provenance and leakage checks."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields, replace
import math
from pathlib import Path
from typing import Iterable


LABELS = {"fall", "non_fall", "unknown"}
SPLITS = {"", "train", "val", "test", "development", "blind_test"}
SOURCE_TYPES = {"video", "image_sequence_dir", "image_sequence_zip"}


@dataclass(frozen=True)
class ManifestRecord:
    dataset: str
    dataset_version: str
    subject_id: str
    subject_provenance: str
    video_id: str
    sequence_id: str
    source_relpath: str
    source_type: str
    action: str
    label: str
    label_start_s: str
    label_end_s: str
    annotation_provenance: str
    timestamp_source: str
    start_timestamp_s: float
    end_timestamp_s: float
    fps: float
    width: int
    height: int
    frame_count: int
    split: str
    source_sha256: str
    license: str
    locked_blind: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_timestamp_s - self.start_timestamp_s


FIELD_NAMES = tuple(field.name for field in fields(ManifestRecord))


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"Invalid boolean value: {value!r}")
    return value.lower() == "true"


def _from_row(row: dict[str, str]) -> ManifestRecord:
    missing = set(FIELD_NAMES) - set(row)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return ManifestRecord(
        **{name: row[name] for name in FIELD_NAMES if name not in {
            "start_timestamp_s", "end_timestamp_s", "fps", "width", "height",
            "frame_count", "locked_blind",
        }},
        start_timestamp_s=float(row["start_timestamp_s"]),
        end_timestamp_s=float(row["end_timestamp_s"]),
        fps=float(row["fps"]),
        width=int(row["width"]),
        height=int(row["height"]),
        frame_count=int(row["frame_count"]),
        locked_blind=_bool(row["locked_blind"]),
    )


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [_from_row(row) for row in csv.DictReader(handle)]


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    rows = list(records)
    validate_manifest(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for record in rows:
            row = asdict(record)
            row["locked_blind"] = str(record.locked_blind).lower()
            writer.writerow(row)


def validate_manifest(records: Iterable[ManifestRecord]) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("Manifest is empty")
    videos: set[str] = set()
    sequences: dict[tuple[str, str], str] = {}
    subjects: dict[tuple[str, str], str] = {}
    for row in rows:
        for name in (
            "dataset", "dataset_version", "subject_id", "subject_provenance",
            "video_id", "sequence_id", "source_relpath", "action",
            "annotation_provenance", "timestamp_source", "license",
        ):
            if not str(getattr(row, name)).strip():
                raise ValueError(f"{row.video_id or '<unknown>'}: {name} is required")
        if Path(row.source_relpath).is_absolute() or ".." in Path(row.source_relpath).parts:
            raise ValueError(f"{row.video_id}: source_relpath must remain inside dataset root")
        if row.video_id in videos:
            raise ValueError(f"Duplicate video_id: {row.video_id}")
        videos.add(row.video_id)
        if row.label not in LABELS:
            raise ValueError(f"{row.video_id}: invalid label {row.label!r}")
        if row.source_type not in SOURCE_TYPES:
            raise ValueError(f"{row.video_id}: invalid source_type {row.source_type!r}")
        if row.split not in SPLITS:
            raise ValueError(f"{row.video_id}: invalid split {row.split!r}")
        numeric = (row.start_timestamp_s, row.end_timestamp_s, row.fps)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"{row.video_id}: timestamps and FPS must be finite")
        if row.start_timestamp_s < 0 or row.end_timestamp_s <= row.start_timestamp_s:
            raise ValueError(f"{row.video_id}: invalid timestamp interval")
        if row.fps <= 0 or row.width <= 0 or row.height <= 0 or row.frame_count <= 0:
            raise ValueError(f"{row.video_id}: invalid media metadata")
        if bool(row.label_start_s) != bool(row.label_end_s):
            raise ValueError(f"{row.video_id}: label_start_s and label_end_s must both be set or empty")
        if row.label_start_s:
            try:
                label_start = float(row.label_start_s)
                label_end = float(row.label_end_s)
            except ValueError as exc:
                raise ValueError(f"{row.video_id}: invalid label timestamp") from exc
            if (not math.isfinite(label_start) or not math.isfinite(label_end) or
                    label_start < row.start_timestamp_s or label_end < label_start or
                    label_end > row.end_timestamp_s):
                raise ValueError(f"{row.video_id}: label interval is outside the media interval")
            if row.label != "fall":
                raise ValueError(f"{row.video_id}: event interval is only valid for fall-labelled media")
        if row.source_sha256 and (len(row.source_sha256) != 64 or
                                  any(c not in "0123456789abcdef" for c in row.source_sha256.lower())):
            raise ValueError(f"{row.video_id}: invalid SHA256")
        if row.locked_blind and row.split != "blind_test":
            raise ValueError(f"{row.video_id}: locked blind data must use split=blind_test")
        if row.split and row.subject_id.lower() in {"unknown", "unverified"}:
            raise ValueError(f"{row.video_id}: cannot split without a verified subject ID")

        sequence_key = (row.dataset, row.sequence_id)
        prior_sequence_split = sequences.setdefault(sequence_key, row.split)
        if prior_sequence_split != row.split:
            raise ValueError(f"Sequence leakage across splits: {sequence_key}")
        if row.split and not row.locked_blind:
            subject_key = (row.dataset, row.subject_id)
            prior_subject_split = subjects.setdefault(subject_key, row.split)
            if prior_subject_split != row.split:
                raise ValueError(f"Subject leakage across splits: {subject_key}")

    return {
        "records": len(rows),
        "datasets": sorted({row.dataset for row in rows}),
        "subjects": len({(row.dataset, row.subject_id) for row in rows}),
        "splits": {split: sum(row.split == split for row in rows) for split in sorted({r.split for r in rows})},
        "locked_blind_records": sum(row.locked_blind for row in rows),
    }


def with_split(record: ManifestRecord, split: str) -> ManifestRecord:
    return replace(record, split=split)
