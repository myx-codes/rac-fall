"""Pose normalization, causal temporal windows, and leakage validation."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from .manifest import ManifestRecord, validate_manifest


def normalize_pose(xy, confidence, valid):
    """Root/scale normalize one COCO pose; missing coordinates become zero."""
    np = __import__("numpy")
    xy = np.asarray(xy, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_)
    usable = (valid & np.isfinite(xy).all(axis=1) & np.isfinite(confidence) &
              (confidence > 0))
    normalized = np.zeros((17, 2), dtype=np.float32)
    if not usable.any():
        return normalized, np.nan_to_num(confidence, nan=0.0), usable

    if usable[11] and usable[12]:
        center = (xy[11] + xy[12]) / 2
    elif usable[5] and usable[6]:
        center = (xy[5] + xy[6]) / 2
    else:
        center = xy[usable].mean(axis=0)

    extent = np.ptp(xy[usable], axis=0)
    body_extent = float(max(extent.max(), 0.0))
    if usable[5] and usable[6] and usable[11] and usable[12]:
        shoulder_mid = (xy[5] + xy[6]) / 2
        hip_mid = (xy[11] + xy[12]) / 2
        torso_scale = float(np.linalg.norm(shoulder_mid - hip_mid))
        # Sideways or occluded poses can collapse the projected torso length.
        # Bound it by visible body extent to avoid exploding normalized values.
        scale = max(torso_scale, body_extent * 0.25)
    else:
        scale = body_extent
    if not np.isfinite(scale) or scale < 1e-6:
        usable[:] = False
        return normalized, np.nan_to_num(confidence, nan=0.0), usable
    normalized[usable] = (xy[usable] - center) / scale
    return normalized, np.nan_to_num(confidence, nan=0.0), usable


def validate_window_leakage(metadata: Iterable[dict]) -> None:
    """Reject subject/sequence reuse or overlapping intervals across splits."""
    rows = list(metadata)
    subject_splits: dict[tuple[str, str], str] = {}
    sequence_splits: dict[tuple[str, str], str] = {}
    intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for row in rows:
        split = row["split"]
        subject_key = (row["dataset"], row["subject_id"])
        sequence_key = (row["dataset"], row["video_id"])
        if subject_key in subject_splits and subject_splits[subject_key] != split:
            raise ValueError(f"Subject window leakage: {subject_key}")
        if sequence_key in sequence_splits and sequence_splits[sequence_key] != split:
            raise ValueError(f"Sequence window leakage: {sequence_key}")
        subject_splits[subject_key] = split
        sequence_splits[sequence_key] = split
        prior = intervals.setdefault(sequence_key, [])
        start, end = int(row["start_frame"]), int(row["end_frame"])
        for old_start, old_end, old_split in prior:
            if old_split != split and max(start, old_start) < min(end, old_end):
                raise ValueError(f"Overlapping-window leakage: {sequence_key}")
        prior.append((start, end, split))


def build_windows(
    records: Iterable[ManifestRecord],
    pose_dir: str | Path,
    output: str | Path,
    *,
    window_frames: int = 30,
    stride_frames: int = 15,
    alignment_dir: str | Path | None = None,
    quality_dir: str | Path | None = None,
    min_track_coverage: float = 0.5,
    allow_internal_test: bool = False,
) -> dict:
    if window_frames <= 1 or stride_frames <= 0 or not 0.0 <= min_track_coverage <= 1.0:
        raise ValueError(
            "window_frames must be > 1, stride_frames must be > 0, and "
            "min_track_coverage must be in [0, 1]"
        )
    rows = list(records)
    validate_manifest(rows)
    if any(row.locked_blind or row.split == "blind_test" for row in rows):
        raise ValueError("Locked blind data cannot be used to build development windows")
    np = __import__("numpy")
    pose_root = Path(pose_dir)
    windows = []
    masks = []
    labels = []
    event_labels = []
    timestamps = []
    track_coverages = []
    source_track_ids = []
    metadata = []
    skipped_low_coverage = 0
    skipped_quarantined_sequences = 0
    ground_label_names = {-1: "unknown", 0: "not_on_ground_due_to_fall", 1: "on_ground_due_to_fall"}
    event_label_names = {-1: "unknown", 0: "no_event_at_endpoint", 1: "inside_verified_event"}
    alignment_root = Path(alignment_dir) if alignment_dir is not None else None
    quality_root = Path(quality_dir) if quality_dir is not None else None
    for record in rows:
        if not record.split:
            raise ValueError(f"{record.video_id}: split must be assigned before window generation")
        if record.label not in {"non_fall", "fall"}:
            raise ValueError(f"{record.video_id}: verified binary label required")
        if record.split == "test" and not allow_internal_test:
            raise ValueError(
                f"{record.video_id}: internal blind test windows require explicit authorization"
            )
        quality_status = "not_provided"
        if quality_root is not None:
            quality_path = quality_root / f"{record.video_id}.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if (quality.get("schema_version") != "racfall.sequence-quality.v1" or
                    quality.get("video_id") != record.video_id):
                raise ValueError(f"Quality provenance does not match {record.video_id}")
            quality_status = str(quality.get("status"))
            if not quality.get("training_eligible", False):
                skipped_quarantined_sequences += 1
                continue
        path = pose_root / f"{record.video_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pose file: {path}")
        with np.load(path, allow_pickle=False) as pose:
            if str(pose["video_id"]) != record.video_id:
                raise ValueError(f"Pose/manifest video mismatch: {path}")
            n_frames = len(pose["frame_index"])
            if alignment_root is not None:
                from .alignment import load_alignment_labels
                frame_labels, alignment_status = load_alignment_labels(
                    alignment_root / f"{record.video_id}.json", record.video_id, n_frames,
                )
            else:
                frame_labels = [0] * n_frames if record.label == "non_fall" else [-1] * n_frames
                alignment_status = "sequence_label_only" if record.label == "non_fall" else "unverified"
            offsets = pose["frame_offsets"]
            detection_tracks = pose["track_id"]
            from .target import (
                CAUSAL_TARGET_TRACK_ID, TargetSelectionConfig, select_causal_target,
            )
            target_config = TargetSelectionConfig()
            selected = select_causal_target(
                offsets, detection_tracks, pose["xy"], pose["confidence"], pose["valid"],
                record.width, record.height, config=target_config,
            )
            features = np.zeros((n_frames, 17, 3), dtype=np.float32)
            validity = np.zeros((n_frames, 17), dtype=np.bool_)
            presence = selected >= 0
            selected_source_tracks = np.full(n_frames, -1, dtype=np.int64)
            for frame_index, detection in enumerate(selected):
                if detection < 0:
                    continue
                selected_source_tracks[frame_index] = int(detection_tracks[detection])
                norm_xy, confidence, valid = normalize_pose(
                    pose["xy"][detection], pose["confidence"][detection], pose["valid"][detection]
                )
                features[frame_index, :, :2] = norm_xy
                features[frame_index, :, 2] = confidence
                validity[frame_index] = valid
            for start in range(0, n_frames - window_frames + 1, stride_frames):
                end = start + window_frames
                track_coverage = float(presence[start:end].mean())
                if track_coverage < min_track_coverage:
                    skipped_low_coverage += 1
                    continue
                windows.append(features[start:end].reshape(window_frames, 51))
                masks.append(validity[start:end])
                window_timestamps = pose["timestamp_s"][start:end]
                timestamps.append(window_timestamps)
                track_coverages.append(track_coverage)
                source_track_ids.append(selected_source_tracks[start:end])
                # A causal target uses only the annotation at the window endpoint.
                # It never scans future frames or infers a transition boundary.
                window_label = int(frame_labels[end - 1])
                if record.label == "non_fall":
                    event_label = 0
                elif record.label_start_s != "" and record.label_end_s != "":
                    current_time = float(window_timestamps[-1])
                    event_label = int(
                        float(record.label_start_s) <= current_time <= float(record.label_end_s)
                    )
                else:
                    event_label = -1
                labels.append(window_label)
                event_labels.append(event_label)
                metadata.append({
                    "dataset": record.dataset, "subject_id": record.subject_id,
                    "video_id": record.video_id, "track_id": CAUSAL_TARGET_TRACK_ID,
                    "split": record.split, "start_frame": start, "end_frame": end,
                    "alignment_status": alignment_status,
                    "quality_status": quality_status,
                })
    if not windows:
        raise ValueError("No complete tracked pose windows were generated")
    validate_window_leakage(metadata)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray("racfall.windows.v3"),
            features=np.asarray(windows, dtype=np.float32),
            valid=np.asarray(masks, dtype=np.bool_), labels=np.asarray(labels, dtype=np.int64),
            ground_state_labels=np.asarray(labels, dtype=np.int64),
            event_labels=np.asarray(event_labels, dtype=np.int64),
            label_semantics=np.asarray("ground_state_at_causal_window_endpoint"),
            timestamp_s=np.asarray(timestamps, dtype=np.float64),
            dataset=np.asarray([row["dataset"] for row in metadata]),
            subject_id=np.asarray([row["subject_id"] for row in metadata]),
            video_id=np.asarray([row["video_id"] for row in metadata]),
            track_id=np.asarray([row["track_id"] for row in metadata], dtype=np.int64),
            split=np.asarray([row["split"] for row in metadata]),
            start_frame=np.asarray([row["start_frame"] for row in metadata], dtype=np.int32),
            end_frame=np.asarray([row["end_frame"] for row in metadata], dtype=np.int32),
            alignment_status=np.asarray([row["alignment_status"] for row in metadata]),
            quality_status=np.asarray([row["quality_status"] for row in metadata]),
            track_coverage=np.asarray(track_coverages, dtype=np.float32),
            min_track_coverage=np.asarray(min_track_coverage, dtype=np.float32),
            source_track_id=np.asarray(source_track_ids, dtype=np.int64),
            target_selection=np.asarray("causal_geometry_without_annotations"),
            target_selection_config=np.asarray(json.dumps(asdict(target_config), sort_keys=True)),
        )
    valid_windows = sum(value >= 0 for value in labels)
    return {
        "windows": len(windows), "window_frames": window_frames,
        "stride_frames": stride_frames, "output": str(output_path),
        "label_counts": {name: labels.count(value) for value, name in ground_label_names.items()},
        "valid_windows": valid_windows,
        "invalid_windows": len(labels) - valid_windows,
        "event_label_counts": {
            name: event_labels.count(value) for value, name in event_label_names.items()
        },
        "label_semantics": "ground_state_at_causal_window_endpoint",
        "min_track_coverage": min_track_coverage,
        "skipped_low_coverage_windows": skipped_low_coverage,
        "skipped_quarantined_sequences": skipped_quarantined_sequences,
    }
