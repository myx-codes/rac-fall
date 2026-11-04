"""Pose normalization, causal temporal windows, and leakage validation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .manifest import ManifestRecord, validate_manifest


def normalize_pose(xy, confidence, valid):
    """Root/scale normalize one COCO pose; missing coordinates become zero."""
    np = __import__("numpy")
    xy = np.asarray(xy, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_)
    usable = valid & np.isfinite(xy).all(axis=1)
    normalized = np.zeros((17, 2), dtype=np.float32)
    if not usable.any():
        return normalized, np.nan_to_num(confidence, nan=0.0), usable

    if usable[11] and usable[12]:
        center = (xy[11] + xy[12]) / 2
    elif usable[5] and usable[6]:
        center = (xy[5] + xy[6]) / 2
    else:
        center = xy[usable].mean(axis=0)

    if usable[5] and usable[6] and usable[11] and usable[12]:
        shoulder_mid = (xy[5] + xy[6]) / 2
        hip_mid = (xy[11] + xy[12]) / 2
        scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    else:
        extent = np.ptp(xy[usable], axis=0)
        scale = float(max(extent.max(), 0.0))
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
) -> dict:
    if window_frames <= 1 or stride_frames <= 0:
        raise ValueError("window_frames must be > 1 and stride_frames must be > 0")
    rows = list(records)
    validate_manifest(rows)
    if any(row.locked_blind or row.split == "blind_test" for row in rows):
        raise ValueError("Locked blind data cannot be used to build development windows")
    np = __import__("numpy")
    pose_root = Path(pose_dir)
    windows = []
    masks = []
    labels = []
    timestamps = []
    metadata = []
    label_names = {-1: "unknown", 0: "non_fall", 1: "fall"}
    for record in rows:
        if not record.split:
            raise ValueError(f"{record.video_id}: split must be assigned before window generation")
        if record.label not in {"non_fall", "fall"}:
            raise ValueError(f"{record.video_id}: verified binary label required")
        path = pose_root / f"{record.video_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pose file: {path}")
        with np.load(path, allow_pickle=False) as pose:
            if str(pose["video_id"]) != record.video_id:
                raise ValueError(f"Pose/manifest video mismatch: {path}")
            n_frames = len(pose["frame_index"])
            offsets = pose["frame_offsets"]
            detection_tracks = pose["track_id"]
            tracks = sorted(int(value) for value in np.unique(detection_tracks) if int(value) >= 0)
            for track_id in tracks:
                features = np.zeros((n_frames, 17, 3), dtype=np.float32)
                validity = np.zeros((n_frames, 17), dtype=np.bool_)
                for frame_index in range(n_frames):
                    candidates = range(int(offsets[frame_index]), int(offsets[frame_index + 1]))
                    detection = next((index for index in candidates
                                      if int(detection_tracks[index]) == track_id), None)
                    if detection is None:
                        continue
                    norm_xy, confidence, valid = normalize_pose(
                        pose["xy"][detection], pose["confidence"][detection], pose["valid"][detection]
                    )
                    features[frame_index, :, :2] = norm_xy
                    features[frame_index, :, 2] = confidence
                    validity[frame_index] = valid
                for start in range(0, n_frames - window_frames + 1, stride_frames):
                    end = start + window_frames
                    windows.append(features[start:end].reshape(window_frames, 51))
                    masks.append(validity[start:end])
                    window_timestamps = pose["timestamp_s"][start:end]
                    timestamps.append(window_timestamps)
                    if record.label == "non_fall":
                        window_label = 0
                    elif record.label_start_s:
                        current_time = float(window_timestamps[-1])
                        window_label = int(float(record.label_start_s) <= current_time <= float(record.label_end_s))
                    else:
                        window_label = -1
                    labels.append(window_label)
                    metadata.append({
                        "dataset": record.dataset, "subject_id": record.subject_id,
                        "video_id": record.video_id, "track_id": track_id, "split": record.split,
                        "start_frame": start, "end_frame": end,
                    })
    if not windows:
        raise ValueError("No complete tracked pose windows were generated")
    validate_window_leakage(metadata)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray("racfall.windows.v1"),
            features=np.asarray(windows, dtype=np.float32),
            valid=np.asarray(masks, dtype=np.bool_), labels=np.asarray(labels, dtype=np.int64),
            timestamp_s=np.asarray(timestamps, dtype=np.float64),
            dataset=np.asarray([row["dataset"] for row in metadata]),
            subject_id=np.asarray([row["subject_id"] for row in metadata]),
            video_id=np.asarray([row["video_id"] for row in metadata]),
            track_id=np.asarray([row["track_id"] for row in metadata], dtype=np.int64),
            split=np.asarray([row["split"] for row in metadata]),
            start_frame=np.asarray([row["start_frame"] for row in metadata], dtype=np.int32),
            end_frame=np.asarray([row["end_frame"] for row in metadata], dtype=np.int32),
        )
    return {
        "windows": len(windows), "window_frames": window_frames,
        "stride_frames": stride_frames, "output": str(output_path),
        "label_counts": {name: labels.count(value) for value, name in label_names.items()},
    }
