"""Causal target-person selection without annotation or future-frame leakage."""
from __future__ import annotations

from dataclasses import dataclass


CAUSAL_TARGET_TRACK_ID = -2


@dataclass(frozen=True)
class TargetSelectionConfig:
    min_valid_keypoints: int = 5
    min_mean_confidence: float = 0.25
    max_center_jump_ratio: float = 0.10
    max_scale_ratio: float = 2.5
    max_gap_frames: int = 20
    allow_untracked_reacquisition: bool = False


def assess_pose_quality(xy, confidence, valid) -> dict:
    """Return measured pose quality without filling or hallucinating keypoints."""
    np = __import__("numpy")
    points = np.asarray(xy)
    scores = np.asarray(confidence)
    mask = (np.asarray(valid, dtype=np.bool_) & np.isfinite(points).all(axis=1) &
            np.isfinite(scores) & (scores > 0))
    count = int(mask.sum())
    if count:
        extent = np.ptp(points[mask], axis=0)
        scale = float(max(extent.max(), 0.0))
        mean_confidence = float(scores[mask].mean())
    else:
        scale = 0.0
        mean_confidence = 0.0
    return {
        "valid_keypoints": count,
        "missing_keypoints": int(len(mask) - count),
        "mean_confidence": mean_confidence,
        "spatial_scale_px": scale,
    }


def select_causal_target(
    frame_offsets,
    track_ids,
    xy,
    confidence,
    valid,
    frame_width: int,
    frame_height: int,
    *,
    config: TargetSelectionConfig = TargetSelectionConfig(),
):
    """Select at most one detection per frame using present/past pose geometry only.

    Source annotation boxes and labels are intentionally absent from this API.
    """
    np = __import__("numpy")
    offsets = np.asarray(frame_offsets)
    ids = np.asarray(track_ids)
    points = np.asarray(xy)
    confidences = np.asarray(confidence)
    validity = np.asarray(valid)
    if len(offsets) < 2 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("Invalid frame offsets or dimensions")
    if not 1 <= config.min_valid_keypoints <= 17 or config.max_gap_frames < 0:
        raise ValueError("Invalid target-selection configuration")
    if not 0.0 <= config.min_mean_confidence <= 1.0:
        raise ValueError("Invalid target-selection confidence threshold")
    if config.max_center_jump_ratio <= 0 or config.max_scale_ratio <= 1:
        raise ValueError("Invalid target-selection geometry thresholds")

    selected = np.full(len(offsets) - 1, -1, dtype=np.int64)
    diagonal = float(np.hypot(frame_width, frame_height))
    last_center = None
    last_scale = None
    last_track = -1
    last_frame = -1
    has_tracked_identity = False

    for frame_index in range(len(selected)):
        candidates = []
        for detection in range(int(offsets[frame_index]), int(offsets[frame_index + 1])):
            usable = (validity[detection] & np.isfinite(points[detection]).all(axis=1) &
                      np.isfinite(confidences[detection]) & (confidences[detection] > 0))
            quality_stats = assess_pose_quality(
                points[detection], confidences[detection], validity[detection]
            )
            if (quality_stats["valid_keypoints"] < config.min_valid_keypoints or
                    quality_stats["mean_confidence"] < config.min_mean_confidence or
                    quality_stats["spatial_scale_px"] <= 0):
                continue
            visible = points[detection][usable]
            center = np.median(visible, axis=0)
            extent = np.ptp(visible, axis=0)
            scale = float(max(extent.max(), 1e-6))
            mean_confidence = quality_stats["mean_confidence"]
            quality = mean_confidence * float(usable.mean()) * (scale / diagonal)
            candidates.append((detection, center, scale, quality, int(ids[detection])))
        if not candidates:
            continue

        if has_tracked_identity and not config.allow_untracked_reacquisition:
            # Once the tracker has established a real identity, anonymous
            # detections are not allowed to take over. They are often transient
            # false positives (for example a person shown on a monitor). Missing
            # data is safer and remains explicit in the output masks.
            candidates = [row for row in candidates if row[4] >= 0]
            if not candidates:
                continue

        gap = frame_index - last_frame if last_frame >= 0 else 0
        if last_center is None:
            chosen = max(candidates, key=lambda row: (row[3], -row[0]))
        else:
            eligible = []
            jump_limit = config.max_center_jump_ratio
            scale_limit = config.max_scale_ratio
            if gap > config.max_gap_frames:
                # Long-gap reacquisition is allowed only near the last known
                # person geometry. This recovers a stable subject after tracker
                # fragmentation without choosing the highest-quality stranger.
                jump_limit *= 0.5
                scale_limit = min(scale_limit, 1.75)
            for row in candidates:
                detection, center, scale, quality, track_id = row
                jump = float(np.linalg.norm(center - last_center) / diagonal)
                scale_ratio = max(scale / last_scale, last_scale / scale)
                if jump <= jump_limit and scale_ratio <= scale_limit:
                    same_track_bonus = 0.05 if track_id >= 0 and track_id == last_track else 0.0
                    score = quality + same_track_bonus - jump
                    eligible.append((score, -detection, row))
            if not eligible:
                # A far-away detection is more safely represented as missing than
                # silently switching to another person or a person on a display.
                continue
            chosen = max(eligible, key=lambda row: (row[0], row[1]))[2]

        detection, last_center, last_scale, _, last_track = chosen
        selected[frame_index] = detection
        last_frame = frame_index
        has_tracked_identity = has_tracked_identity or last_track >= 0
    return selected
