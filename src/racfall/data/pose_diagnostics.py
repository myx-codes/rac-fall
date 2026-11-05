"""Inference-only pose recovery diagnostics without annotation-box input."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .pose import _plain, iter_frames


def _has_usable_pose(result_rows, keypoint_confidence: float, minimum_keypoints: int = 5) -> bool:
    np = __import__("numpy")
    result = result_rows[0] if result_rows else None
    if result is None or getattr(result, "keypoints", None) is None:
        return False
    xy_rows = _plain(getattr(result.keypoints, "xy", None))
    confidence_rows = _plain(getattr(result.keypoints, "conf", None))
    for index, xy in enumerate(xy_rows):
        if index >= len(confidence_rows):
            continue
        points = np.asarray(xy, dtype=np.float32)
        confidence = np.asarray(confidence_rows[index], dtype=np.float32)
        if len(points) != 17 or len(confidence) != 17:
            continue
        valid = (np.isfinite(points).all(axis=1) & np.isfinite(confidence) &
                 (confidence >= keypoint_confidence) & np.any(points != 0, axis=1))
        if int(valid.sum()) >= minimum_keypoints:
            return True
    return False


def analyze_pose_recovery(
    record,
    dataset_root: str | Path,
    baseline_pose: str | Path,
    output: str | Path,
    *,
    device: str = "mps",
    imgsz: int = 416,
    detector_confidences: tuple[float, ...] = (0.25, 0.10, 0.05),
    keypoint_confidence: float = 0.25,
    limit_missing: int | None = None,
    model_factory=None,
) -> dict:
    """Measure lower-confidence and annotation-free overlapping-tile recovery."""
    np = __import__("numpy")
    if not detector_confidences or any(not 0 <= value <= 1 for value in detector_confidences):
        raise ValueError("Detector confidence values must be in [0, 1]")
    if model_factory is None:
        model_factory = __import__("ultralytics", fromlist=["YOLO"]).YOLO
    with np.load(Path(baseline_pose), allow_pickle=False) as pose:
        if str(pose["video_id"]) != record.video_id:
            raise ValueError("Baseline pose does not match record")
        missing = np.flatnonzero(np.diff(pose["frame_offsets"]) == 0).tolist()
    if limit_missing is not None:
        if limit_missing <= 0:
            raise ValueError("limit_missing must be positive")
        missing = missing[:limit_missing]
    missing_set = set(missing)
    model = model_factory("yolo26n-pose.pt")
    methods = {f"full_conf_{value:g}": 0 for value in detector_confidences}
    methods[f"overlap_tiles_conf_{detector_confidences[0]:g}"] = 0
    per_frame = []
    started = time.perf_counter()
    for frame_index, _timestamp, frame in iter_frames(dataset_root, record):
        if frame_index not in missing_set:
            continue
        recovered = {}
        for confidence in detector_confidences:
            name = f"full_conf_{confidence:g}"
            rows = model.predict(
                frame, device=device, imgsz=imgsz, conf=confidence,
                verbose=False, save=False, show=False,
            )
            recovered[name] = _has_usable_pose(rows, keypoint_confidence)
            methods[name] += int(recovered[name])
        height, width = frame.shape[:2]
        crop_width, crop_height = max(1, int(width * 0.6)), max(1, int(height * 0.6))
        crops = [
            frame[y:y + crop_height, x:x + crop_width]
            for y in (0, height - crop_height)
            for x in (0, width - crop_width)
        ]
        tile_name = f"overlap_tiles_conf_{detector_confidences[0]:g}"
        recovered[tile_name] = any(
            _has_usable_pose(model.predict(
                crop, device=device, imgsz=imgsz, conf=detector_confidences[0],
                verbose=False, save=False, show=False,
            ), keypoint_confidence)
            for crop in crops
        )
        methods[tile_name] += int(recovered[tile_name])
        per_frame.append({"frame_index_zero_based": frame_index, "recovered": recovered})
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "racfall.pose-recovery-diagnostic.v1",
        "video_id": record.video_id,
        "source": "canonical PNG frames",
        "annotation_boxes_used": False,
        "production_policy_changed": False,
        "baseline_missing_frames": len(missing),
        "detector_confidences": list(detector_confidences),
        "keypoint_confidence": keypoint_confidence,
        "imgsz": imgsz,
        "device": device,
        "methods_recovered_frames": methods,
        "elapsed_s": round(elapsed, 6),
        "per_frame": per_frame,
        "note": (
            "Overlapping tiles are diagnostic only. Tile detections are not merged or used for "
            "training until a frozen, annotation-free coordinate/identity policy is validated."
        ),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
