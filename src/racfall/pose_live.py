"""Live, privacy-preserving pose visualization (no fall decision or recording)."""
from __future__ import annotations

from collections import deque
import time
from typing import Any, Callable


# COCO's 17 keypoints: nose; eyes; ears; shoulders; elbows; wrists; hips;
# knees; ankles. Connections are zero-based indices into that ordering.
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

WINDOW_NAME = "RAC-Fall Pose (Q quit, P pause)"


def _as_list(value: Any) -> list:
    """Convert a Torch/NumPy-like value to plain lists without retaining it."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _select_device(torch: Any, requested: str) -> str:
    mps_ready = bool(torch.backends.mps.is_available())
    if requested == "auto":
        return "mps" if mps_ready else "cpu"
    if requested == "mps" and not mps_ready:
        raise RuntimeError("MPS requested but not available on this machine; use --device cpu")
    return requested


def _pose_rows(result: Any) -> list[tuple[str, list, list, float]]:
    """Return ephemeral (track ID, xy, confidence, mean confidence) rows."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return []
    xy_rows = _as_list(getattr(keypoints, "xy", None))
    confidence_rows = _as_list(getattr(keypoints, "conf", None))
    boxes = getattr(result, "boxes", None)
    ids = _as_list(getattr(boxes, "id", None)) if boxes is not None else []
    box_confidence = _as_list(getattr(boxes, "conf", None)) if boxes is not None else []

    poses = []
    for index, xy in enumerate(xy_rows):
        confidence = confidence_rows[index] if index < len(confidence_rows) else []
        valid = [float(value) for value in confidence if value is not None]
        if valid:
            pose_confidence = sum(valid) / len(valid)
        elif index < len(box_confidence):
            pose_confidence = float(box_confidence[index])
        else:
            pose_confidence = 0.0
        tracking_id = str(int(ids[index])) if index < len(ids) and ids[index] is not None else "?"
        poses.append((tracking_id, xy, confidence, pose_confidence))
    return poses


def draw_pose_overlay(
    frame: Any,
    result: Any,
    cv2: Any,
    fps: float,
    inference_ms: float,
    keypoint_threshold: float = 0.25,
) -> int:
    """Draw tracked COCO poses in place and return the pose count."""
    poses = _pose_rows(result)
    for tracking_id, xy, confidence, pose_confidence in poses:
        visible = []
        for index, point in enumerate(xy):
            conf = float(confidence[index]) if index < len(confidence) else pose_confidence
            valid = len(point) >= 2 and conf >= keypoint_threshold and not (point[0] == 0 and point[1] == 0)
            visible.append(valid)

        for start, end in COCO_SKELETON:
            if start < len(xy) and end < len(xy) and visible[start] and visible[end]:
                p1 = (int(round(xy[start][0])), int(round(xy[start][1])))
                p2 = (int(round(xy[end][0])), int(round(xy[end][1])))
                cv2.line(frame, p1, p2, (60, 220, 60), 2, cv2.LINE_AA)
        for index, point in enumerate(xy):
            if visible[index]:
                center = (int(round(point[0])), int(round(point[1])))
                cv2.circle(frame, center, 4, (0, 190, 255), -1, cv2.LINE_AA)

        anchors = [xy[i] for i in range(len(xy)) if visible[i]]
        if anchors:
            label_at = (int(min(p[0] for p in anchors)), max(20, int(min(p[1] for p in anchors)) - 8))
            cv2.putText(
                frame, f"ID {tracking_id}  pose {pose_confidence:.2f}", label_at,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        frame, f"FPS {fps:5.1f}   inference {inference_ms:5.1f} ms   poses {len(poses)}",
        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return len(poses)


def pose_live(
    source: str | int = 0,
    imgsz: int = 416,
    device: str = "mps",
    *,
    model_factory: Callable[[str], Any] | None = None,
    cv2_module: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Run live tracked pose visualization and return non-biometric run metrics."""
    if imgsz < 32:
        raise ValueError("imgsz must be >= 32")
    if device not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cpu")
    if isinstance(source, str):
        if not source.isdigit():
            raise ValueError("pose-live source must be a non-negative webcam index")
        source = int(source)
    if source < 0:
        raise ValueError("pose-live source must be a non-negative webcam index")

    try:
        if torch_module is None:
            import torch as torch_module
        if cv2_module is None:
            import cv2 as cv2_module
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
    except ImportError as exc:
        raise RuntimeError("Install the vision extra: python -m pip install -e '.[vision]'") from exc

    selected_device = _select_device(torch_module, device)
    model = model_factory("yolo26n-pose.pt")
    capture = cv2_module.VideoCapture(source, cv2_module.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Camera {source} could not be opened (check macOS camera permission)")

    started = clock()
    frame_times: deque[float] = deque(maxlen=31)
    frames_processed = 0
    inference_total_ms = 0.0
    paused = False
    last_frame = None
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Camera {source} stopped returning frames")
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=selected_device,
                    imgsz=imgsz,
                    verbose=False,
                    save=False,
                    show=False,
                )
                result = results[0] if results else None
                inference_ms = (float(getattr(result, "speed", {}).get("inference", 0.0))
                                if result is not None else 0.0)
                inference_total_ms += inference_ms
                frames_processed += 1
                now = clock()
                frame_times.append(now)
                fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                       if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)
                if result is not None:
                    draw_pose_overlay(frame, result, cv2_module, fps, inference_ms)
                last_frame = frame

            if last_frame is not None:
                display = last_frame.copy() if paused else last_frame
                if paused:
                    cv2_module.putText(
                        display, "PAUSED", (10, 64), cv2_module.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 200, 255), 2, cv2_module.LINE_AA,
                    )
                cv2_module.imshow(WINDOW_NAME, display)
            key = cv2_module.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
    finally:
        capture.release()
        cv2_module.destroyAllWindows()

    elapsed = clock() - started
    return {
        "model": "yolo26n-pose.pt",
        "selected_device": selected_device,
        "imgsz": imgsz,
        "frames_processed": frames_processed,
        "elapsed_s_including_display": round(elapsed, 4),
        "average_loop_fps": round(frames_processed / elapsed, 3) if elapsed > 0 else None,
        "mean_inference_latency_ms": round(inference_total_ms / frames_processed, 3) if frames_processed else None,
        "is_fall_detector": False,
        "stores_rgb_or_poses": False,
    }
