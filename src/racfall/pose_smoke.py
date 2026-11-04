"""Local YOLO pose feasibility check. No fall decision, no RGB retention."""
from __future__ import annotations

import time


def pose_smoke(source: str, max_frames: int = 50, device: str = "auto", imgsz: int = 640) -> dict:
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if imgsz < 32:
        raise ValueError("imgsz must be >= 32")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the vision extra: python -m pip install -e '.[vision]'") from exc

    mps_ready = torch.backends.mps.is_available()
    if device == "auto":
        selected = "mps" if mps_ready else "cpu"
    elif device == "mps" and not mps_ready:
        raise RuntimeError("MPS requested but not available on this machine")
    else:
        selected = device
    source_arg = int(source) if source.isdigit() else source
    model = YOLO("yolo26n-pose.pt")  # official weights; auto-downloaded first time
    n_frames = 0
    n_pose_detections = 0
    started = time.perf_counter()

    if isinstance(source_arg, int):
        import cv2
        cap = cv2.VideoCapture(source_arg, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened")
        try:
            while n_frames < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                results = model.predict(
                    frame, device=selected, imgsz=imgsz,
                    verbose=False, save=False, show=False
                )
                for result in results:
                    if result.keypoints is not None:
                        n_pose_detections += len(result.keypoints.xy)
                n_frames += 1
        finally:
            cap.release()
    else:
        for result in model.predict(
            source=source_arg, stream=True, device=selected,
            imgsz=imgsz, verbose=False, save=False, show=False
        ):
            n_frames += 1
            if result.keypoints is not None:
                n_pose_detections += len(result.keypoints.xy)
            if n_frames >= max_frames:
                break
    seconds = time.perf_counter() - started
    return {
        "model": "yolo26n-pose.pt", "selected_device": selected,
        "frames_processed": n_frames, "pose_detections": n_pose_detections,
        "elapsed_s_including_io": round(seconds, 4),
        "smoke_throughput_fps": round(n_frames / seconds, 3) if seconds > 0 else None,
        "is_fall_detector": False,
        "stores_rgb_or_poses": False,
    }
