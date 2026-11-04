"""YOLO26n-Pose extraction to compact, frame-addressable NPZ files."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import time
from typing import Any, Iterator
import zipfile

from .adapters import _natural_key
from .manifest import ManifestRecord


def _plain(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _source_path(root: Path, relative: str) -> Path:
    source = (root / relative).resolve()
    if source != root and root not in source.parents:
        raise ValueError(f"Source path escapes dataset root: {relative}")
    return source


def _urfd_timestamps(root: Path, record: ManifestRecord) -> list[float]:
    sync_relpath = record.timestamp_source.removesuffix(" milliseconds")
    path = _source_path(root, sync_relpath)
    text = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel
    timestamps = []
    for row in csv.reader(io.StringIO(text), dialect):
        numeric = []
        for cell in row:
            try:
                numeric.append(float(cell.strip()))
            except ValueError:
                pass
        if len(numeric) >= 2:
            timestamps.append(numeric[1] / 1000.0)
    return timestamps


def iter_frames(root: str | Path, record: ManifestRecord) -> Iterator[tuple[int, float, Any]]:
    """Yield frames without copying or modifying original dataset files."""
    root_path = Path(root).resolve()
    source = _source_path(root_path, record.source_relpath)
    try:
        cv2 = __import__("cv2")
        np = __import__("numpy")
    except ImportError as exc:
        raise RuntimeError("Install the vision extra before pose extraction") from exc
    if record.source_type == "video":
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open source video: {record.source_relpath}")
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield index, record.start_timestamp_s + index / record.fps, frame
                index += 1
        finally:
            capture.release()
        return

    timestamps = _urfd_timestamps(root_path, record)
    if record.source_type == "image_sequence_zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted((name for name in archive.namelist() if name.lower().endswith(".png")),
                           key=_natural_key)
            for index, name in enumerate(names):
                frame = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Cannot decode {record.source_relpath}:{name}")
                yield index, timestamps[index], frame
    elif record.source_type == "image_sequence_dir":
        images = sorted((path for path in source.iterdir() if path.suffix.lower() == ".png"),
                        key=lambda path: _natural_key(path.name))
        for index, path in enumerate(images):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Cannot decode {path}")
            yield index, timestamps[index], frame
    else:
        raise ValueError(f"Unsupported source_type: {record.source_type}")


class PoseExtractor:
    def __init__(
        self,
        *,
        device: str = "mps",
        imgsz: int = 416,
        confidence_threshold: float = 0.25,
        model_factory=None,
    ):
        if imgsz < 32 or not 0 <= confidence_threshold <= 1:
            raise ValueError("Invalid imgsz or confidence threshold")
        if model_factory is None:
            try:
                model_factory = __import__("ultralytics", fromlist=["YOLO"]).YOLO
            except ImportError as exc:
                raise RuntimeError("Install the vision extra before pose extraction") from exc
        self.model = model_factory("yolo26n-pose.pt")
        self.device = device
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold

    def extract(self, root: str | Path, record: ManifestRecord, output: str | Path) -> dict:
        if record.locked_blind or record.split == "blind_test" or record.dataset == "GMDCSA-24":
            raise ValueError("Pose extraction is forbidden for the locked GMDCSA-24 blind test")
        np = __import__("numpy")
        frame_indices: list[int] = []
        timestamps: list[float] = []
        offsets = [0]
        track_ids: list[int] = []
        all_xy: list[Any] = []
        all_confidence: list[Any] = []
        all_valid: list[Any] = []
        inference_ms: list[float] = []
        started = time.perf_counter()
        for frame_index, timestamp_s, frame in iter_frames(root, record):
            result_rows = self.model.track(
                frame, persist=frame_index > 0, tracker="bytetrack.yaml",
                device=self.device, imgsz=self.imgsz, verbose=False, save=False, show=False,
            )
            result = result_rows[0] if result_rows else None
            frame_indices.append(frame_index)
            timestamps.append(timestamp_s)
            if result is not None:
                inference_ms.append(float(getattr(result, "speed", {}).get("inference", 0.0)))
            if result is None or getattr(result, "keypoints", None) is None:
                offsets.append(len(track_ids))
                continue
            xy_rows = _plain(getattr(result.keypoints, "xy", None))
            confidence_rows = _plain(getattr(result.keypoints, "conf", None))
            boxes = getattr(result, "boxes", None)
            ids = _plain(getattr(boxes, "id", None)) if boxes is not None else []
            for detection_index, xy in enumerate(xy_rows):
                confidence = (confidence_rows[detection_index]
                              if detection_index < len(confidence_rows) else [float("nan")] * 17)
                if len(xy) != 17 or len(confidence) != 17:
                    raise RuntimeError(f"Expected 17 COCO keypoints in {record.video_id}")
                xy_array = np.asarray(xy, dtype=np.float32)
                confidence_array = np.asarray(confidence, dtype=np.float32)
                valid = (np.isfinite(xy_array).all(axis=1) & np.isfinite(confidence_array) &
                         (confidence_array >= self.confidence_threshold) &
                         np.any(xy_array != 0, axis=1))
                xy_array[~valid] = np.nan
                track_id = int(ids[detection_index]) if detection_index < len(ids) else -1
                track_ids.append(track_id)
                all_xy.append(xy_array)
                all_confidence.append(confidence_array)
                all_valid.append(valid)
            offsets.append(len(track_ids))

        if len(frame_indices) != record.frame_count:
            raise RuntimeError(
                f"{record.video_id}: decoded {len(frame_indices)} frames, manifest declares {record.frame_count}"
            )
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray("racfall.pose.v1"),
                dataset=np.asarray(record.dataset), video_id=np.asarray(record.video_id),
                subject_id=np.asarray(record.subject_id), split=np.asarray(record.split),
                label=np.asarray(record.label), annotation_provenance=np.asarray(record.annotation_provenance),
                frame_index=np.asarray(frame_indices, dtype=np.int32),
                timestamp_s=np.asarray(timestamps, dtype=np.float64),
                frame_offsets=np.asarray(offsets, dtype=np.int64),
                track_id=np.asarray(track_ids, dtype=np.int64),
                xy=np.asarray(all_xy, dtype=np.float32).reshape(-1, 17, 2),
                confidence=np.asarray(all_confidence, dtype=np.float32).reshape(-1, 17),
                valid=np.asarray(all_valid, dtype=np.bool_).reshape(-1, 17),
                frame_width=np.asarray(record.width, dtype=np.int32),
                frame_height=np.asarray(record.height, dtype=np.int32),
                imgsz=np.asarray(self.imgsz, dtype=np.int32), device=np.asarray(self.device),
            )
        elapsed = time.perf_counter() - started
        return {
            "video_id": record.video_id,
            "frames": len(frame_indices),
            "pose_detections": len(track_ids),
            "elapsed_s": round(elapsed, 4),
            "throughput_fps": round(len(frame_indices) / elapsed, 3) if elapsed else None,
            "mean_inference_ms": round(sum(inference_ms) / len(inference_ms), 3) if inference_ms else None,
            "output": str(output_path),
        }
