"""Fail-closed adapters for CAUCAFall videos and URFD RGB sequences."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

from .manifest import ManifestRecord, validate_manifest


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash relative names and bytes without changing an extracted source tree."""
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def load_label_map(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Label map must be a non-empty JSON object")
    labels = {str(key): str(value) for key, value in data.items()}
    invalid = {value for value in labels.values() if value not in {"fall", "non_fall"}}
    if invalid:
        raise ValueError(f"Invalid label-map values: {sorted(invalid)}")
    return labels


class CAUCAFallAdapter:
    """Inspect CAUCAFall AVI files without modifying the source tree."""

    def __init__(self, root: str | Path, label_map: dict[str, str], *, hash_sources: bool = True):
        self.root = Path(root).resolve()
        self.label_map = label_map
        self.hash_sources = hash_sources

    def scan(self) -> list[ManifestRecord]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"CAUCAFall root not found: {self.root}")
        try:
            cv2 = __import__("cv2")
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to inspect CAUCAFall videos") from exc
        videos = sorted(path for path in self.root.rglob("*") if path.suffix.lower() == ".avi")
        if not videos:
            raise ValueError("No CAUCAFall .avi files found")
        records = []
        for video in videos:
            relative = video.relative_to(self.root)
            if len(relative.parts) < 3:
                raise ValueError(f"Expected subject/activity/video layout: {relative}")
            subject_id, action = relative.parts[0], relative.parts[-2]
            if action not in self.label_map:
                raise ValueError(
                    f"Unmapped CAUCAFall activity {action!r}; provide an explicit source-verified label map"
                )
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video: {relative}")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                capture.release()
            if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid video metadata: {relative}")
            sequence = relative.parent.as_posix()
            records.append(ManifestRecord(
                dataset="CAUCAFall", dataset_version="v5", subject_id=subject_id,
                subject_provenance="top-level source directory name",
                video_id=f"caucafall-{_safe_id(relative.with_suffix('').as_posix())}",
                sequence_id=f"caucafall-{_safe_id(sequence)}",
                source_relpath=relative.as_posix(), source_type="video", action=action,
                label=self.label_map[action], label_start_s="", label_end_s="",
                annotation_provenance="sequence label from explicit researcher-reviewed activity map; no event boundary",
                timestamp_source="AVI frame index / container FPS", start_timestamp_s=0.0,
                end_timestamp_s=frame_count / fps, fps=fps, width=width, height=height,
                frame_count=frame_count, split="",
                source_sha256=sha256_file(video) if self.hash_sources else "",
                license="CC BY 4.0", locked_blind=False,
            ))
        validate_manifest(records)
        return records


class CAUCAFallCanonicalAdapter:
    """Inspect official PNG sequences; AVI supplies timing rate only."""

    def __init__(self, root: str | Path, label_map: dict[str, str], *, hash_sources: bool = True):
        self.root = Path(root).resolve()
        self.label_map = label_map
        self.hash_sources = hash_sources

    def scan(self) -> list[ManifestRecord]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"CAUCAFall root not found: {self.root}")
        cv2 = __import__("cv2")
        from .canonical import _indexed_pngs
        activity_dirs = sorted({path.parent for path in self.root.rglob("*.png")})
        if not activity_dirs:
            raise ValueError("No CAUCAFall PNG sequences found")
        records = []
        for directory in activity_dirs:
            relative = directory.relative_to(self.root)
            if len(relative.parts) != 2:
                raise ValueError(f"Expected subject/activity PNG layout: {relative}")
            subject_id, action = relative.parts
            if action not in self.label_map:
                raise ValueError(f"Unmapped CAUCAFall activity {action!r}")
            pngs = _indexed_pngs(directory)
            first = cv2.imread(str(pngs[0][2]))
            if first is None:
                raise ValueError(f"Cannot decode canonical PNG: {pngs[0][2]}")
            height, width = map(int, first.shape[:2])
            avis = sorted(directory.glob("*.avi"))
            if len(avis) != 1:
                raise ValueError(f"Expected exactly one secondary AVI in {relative}, found {len(avis)}")
            capture = cv2.VideoCapture(str(avis[0]))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open secondary AVI: {avis[0]}")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid secondary AVI FPS: {avis[0]}")
            sequence_id = f"caucafall-{_safe_id(relative.as_posix())}"
            records.append(ManifestRecord(
                dataset="CAUCAFall", dataset_version="v5", subject_id=subject_id,
                subject_provenance="top-level source directory name",
                video_id=f"{sequence_id}-png", sequence_id=sequence_id,
                source_relpath=relative.as_posix(), source_type="image_sequence_dir",
                action=action, label=self.label_map[action], label_start_s="", label_end_s="",
                annotation_provenance=(
                    "same-index official PNG/TXT ground-state label; missing TXT remains unknown; "
                    "no event boundary"
                ),
                timestamp_source=(
                    f"canonical PNG ordinal / secondary AVI {avis[0].name} container FPS; "
                    "no AVI frame-index equivalence"
                ),
                start_timestamp_s=0.0, end_timestamp_s=len(pngs) / fps, fps=fps,
                width=width, height=height, frame_count=len(pngs), split="",
                source_sha256=sha256_tree(directory) if self.hash_sources else "",
                license="CC BY 4.0", locked_blind=False,
            ))
        validate_manifest(records)
        return records


def _natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _read_image_metadata(source: Path, source_type: str, names: list[str]) -> tuple[int, int]:
    try:
        cv2 = __import__("cv2")
        np = __import__("numpy")
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to inspect URFD images") from exc
    if source_type == "image_sequence_zip":
        with zipfile.ZipFile(source) as archive:
            image = cv2.imdecode(np.frombuffer(archive.read(names[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(source / names[0]))
    if image is None:
        raise ValueError(f"Cannot decode first RGB frame from {source}")
    return int(image.shape[1]), int(image.shape[0])


def _sync_timestamps(path: Path) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"URFD synchronization CSV not found: {path.name}")
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel
    values = []
    for row in csv.reader(io.StringIO(text), dialect):
        numeric = []
        for cell in row:
            try:
                numeric.append(float(cell.strip()))
            except ValueError:
                pass
        if len(numeric) >= 2 and math.isfinite(numeric[1]):
            values.append(numeric[1] / 1000.0)
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"Invalid or insufficient URFD timestamps: {path.name}")
    return values


class URFDRGBAdapter:
    """Inspect extracted URFD RGB directories or original RGB ZIP archives."""

    PATTERN = re.compile(r"^(fall|adl)-(\d+)-cam([01])-rgb$", re.IGNORECASE)

    def __init__(self, root: str | Path, *, hash_sources: bool = True):
        self.root = Path(root).resolve()
        self.hash_sources = hash_sources

    def scan(self) -> list[ManifestRecord]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"URFD root not found: {self.root}")
        sources = sorted(list(self.root.rglob("*-rgb.zip")) + [
            path for path in self.root.rglob("*-rgb") if path.is_dir()
        ])
        if not sources:
            raise ValueError("No URFD *-rgb.zip archives or extracted *-rgb directories found")
        records = []
        for source in sources:
            stem = source.name[:-4] if source.suffix.lower() == ".zip" else source.name
            match = self.PATTERN.fullmatch(stem)
            if not match:
                raise ValueError(f"Unrecognized URFD RGB sequence name: {source.name}")
            activity, number, camera = match.groups()
            activity = activity.lower()
            if source.suffix.lower() == ".zip":
                source_type = "image_sequence_zip"
                with zipfile.ZipFile(source) as archive:
                    names = sorted((name for name in archive.namelist()
                                    if name.lower().endswith(".png") and not name.endswith("/")), key=_natural_key)
            else:
                source_type = "image_sequence_dir"
                names = sorted((path.name for path in source.iterdir()
                                if path.is_file() and path.suffix.lower() == ".png"), key=_natural_key)
            if not names:
                raise ValueError(f"No PNG frames in {source.name}")
            sync_name = f"{activity}-{number}-data.csv"
            sync_path = next(iter(sorted(self.root.rglob(sync_name))), self.root / sync_name)
            timestamps = _sync_timestamps(sync_path)
            if len(timestamps) != len(names):
                raise ValueError(
                    f"{stem}: RGB frame count ({len(names)}) differs from sync timestamps ({len(timestamps)})"
                )
            differences = [b - a for a, b in zip(timestamps, timestamps[1:])]
            fps = 1.0 / statistics.median(differences)
            width, height = _read_image_metadata(source, source_type, names)
            relative = source.relative_to(self.root)
            checksum = (sha256_file(source) if source.is_file() else sha256_tree(source)) if self.hash_sources else ""
            records.append(ManifestRecord(
                dataset="URFD", dataset_version="public-web-release", subject_id="unknown",
                subject_provenance="not published on dataset download page; unresolved",
                video_id=f"urfd-{activity}-{number}-cam{camera}",
                sequence_id=f"urfd-{activity}-{number}", source_relpath=relative.as_posix(),
                source_type=source_type, action=activity,
                label="fall" if activity == "fall" else "non_fall",
                label_start_s="", label_end_s="",
                annotation_provenance="sequence class from official URFD sequence name; no event boundary",
                timestamp_source=f"{sync_path.relative_to(self.root).as_posix()} milliseconds",
                start_timestamp_s=timestamps[0], end_timestamp_s=timestamps[-1] + statistics.median(differences),
                fps=fps, width=width, height=height, frame_count=len(names), split="",
                source_sha256=checksum, license="CC BY-NC-SA 4.0; non-commercial academic use",
                locked_blind=False,
            ))
        validate_manifest(records)
        return records
