"""Fail-closed CAUCAFall frame/annotation alignment and provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .manifest import ManifestRecord


SCHEMA_VERSION = "racfall.caucafall-alignment.v1"
TERMINAL_DUPLICATE_MAE_MAX = 0.1
MIN_KEYPOINTS_INSIDE_BOX = 0.8
_INDEX_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameAnnotation:
    index: int
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    filename: str


@dataclass(frozen=True)
class AlignmentEvidence:
    """Only measured evidence used by the deterministic index resolver."""

    representative_best_matches: dict[int, int]
    required_exact_indices: tuple[int, ...]
    terminal_pair_mae: float | None
    bbox_inside_fraction: dict[int, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frame_annotations(
    directory: str | Path, *, require_contiguous: bool = True,
) -> list[FrameAnnotation]:
    """Read original YOLO rows without changing or filling any annotation."""
    root = Path(directory)
    rows = []
    for path in root.iterdir():
        match = _INDEX_RE.match(path.name)
        if not match:
            continue
        parts = path.read_text(encoding="utf-8").strip().split()
        if len(parts) != 5:
            raise ValueError(f"{path}: expected one five-field YOLO annotation")
        try:
            class_id = int(parts[0])
            box = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}: invalid YOLO annotation") from exc
        if class_id not in {0, 1}:
            raise ValueError(f"{path}: unsupported class {class_id}")
        if any(not 0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"{path}: normalized box is outside [0, 1]")
        rows.append(FrameAnnotation(int(match.group("index")), class_id, *box, path.name))
    rows.sort(key=lambda row: row.index)
    if not rows:
        raise ValueError(f"No indexed TXT annotations found in {root}")
    expected = list(range(1, rows[-1].index + 1))
    observed = [row.index for row in rows]
    if require_contiguous and observed != expected:
        missing = sorted(set(expected) - set(observed))
        raise ValueError(f"Annotation indices are not contiguous from 1: missing={missing[:10]}")
    return rows


def resolve_index_alignment(
    decoded_frames: int,
    annotations: Iterable[FrameAnnotation],
    evidence: AlignmentEvidence | None,
) -> dict:
    """Resolve 1-based source annotations to 0-based decoded frames, or quarantine."""
    rows = list(annotations)
    indices = [row.index for row in rows]
    reasons = []
    if decoded_frames <= 0:
        reasons.append("video has no decoded frames")
    if indices != list(range(1, len(rows) + 1)):
        reasons.append("annotation indices are not contiguous from 1")

    terminal_extra = len(rows) == decoded_frames + 1
    if len(rows) == decoded_frames or terminal_extra:
        status = "verified_terminal_extra" if terminal_extra else "verified_exact_count"
        if evidence is None:
            reasons.append("alignment has no measured PNG/bounding-box evidence")
        else:
            for index in evidence.required_exact_indices:
                if evidence.representative_best_matches.get(index) != index:
                    reasons.append(f"reference PNG {index} does not best-match decoded frame {index}")
            if terminal_extra:
                if evidence.representative_best_matches.get(decoded_frames + 1) != decoded_frames:
                    reasons.append("extra terminal PNG does not best-match the final decoded frame")
                if (evidence.terminal_pair_mae is None or
                        evidence.terminal_pair_mae > TERMINAL_DUPLICATE_MAE_MAX):
                    reasons.append("terminal PNG pair is not a verified near-duplicate")
            for index in evidence.required_exact_indices:
                fraction = evidence.bbox_inside_fraction.get(index, 0.0)
                if fraction < MIN_KEYPOINTS_INSIDE_BOX:
                    reasons.append(f"annotation box {index} fails pose correspondence ({fraction:.3f})")
    else:
        status = "quarantined"
        reasons.append(
            f"unsupported count relationship: decoded={decoded_frames}, annotations={len(rows)}"
        )

    if reasons:
        return {
            "status": "quarantined",
            "sequence_quarantined": True,
            "reasons": reasons,
            "frame_labels": [-1] * max(decoded_frames, 0),
            "quarantined_annotations": indices,
        }
    mapped = rows[:decoded_frames]
    return {
        "status": status,
        "sequence_quarantined": False,
        "reasons": [],
        "frame_labels": [row.class_id for row in mapped],
        "quarantined_annotations": indices[decoded_frames:],
    }


def _bbox_inside_fraction(annotation: FrameAnnotation, xy, valid, width: int, height: int) -> float:
    np = __import__("numpy")
    usable = np.asarray(valid, dtype=np.bool_) & np.isfinite(xy).all(axis=1)
    if not usable.any():
        return 0.0
    x0 = (annotation.x_center - annotation.width / 2) * width
    x1 = (annotation.x_center + annotation.width / 2) * width
    y0 = (annotation.y_center - annotation.height / 2) * height
    y1 = (annotation.y_center + annotation.height / 2) * height
    points = xy[usable]
    inside = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
              (points[:, 1] >= y0) & (points[:, 1] <= y1))
    return float(inside.mean())


def audit_caucafall_alignment(
    dataset_root: str | Path,
    record: ManifestRecord,
    reference_png_dir: str | Path,
    pose_file: str | Path,
    output: str | Path,
) -> dict:
    """Measure representative PNG/TXT/video agreement and write an immutable audit record."""
    if record.dataset != "CAUCAFall" or record.source_type != "video":
        raise ValueError("CAUCAFall video record required")
    cv2 = __import__("cv2")
    np = __import__("numpy")
    root = Path(dataset_root).resolve()
    video = (root / record.source_relpath).resolve()
    if root not in video.parents:
        raise ValueError("Video path escapes dataset root")
    annotations = read_frame_annotations(video.parent, require_contiguous=False)
    classes_path = video.parent / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Missing official class list: {classes_path}")
    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    if class_names != ["nofall", "fall"]:
        raise ValueError(f"Unexpected CAUCAFall classes: {class_names}")
    observed_indices = [row.index for row in annotations]
    expected_indices = list(range(1, annotations[-1].index + 1))
    if observed_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(observed_indices))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset": record.dataset,
            "video_id": record.video_id,
            "subject_id": record.subject_id,
            "source_relpath": record.source_relpath,
            "status": "quarantined",
            "sequence_quarantined": True,
            "reasons": [f"annotation indices are not contiguous from 1: missing={missing}"],
            "frame_labels": [-1] * record.frame_count,
            "quarantined_annotations": observed_indices,
            "annotation_metadata": {
                "count": len(annotations), "observed_indices": observed_indices,
                "official_class_names": class_names,
            },
            "mapping": {"rule": None, "event_onset_s": None, "event_end_s": None},
        }
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return payload
    decoded = []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open {video}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
    finally:
        capture.release()
    n_frames = len(decoded)
    if n_frames != record.frame_count:
        raise RuntimeError(f"Decoded {n_frames} frames, manifest declares {record.frame_count}")

    transitions = [annotations[index].index for index in range(1, len(annotations))
                   if annotations[index].class_id != annotations[index - 1].class_id]
    representative = {1, (n_frames + 1) // 2, n_frames}
    for index in transitions:
        representative.update({index - 1, index})
    if len(annotations) == n_frames + 1:
        representative.add(n_frames + 1)
    reference_root = Path(reference_png_dir)
    prefix = Path(annotations[0].filename).stem.rsplit("-", 1)[0]
    reference_images = {}
    reference_paths = {}
    for index in sorted(representative):
        path = reference_root / f"{prefix}-{index:05d}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Missing official reference PNG: {path}")
        if image.shape != decoded[0].shape:
            raise ValueError(f"Reference PNG resolution mismatch: {path}")
        reference_images[index] = image
        reference_paths[index] = path

    matches = {}
    match_mae = {}
    for index, image in reference_images.items():
        errors = [float(np.abs(frame.astype(np.int16) - image.astype(np.int16)).mean())
                  for frame in decoded]
        best = int(np.argmin(errors)) + 1
        matches[index] = best
        match_mae[index] = {
            "best": errors[best - 1],
            "at_same_index": errors[index - 1] if index <= n_frames else None,
        }

    required_exact = sorted(index for index in representative if 1 < index <= n_frames)
    bbox_inside = {}
    with np.load(Path(pose_file), allow_pickle=False) as pose:
        if str(pose["video_id"]) != record.video_id or len(pose["frame_index"]) != n_frames:
            raise ValueError("Pose file does not match alignment video")
        offsets = pose["frame_offsets"]
        for index in sorted(representative):
            if index > n_frames:
                continue
            start, end = int(offsets[index - 1]), int(offsets[index])
            fractions = [
                _bbox_inside_fraction(annotations[index - 1], pose["xy"][detection],
                                      pose["valid"][detection], record.width, record.height)
                for detection in range(start, end)
            ]
            bbox_inside[index] = max(fractions, default=0.0)

    terminal_mae = None
    if n_frames in reference_images and n_frames + 1 in reference_images:
        terminal_mae = float(np.abs(
            reference_images[n_frames].astype(np.int16) -
            reference_images[n_frames + 1].astype(np.int16)
        ).mean())
    evidence = AlignmentEvidence(matches, tuple(required_exact), terminal_mae, bbox_inside)
    resolution = resolve_index_alignment(n_frames, annotations, evidence)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": record.dataset,
        "video_id": record.video_id,
        "subject_id": record.subject_id,
        "source_relpath": record.source_relpath,
        "source_sha256": record.source_sha256 or _sha256(video),
        "video_metadata": {
            "decoded_frames": n_frames, "fps": record.fps,
            "width": record.width, "height": record.height,
            "timestamp_formula": "timestamp_s = decoded_frame_index_zero_based / fps",
            "first_timestamp_s": 0.0,
            "last_timestamp_s": (n_frames - 1) / record.fps,
        },
        "annotation_metadata": {
            "count": len(annotations), "first_index": annotations[0].index,
            "last_index": annotations[-1].index,
            "index_base": 1, "filenames": [row.filename for row in annotations],
            "official_class_names": class_names,
            "classes_sha256": _sha256(classes_path),
            "class_semantics": {
                "0": "not on ground due to a fall",
                "1": "on ground due to a fall; not a fall-event onset annotation",
            },
        },
        "mapping": {
            "rule": "source annotation n maps to decoded frame n-1 only after evidence validation",
            "mapped_source_index_range_inclusive": [1, n_frames],
            "mapped_decoded_index_range_inclusive": [0, n_frames - 1],
            "event_onset_s": None, "event_end_s": None,
        },
        "evidence": {
            **asdict(evidence),
            "representative_match_mae": match_mae,
            "reference_png_sha256": {
                str(index): _sha256(path) for index, path in reference_paths.items()
            },
            "reference_png_filenames": {
                str(index): path.name for index, path in reference_paths.items()
            },
            "terminal_duplicate_mae_threshold": TERMINAL_DUPLICATE_MAE_MAX,
            "minimum_keypoints_inside_box": MIN_KEYPOINTS_INSIDE_BOX,
            "opening_frame_note": "opening static content may make the best pixel match non-unique",
        },
        **resolution,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_alignment_labels(path: str | Path, video_id: str, frame_count: int) -> tuple[list[int], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("video_id") != video_id:
        raise ValueError(f"Alignment provenance does not match {video_id}")
    labels = payload.get("frame_labels", [])
    if len(labels) != frame_count or any(value not in {-1, 0, 1} for value in labels):
        raise ValueError(f"Invalid frame labels for {video_id}")
    if payload.get("sequence_quarantined"):
        return [-1] * frame_count, "quarantined"
    return labels, str(payload["status"])
