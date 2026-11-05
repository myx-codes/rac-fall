"""Canonical CAUCAFall PNG/TXT inventory and fail-closed quality gates."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re

from .alignment import FrameAnnotation, read_caucafall_classes, read_frame_annotations
from .manifest import ManifestRecord
from .target import TargetSelectionConfig, select_causal_target


SCHEMA_VERSION = "racfall.caucafall-canonical.v1"
QUALITY_SCHEMA_VERSION = "racfall.sequence-quality.v1"
_PNG_RE = re.compile(
    r"^(?P<prefix>.+?)(?P<index>\d{5})(?: - copia)?\.png$", re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed_pngs(directory: str | Path) -> list[tuple[int, str, Path]]:
    root = Path(directory)
    rows = []
    for path in root.iterdir():
        match = _PNG_RE.fullmatch(path.name)
        if match:
            rows.append((int(match.group("index")), match.group("prefix"), path))
    rows.sort(key=lambda row: (row[0], row[2].name))
    if not rows:
        raise ValueError(f"No indexed CAUCAFall PNG frames found in {root}")
    indices = [row[0] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate PNG indices in {root}")
    prefixes = {row[1] for row in rows}
    if len(prefixes) != 1:
        raise ValueError(f"Multiple PNG prefixes in one sequence: {sorted(prefixes)}")
    return rows


def audit_canonical_sequence(
    image_dir: str | Path,
    annotation_dir: str | Path,
    *,
    sequence_label: str,
    video_id: str,
    subject_id: str,
    action: str,
    timing_fps: float,
    timing_provenance: str,
    output: str | Path | None = None,
) -> dict:
    """Pair official PNG/TXT indices without consulting or aligning AVI frames."""
    if timing_fps <= 0:
        raise ValueError("A positive secondary timing FPS is required")
    cv2 = __import__("cv2")
    pngs = _indexed_pngs(image_dir)
    annotations = read_frame_annotations(annotation_dir, require_contiguous=False)
    classes = read_caucafall_classes(Path(annotation_dir) / "classes.txt", sequence_label)
    png_indices = [row[0] for row in pngs]
    annotation_by_index: dict[int, FrameAnnotation] = {row.index: row for row in annotations}
    annotation_indices = sorted(annotation_by_index)
    expected_png_indices = list(range(png_indices[0], png_indices[-1] + 1))
    missing_png_indices = sorted(set(expected_png_indices) - set(png_indices))
    missing_txt_indices = sorted(set(png_indices) - set(annotation_indices))
    orphan_txt_indices = sorted(set(annotation_indices) - set(png_indices))

    prefixes = {row[1] for row in pngs}
    png_prefix = next(iter(prefixes))
    txt_prefixes = {
        re.sub(r"\d{5}\.txt$", "", row.filename, flags=re.IGNORECASE)
        for row in annotations
    }
    reasons = []
    if png_indices[0] != 1:
        reasons.append(f"PNG indexing does not start at 1: first={png_indices[0]}")
    if missing_png_indices:
        reasons.append(f"PNG indices are not contiguous: missing={missing_png_indices[:20]}")
    if len(txt_prefixes) != 1 or next(iter(txt_prefixes), None) != png_prefix:
        reasons.append(
            f"PNG/TXT filename prefixes differ: png={png_prefix!r}, txt={sorted(txt_prefixes)!r}"
        )
    if orphan_txt_indices:
        reasons.append(f"TXT annotations have no canonical PNG: {orphan_txt_indices[:20]}")

    width = height = None
    image_errors = []
    frames = []
    for ordinal, (source_index, _prefix, path) in enumerate(pngs):
        image = cv2.imread(str(path))
        if image is None:
            image_errors.append(f"cannot decode {path.name}")
            current_width = current_height = None
        else:
            current_height, current_width = map(int, image.shape[:2])
            if width is None:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                image_errors.append(
                    f"resolution mismatch {path.name}: {current_width}x{current_height}"
                )
        annotation = annotation_by_index.get(source_index)
        frames.append({
            "ordinal_zero_based": ordinal,
            "source_index_one_based": source_index,
            "timestamp_s": ordinal / timing_fps,
            "png_filename": path.name,
            "png_sha256": _sha256(path),
            "txt_filename": annotation.filename if annotation else None,
            "txt_sha256": (
                _sha256(Path(annotation_dir) / annotation.filename) if annotation else None
            ),
            "ground_state_label": annotation.class_id if annotation else -1,
            "ground_state_valid": annotation is not None,
            "event_label": -1,
            "event_provenance": "unknown; no source event boundary",
        })
    reasons.extend(image_errors[:20])
    sequence_quarantined = bool(reasons)
    if sequence_quarantined:
        status = "quarantined"
    elif missing_txt_indices:
        status = "verified_with_invalid_labels"
    else:
        status = "verified"
    frame_labels = [frame["ground_state_label"] for frame in frames]
    label_valid = [frame["ground_state_valid"] for frame in frames]
    effective_labels = [-1] * len(frames) if sequence_quarantined else frame_labels
    effective_valid = [False] * len(frames) if sequence_quarantined else label_valid
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "CAUCAFall",
        "video_id": video_id,
        "subject_id": subject_id,
        "action": action,
        "canonical_source": "official PNG ordinal and same-index TXT annotation",
        "avi_role": "secondary timing/video source only; no PNG-to-AVI index equivalence",
        "timing": {
            "fps": timing_fps,
            "provenance": timing_provenance,
            "formula": "timestamp_s = canonical_png_ordinal_zero_based / fps",
        },
        "class_names": classes,
        "class_semantics": {
            "0": "not on ground due to a fall",
            "1": "on ground due to a fall; not fall-event initiation",
            "-1": "unknown because the same-index TXT is missing or sequence is quarantined",
        },
        "status": status,
        "sequence_quarantined": sequence_quarantined,
        "reasons": reasons,
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "png_index_range": [png_indices[0], png_indices[-1]],
        "annotation_index_range": (
            [annotation_indices[0], annotation_indices[-1]] if annotation_indices else None
        ),
        "missing_png_indices": missing_png_indices,
        "missing_txt_indices": missing_txt_indices,
        "orphan_txt_indices": orphan_txt_indices,
        "paired_txt_frames": sum(label_valid),
        "valid_ground_state_frames": sum(effective_valid),
        "invalid_ground_state_frames": len(effective_valid) - sum(effective_valid),
        "source_ground_state_class_counts": {
            "0": frame_labels.count(0), "1": frame_labels.count(1), "-1": frame_labels.count(-1),
        },
        "ground_state_class_counts": {
            "0": effective_labels.count(0), "1": effective_labels.count(1),
            "-1": effective_labels.count(-1),
        },
        "frame_labels": effective_labels,
        "label_valid": effective_valid,
        "event_labels": [-1] * len(frames),
        "frames": frames,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        os.replace(temporary, output_path)
    return payload


def assess_sequence_quality(
    canonical_audit: str | Path | dict,
    pose_file: str | Path,
    record: ManifestRecord,
    *,
    min_pose_coverage: float = 0.5,
    output: str | Path | None = None,
) -> dict:
    """Create an annotation-free inference quality gate plus label-availability gate."""
    if not 0.0 <= min_pose_coverage <= 1.0:
        raise ValueError("min_pose_coverage must be in [0, 1]")
    np = __import__("numpy")
    audit = (canonical_audit if isinstance(canonical_audit, dict) else
             json.loads(Path(canonical_audit).read_text(encoding="utf-8")))
    if audit.get("schema_version") != SCHEMA_VERSION or audit.get("video_id") != record.video_id:
        raise ValueError("Canonical audit does not match manifest record")
    with np.load(Path(pose_file), allow_pickle=False) as pose:
        if str(pose["video_id"]) != record.video_id:
            raise ValueError("Pose artifact does not match manifest record")
        if len(pose["frame_index"]) != audit["frame_count"]:
            raise ValueError("Pose frame count does not match canonical PNG inventory")
        config = TargetSelectionConfig()
        selected = select_causal_target(
            pose["frame_offsets"], pose["track_id"], pose["xy"], pose["confidence"],
            pose["valid"], record.width, record.height, config=config,
        )
        present = selected >= 0
        selected_detections = selected[present]
        valid_keypoints = int(pose["valid"][selected_detections].sum()) if present.any() else 0
        possible_keypoints = int(len(selected_detections) * 17)
        detection_frames = int(np.diff(pose["frame_offsets"]).astype(bool).sum())
    pose_coverage = float(present.mean()) if len(present) else 0.0
    valid_labels = int(sum(audit["label_valid"]))
    reasons = list(audit.get("reasons", []))
    if audit.get("sequence_quarantined"):
        reasons.append("canonical PNG/TXT audit is quarantined")
    if valid_labels == 0:
        reasons.append("no verified ground-state labels")
    if pose_coverage < min_pose_coverage:
        reasons.append(
            f"causal target pose coverage {pose_coverage:.6f} is below {min_pose_coverage:.6f}"
        )
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "dataset": record.dataset,
        "video_id": record.video_id,
        "subject_id": record.subject_id,
        "split": record.split,
        "status": "verified" if not reasons else "quarantined",
        "training_eligible": not reasons,
        "reasons": reasons,
        "canonical_status": audit["status"],
        "frames": audit["frame_count"],
        "valid_ground_state_frames": valid_labels,
        "invalid_ground_state_frames": audit["frame_count"] - valid_labels,
        "ground_state_class_counts": audit["ground_state_class_counts"],
        "event_labels_known": 0,
        "frames_with_any_pose_detection": detection_frames,
        "causal_target_frames": int(present.sum()),
        "missing_target_frames": int((~present).sum()),
        "missing_target_rate": float((~present).mean()) if len(present) else 1.0,
        "valid_keypoints": valid_keypoints,
        "invalid_keypoints": possible_keypoints - valid_keypoints,
        "invalid_keypoint_rate": (
            (possible_keypoints - valid_keypoints) / possible_keypoints
            if possible_keypoints else 1.0
        ),
        "min_pose_coverage": min_pose_coverage,
        "target_selection": "causal geometry without annotation boxes",
        "target_selection_config": asdict(config),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        os.replace(temporary, output_path)
    return payload
