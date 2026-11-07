"""Training-only, frame-accurate fall-event annotation and review.

Source ground-state labels are intentionally not loaded or displayed here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

from ..data.canonical import _indexed_pngs
from ..data.manifest import ManifestRecord


SCHEMA = "racfall.phase4c-event-annotations.v1"
TOOL_VERSION = "phase4c-v1"
LANDMARKS = (
    "fall_initiation", "ground_contact", "settled_ground_state",
    "recovery_start", "recovery_end",
)
STATUSES = {"exact", "unknown", "ambiguous", "not_applicable"}
CONFIDENCES = {"high", "medium", "low", "unknown"}
EVENT_VALID = {"1", "0", "-1"}
BASE_FIELDS = (
    "dataset", "video_id", "subject_id", "action", "split", "event_id",
    "event_valid", "contact_region", "unknown_reason", "reviewer_id",
    "review_round", "annotation_confidence", "annotation_tool_version",
    "created_at_utc", "source_sha256", "canonical_audit_sha256",
    "frame_count", "fps_proxy", "fps_provenance", "fps_uncertainty_s",
    "provenance_note", "review_status",
)
FIELDS = BASE_FIELDS + tuple(
    field for name in LANDMARKS for field in (f"{name}_frame", f"{name}_status")
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema_version") != "racfall.phase4c-config.v1" or
            config.get("status") != "frozen_before_training_only_pilot"):
        raise ValueError("Phase 4C requires the frozen pilot configuration")
    if tuple(config.get("landmarks", ())) != LANDMARKS:
        raise ValueError("Phase 4C landmark contract mismatch")
    return config


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def prepare_pilot(
    records: list[ManifestRecord], dataset_root: str | Path,
    canonical_audit_dir: str | Path, output_dir: str | Path,
    config_path: str | Path, reviewer_id: str,
) -> dict:
    """Create a training-only queue and blank immutable-review template."""
    config = _load_config(config_path)
    if not reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    allowed, forbidden = set(config["allowed_subjects"]), set(config["forbidden_subjects"])
    requested = [tuple(value) for value in config["pilot_sequences"]]
    by_key = {(row.subject_id, row.action): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("Manifest has duplicate subject/action records")
    root, audit_root = Path(dataset_root), Path(canonical_audit_dir)
    queue, template = [], []
    for subject, action in requested:
        if subject not in allowed or subject in forbidden:
            raise ValueError(f"Pilot contains forbidden subject: {subject}")
        record = by_key.get((subject, action))
        if record is None:
            raise ValueError(f"Pilot sequence missing from manifest: {subject}/{action}")
        if record.split != "train" or record.locked_blind:
            raise ValueError(f"Pilot sequence is not training-only: {record.video_id}")
        audit_path = audit_root / f"{record.video_id}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit.get("status") != "verified" or audit.get("sequence_quarantined") or
                audit.get("video_id") != record.video_id):
            raise ValueError(f"Sequence lacks verified canonical provenance: {record.video_id}")
        images = _indexed_pngs(root / record.source_relpath)
        if len(images) != record.frame_count:
            raise ValueError(f"Frame inventory mismatch: {record.video_id}")
        fps = float(config["timing"]["fps_proxy"])
        if not math.isclose(record.fps, fps, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"FPS proxy mismatch: {record.video_id}")
        audit_sha = _sha256(audit_path)
        queue.append({
            "video_id": record.video_id, "subject_id": subject, "action": action,
            "split": record.split, "image_directory": record.source_relpath,
            "frame_count": record.frame_count,
            "first_source_index": images[0][0], "last_source_index": images[-1][0],
            "fps_proxy": fps, "frame_period_s": 1.0 / fps,
            "fps_quantization_uncertainty_s": config["timing"]["quantization_uncertainty_frames"] / fps,
            "fps_systematic_uncertainty": config["timing"]["systematic_png_avi_offset"],
            "source_sha256": record.source_sha256,
            "canonical_audit_sha256": audit_sha,
            "ground_state_labels_exposed_to_tool": False,
        })
        row = {field: "" for field in FIELDS}
        row.update({
            "dataset": record.dataset, "video_id": record.video_id,
            "subject_id": subject, "action": action, "split": "train",
            "event_id": f"{record.video_id}-event-1",
            "event_valid": "-1", "unknown_reason": "pending_independent_review",
            "reviewer_id": reviewer_id, "review_round": "independent-1",
            "annotation_confidence": "unknown", "annotation_tool_version": TOOL_VERSION,
            "source_sha256": record.source_sha256,
            "canonical_audit_sha256": audit_sha, "frame_count": str(record.frame_count),
            "fps_proxy": str(fps),
            "fps_provenance": "canonical PNG ordinal / configured FPS proxy; no AVI index equivalence",
            "fps_uncertainty_s": str(config["timing"]["quantization_uncertainty_frames"] / fps),
            "provenance_note": config["timing"]["systematic_png_avi_offset"],
            "review_status": "pending",
        })
        for landmark in LANDMARKS:
            row[f"{landmark}_frame"] = "-1"
            row[f"{landmark}_status"] = "unknown"
        template.append(row)
    output = Path(output_dir)
    queue_path, annotation_path = output / "pilot-queue.json", output / f"annotations-{reviewer_id}.csv"
    _atomic_json(queue_path, {
        "schema_version": SCHEMA, "config_sha256": _sha256(config_path),
        "scope": "training_only", "reviewer_id": reviewer_id,
        "sequence_count": len(queue), "sequences": queue,
    })
    _write_csv(annotation_path, template)
    return {
        "sequence_count": len(queue),
        "fall_sequences": sum(row["action"].lower().startswith("fall") for row in queue),
        "adl_sequences": sum(not row["action"].lower().startswith("fall") for row in queue),
        "completed_reviews": 0, "exact_landmarks": 0,
        "queue": str(queue_path), "annotations": str(annotation_path),
    }


def _read_annotations(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Invalid Phase 4C annotation schema")
        return list(reader)


def validate_annotations(
    path: str | Path, config_path: str | Path, *, require_complete: bool = True,
    output: str | Path | None = None,
) -> dict:
    """Validate independent/adjudicated rows and derive timestamps, never boundaries."""
    config = _load_config(config_path)
    allowed, forbidden = set(config["allowed_subjects"]), set(config["forbidden_subjects"])
    rows = _read_annotations(path)
    if not rows:
        raise ValueError("Annotation file is empty")
    derived, pending = [], 0
    seen = set()
    for line, row in enumerate(rows, start=2):
        key = (row["video_id"], row["reviewer_id"], row["review_round"])
        if key in seen:
            raise ValueError(f"Duplicate annotation decision at line {line}")
        seen.add(key)
        if row["subject_id"] not in allowed or row["subject_id"] in forbidden or row["split"] != "train":
            raise ValueError(f"Non-training subject in annotations: {row['subject_id']}")
        if not row["reviewer_id"].strip() or not row["review_round"].strip():
            raise ValueError(f"Reviewer provenance is missing at line {line}")
        for checksum_field in ("source_sha256", "canonical_audit_sha256"):
            checksum = row[checksum_field]
            if len(checksum) != 64 or any(value not in "0123456789abcdef" for value in checksum.lower()):
                raise ValueError(f"Invalid {checksum_field} at line {line}")
        if row["event_valid"] not in EVENT_VALID:
            raise ValueError(f"Invalid event_valid at line {line}")
        if row["annotation_confidence"] not in CONFIDENCES:
            raise ValueError(f"Invalid annotation confidence at line {line}")
        if row["review_status"] not in {"pending", "reviewed", "adjudicated"}:
            raise ValueError(f"Invalid review status at line {line}")
        pending += row["review_status"] == "pending"
        if require_complete and row["review_status"] == "pending":
            raise ValueError("Annotation review is incomplete")
        if row["review_status"] != "pending" and not row["created_at_utc"].strip():
            raise ValueError(f"Completed review lacks created_at_utc at line {line}")
        frame_count, fps = int(row["frame_count"]), float(row["fps_proxy"])
        if frame_count <= 0 or fps <= 0 or not math.isfinite(fps):
            raise ValueError(f"Invalid timing provenance at line {line}")
        exact = []
        timestamps = {}
        unresolved = False
        for landmark in LANDMARKS:
            status = row[f"{landmark}_status"]
            if status not in STATUSES:
                raise ValueError(f"Invalid {landmark} status at line {line}")
            frame = int(row[f"{landmark}_frame"])
            if status == "exact":
                if not 0 <= frame < frame_count:
                    raise ValueError(f"{landmark} frame out of bounds at line {line}")
                exact.append((landmark, frame))
                timestamps[f"{landmark}_s"] = frame / fps
            elif frame != -1:
                raise ValueError(f"Non-exact {landmark} must use frame -1 at line {line}")
            else:
                timestamps[f"{landmark}_s"] = None
                unresolved |= status in {"unknown", "ambiguous"}
        frames = [frame for _name, frame in exact]
        if frames != sorted(frames):
            raise ValueError(f"Landmarks are not temporally ordered at line {line}")
        if unresolved and not row["unknown_reason"].strip():
            raise ValueError(f"Unknown/ambiguous boundary lacks reason at line {line}")
        if row["event_valid"] == "0" and any(row[f"{name}_status"] != "not_applicable" for name in LANDMARKS):
            raise ValueError(f"Reviewed non-fall must mark all landmarks not_applicable at line {line}")
        if row["event_valid"] == "1" and row["fall_initiation_status"] == "not_applicable":
            raise ValueError(f"Valid fall cannot have not_applicable initiation at line {line}")
        derived.append({
            "video_id": row["video_id"], "event_id": row["event_id"],
            "reviewer_id": row["reviewer_id"], "event_valid": int(row["event_valid"]),
            "review_status": row["review_status"], "timestamps_s": timestamps,
            "timestamp_quantization_uncertainty_s": float(row["fps_uncertainty_s"]),
            "timestamp_systematic_uncertainty": row["provenance_note"],
        })
    result = {
        "schema_version": SCHEMA, "rows": len(rows), "pending_rows": pending,
        "reviewed_rows": len(rows) - pending,
        "exact_landmarks": sum(
            row[f"{name}_status"] == "exact" for row in rows for name in LANDMARKS
        ),
        "unknown_or_ambiguous_landmarks": sum(
            row[f"{name}_status"] in {"unknown", "ambiguous"}
            for row in rows for name in LANDMARKS
        ),
        "derived_timing": derived,
    }
    if output is not None:
        _atomic_json(Path(output), result)
    return result


def compare_reviews(
    first_path: str | Path, second_path: str | Path, config_path: str | Path,
    output: str | Path | None = None,
) -> dict:
    """Compare two independent reviews and identify required adjudications."""
    config = _load_config(config_path)
    validate_annotations(first_path, config_path)
    validate_annotations(second_path, config_path)
    tolerance = int(config["review"]["frame_tolerance_for_agreement"])
    first, second = _read_annotations(first_path), _read_annotations(second_path)
    left, right = {row["video_id"]: row for row in first}, {row["video_id"]: row for row in second}
    if set(left) != set(right) or len(left) != len(first) or len(right) != len(second):
        raise ValueError("Review files must contain the same unique videos")
    details, exact_total, within_total, compared_total = [], 0, 0, 0
    validity_agreement = 0
    validity_pairs = []
    for video_id in sorted(left):
        a, b = left[video_id], right[video_id]
        if a["reviewer_id"] == b["reviewer_id"]:
            raise ValueError("IAA requires different reviewer IDs")
        validity_same = a["event_valid"] == b["event_valid"]
        validity_agreement += validity_same
        validity_pairs.append((a["event_valid"], b["event_valid"]))
        disagreements = []
        landmark_rows = {}
        for name in LANDMARKS:
            sa, sb = a[f"{name}_status"], b[f"{name}_status"]
            fa, fb = int(a[f"{name}_frame"]), int(b[f"{name}_frame"])
            difference = abs(fa - fb) if sa == sb == "exact" else None
            exact_agree = difference == 0
            within = difference is not None and difference <= tolerance
            compared_total += difference is not None
            exact_total += exact_agree
            within_total += within
            if sa != sb or (difference is not None and difference > tolerance):
                disagreements.append(name)
            landmark_rows[name] = {
                "reviewer_1_status": sa, "reviewer_2_status": sb,
                "absolute_frame_difference": difference,
                "within_tolerance": within,
            }
        adjudicate = not validity_same or bool(disagreements)
        details.append({
            "video_id": video_id, "event_valid_agreement": validity_same,
            "landmarks": landmark_rows, "disagreements": disagreements,
            "adjudication_required": adjudicate,
        })
    categories = sorted(EVENT_VALID)
    observed = validity_agreement / len(details)
    expected = sum(
        (sum(a == category for a, _b in validity_pairs) / len(details)) *
        (sum(b == category for _a, b in validity_pairs) / len(details))
        for category in categories
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else None)
    result = {
        "schema_version": "racfall.phase4c-review-comparison.v1",
        "videos": len(details), "frame_tolerance": tolerance,
        "event_valid_raw_agreement": observed,
        "event_valid_cohen_kappa": kappa,
        "exact_frame_agreement": exact_total / compared_total if compared_total else None,
        "within_tolerance_agreement": within_total / compared_total if compared_total else None,
        "known_frame_comparisons": compared_total,
        "adjudication_required": sum(row["adjudication_required"] for row in details),
        "details": details,
    }
    if output is not None:
        _atomic_json(Path(output), result)
    return result


def annotate_sequence(
    queue_path: str | Path, annotations_path: str | Path,
    dataset_root: str | Path, video_id: str,
) -> None:
    """Open a local OpenCV frame navigator and update one review row in place."""
    cv2 = __import__("cv2")
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    sequence = next((row for row in queue["sequences"] if row["video_id"] == video_id), None)
    if sequence is None:
        raise ValueError(f"Video is not in the pilot queue: {video_id}")
    rows = _read_annotations(annotations_path)
    row = next((value for value in rows if value["video_id"] == video_id), None)
    if row is None:
        raise ValueError(f"Video is not in the annotation file: {video_id}")
    images = _indexed_pngs(Path(dataset_root) / sequence["image_directory"])
    position, selected, message = 0, 0, ""
    window = "RAC-Fall Phase 4C (local; no image retention)"
    try:
        while True:
            image = cv2.imread(str(images[position][2]))
            if image is None:
                raise RuntimeError(f"Cannot decode {images[position][2]}")
            lines = [
                f"{video_id} frame {position}/{len(images)-1} t={position/sequence['fps_proxy']:.3f}s",
                "left/right or j/l: frame | 1-5: select+set exact | x: unknown | a: ambiguous",
                "0: non-fall | f: valid fall | ?: unresolved | h/m/o: confidence | w: save | q: quit",
            ]
            for index, name in enumerate(LANDMARKS):
                marker = ">" if index == selected else " "
                lines.append(f"{marker}{index+1} {name}: {row[f'{name}_status']} {row[f'{name}_frame']}")
            lines.append(f"event_valid={row['event_valid']} confidence={row['annotation_confidence']} {message}")
            y = 28
            for text in lines:
                cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                y += 24
            cv2.imshow(window, image)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (81, ord("j")):
                position = max(0, position - 1)
            elif key in (83, ord("l")):
                position = min(len(images) - 1, position + 1)
            elif ord("1") <= key <= ord("5"):
                selected = key - ord("1")
                name = LANDMARKS[selected]
                row[f"{name}_frame"], row[f"{name}_status"] = str(position), "exact"
            elif key in (ord("x"), ord("a")):
                name = LANDMARKS[selected]
                row[f"{name}_frame"] = "-1"
                row[f"{name}_status"] = "unknown" if key == ord("x") else "ambiguous"
                row["unknown_reason"] = (
                    f"{name}_not_visible" if key == ord("x") else f"{name}_visually_ambiguous"
                )
            elif key == ord("0"):
                row["event_valid"] = "0"
                row["unknown_reason"] = "reviewed_non_fall"
                for name in LANDMARKS:
                    row[f"{name}_frame"], row[f"{name}_status"] = "-1", "not_applicable"
            elif key == ord("f"):
                row["event_valid"] = "1"
            elif key == ord("?"):
                row["event_valid"] = "-1"
            elif key in (ord("h"), ord("m"), ord("o")):
                row["annotation_confidence"] = {ord("h"): "high", ord("m"): "medium", ord("o"): "low"}[key]
            elif key == ord("w"):
                if row["event_valid"] == "1" and all(
                        row[f"{name}_status"] == "exact" for name in LANDMARKS):
                    row["unknown_reason"] = ""
                row["created_at_utc"] = datetime.now(timezone.utc).isoformat()
                row["review_status"] = "reviewed"
                _write_csv(Path(annotations_path), rows)
                message = "saved"
    finally:
        cv2.destroyAllWindows()
