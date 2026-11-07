"""Local scene-object detection and training-only feasibility audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..data.canonical import _indexed_pngs
from ..data.manifest import ManifestRecord
from ..data.target import TargetSelectionConfig, select_causal_target


AUDIT_SCHEMA = "racfall.phase4b-scene-audit.v1"
MANUAL_FIELDS = (
    "subject_id", "video_id", "frame_ordinal", "source_index", "image_relpath",
    "image_sha256", "predicted_classes", "predicted_max_confidence",
    "predicted_relation", "predicted_iou", "predicted_proximity",
    "manual_chair_present", "manual_couch_present", "manual_bed_present",
    "manual_relation", "relation_plausible", "occluded_or_ambiguous",
    "reviewer", "review_status", "notes",
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


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_relationship(person_box, object_box, frame_width: int, frame_height: int, *,
                      near_edge_distance_ratio: float = 0.15,
                      overlap_iou_threshold: float = 0.05) -> dict:
    """Return scale-normalized spatial relations without semantic safety claims."""
    if frame_width <= 0 or frame_height <= 0 or near_edge_distance_ratio <= 0:
        raise ValueError("Invalid frame geometry")
    px1, py1, px2, py2 = map(float, person_box)
    ox1, oy1, ox2, oy2 = map(float, object_box)
    if px2 <= px1 or py2 <= py1 or ox2 <= ox1 or oy2 <= oy1:
        raise ValueError("Boxes must have positive area")
    ix1, iy1 = max(px1, ox1), max(py1, oy1)
    ix2, iy2 = min(px2, ox2), min(py2, oy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((px2 - px1) * (py2 - py1) + (ox2 - ox1) * (oy2 - oy1) - intersection)
    iou = intersection / union if union else 0.0
    dx = max(ox1 - px2, px1 - ox2, 0.0)
    dy = max(oy1 - py2, py1 - oy2, 0.0)
    edge_distance_ratio = math.hypot(dx, dy) / math.hypot(frame_width, frame_height)
    proximity = _clip(1.0 - edge_distance_ratio / near_edge_distance_ratio)
    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    center_distance_ratio = math.dist(person_center, object_center) / math.hypot(
        frame_width, frame_height,
    )
    if iou >= overlap_iou_threshold:
        relation = "overlap"
    elif proximity > 0:
        relation = "near"
    else:
        relation = "far"
    return {
        "relation": relation, "iou": float(iou), "proximity": proximity,
        "edge_distance_ratio": float(edge_distance_ratio),
        "center_distance_ratio": float(center_distance_ratio),
        "person_bottom_minus_object_top_ratio": float((py2 - oy1) / frame_height),
    }


def pose_bbox(xy, valid, frame_width: int, frame_height: int, padding: float = 0.10):
    np = __import__("numpy")
    points = np.asarray(xy, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_) & np.isfinite(points).all(axis=1)
    if int(mask.sum()) < 5:
        return None
    x1, y1 = points[mask].min(axis=0)
    x2, y2 = points[mask].max(axis=0)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    return [
        max(0.0, x1 - padding * width), max(0.0, y1 - padding * height),
        min(float(frame_width), x2 + padding * width),
        min(float(frame_height), y2 + padding * height),
    ]


class SceneDetector:
    """COCO scene-object detector. It never saves input images or annotated renders."""

    def __init__(self, weights: str | Path, config: dict, *, device: str = "mps"):
        expected = config["detector"]["weights_sha256"]
        if _sha256(weights) != expected:
            raise ValueError("Scene detector checksum mismatch")
        ultralytics = __import__("ultralytics")
        self.model = ultralytics.YOLO(str(weights))
        self.device = device
        self.imgsz = int(config["detector"]["imgsz"])
        self.confidence = float(config["detector"]["confidence_threshold"])
        self.class_ids = {name: int(value) for name, value in config["detector"]["classes"].items()}
        for name, class_id in self.class_ids.items():
            if str(self.model.names.get(class_id)) != name:
                raise ValueError(f"Detector class mismatch for {name}: {class_id}")

    def detect(self, image_path: str | Path) -> tuple[list[dict], float]:
        started = time.perf_counter()
        results = self.model.predict(
            source=str(image_path), device=self.device, imgsz=self.imgsz,
            conf=self.confidence, classes=list(self.class_ids.values()),
            verbose=False, save=False, show=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy()
                reverse = {value: key for key, value in self.class_ids.items()}
                for box, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
                    detections.append({
                        "class": reverse[int(class_id)], "class_id": int(class_id),
                        "confidence": float(confidence), "bbox_xyxy": [float(v) for v in box],
                    })
        return detections, elapsed_ms


def _sample_positions(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0 or count > frame_count:
        raise ValueError("Invalid deterministic sample request")
    np = __import__("numpy")
    # Avoid the first/last 5% where the subject often only enters/leaves.
    values = np.linspace(0.05, 0.95, count)
    positions = [min(frame_count - 1, int(round(value * (frame_count - 1)))) for value in values]
    if len(set(positions)) != count:
        raise ValueError("Frame count is too small for unique audit samples")
    return positions


def _load_target_boxes(pose_path: Path, record: ManifestRecord):
    np = __import__("numpy")
    with np.load(pose_path, allow_pickle=False) as pose:
        if str(pose["video_id"]) != record.video_id or str(pose["split"]) != "train":
            raise ValueError("Training-only pose provenance mismatch")
        selected = select_causal_target(
            pose["frame_offsets"], pose["track_id"], pose["xy"], pose["confidence"],
            pose["valid"], record.width, record.height, config=TargetSelectionConfig(),
        )
        boxes = []
        for detection in selected:
            boxes.append(None if detection < 0 else pose_bbox(
                pose["xy"][detection], pose["valid"][detection], record.width, record.height,
            ))
    return boxes


def audit_training_scene_context(
    records: list[ManifestRecord], dataset_root: str | Path, pose_dir: str | Path,
    weights: str | Path, config_path: str | Path, output_dir: str | Path, *, device: str = "mps",
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (config.get("schema_version") != "racfall.phase4b-config.v1" or
            config.get("status") != "preregistered_before_training_only_scene_audit"):
        raise ValueError("Phase 4B audit requires the preregistered configuration")
    audit = config["training_only_audit"]
    subjects, actions = set(audit["subjects"]), set(audit["actions"])
    selected_records = [
        row for row in records
        if row.subject_id in subjects and row.action in actions and row.split == "train"
    ]
    expected = {(subject, action) for subject in subjects for action in actions}
    observed = {(row.subject_id, row.action) for row in selected_records}
    if observed != expected or any(row.locked_blind for row in selected_records):
        raise ValueError(f"Training-only audit selection mismatch: missing={sorted(expected-observed)}")
    if any(row.subject_id in {"Subject.3", "Subject.7", "Subject.9", "Subject.10"}
           for row in selected_records):
        raise ValueError("Validation or sealed subject entered the training-only scene audit")

    detector = SceneDetector(weights, config, device=device)
    root, poses = Path(dataset_root), Path(pose_dir)
    relationships = config["relationships"]
    frame_rows = []
    sequence_rows = []
    inference_ms = []
    for record in sorted(selected_records, key=lambda row: (row.subject_id, row.action)):
        image_dir = root / record.source_relpath
        images = _indexed_pngs(image_dir)
        if len(images) != record.frame_count:
            raise ValueError(f"Canonical frame count mismatch: {record.video_id}")
        target_boxes = _load_target_boxes(poses / f"{record.video_id}.npz", record)
        positions = _sample_positions(record.frame_count, int(audit["frames_per_sequence"]))
        object_frames = 0
        target_frames = 0
        context_frames = 0
        for position in positions:
            source_index, _prefix, image_path = images[position]
            detections, elapsed_ms = detector.detect(image_path)
            inference_ms.append(elapsed_ms)
            target = target_boxes[position]
            target_frames += target is not None
            object_frames += bool(detections)
            relations = []
            if target is not None:
                for detection in detections:
                    relations.append({
                        **detection,
                        **bbox_relationship(
                            target, detection["bbox_xyxy"], record.width, record.height,
                            near_edge_distance_ratio=relationships["near_edge_distance_ratio"],
                            overlap_iou_threshold=relationships["overlap_iou_threshold"],
                        ),
                    })
            context_frames += bool(relations)
            best = max(
                relations,
                key=lambda row: (row["iou"], row["proximity"], row["confidence"]),
                default=None,
            )
            frame_rows.append({
                "subject_id": record.subject_id, "video_id": record.video_id,
                "action": record.action, "split": record.split,
                "frame_ordinal": position, "source_index": source_index,
                "image_relpath": image_path.relative_to(root).as_posix(),
                "image_sha256": _sha256(image_path),
                "target_pose_available": target is not None,
                "detections": detections, "best_relationship": best,
                "context_status": (
                    "target_missing" if target is None else
                    "object_not_detected_or_absent" if not detections else
                    "relationship_available"
                ),
                "inference_ms": elapsed_ms,
            })
        persistence = object_frames / len(positions)
        sequence_rows.append({
            "subject_id": record.subject_id, "video_id": record.video_id,
            "action": record.action, "sampled_frames": len(positions),
            "target_pose_frames": target_frames, "object_detection_frames": object_frames,
            "relationship_frames": context_frames, "object_frame_fraction": persistence,
            "persistent_context": persistence >= audit["persistence_minimum_frame_fraction"],
        })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manual_count = int(audit["manual_frames_per_sequence"])
    manual_rows = []
    for sequence in sequence_rows:
        candidates = [row for row in frame_rows if row["video_id"] == sequence["video_id"]]
        indices = _sample_positions(len(candidates), manual_count)
        for index in indices:
            row = candidates[index]
            detections = row["detections"]
            relationship = row["best_relationship"]
            manual_rows.append({
                "subject_id": row["subject_id"], "video_id": row["video_id"],
                "frame_ordinal": row["frame_ordinal"], "source_index": row["source_index"],
                "image_relpath": row["image_relpath"], "image_sha256": row["image_sha256"],
                "predicted_classes": ";".join(sorted({d["class"] for d in detections})),
                "predicted_max_confidence": max(
                    (d["confidence"] for d in detections), default=0.0,
                ),
                "predicted_relation": relationship["relation"] if relationship else "",
                "predicted_iou": relationship["iou"] if relationship else "",
                "predicted_proximity": relationship["proximity"] if relationship else "",
                "manual_chair_present": "", "manual_couch_present": "",
                "manual_bed_present": "", "manual_relation": "",
                "relation_plausible": "", "occluded_or_ambiguous": "",
                "reviewer": "", "review_status": "pending", "notes": "",
            })
    template_path = output / "manual-audit.csv"
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS)
        writer.writeheader()
        writer.writerows(manual_rows)

    target_rate = sum(row["target_pose_available"] for row in frame_rows) / len(frame_rows)
    persistent_rate = sum(row["persistent_context"] for row in sequence_rows) / len(sequence_rows)
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "pending_manual_review",
        "scope": {
            "split": "train", "subjects": sorted(subjects), "actions": sorted(actions),
            "sequences": len(sequence_rows), "frames": len(frame_rows),
            "validation_subjects_accessed": [], "sealed_subjects_accessed": [],
            "raw_images_retained": False,
        },
        "provenance": {
            "config_sha256": _sha256(config_path), "weights_sha256": _sha256(weights),
            "dataset_license": config["dataset_license"],
            "detector_license": config["detector"]["software_license"],
        },
        "automatic_metrics": {
            "target_pose_availability": target_rate,
            "object_detection_frame_fraction": sum(bool(row["detections"]) for row in frame_rows) / len(frame_rows),
            "relationship_frame_fraction": sum(row["best_relationship"] is not None for row in frame_rows) / len(frame_rows),
            "sequences_with_persistent_context_fraction": persistent_rate,
            "mean_inference_ms": sum(inference_ms) / len(inference_ms),
            "total_inference_s": sum(inference_ms) / 1000,
        },
        "automatic_gate": {
            "target_pose_availability_pass": target_rate >= audit["minimum_target_pose_availability"],
            "persistence_pass": persistent_rate >= audit["minimum_sequences_with_persistent_context"],
            "manual_review_required": True,
            "passed": False,
        },
        "sequences": sequence_rows, "frames": frame_rows,
        "manual_audit": str(template_path),
    }
    _atomic_json(output / "automatic-audit.json", result)
    return result


def validate_manual_scene_audit(
    path: str | Path, config_path: str | Path, output: str | Path | None = None,
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    audit = config["training_only_audit"]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != set(MANUAL_FIELDS):
        raise ValueError("Invalid manual scene-audit schema")
    if any(row["review_status"] != "reviewed" for row in rows):
        raise ValueError("Manual scene audit is incomplete")
    truth_positive = predicted_positive = true_positive = 0
    plausible = reviewed_relations = ambiguous = 0
    for row in rows:
        if row["subject_id"] not in set(audit["subjects"]):
            raise ValueError("Manual audit contains a non-training subject")
        manual = {
            name for name, field in (
                ("chair", "manual_chair_present"), ("couch", "manual_couch_present"),
                ("bed", "manual_bed_present"),
            ) if row[field] == "1"
        }
        predicted = {value for value in row["predicted_classes"].split(";") if value}
        truth_positive += len(manual)
        predicted_positive += len(predicted)
        true_positive += len(manual & predicted)
        if row["predicted_relation"]:
            reviewed_relations += 1
            plausible += row["relation_plausible"] == "1"
        ambiguous += row["occluded_or_ambiguous"] == "1"
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / truth_positive if truth_positive else 0.0
    relation_plausibility = plausible / reviewed_relations if reviewed_relations else 0.0
    passed = (
        precision >= audit["minimum_manual_object_precision"] and
        recall >= audit["minimum_manual_object_recall"] and
        relation_plausibility >= audit["minimum_relation_plausibility"]
    )
    result = {
        "frames": len(rows), "object_true_positives": true_positive,
        "object_false_positives": predicted_positive - true_positive,
        "object_false_negatives": truth_positive - true_positive,
        "object_precision": precision, "object_recall": recall,
        "relation_plausibility": relation_plausibility,
        "ambiguous_frames": ambiguous, "passed": passed,
    }
    if output is not None:
        _atomic_json(Path(output), result)
    return result
