import csv
from pathlib import Path

import pytest

from racfall.context.scene import (
    MANUAL_FIELDS,
    bbox_relationship,
    pose_bbox,
    validate_manual_scene_audit,
)


def test_bbox_relationship_overlap_near_and_far():
    overlap = bbox_relationship([10, 10, 50, 90], [30, 60, 80, 100], 200, 100)
    near = bbox_relationship([10, 10, 40, 80], [45, 20, 80, 70], 200, 100)
    far = bbox_relationship([0, 0, 20, 20], [170, 70, 200, 100], 200, 100)
    assert overlap["relation"] == "overlap" and overlap["iou"] > 0
    assert near["relation"] == "near" and near["proximity"] > 0
    assert far["relation"] == "far" and far["proximity"] == 0


def test_pose_bbox_missing_is_explicit():
    np = pytest.importorskip("numpy")
    xy = np.zeros((17, 2), dtype=np.float32)
    valid = np.zeros(17, dtype=bool)
    assert pose_bbox(xy, valid, 100, 100) is None
    valid[:5] = True
    xy[:5] = [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]
    assert pose_bbox(xy, valid, 100, 100) is not None


def test_relationship_has_no_safe_activity_field():
    result = bbox_relationship([10, 10, 50, 90], [30, 60, 80, 100], 200, 100)
    assert "safe" not in result
    assert "activity" not in result


def test_manual_audit_rejects_incomplete_review(tmp_path: Path):
    row = {field: "" for field in MANUAL_FIELDS}
    row.update({"subject_id": "Subject.2", "review_status": "pending"})
    path = tmp_path / "audit.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS)
        writer.writeheader(); writer.writerow(row)
    with pytest.raises(ValueError, match="incomplete"):
        validate_manual_scene_audit(path, "configs/phase4b_scene_context.json")


def test_manual_audit_is_training_only_and_scores_classes(tmp_path: Path):
    rows = []
    for predicted, chair in (("chair", "1"), ("bed", "0")):
        row = {field: "" for field in MANUAL_FIELDS}
        row.update({
            "subject_id": "Subject.2", "predicted_classes": predicted,
            "predicted_relation": "near",
            "manual_chair_present": chair, "manual_couch_present": "0",
            "manual_bed_present": "0", "manual_relation": "near",
            "relation_plausible": "1", "occluded_or_ambiguous": "0",
            "review_status": "reviewed",
        })
        rows.append(row)
    path = tmp_path / "audit.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    result = validate_manual_scene_audit(path, "configs/phase4b_scene_context.json")
    assert result["object_precision"] == pytest.approx(0.5)
    assert result["object_recall"] == pytest.approx(1.0)
    assert not result["passed"]


def test_manual_audit_rejects_validation_subject(tmp_path: Path):
    row = {field: "" for field in MANUAL_FIELDS}
    row.update({
        "subject_id": "Subject.3", "review_status": "reviewed",
        "manual_chair_present": "0", "manual_couch_present": "0",
        "manual_bed_present": "0", "relation_plausible": "0",
        "occluded_or_ambiguous": "0",
    })
    path = tmp_path / "audit.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="non-training subject"):
        validate_manual_scene_audit(path, "configs/phase4b_scene_context.json")
