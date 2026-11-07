import csv
from pathlib import Path

import pytest

from racfall.annotation.events import FIELDS, LANDMARKS, compare_reviews, validate_annotations
from racfall.evaluation.events import Video, read_reviewed_events


CONFIG = "configs/phase4c_event_annotation.json"


def annotation_row(*, reviewer="r1", subject="Subject.2", valid="1", status="reviewed"):
    row = {field: "" for field in FIELDS}
    row.update({
        "dataset": "CAUCAFall", "video_id": "training-fall", "subject_id": subject,
        "action": "Fall forward", "split": "train", "event_id": "event-1",
        "event_valid": valid, "contact_region": "hip", "unknown_reason": "",
        "reviewer_id": reviewer, "review_round": "independent-1",
        "annotation_confidence": "high", "annotation_tool_version": "phase4c-v1",
        "created_at_utc": "2026-11-08T00:00:00+00:00", "source_sha256": "a" * 64,
        "canonical_audit_sha256": "b" * 64, "frame_count": "100",
        "fps_proxy": "20", "fps_provenance": "PNG ordinal / proxy",
        "fps_uncertainty_s": "0.025", "provenance_note": "systematic offset unknown",
        "review_status": status,
    })
    for frame, landmark in zip((10, 20, 30, 60, 80), LANDMARKS, strict=True):
        row[f"{landmark}_frame"] = str(frame)
        row[f"{landmark}_status"] = "exact"
    return row


def write_rows(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_event_annotation_validation_derives_proxy_timestamps(tmp_path):
    path = tmp_path / "review.csv"
    write_rows(path, [annotation_row()])
    result = validate_annotations(path, CONFIG)
    timing = result["derived_timing"][0]
    assert result["exact_landmarks"] == 5
    assert timing["timestamps_s"]["fall_initiation_s"] == pytest.approx(0.5)
    assert timing["timestamp_quantization_uncertainty_s"] == pytest.approx(0.025)


def test_pending_ambiguous_and_temporal_order_are_fail_closed(tmp_path):
    path = tmp_path / "review.csv"
    pending = annotation_row(status="pending")
    write_rows(path, [pending])
    with pytest.raises(ValueError, match="incomplete"):
        validate_annotations(path, CONFIG)
    assert validate_annotations(path, CONFIG, require_complete=False)["pending_rows"] == 1

    reversed_row = annotation_row()
    reversed_row["ground_contact_frame"] = "5"
    write_rows(path, [reversed_row])
    with pytest.raises(ValueError, match="temporally ordered"):
        validate_annotations(path, CONFIG)


def test_nonfall_and_forbidden_subject_rules(tmp_path):
    path = tmp_path / "review.csv"
    nonfall = annotation_row(valid="0")
    write_rows(path, [nonfall])
    with pytest.raises(ValueError, match="not_applicable"):
        validate_annotations(path, CONFIG)
    forbidden = annotation_row(subject="Subject.3")
    write_rows(path, [forbidden])
    with pytest.raises(ValueError, match="Non-training"):
        validate_annotations(path, CONFIG)


def test_review_comparison_flags_known_unknown_disagreement(tmp_path):
    first, second = tmp_path / "r1.csv", tmp_path / "r2.csv"
    a, b = annotation_row(reviewer="r1"), annotation_row(reviewer="r2")
    b["ground_contact_frame"], b["ground_contact_status"] = "-1", "ambiguous"
    b["unknown_reason"] = "contact occluded"
    write_rows(first, [a]); write_rows(second, [b])
    result = compare_reviews(first, second, CONFIG)
    assert result["adjudication_required"] == 1
    assert result["details"][0]["disagreements"] == ["ground_contact"]
    assert result["event_valid_cohen_kappa"] == pytest.approx(1.0)


def test_reviewed_event_reader_excludes_ambiguous_onset(tmp_path):
    path = tmp_path / "review.csv"
    row = annotation_row(status="adjudicated")
    row["fall_initiation_frame"], row["fall_initiation_status"] = "-1", "ambiguous"
    row["unknown_reason"] = "event begins behind occlusion"
    write_rows(path, [row])
    videos = {"training-fall": Video(
        "training-fall", "CAUCAFall", "Subject.2", "train", "Fall forward", 5.0, True,
    )}
    events, audit = read_reviewed_events(path, videos)
    assert events == []
    assert audit["excluded_ambiguous_events"] == 1


def test_reviewed_event_reader_requires_adjudication_and_preserves_contact(tmp_path):
    path = tmp_path / "review.csv"
    row = annotation_row(status="reviewed")
    write_rows(path, [row])
    videos = {"training-fall": Video(
        "training-fall", "CAUCAFall", "Subject.2", "train", "Fall forward", 5.0, True,
    )}
    events, audit = read_reviewed_events(path, videos)
    assert events == [] and audit["excluded_unreviewed_rows"] == 1
    row["review_status"] = "adjudicated"
    write_rows(path, [row])
    events, audit = read_reviewed_events(path, videos)
    assert events[0].onset_s == pytest.approx(0.5)
    assert events[0].ground_contact_s == pytest.approx(1.0)
    assert audit["maximum_timestamp_quantization_uncertainty_s"] == pytest.approx(0.025)


def test_annotation_contract_does_not_contain_ground_state_labels():
    assert "ground_state_label" not in FIELDS
    assert "source_ground_state_transition" not in FIELDS
