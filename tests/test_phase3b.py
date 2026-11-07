import json
from pathlib import Path

import numpy as np
import pytest

from racfall.training.phase3b import (
    activity_from_video_id,
    aggregate_diagnostics,
    error_clusters,
    sha256_file,
    validate_validation_scope,
    window_diagnostics,
)


def test_activity_from_canonical_video_id():
    assert activity_from_video_id(
        "caucafall-Subject.3-Sit-down-png", "Subject.3",
    ) == "Sit down"


def test_window_diagnostics_respects_validity_and_tracks():
    features = np.zeros((3, 51), dtype=np.float32)
    pose = features.reshape(3, 17, 3)
    pose[:, :, 2] = 0.8
    pose[1, 0, 0] = 1.0
    pose[2, 0, 0] = 2.0
    valid = np.zeros((3, 17), dtype=bool)
    valid[:, 0] = True
    row = window_diagnostics(features, valid, np.asarray([4, 4, 7]))
    assert row["mean_velocity_normalized_per_frame"] == pytest.approx(1.0)
    assert row["endpoint_velocity_normalized_per_frame"] == pytest.approx(1.0)
    assert row["mean_visible_confidence"] == pytest.approx(0.8)
    assert row["missing_keypoint_rate"] == pytest.approx(16 / 17)
    assert row["fully_missing_frames"] == 0
    assert row["track_switches"] == 1


def test_aggregate_diagnostics_empty_is_explicit():
    result = aggregate_diagnostics([])
    assert result["windows"] == 0
    assert result["missing_keypoint_rate"] == {"mean": None, "median": None}


def test_error_clusters_merge_overlap_but_not_touching_windows():
    base = {
        "video_id": "v", "subject_id": "Subject.3", "activity": "Sit down",
        "correct": False, "error_type": "false_positive",
    }
    rows = [
        {**base, "start_frame": 0, "end_frame": 30},
        {**base, "start_frame": 15, "end_frame": 45},
        {**base, "start_frame": 45, "end_frame": 75},
    ]
    clusters = error_clusters(rows)
    assert [(row["start_frame"], row["end_frame"], row["window_count"]) for row in clusters] == [
        (0, 45, 2), (45, 75, 1),
    ]


def test_sha256_file_and_registry_json_are_stable(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"racfall")
    assert sha256_file(artifact) == "5e98741159e6db2fe0874084ba4f18ce52aa82b47bc6efbdac695be2db360d4b"
    registry = json.loads(Path("configs/phase3b_frozen_baseline.json").read_text())
    assert registry["status"] == "frozen"
    assert registry["selection"] == {
        "seed": 29,
        "epoch": 6,
        "criterion": "minimum Phase 3A validation BCE",
        "validation_loss": 0.1356429398059845,
    }
    assert registry["development_scope"]["sealed_subjects"] == ["Subject.9", "Subject.10"]


def test_validation_scope_is_fail_closed_for_sealed_or_partial_subjects():
    validate_validation_scope(["Subject.3", "Subject.7"])
    with pytest.raises(ValueError, match="restricted"):
        validate_validation_scope(["Subject.3"])
    with pytest.raises(ValueError, match="restricted"):
        validate_validation_scope(["Subject.3", "Subject.7", "Subject.9"])
