from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from racfall.data.adapters import URFDRGBAdapter
from racfall.data.manifest import ManifestRecord, read_manifest, validate_manifest, write_manifest
from racfall.data.pose import PoseExtractor
from racfall.data.sequences import build_windows, normalize_pose, validate_window_leakage
from racfall.data.splits import assign_subject_splits
from racfall.data.torch_data import PoseWindowDataset, make_dataloader


def record(subject="S01", video="v1", split="", label="fall", **changes):
    base = ManifestRecord(
        dataset="CAUCAFall", dataset_version="v5", subject_id=subject,
        subject_provenance="source directory", video_id=video, sequence_id=video,
        source_relpath=f"{subject}/fall/{video}.avi", source_type="video", action="fall",
        label=label, label_start_s="", label_end_s="",
        annotation_provenance="reviewed activity map; no event boundary",
        timestamp_source="container FPS", start_timestamp_s=0.0, end_timestamp_s=1.0,
        fps=2.0, width=16, height=16, frame_count=2, split=split,
        source_sha256="", license="CC BY 4.0", locked_blind=False,
    )
    return replace(base, **changes)


def test_manifest_roundtrip_and_leakage(tmp_path):
    path = tmp_path / "manifest.csv"
    rows = [record("S01", "a", split="train"), record("S02", "b", split="val")]
    write_manifest(rows, path)
    loaded = read_manifest(path)
    assert loaded == rows
    assert validate_manifest(loaded)["records"] == 2
    with pytest.raises(ValueError, match="Subject leakage"):
        validate_manifest([record("S01", "a", split="train"), record("S01", "b", split="test")])
    with pytest.raises(ValueError, match="locked blind"):
        validate_manifest([record(split="test", locked_blind=True)])


def test_reproducible_caucafall_622_split():
    rows = [record(f"S{i:02d}", f"v{i}") for i in range(1, 11)]
    first, assignment = assign_subject_splits(rows, seed=7)
    second, second_assignment = assign_subject_splits(rows, seed=7)
    assert assignment == second_assignment
    assert [row.split for row in first] == [row.split for row in second]
    assert {key: len(assignment[key]) for key in ("train", "val", "test")} == {
        "train": 6, "val": 2, "test": 2,
    }


def test_urfd_zip_adapter_uses_sync_timestamps_and_unknown_subject(tmp_path):
    cv2 = __import__("cv2")
    archive_path = tmp_path / "fall-01-cam0-rgb.zip"
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("fall-01-cam0-rgb-001.png", encoded.tobytes())
        archive.writestr("fall-01-cam0-rgb-002.png", encoded.tobytes())
    (tmp_path / "fall-01-data.csv").write_text("1,0\n2,40\n", encoding="utf-8")

    rows = URFDRGBAdapter(tmp_path, hash_sources=False).scan()

    assert len(rows) == 1
    assert rows[0].subject_id == "unknown" and rows[0].label == "fall"
    assert rows[0].fps == pytest.approx(25.0)
    assert (rows[0].width, rows[0].height, rows[0].frame_count) == (12, 8, 2)
    with pytest.raises(ValueError, match="cannot split"):
        validate_manifest([replace(rows[0], split="development")])


class FakePoseModel:
    def track(self, _frame, **_kwargs):
        xy = [[[float(i + 1), float(i + 2)] for i in range(17)]]
        return [SimpleNamespace(
            keypoints=SimpleNamespace(xy=xy, conf=[[0.9] * 16 + [0.1]]),
            boxes=SimpleNamespace(id=[3]), speed={"inference": 5.0,
            },
        )]


def test_pose_extract_normalize_windows_and_torch_loader(tmp_path):
    cv2 = __import__("cv2")
    source = tmp_path / "frames"
    source.mkdir()
    for index in range(3):
        assert cv2.imwrite(str(source / f"frame-{index:03d}.png"), np.zeros((16, 16, 3), dtype=np.uint8))
    (tmp_path / "sync.csv").write_text("1,0\n2,40\n3,80\n", encoding="utf-8")
    row = record(
        video="pilot", split="train", label="non_fall", source_relpath="frames",
        source_type="image_sequence_dir", timestamp_source="sync.csv milliseconds",
        end_timestamp_s=0.12, fps=25.0, frame_count=3,
    )
    pose_path = tmp_path / "poses" / "pilot.npz"
    extractor = PoseExtractor(device="cpu", imgsz=416, model_factory=lambda _: FakePoseModel())
    result = extractor.extract(tmp_path, row, pose_path)
    assert result["frames"] == 3 and result["pose_detections"] == 3
    with np.load(pose_path, allow_pickle=False) as pose:
        assert pose["xy"].shape == (3, 17, 2)
        assert not pose["valid"][:, 16].any()
        assert np.isnan(pose["xy"][:, 16]).all()
        assert pose["frame_offsets"].tolist() == [0, 1, 2, 3]

    windows_path = tmp_path / "windows.npz"
    summary = build_windows([row], pose_path.parent, windows_path, window_frames=2, stride_frames=1)
    assert summary["windows"] == 2
    dataset = PoseWindowDataset(windows_path, split="train")
    assert len(dataset) == 2
    sample = dataset[0]
    assert tuple(sample["features"].shape) == (2, 51)
    assert tuple(sample["valid"].shape) == (2, 17)
    assert sample["label"].item() == 0
    loader = make_dataloader(windows_path, split="train", batch_size=2, shuffle=False)
    assert next(iter(loader))["features"].shape[0] == 2
    dataset.close()

    unknown_row = replace(row, video_id="pilot-fall", sequence_id="pilot-fall", label="fall")
    unknown_pose = pose_path.parent / "pilot-fall.npz"
    with np.load(pose_path, allow_pickle=False) as source_pose, unknown_pose.open("wb") as handle:
        values = {name: source_pose[name] for name in source_pose.files}
        values["video_id"] = np.asarray("pilot-fall")
        values["label"] = np.asarray("fall")
        np.savez_compressed(handle, **values)
    unknown_windows = tmp_path / "unknown-windows.npz"
    build_windows([unknown_row], pose_path.parent, unknown_windows, window_frames=2, stride_frames=1)
    assert len(PoseWindowDataset(unknown_windows, split="train")) == 0
    unknown_dataset = PoseWindowDataset(unknown_windows, split="train", include_unknown=True)
    assert len(unknown_dataset) == 2
    unknown_dataset.close()
    with pytest.raises(ValueError, match="No eligible windows"):
        make_dataloader(unknown_windows, split="train")

    bounded_row = replace(unknown_row, label_start_s="0.05", label_end_s="0.10")
    bounded_windows = tmp_path / "bounded-windows.npz"
    build_windows([bounded_row], pose_path.parent, bounded_windows, window_frames=2, stride_frames=1)
    with np.load(bounded_windows, allow_pickle=False) as bounded:
        assert bounded["labels"].tolist() == [0, 1]


def test_normalization_and_window_leakage_guards():
    xy = np.full((17, 2), np.nan, dtype=np.float32)
    normalized, confidence, valid = normalize_pose(xy, np.zeros(17), np.zeros(17, dtype=bool))
    assert np.all(normalized == 0) and not valid.any() and np.all(confidence == 0)
    with pytest.raises(ValueError, match="Subject window leakage"):
        validate_window_leakage([
            {"dataset": "D", "subject_id": "S", "video_id": "a", "split": "train",
             "start_frame": 0, "end_frame": 10},
            {"dataset": "D", "subject_id": "S", "video_id": "b", "split": "test",
             "start_frame": 0, "end_frame": 10},
        ])


def test_locked_blind_pose_extraction_is_refused(tmp_path):
    row = record(dataset="GMDCSA-24", split="blind_test", locked_blind=True)
    extractor = PoseExtractor(device="cpu", model_factory=lambda _: FakePoseModel())
    with pytest.raises(ValueError, match="forbidden"):
        extractor.extract(tmp_path, row, tmp_path / "never.npz")
