from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from racfall.data.adapters import (
    CAUCAFallCanonicalAdapter, URFDRGBAdapter, sha256_canonical_tree, sha256_tree,
)
from racfall.data.batch import preprocess_canonical_batch
from racfall.data.canonical import audit_canonical_sequence
from racfall.data.download import load_official_inventory, plan_download
from racfall.data.alignment import (
    AlignmentEvidence,
    FrameAnnotation,
    SCHEMA_VERSION as ALIGNMENT_SCHEMA_VERSION,
    audit_monotonic_source_skips,
    read_caucafall_classes,
    read_frame_annotations,
    resolve_index_alignment,
)
from racfall.data.manifest import ManifestRecord, read_manifest, validate_manifest, write_manifest
from racfall.data.pose import PoseExtractor
from racfall.data.pose_diagnostics import analyze_pose_recovery
from racfall.data.sequences import build_windows, normalize_pose, validate_window_leakage
from racfall.data.splits import (
    assign_frozen_2b4_splits, assign_frozen_2b5_trainval_splits,
    assign_subject_ids, assign_subject_splits,
)
from racfall.data.target import assess_pose_quality, select_causal_target
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
    assert assign_subject_ids(reversed([f"S{i:02d}" for i in range(1, 11)]), seed=7) == {
        key: assignment[key] for key in ("seed", "algorithm", "train", "val", "test")
    }
    assert assignment["distribution"]["train"]["records"] == 6


def test_development_exposed_subjects_cannot_remain_in_internal_test():
    with pytest.raises(ValueError, match="Development-exposed subjects"):
        assign_subject_ids([f"Subject.{i}" for i in range(1, 11)], seed=20261104)


def test_frozen_2b4_split_reserves_untouched_subjects_9_and_10():
    rows = [record(f"Subject.{i}", f"v{i}") for i in range(1, 11)]
    assigned, provenance = assign_frozen_2b4_splits(reversed(rows))
    by_subject = {row.subject_id: row.split for row in assigned}
    assert {subject for subject, split in by_subject.items() if split == "test"} == {
        "Subject.9", "Subject.10",
    }
    assert provenance["development_exposed_subjects"] == ["Subject.3", "Subject.5"]
    assert by_subject["Subject.3"] == "val"
    assert by_subject["Subject.5"] == "train"


def test_frozen_2b5_trainval_split_requires_sealed_subjects_absent():
    rows = [record(f"Subject.{i}", f"v{i}") for i in range(1, 9)]
    assigned, provenance = assign_frozen_2b5_trainval_splits(reversed(rows))
    by_subject = {row.subject_id: row.split for row in assigned}
    assert by_subject["Subject.3"] == "val"
    assert by_subject["Subject.5"] == "train"
    assert provenance["sealed_absent_subjects"] == ["Subject.9", "Subject.10"]
    with pytest.raises(ValueError, match="exactly downloaded train/validation subjects"):
        assign_frozen_2b5_trainval_splits(rows + [record("Subject.9", "v9")])


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


class FakeRecoveryModel(FakePoseModel):
    def predict(self, frame, **kwargs):
        return self.track(frame, **kwargs)


def _write_canonical_fixture(root: Path, *, missing_txt: frozenset[int] = frozenset()):
    cv2 = __import__("cv2")
    root.mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for index in range(1, 4):
        assert cv2.imwrite(str(root / f"seq{index:05d}.png"), image + index)
        if index not in missing_txt:
            (root / f"seq{index:05d}.txt").write_text(
                f"{int(index == 3)} 0.5 0.5 0.5 0.8\n", encoding="utf-8",
            )
    (root / "classes.txt").write_text("nofall\nfall\n", encoding="utf-8")


def test_canonical_png_txt_pairing_preserves_missing_label_mask(tmp_path):
    source = tmp_path / "Subject.1" / "Fall forward"
    _write_canonical_fixture(source, missing_txt=frozenset({2}))
    output = tmp_path / "audit.json"
    audit = audit_canonical_sequence(
        source, source, sequence_label="fall", video_id="canonical", subject_id="Subject.1",
        action="Fall forward", timing_fps=20.0,
        timing_provenance="secondary AVI FPS; no index equivalence", output=output,
    )
    assert audit["status"] == "verified_with_invalid_labels"
    assert audit["frame_labels"] == [0, -1, 1]
    assert audit["label_valid"] == [True, False, True]
    assert audit["event_labels"] == [-1, -1, -1]
    assert audit["missing_txt_indices"] == [2]
    assert [row["timestamp_s"] for row in audit["frames"]] == [0.0, 0.05, 0.1]

    (source / "seq00004.txt").write_text("1 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    quarantined = audit_canonical_sequence(
        source, source, sequence_label="fall", video_id="canonical", subject_id="Subject.1",
        action="Fall forward", timing_fps=20.0,
        timing_provenance="secondary AVI FPS; no index equivalence",
    )
    assert quarantined["sequence_quarantined"]
    assert quarantined["frame_labels"] == [-1, -1, -1]


def test_canonical_inventory_accepts_documented_official_copia_filename(tmp_path):
    source = tmp_path / "sequence"
    _write_canonical_fixture(source)
    (source / "seq00002.png").rename(source / "seq00002 - copia.png")
    audit = audit_canonical_sequence(
        source, source, sequence_label="fall", video_id="copy-name",
        subject_id="Subject.2", action="Fall backwards", timing_fps=20.0,
        timing_provenance="secondary AVI FPS; no index equivalence",
    )
    assert audit["status"] == "verified"
    assert audit["png_index_range"] == [1, 3]
    assert audit["frames"][1]["png_filename"] == "seq00002 - copia.png"
    assert audit["frames"][1]["txt_filename"] == "seq00002.txt"


def test_canonical_index_gap_quarantines_even_when_png_txt_pairs_match(tmp_path):
    source = tmp_path / "sequence"
    _write_canonical_fixture(source)
    (source / "seq00002.png").unlink()
    (source / "seq00002.txt").unlink()
    audit = audit_canonical_sequence(
        source, source, sequence_label="fall", video_id="gap",
        subject_id="Subject.1", action="Sit down", timing_fps=20.0,
        timing_provenance="frozen ordinal proxy",
    )
    assert audit["status"] == "quarantined"
    assert audit["missing_png_indices"] == [2]
    assert audit["frame_labels"] == [-1, -1]
    assert audit["valid_ground_state_frames"] == 0


def test_canonical_adapter_uses_png_count_and_avi_timing_only(tmp_path):
    cv2 = __import__("cv2")
    source = tmp_path / "Subject.1" / "Fall forward"
    _write_canonical_fixture(source)
    writer = cv2.VideoWriter(
        str(source / "secondary.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (16, 16),
    )
    assert writer.isOpened()
    try:
        # Deliberately differs from the three canonical PNGs.
        for _ in range(2):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()
    rows = CAUCAFallCanonicalAdapter(
        tmp_path, {"Fall forward": "fall"}, hash_sources=False,
    ).scan()
    assert len(rows) == 1
    assert rows[0].frame_count == 3
    assert rows[0].source_type == "image_sequence_dir"
    assert rows[0].fps == pytest.approx(20.0)
    assert "no AVI frame-index equivalence" in rows[0].timestamp_source


def test_canonical_adapter_supports_frozen_proxy_without_avi(tmp_path):
    source = tmp_path / "Subject.1" / "Fall forward"
    _write_canonical_fixture(source)
    rows = CAUCAFallCanonicalAdapter(
        tmp_path, {"Fall forward": "fall"}, timing_fps=20.0,
    ).scan()
    assert rows[0].fps == 20.0
    assert "frozen 20 FPS development proxy" in rows[0].timestamp_source
    first_hash = rows[0].source_sha256
    (source / "ignored.avi").write_bytes(b"not part of the canonical source")
    assert sha256_canonical_tree(source) == first_hash
    (source / "seq00001.txt").write_text("1 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    assert sha256_canonical_tree(source) != first_hash


def test_official_inventory_and_plan_require_exact_checksum(tmp_path):
    import hashlib
    import json

    content = b"0 0.5 0.5 0.5 0.8\n"
    digest = hashlib.sha256(content).hexdigest()
    folder_id = "action-folder"
    folders = [
        {"id": "subject-folder", "name": "Subject.1", "parent_id": "root"},
        {"id": folder_id, "name": "Walk", "parent_id": "subject-folder"},
    ]
    (tmp_path / "folders.json").write_text(json.dumps(folders), encoding="utf-8")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    file_id = "official-file"
    url = (
        "https://data.mendeley.com/public-files/datasets/7w7fccy7ky/files/"
        f"{file_id}/file_downloaded"
    )
    listing = [{
        "filename": "walk00001.txt", "id": file_id, "size": len(content),
        "content_details": {
            "size": len(content), "sha256_hash": digest, "download_url": url,
        },
    }]
    (metadata / f"Subject.1_{folder_id}.json").write_text(
        json.dumps(listing), encoding="utf-8",
    )
    inventory = load_official_inventory(
        tmp_path / "folders.json", metadata, subjects=("Subject.1",),
    )
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    (reuse / "copy.txt").write_bytes(content)
    destination = tmp_path / "destination"
    plan = plan_download(inventory, destination, reuse)
    assert plan["counts"] == {"verified": 0, "reusable": 1, "network": 0}
    target = destination / "Subject.1" / "Walk" / "walk00001.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="fails official size/SHA-256"):
        plan_download(inventory, destination, reuse)


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
    assert sample["track_coverage"].item() == pytest.approx(1.0)
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
        assert bounded["labels"].tolist() == [-1, -1]
        assert bounded["event_labels"].tolist() == [0, 1]


def test_resumable_batch_validates_checksum_and_seals_internal_test(tmp_path):
    source = tmp_path / "dataset" / "Subject.1" / "Fall forward"
    _write_canonical_fixture(source)
    row = record(
        subject="Subject.1", video="canonical", split="train", label="fall",
        source_relpath="Subject.1/Fall forward", source_type="image_sequence_dir",
        timestamp_source="canonical PNG ordinal / secondary AVI FPS; no index equivalence",
        end_timestamp_s=0.15, fps=20.0, width=16, height=16, frame_count=3,
        source_sha256=sha256_tree(source),
    )
    audits = tmp_path / "audits"
    audit_canonical_sequence(
        source, source, sequence_label="fall", video_id=row.video_id,
        subject_id=row.subject_id, action=row.action, timing_fps=row.fps,
        timing_provenance=row.timestamp_source, output=audits / "canonical.json",
    )
    output = tmp_path / "processed"
    first = preprocess_canonical_batch(
        [row], tmp_path / "dataset", audits, output, device="cpu",
        model_factory=lambda _: FakePoseModel(),
    )
    assert first["processed"] == 1 and first["verified_sequences"] == 1
    assert first["ground_state_class_counts"] == {"-1": 0, "0": 2, "1": 1}
    windows = tmp_path / "canonical-windows.npz"
    window_summary = build_windows(
        [row], output / "poses", windows, window_frames=2, stride_frames=1,
        alignment_dir=audits, quality_dir=output / "quality",
    )
    assert window_summary["valid_windows"] == 2
    with np.load(windows, allow_pickle=False) as data:
        assert data["ground_state_labels"].tolist() == [0, 1]
        assert data["event_labels"].tolist() == [-1, -1]
        assert data["quality_status"].tolist() == ["verified", "verified"]
    second = preprocess_canonical_batch(
        [row], tmp_path / "dataset", audits, output, device="cpu",
        model_factory=lambda _: FakePoseModel(),
    )
    assert second["resumed_verified"] == 1 and second["processed"] == 0
    assert second["results"][0]["status"] == "resumed_verified"
    audit_path = audits / "canonical.json"
    audit_path.write_text(audit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed_provenance = preprocess_canonical_batch(
        [row], tmp_path / "dataset", audits, output, device="cpu",
        model_factory=lambda _: FakePoseModel(),
    )
    assert changed_provenance["processed"] == 1
    assert changed_provenance["resumed_verified"] == 0

    sealed = preprocess_canonical_batch(
        [replace(row, split="test")], tmp_path / "dataset", audits, tmp_path / "sealed",
        device="cpu", model_factory=lambda _: FakePoseModel(),
    )
    assert sealed["sealed_internal_test"] == 1

    (source / "seq00001.txt").write_text("1 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    mismatch = preprocess_canonical_batch(
        [row], tmp_path / "dataset", audits, tmp_path / "mismatch", device="cpu",
        model_factory=lambda _: FakePoseModel(),
    )
    assert mismatch["failed"] == 1
    assert "checksum mismatch" in mismatch["results"][0]["error"]

    with pytest.raises(ValueError, match="internal blind test windows"):
        build_windows(
            [replace(row, split="test")], output / "poses", tmp_path / "test-windows.npz",
            window_frames=2, stride_frames=1,
        )


def test_pose_recovery_diagnostic_is_annotation_free(tmp_path):
    source = tmp_path / "dataset" / "frames"
    _write_canonical_fixture(source)
    row = record(
        video="diagnostic", split="development", source_relpath="frames",
        source_type="image_sequence_dir", timestamp_source="PNG ordinal / FPS",
        end_timestamp_s=0.15, fps=20.0, width=16, height=16, frame_count=3,
    )
    baseline = tmp_path / "baseline.npz"
    with baseline.open("wb") as handle:
        np.savez_compressed(
            handle, video_id=np.asarray("diagnostic"),
            frame_offsets=np.asarray([0, 0, 0, 0], dtype=np.int64),
        )
    report = analyze_pose_recovery(
        row, tmp_path / "dataset", baseline, tmp_path / "diagnostic.json", device="cpu",
        model_factory=lambda _: FakeRecoveryModel(),
    )
    assert report["annotation_boxes_used"] is False
    assert report["production_policy_changed"] is False
    assert report["baseline_missing_frames"] == 3
    assert set(report["methods_recovered_frames"].values()) == {3}


def annotation(index, class_id=0):
    return FrameAnnotation(index, class_id, 0.5, 0.5, 0.5, 0.8, f"seq-{index:05d}.txt")


def test_terminal_extra_requires_verified_unshifted_alignment():
    rows = [annotation(index, int(index >= 4)) for index in range(1, 7)]
    evidence = AlignmentEvidence(
        representative_best_matches={3: 3, 4: 4, 5: 5, 6: 5},
        required_exact_indices=(3, 4, 5), terminal_pair_mae=0.01,
        bbox_inside_fraction={3: 1.0, 4: 0.9, 5: 1.0},
    )
    result = resolve_index_alignment(5, rows, evidence)
    assert result["status"] == "verified_terminal_extra"
    assert result["frame_labels"] == [0, 0, 0, 1, 1]
    assert result["quarantined_annotations"] == [6]

    shifted = replace(evidence, representative_best_matches={3: 2, 4: 3, 5: 4, 6: 5})
    rejected = resolve_index_alignment(5, rows, shifted)
    assert rejected["sequence_quarantined"]
    assert rejected["frame_labels"] == [-1] * 5
    assert any("does not best-match" in reason for reason in rejected["reasons"])

    exact_without_evidence = resolve_index_alignment(5, rows[:5], None)
    assert exact_without_evidence["sequence_quarantined"]
    assert exact_without_evidence["frame_labels"] == [-1] * 5

    two_extras = rows + [annotation(7, 1)]
    two_extra_evidence = replace(
        evidence, representative_best_matches={3: 3, 4: 4, 5: 5, 6: 5, 7: 5},
    )
    verified_two = resolve_index_alignment(5, two_extras, two_extra_evidence)
    assert verified_two["status"] == "verified_terminal_extras"
    assert verified_two["quarantined_annotations"] == [6, 7]

    moving_terminal = replace(two_extra_evidence, terminal_pair_mae=1.01)
    assert resolve_index_alignment(5, two_extras, moving_terminal)["sequence_quarantined"]


def test_monotonic_alignment_reports_duplicate_frame_ambiguity_without_shifting():
    # Two identical source rows can be skipped at equal cost; neither is silently chosen.
    costs = np.asarray([
        [0.0, 9.0, 9.0],
        [9.0, 0.0, 9.0],
        [9.0, 0.0, 9.0],
        [9.0, 9.0, 0.0],
    ])
    audit = audit_monotonic_source_skips(costs)
    assert audit["ambiguous"]
    assert audit["exact_best_candidate_count"] == 2
    assert [row["skipped_source_indices"] for row in audit["candidates"][:2]] == [[2], [3]]


def test_alignment_quarantines_missing_or_ambiguous_labels(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "seq-00001.txt").write_text("0 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    (annotation_dir / "seq-00003.txt").write_text("1 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not contiguous"):
        read_frame_annotations(annotation_dir)

    unsupported = resolve_index_alignment(5, [annotation(i) for i in range(1, 8)], None)
    assert unsupported["sequence_quarantined"]
    assert unsupported["quarantined_annotations"] == list(range(1, 8))


def test_caucafall_annotation_names_allow_official_dashless_variant(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "cas200001.txt").write_text("0 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    (annotation_dir / "cas200002.txt").write_text("1 0.5 0.5 0.5 0.8\n", encoding="utf-8")
    rows = read_frame_annotations(annotation_dir)
    assert [row.index for row in rows] == [1, 2]
    assert [row.filename for row in rows] == ["cas200001.txt", "cas200002.txt"]
    classes = annotation_dir / "classes.txt"
    classes.write_text("nofall\n", encoding="utf-8")
    assert read_caucafall_classes(classes, "non_fall") == ["nofall"]
    with pytest.raises(ValueError, match="Unexpected"):
        read_caucafall_classes(classes, "fall")


def test_verified_ground_state_labels_are_causal_and_keep_unknown(tmp_path):
    cv2 = __import__("cv2")
    source = tmp_path / "frames"
    source.mkdir()
    for index in range(5):
        assert cv2.imwrite(str(source / f"frame-{index:03d}.png"), np.zeros((16, 16, 3), dtype=np.uint8))
    (tmp_path / "sync.csv").write_text(
        "1,0\n2,40\n3,80\n4,120\n5,160\n", encoding="utf-8",
    )
    row = record(
        video="causal", split="train", label="fall", source_relpath="frames",
        source_type="image_sequence_dir", timestamp_source="sync.csv milliseconds",
        end_timestamp_s=0.2, fps=25.0, frame_count=5,
    )
    pose_dir = tmp_path / "poses"
    PoseExtractor(device="cpu", imgsz=416, model_factory=lambda _: FakePoseModel()).extract(
        tmp_path, row, pose_dir / "causal.npz",
    )
    alignment_dir = tmp_path / "alignment"
    alignment_dir.mkdir()
    (alignment_dir / "causal.json").write_text(__import__("json").dumps({
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "video_id": "causal",
        "status": "verified_exact_count",
        "sequence_quarantined": False,
        # Frame 4 is deliberately unknown; frame 5 must not back-fill it.
        "frame_labels": [0, 0, 1, -1, 1],
    }), encoding="utf-8")
    output = tmp_path / "causal-windows.npz"
    summary = build_windows(
        [row], pose_dir, output, window_frames=2, stride_frames=1,
        alignment_dir=alignment_dir,
    )
    assert summary["valid_windows"] == 3
    assert summary["invalid_windows"] == 1
    with np.load(output, allow_pickle=False) as data:
        # Endpoint frames are 2, 3, 4, 5: no future frame is consulted.
        assert data["ground_state_labels"].tolist() == [0, 1, -1, 1]
        assert data["labels"].tolist() == [0, 1, -1, 1]
        assert data["event_labels"].tolist() == [-1, -1, -1, -1]
        assert str(data["label_semantics"]) == "ground_state_at_causal_window_endpoint"


def test_sparse_tracks_do_not_create_mostly_empty_training_windows(tmp_path):
    row = record(video="sparse", split="train", label="non_fall", frame_count=4, end_timestamp_s=2.0)
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    xy = np.ones((1, 17, 2), dtype=np.float32)
    with (pose_dir / "sparse.npz").open("wb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray("racfall.pose.v1"), video_id=np.asarray("sparse"),
            frame_index=np.arange(4, dtype=np.int32), timestamp_s=np.arange(4) / 2.0,
            frame_offsets=np.asarray([0, 1, 1, 1, 1]), track_id=np.asarray([9]),
            xy=xy, confidence=np.ones((1, 17), dtype=np.float32),
            valid=np.ones((1, 17), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="No complete tracked pose windows"):
        build_windows([row], pose_dir, tmp_path / "none.npz", window_frames=4, stride_frames=1)
    summary = build_windows(
        [row], pose_dir, tmp_path / "qa.npz", window_frames=4, stride_frames=1,
        min_track_coverage=0.0,
    )
    assert summary["windows"] == 1
    assert summary["skipped_low_coverage_windows"] == 0


def test_causal_target_selection_stitches_nearby_ids_and_rejects_far_person():
    def pose_at(center_x, center_y, scale):
        offsets = np.linspace(-scale / 2, scale / 2, 17, dtype=np.float32)
        return np.column_stack((center_x + offsets, center_y + offsets))

    xy = np.asarray([
        pose_at(100, 200, 40),
        pose_at(105, 202, 42),
        pose_at(300, 100, 30),
        pose_at(302, 102, 30),
    ], dtype=np.float32)
    selected = select_causal_target(
        np.asarray([0, 1, 3, 4]), np.asarray([1, 2, 9, 9]), xy,
        np.full((4, 17), 0.9, dtype=np.float32), np.ones((4, 17), dtype=np.bool_),
        720, 480,
    )
    assert selected.tolist() == [0, 1, -1]
    repeated = select_causal_target(
        np.asarray([0, 1, 3, 4]), np.asarray([1, 2, 9, 9]), xy,
        np.full((4, 17), 0.9, dtype=np.float32), np.ones((4, 17), dtype=np.bool_),
        720, 480,
    )
    assert repeated.tolist() == selected.tolist()


def test_target_selection_rejects_anonymous_takeover_and_long_gap():
    xy = np.asarray([
        np.column_stack((np.arange(17) + 100, np.arange(17) + 100)),
        np.column_stack((np.arange(17) + 102, np.arange(17) + 102)),
        np.column_stack((np.arange(17) + 300, np.arange(17) + 300)),
    ], dtype=np.float32)
    confidence = np.full((3, 17), 0.9, dtype=np.float32)
    valid = np.ones((3, 17), dtype=np.bool_)
    # Detection 1 is anonymous after a tracked identity. Detection 2 arrives
    # beyond the fail-closed 20-frame reacquisition interval.
    offsets = np.zeros(23, dtype=np.int64)
    offsets[1:22] = 1
    offsets[22] = 3
    selected = select_causal_target(
        offsets, np.asarray([1, -1, 2]), xy, confidence, valid, 720, 480,
    )
    assert selected[0] == 0
    assert selected[-1] == -1
    assert np.count_nonzero(selected >= 0) == 1

    low = assess_pose_quality(xy[0], np.zeros(17, dtype=np.float32), valid[0])
    assert low == {
        "valid_keypoints": 0, "missing_keypoints": 17,
        "mean_confidence": 0.0, "spatial_scale_px": 0.0,
    }


def test_normalization_and_window_leakage_guards():
    xy = np.full((17, 2), np.nan, dtype=np.float32)
    normalized, confidence, valid = normalize_pose(xy, np.zeros(17), np.zeros(17, dtype=bool))
    assert np.all(normalized == 0) and not valid.any() and np.all(confidence == 0)
    collapsed = np.asarray([[0.0, float(i)] for i in range(17)], dtype=np.float32)
    collapsed[5:7] = [[0.0, 4.0], [0.0, 5.0]]
    collapsed[11:13] = [[0.0, 4.1], [0.0, 5.1]]
    stable, _, stable_valid = normalize_pose(collapsed, np.ones(17), np.ones(17, dtype=bool))
    assert stable_valid.all()
    assert np.abs(stable).max() <= 4.0
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
