"""Resumable, checksum-validated canonical pose preprocessing batches."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterable

from .adapters import sha256_file, sha256_tree
from .canonical import assess_sequence_quality
from .manifest import ManifestRecord, validate_manifest
from .pose import PoseExtractor


LEDGER_SCHEMA_VERSION = "racfall.preprocess-ledger.v1"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _signature(record: ManifestRecord, config: dict) -> str:
    encoded = json.dumps(
        {"record": asdict(record), "config": config}, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_checksum(root: Path, record: ManifestRecord) -> str:
    source = (root / record.source_relpath).resolve()
    if source != root and root not in source.parents:
        raise ValueError(f"Source path escapes dataset root: {record.source_relpath}")
    if source.is_file():
        return sha256_file(source)
    if source.is_dir():
        return sha256_tree(source)
    raise FileNotFoundError(f"Missing source: {source}")


def preprocess_canonical_batch(
    records: Iterable[ManifestRecord],
    dataset_root: str | Path,
    canonical_audit_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "mps",
    imgsz: int = 416,
    detector_confidence: float = 0.25,
    keypoint_confidence: float = 0.25,
    min_pose_coverage: float = 0.5,
    allow_internal_test: bool = False,
    limit: int | None = None,
    model_factory=None,
) -> dict:
    """Run or resume a batch; completed artifacts are checksum-verified before reuse."""
    rows = list(records)
    validate_manifest(rows)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if any(row.dataset != "CAUCAFall" or row.source_type != "image_sequence_dir" for row in rows):
        raise ValueError("Canonical CAUCAFall image-sequence records are required")
    root = Path(dataset_root).resolve()
    audit_root = Path(canonical_audit_dir)
    destination = Path(output_dir)
    pose_dir = destination / "poses"
    quality_dir = destination / "quality"
    ledger_path = destination / "preprocess-ledger.json"
    destination.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ValueError("Unsupported preprocessing ledger schema")
    else:
        ledger = {"schema_version": LEDGER_SCHEMA_VERSION, "entries": {}}

    config = {
        "model": "yolo26n-pose.pt", "device": device, "imgsz": imgsz,
        "detector_confidence": detector_confidence,
        "keypoint_confidence": keypoint_confidence,
        "min_pose_coverage": min_pose_coverage,
        "canonical_schema": "racfall.caucafall-canonical.v1",
        "pose_schema": "racfall.pose.v1",
        "quality_schema": "racfall.sequence-quality.v1",
    }
    extractor = None
    started = time.perf_counter()
    disk_before = shutil.disk_usage(destination)
    results = []
    processed = skipped = failed = sealed = 0
    for record in rows:
        if record.split == "test" and not allow_internal_test:
            sealed += 1
            results.append({
                "video_id": record.video_id, "status": "sealed_internal_test",
                "reason": "internal blind test preprocessing requires explicit allow_internal_test",
            })
            continue
        signature = _signature(record, config)
        pose_path = pose_dir / f"{record.video_id}.npz"
        quality_path = quality_dir / f"{record.video_id}.json"
        audit_path = audit_root / f"{record.video_id}.json"
        prior = ledger["entries"].get(record.video_id)
        if (prior and prior.get("status") == "complete" and
                prior.get("signature") == signature and pose_path.is_file() and
                quality_path.is_file() and audit_path.is_file() and
                sha256_file(audit_path) == prior.get("canonical_audit_sha256") and
                sha256_file(pose_path) == prior.get("pose_sha256") and
                sha256_file(quality_path) == prior.get("quality_sha256")):
            skipped += 1
            results.append({**prior, "status": "resumed_verified"})
            continue
        item_started = time.perf_counter()
        try:
            if not record.source_sha256:
                raise ValueError("Manifest source_sha256 is required for controlled preprocessing")
            actual_source_sha256 = _source_checksum(root, record)
            if actual_source_sha256 != record.source_sha256:
                raise ValueError(
                    f"Source checksum mismatch: manifest={record.source_sha256}, "
                    f"actual={actual_source_sha256}"
                )
            if not audit_path.is_file():
                raise FileNotFoundError(f"Missing canonical audit: {audit_path}")
            if extractor is None:
                extractor = PoseExtractor(
                    device=device, imgsz=imgsz, confidence_threshold=keypoint_confidence,
                    detector_confidence=detector_confidence, model_factory=model_factory,
                )
            temporary_pose = pose_path.with_name(pose_path.name + ".partial")
            extraction = extractor.extract(root, record, temporary_pose)
            os.replace(temporary_pose, pose_path)
            temporary_quality = quality_path.with_name(quality_path.name + ".partial")
            quality = assess_sequence_quality(
                audit_path, pose_path, record, min_pose_coverage=min_pose_coverage,
                output=temporary_quality,
            )
            os.replace(temporary_quality, quality_path)
            entry = {
                "video_id": record.video_id,
                "status": "complete",
                "signature": signature,
                "source_sha256": actual_source_sha256,
                "canonical_audit_sha256": sha256_file(audit_path),
                "pose_sha256": sha256_file(pose_path),
                "quality_sha256": sha256_file(quality_path),
                "pose_bytes": pose_path.stat().st_size,
                "quality_bytes": quality_path.stat().st_size,
                "frames": extraction["frames"],
                "pose_detections": extraction["pose_detections"],
                "extraction_elapsed_s": extraction["elapsed_s"],
                "extraction_throughput_fps": extraction["throughput_fps"],
                "quality_status": quality["status"],
                "training_eligible": quality["training_eligible"],
                "ground_state_class_counts": quality["ground_state_class_counts"],
                "valid_ground_state_frames": quality["valid_ground_state_frames"],
                "invalid_ground_state_frames": quality["invalid_ground_state_frames"],
                "missing_target_frames": quality["missing_target_frames"],
                "missing_target_rate": quality["missing_target_rate"],
                "elapsed_s": round(time.perf_counter() - item_started, 6),
            }
            ledger["entries"][record.video_id] = entry
            _atomic_json(ledger_path, ledger)
            results.append(entry)
            processed += 1
        except Exception as exc:
            failed += 1
            entry = {
                "video_id": record.video_id, "status": "failed",
                "signature": signature, "error": f"{type(exc).__name__}: {exc}",
            }
            ledger["entries"][record.video_id] = entry
            _atomic_json(ledger_path, ledger)
            results.append(entry)

    elapsed = time.perf_counter() - started
    disk_after = shutil.disk_usage(destination)
    completed = [row for row in results if row.get("status") in {"complete", "resumed_verified"}]
    processed_rows = [row for row in results if row.get("status") == "complete"]
    frames = sum(int(row.get("frames", 0)) for row in completed)
    processed_frames = sum(int(row.get("frames", 0)) for row in processed_rows)
    class_counts = {key: 0 for key in ("-1", "0", "1")}
    for row in completed:
        for key, value in row.get("ground_state_class_counts", {}).items():
            class_counts[str(key)] += int(value)
    missing_target_frames = sum(int(row.get("missing_target_frames", 0)) for row in completed)
    summary = {
        "schema_version": "racfall.preprocess-batch-summary.v1",
        "config": config,
        "records_requested": len(rows),
        "processed": processed, "resumed_verified": skipped,
        "failed": failed, "sealed_internal_test": sealed,
        "verified_sequences": sum(bool(row.get("training_eligible")) for row in completed),
        "quarantined_sequences": sum(
            row.get("quality_status") == "quarantined" for row in completed
        ),
        "frames": frames,
        "frames_processed_this_run": processed_frames,
        "ground_state_class_counts": class_counts,
        "valid_ground_state_frames": sum(
            int(row.get("valid_ground_state_frames", 0)) for row in completed
        ),
        "invalid_ground_state_frames": sum(
            int(row.get("invalid_ground_state_frames", 0)) for row in completed
        ),
        "missing_target_frames": missing_target_frames,
        "missing_target_rate": missing_target_frames / frames if frames else None,
        "wall_elapsed_s": round(elapsed, 6),
        "wall_throughput_fps": (
            round(processed_frames / elapsed, 3) if elapsed and processed_frames else None
        ),
        "disk": {
            "total_bytes": disk_after.total,
            "free_bytes_before": disk_before.free,
            "free_bytes_after": disk_after.free,
            "bytes_consumed": disk_before.free - disk_after.free,
            "artifact_bytes": sum(
                int(row.get("pose_bytes", 0)) + int(row.get("quality_bytes", 0))
                for row in completed
            ),
        },
        "ledger": str(ledger_path),
        "results": results,
    }
    _atomic_json(destination / "batch-summary.json", summary)
    return summary
