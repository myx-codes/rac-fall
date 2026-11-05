from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import print_diagnostic
from .evaluation.events import evaluate, read_alerts, read_events, read_manifest


def main():
    parser = argparse.ArgumentParser(description="RAC-Fall research tools (not a safety product)")
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("env-check", help="Check local Python / Apple MPS support")
    ev = command.add_parser("eval-events", help="Reference event-level evaluation for annotated CSVs")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--ground-truth", required=True)
    ev.add_argument("--alerts", required=True)
    ev.add_argument("--cooldown-s", type=float, default=10.0)
    ev.add_argument("--match-window-s", type=float, default=5.0)
    smoke = command.add_parser("pose-smoke", help="Run pretrained pose inference only (no fall decision)")
    smoke.add_argument("--source", required=True, help="MP4 path or webcam index 0")
    smoke.add_argument("--max-frames", type=int, default=50)
    smoke.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    smoke.add_argument("--imgsz", type=int, default=640)
    live = command.add_parser("pose-live", help="Live tracked pose display (no fall decision or recording)")
    live.add_argument("--source", default="0", help="Webcam index (default: 0)")
    live.add_argument("--device", choices=["auto", "mps", "cpu"], default="mps")
    live.add_argument("--imgsz", type=int, default=416)
    manifest_cauca = command.add_parser("manifest-caucafall", help="Audit CAUCAFall videos into a strict manifest")
    manifest_cauca.add_argument("--root", required=True)
    manifest_cauca.add_argument("--label-map", required=True, help="Reviewed JSON mapping activity folders to fall/non_fall")
    manifest_cauca.add_argument("--output", required=True)
    manifest_cauca.add_argument("--skip-sha256", action="store_true")
    manifest_cauca_png = command.add_parser(
        "manifest-caucafall-png", help="Build canonical CAUCAFall PNG/TXT manifest",
    )
    manifest_cauca_png.add_argument("--root", required=True)
    manifest_cauca_png.add_argument("--label-map", required=True)
    manifest_cauca_png.add_argument("--output", required=True)
    manifest_cauca_png.add_argument("--skip-sha256", action="store_true")
    manifest_urfd = command.add_parser("manifest-urfd", help="Audit URFD RGB sequences into a strict manifest")
    manifest_urfd.add_argument("--root", required=True)
    manifest_urfd.add_argument("--output", required=True)
    manifest_urfd.add_argument("--skip-sha256", action="store_true")
    validate_data = command.add_parser("validate-dataset-manifest", help="Validate Phase 2B manifest and leakage rules")
    validate_data.add_argument("--manifest", required=True)
    split_cauca = command.add_parser("split-caucafall", help="Create reproducible 6/2/2 subject splits")
    split_cauca.add_argument("--manifest", required=True)
    split_cauca.add_argument("--output", required=True)
    split_cauca.add_argument("--assignment-output", required=True)
    split_cauca.add_argument("--seed", type=int, default=20261104)
    frozen_split = command.add_parser(
        "split-caucafall-frozen", help="Apply the frozen Phase 2B.4 subject split",
    )
    frozen_split.add_argument("--manifest", required=True)
    frozen_split.add_argument("--output", required=True)
    frozen_split.add_argument("--assignment-output", required=True)
    extract = command.add_parser("extract-poses", help="Extract tracked COCO poses to compact NPZ files")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--dataset-root", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--device", choices=["mps", "cpu"], default="mps")
    extract.add_argument("--imgsz", type=int, default=416)
    extract.add_argument("--confidence", type=float, default=0.25)
    extract.add_argument("--detector-confidence", type=float, default=0.25)
    extract.add_argument("--limit", type=int, help="Pilot limit; omitted means all manifest records")
    sequences = command.add_parser("build-pose-sequences", help="Normalize poses and generate causal windows")
    sequences.add_argument("--manifest", required=True)
    sequences.add_argument("--pose-dir", required=True)
    sequences.add_argument("--output", required=True)
    sequences.add_argument("--window-frames", type=int, default=30)
    sequences.add_argument("--stride-frames", type=int, default=15)
    sequences.add_argument("--alignment-dir", help="Verified per-video annotation-alignment JSON directory")
    sequences.add_argument("--quality-dir", help="Sequence quality-gate JSON directory")
    sequences.add_argument("--min-track-coverage", type=float, default=0.5)
    sequences.add_argument("--allow-internal-test", action="store_true")
    align = command.add_parser(
        "align-caucafall-annotations",
        help="Validate CAUCAFall TXT/PNG/video alignment and write fail-closed provenance",
    )
    align.add_argument("--manifest", required=True)
    align.add_argument("--dataset-root", required=True)
    align.add_argument("--reference-png-dir", required=True)
    align.add_argument("--pose-dir", required=True)
    align.add_argument("--output-dir", required=True)
    align.add_argument("--limit", type=int, default=1)
    canonical = command.add_parser(
        "audit-caucafall-canonical", help="Audit same-index canonical PNG/TXT ground-state labels",
    )
    canonical.add_argument("--manifest", required=True)
    canonical.add_argument("--dataset-root", required=True)
    canonical.add_argument("--output-dir", required=True)
    canonical.add_argument(
        "--annotation-dir", help="Pilot-only exact TXT directory; requires --limit 1",
    )
    canonical.add_argument("--limit", type=int)
    batch = command.add_parser(
        "preprocess-caucafall-batch", help="Resume checksum-validated canonical preprocessing",
    )
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--dataset-root", required=True)
    batch.add_argument("--canonical-audit-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--device", choices=["mps", "cpu"], default="mps")
    batch.add_argument("--imgsz", type=int, default=416)
    batch.add_argument("--detector-confidence", type=float, default=0.25)
    batch.add_argument("--keypoint-confidence", type=float, default=0.25)
    batch.add_argument("--min-pose-coverage", type=float, default=0.5)
    batch.add_argument("--allow-internal-test", action="store_true")
    batch.add_argument("--limit", type=int)
    recovery = command.add_parser(
        "analyze-pose-recovery", help="Diagnostic lower-confidence/tiled inference on missing poses",
    )
    recovery.add_argument("--manifest", required=True)
    recovery.add_argument("--dataset-root", required=True)
    recovery.add_argument("--pose-dir", required=True)
    recovery.add_argument("--output-dir", required=True)
    recovery.add_argument("--device", choices=["mps", "cpu"], default="mps")
    recovery.add_argument("--imgsz", type=int, default=416)
    recovery.add_argument("--limit", type=int, default=1)
    recovery.add_argument("--limit-missing", type=int)
    args = parser.parse_args()
    if args.command == "env-check":
        print_diagnostic()
    elif args.command == "eval-events":
        manifest = read_manifest(args.manifest)
        events = read_events(args.ground_truth, manifest)
        alerts = read_alerts(args.alerts, manifest)
        print(json.dumps(evaluate(manifest, events, alerts, cooldown_s=args.cooldown_s,
                                  match_window_s=args.match_window_s), indent=2))
    elif args.command == "pose-smoke":
        from .pose_smoke import pose_smoke
        print(json.dumps(pose_smoke(args.source, args.max_frames, args.device, args.imgsz), indent=2))
    elif args.command == "pose-live":
        from .pose_live import pose_live
        print(json.dumps(pose_live(args.source, args.imgsz, args.device), indent=2))
    elif args.command == "manifest-caucafall":
        from .data.adapters import CAUCAFallAdapter, load_label_map
        from .data.manifest import write_manifest
        records = CAUCAFallAdapter(
            args.root, load_label_map(args.label_map), hash_sources=not args.skip_sha256,
        ).scan()
        write_manifest(records, args.output)
        print(json.dumps({"records": len(records), "output": args.output}, indent=2))
    elif args.command == "manifest-caucafall-png":
        from .data.adapters import CAUCAFallCanonicalAdapter, load_label_map
        from .data.manifest import write_manifest
        records = CAUCAFallCanonicalAdapter(
            args.root, load_label_map(args.label_map), hash_sources=not args.skip_sha256,
        ).scan()
        write_manifest(records, args.output)
        print(json.dumps({"records": len(records), "output": args.output}, indent=2))
    elif args.command == "manifest-urfd":
        from .data.adapters import URFDRGBAdapter
        from .data.manifest import write_manifest
        records = URFDRGBAdapter(args.root, hash_sources=not args.skip_sha256).scan()
        write_manifest(records, args.output)
        print(json.dumps({"records": len(records), "output": args.output}, indent=2))
    elif args.command == "validate-dataset-manifest":
        from .data.manifest import read_manifest, validate_manifest
        print(json.dumps(validate_manifest(read_manifest(args.manifest)), indent=2))
    elif args.command == "split-caucafall":
        from .data.manifest import read_manifest, write_manifest
        from .data.splits import assign_subject_splits
        records, assignment = assign_subject_splits(read_manifest(args.manifest), seed=args.seed)
        write_manifest(records, args.output)
        assignment_path = Path(args.assignment_output)
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_path.write_text(json.dumps(assignment, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"records": len(records), "output": args.output, **assignment}, indent=2))
    elif args.command == "split-caucafall-frozen":
        from .data.manifest import read_manifest, write_manifest
        from .data.splits import assign_frozen_2b4_splits
        records, assignment = assign_frozen_2b4_splits(read_manifest(args.manifest))
        write_manifest(records, args.output)
        assignment_path = Path(args.assignment_output)
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_path.write_text(json.dumps(assignment, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"records": len(records), "output": args.output, **assignment}, indent=2))
    elif args.command == "extract-poses":
        from .data.manifest import read_manifest
        from .data.pose import PoseExtractor
        records = read_manifest(args.manifest)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be > 0")
            records = records[:args.limit]
        extractor = PoseExtractor(
            device=args.device, imgsz=args.imgsz, confidence_threshold=args.confidence,
            detector_confidence=args.detector_confidence,
        )
        output_dir = Path(args.output_dir)
        results = [extractor.extract(
            args.dataset_root, record, output_dir / f"{record.video_id}.npz",
        ) for record in records]
        print(json.dumps({"processed": len(results), "results": results}, indent=2))
    elif args.command == "audit-caucafall-canonical":
        from .data.canonical import audit_canonical_sequence
        from .data.manifest import read_manifest
        rows = read_manifest(args.manifest)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be > 0")
            rows = rows[:args.limit]
        if args.annotation_dir and len(rows) != 1:
            raise ValueError("--annotation-dir is pilot-only and requires exactly one record")
        root = Path(args.dataset_root).resolve()
        output_dir = Path(args.output_dir)
        results = []
        for row in rows:
            image_dir = (root / row.source_relpath).resolve()
            annotation_dir = Path(args.annotation_dir).resolve() if args.annotation_dir else image_dir
            payload = audit_canonical_sequence(
                image_dir, annotation_dir, sequence_label=row.label, video_id=row.video_id,
                subject_id=row.subject_id, action=row.action, timing_fps=row.fps,
                timing_provenance=row.timestamp_source,
                output=output_dir / f"{row.video_id}.json",
            )
            results.append({
                "video_id": row.video_id, "status": payload["status"],
                "frames": payload["frame_count"],
                "valid_ground_state_frames": payload["valid_ground_state_frames"],
                "invalid_ground_state_frames": payload["invalid_ground_state_frames"],
                "class_counts": payload["ground_state_class_counts"],
            })
        print(json.dumps({"processed": len(results), "results": results}, indent=2))
    elif args.command == "preprocess-caucafall-batch":
        from .data.batch import preprocess_canonical_batch
        from .data.manifest import read_manifest
        result = preprocess_canonical_batch(
            read_manifest(args.manifest), args.dataset_root, args.canonical_audit_dir,
            args.output_dir, device=args.device, imgsz=args.imgsz,
            detector_confidence=args.detector_confidence,
            keypoint_confidence=args.keypoint_confidence,
            min_pose_coverage=args.min_pose_coverage,
            allow_internal_test=args.allow_internal_test, limit=args.limit,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "analyze-pose-recovery":
        from .data.manifest import read_manifest
        from .data.pose_diagnostics import analyze_pose_recovery
        rows = read_manifest(args.manifest)[:args.limit]
        results = [analyze_pose_recovery(
            row, args.dataset_root, Path(args.pose_dir) / f"{row.video_id}.npz",
            Path(args.output_dir) / f"{row.video_id}.json", device=args.device,
            imgsz=args.imgsz, limit_missing=args.limit_missing,
        ) for row in rows]
        print(json.dumps({"processed": len(results), "results": results}, indent=2))
    elif args.command == "build-pose-sequences":
        from .data.manifest import read_manifest
        from .data.sequences import build_windows
        result = build_windows(
            read_manifest(args.manifest), args.pose_dir, args.output,
            window_frames=args.window_frames, stride_frames=args.stride_frames,
            alignment_dir=args.alignment_dir,
            quality_dir=args.quality_dir,
            min_track_coverage=args.min_track_coverage,
            allow_internal_test=args.allow_internal_test,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "align-caucafall-annotations":
        from .data.alignment import audit_caucafall_alignment
        from .data.manifest import read_manifest
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        rows = read_manifest(args.manifest)[:args.limit]
        output_dir = Path(args.output_dir)
        results = []
        for row in rows:
            payload = audit_caucafall_alignment(
                args.dataset_root, row, args.reference_png_dir,
                Path(args.pose_dir) / f"{row.video_id}.npz",
                output_dir / f"{row.video_id}.json",
            )
            results.append({
                "video_id": row.video_id, "status": payload["status"],
                "quarantined_annotation_count": len(payload["quarantined_annotations"]),
                "quarantined_annotations": (
                    payload["quarantined_annotations"]
                    if len(payload["quarantined_annotations"]) <= 10 else None
                ),
            })
        print(json.dumps({"processed": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
