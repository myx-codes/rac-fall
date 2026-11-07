"""Frozen Phase 3A baseline registration and validation-only failure analysis."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import time

from .metrics import binary_metrics, subject_metrics
from .models import CausalResidualTCN


SCHEMA_VERSION = "racfall.phase3b-analysis.v1"
ALLOWED_VALIDATION_SUBJECTS = {"Subject.3", "Subject.7"}
SEALED_SUBJECTS = {"Subject.9", "Subject.10"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def activity_from_video_id(video_id: str, subject_id: str) -> str:
    value = video_id.removeprefix("caucafall-")
    value = value.removeprefix(subject_id + "-").removesuffix("-png")
    return value.replace("-", " ")


def validate_validation_scope(subjects) -> None:
    observed = {str(subject) for subject in subjects}
    if observed & SEALED_SUBJECTS or observed != ALLOWED_VALIDATION_SUBJECTS:
        raise ValueError("Analysis is restricted to frozen validation Subjects 3 and 7")


def window_diagnostics(features, valid, source_track_id) -> dict:
    """Summarize normalized pose motion and input quality without raw imagery."""
    np = __import__("numpy")
    pose = np.asarray(features, dtype=np.float64).reshape(-1, 17, 3)
    mask = np.asarray(valid, dtype=np.bool_)
    tracks = np.asarray(source_track_id, dtype=np.int64)
    confidence = pose[..., 2]
    visible_confidence = confidence[mask]
    pair_mask = mask[1:] & mask[:-1]
    displacement = np.linalg.norm(pose[1:, :, :2] - pose[:-1, :, :2], axis=-1)
    velocities = displacement[pair_mask]
    endpoint_velocities = displacement[-1][pair_mask[-1]] if len(pair_mask) else np.asarray([])
    nonnegative_tracks = tracks[tracks >= 0]
    track_switches = int((nonnegative_tracks[1:] != nonnegative_tracks[:-1]).sum())
    return {
        "mean_velocity_normalized_per_frame": float(velocities.mean()) if len(velocities) else 0.0,
        "p95_velocity_normalized_per_frame": float(np.quantile(velocities, 0.95)) if len(velocities) else 0.0,
        "endpoint_velocity_normalized_per_frame": (
            float(endpoint_velocities.mean()) if len(endpoint_velocities) else 0.0
        ),
        "mean_visible_confidence": (
            float(visible_confidence.mean()) if len(visible_confidence) else 0.0
        ),
        "missing_keypoint_rate": float(1.0 - mask.mean()),
        "fully_missing_frames": int((~mask.any(axis=1)).sum()),
        "track_switches": track_switches,
    }


def aggregate_diagnostics(rows: list[dict]) -> dict:
    np = __import__("numpy")
    fields = (
        "mean_velocity_normalized_per_frame", "p95_velocity_normalized_per_frame",
        "endpoint_velocity_normalized_per_frame", "mean_visible_confidence",
        "missing_keypoint_rate", "fully_missing_frames", "track_switches", "track_coverage",
    )
    if not rows:
        return {"windows": 0, **{name: {"mean": None, "median": None} for name in fields}}
    return {
        "windows": len(rows),
        **{
            name: {
                "mean": float(np.mean([float(row[name]) for row in rows])),
                "median": float(np.median([float(row[name]) for row in rows])),
            }
            for name in fields
        },
    }


def error_clusters(rows: list[dict]) -> list[dict]:
    """Merge only overlapping erroneous windows from the same sequence/type."""
    clusters = []
    grouped = defaultdict(list)
    for row in rows:
        if not row["correct"]:
            grouped[(row["video_id"], row["error_type"])].append(row)
    for (video_id, error_type), values in sorted(grouped.items()):
        current = None
        for row in sorted(values, key=lambda item: item["start_frame"]):
            if current is None or row["start_frame"] >= current["end_frame"]:
                if current is not None:
                    clusters.append(current)
                current = {
                    "video_id": video_id, "subject_id": row["subject_id"],
                    "activity": row["activity"], "error_type": error_type,
                    "start_frame": row["start_frame"], "end_frame": row["end_frame"],
                    "window_count": 1,
                }
            else:
                current["end_frame"] = max(current["end_frame"], row["end_frame"])
                current["window_count"] += 1
        if current is not None:
            clusters.append(current)
    return clusters


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sequence_svg(video_id: str, rows: list[dict], threshold: float) -> str:
    width, height = 960, 320
    left, right, top, bottom = 70, 25, 45, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    endpoints = [row["end_frame"] for row in rows]
    minimum, maximum = min(endpoints), max(endpoints)
    span = max(1, maximum - minimum)
    x = lambda frame: left + (frame - minimum) / span * plot_w
    y = lambda probability: top + (1.0 - probability) * plot_h
    points = " ".join(
        f'{x(row["end_frame"]):.2f},{y(row["probability_ground_state_1"]):.2f}'
        for row in rows
    )
    marks = []
    for row in rows:
        color = "#1f77b4" if row["label"] == 0 else "#d62728"
        stroke = "#111" if not row["correct"] else "none"
        marks.append(
            f'<circle cx="{x(row["end_frame"]):.2f}" '
            f'cy="{y(row["probability_ground_state_1"]):.2f}" r="4" '
            f'fill="{color}" stroke="{stroke}" stroke-width="2"><title>'
            f'frames {row["start_frame"]}-{row["end_frame"]}; label={row["label"]}; '
            f'p={row["probability_ground_state_1"]:.6f}; {escape(row["error_type"])}'
            f'</title></circle>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<title>{escape(video_id)} temporal ground-state probability</title>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="16">'
        f'{escape(video_id)} — frozen seed 29</text>'
        f'<line x1="{left}" y1="{y(threshold):.2f}" x2="{width-right}" '
        f'y2="{y(threshold):.2f}" stroke="#777" stroke-dasharray="6 4"/>'
        f'<polyline points="{points}" fill="none" stroke="#333" stroke-width="2"/>'
        + "".join(marks)
        + f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#000"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#000"/>'
        f'<text x="12" y="{top+8}" font-family="sans-serif" font-size="12">1.0</text>'
        f'<text x="12" y="{y(threshold)+4:.2f}" font-family="sans-serif" font-size="12">0.5</text>'
        f'<text x="12" y="{height-bottom+4}" font-family="sans-serif" font-size="12">0.0</text>'
        f'<text x="{width/2-55}" y="{height-15}" font-family="sans-serif" font-size="12">window endpoint frame</text>'
        f'<text x="{left}" y="{height-15}" font-family="sans-serif" font-size="11">{minimum}</text>'
        f'<text x="{width-right-30}" y="{height-15}" font-family="sans-serif" font-size="11">{maximum}</text>'
        f'<text x="{left+300}" y="25" font-family="sans-serif" font-size="11">'
        f'blue=label 0, red=label 1, black outline=error</text></svg>'
    )


def _predict(model, features, valid, device: str):
    torch = __import__("torch")
    with torch.inference_mode():
        return torch.sigmoid(model(features.to(device), valid.to(device))).cpu()


def _synchronize(torch, device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def inference_consistency(model, features, valid, device: str, repeats: int) -> dict:
    if repeats < 2:
        raise ValueError("repeat inference count must be at least 2")
    torch = __import__("torch")
    np = __import__("numpy")
    runs = []
    _predict(model, features, valid, device)
    _synchronize(torch, device)
    started = time.perf_counter()
    for _ in range(repeats):
        runs.append(_predict(model, features, valid, device).numpy())
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started
    values = np.stack(runs)
    reference = values[0]
    return {
        "repeats": repeats,
        "windows_per_repeat": int(values.shape[1]),
        "elapsed_s": elapsed,
        "mean_ms_per_repeat": elapsed * 1000 / repeats,
        "max_absolute_probability_delta": float(np.max(np.abs(values - reference))),
        "maximum_probability_std": float(values.std(axis=0).max()),
        "decision_disagreements": int(((values >= 0.5) != (reference >= 0.5)).sum()),
    }


def robustness_scenarios(model, features, valid, labels, device: str, seed: int) -> dict:
    """Fixed sensitivity checks; these are diagnostics, not tuned augmentations."""
    torch = __import__("torch")
    np = __import__("numpy")
    scenarios = {}
    baseline = _predict(model, features, valid, device).numpy()

    generator = torch.Generator().manual_seed(seed)
    jittered = features.clone()
    pose = jittered.reshape(*jittered.shape[:2], 17, 3)
    noise = torch.randn(pose[..., :2].shape, generator=generator) * 0.01
    pose[..., :2] += noise * valid.unsqueeze(-1)
    jitter_scores = _predict(model, jittered, valid, device).numpy()

    generator = torch.Generator().manual_seed(seed + 1)
    dropped_valid = valid.clone()
    drop = (torch.rand(valid.shape, generator=generator) < 0.10) & valid
    dropped_valid[drop] = False
    dropped_features = features.clone().reshape(*features.shape[:2], 17, 3)
    dropped_features[drop] = 0.0
    dropped_scores = _predict(
        model, dropped_features.reshape_as(features), dropped_valid, device,
    ).numpy()

    outage_valid = valid.clone()
    outage_features = features.clone().reshape(*features.shape[:2], 17, 3)
    outage_valid[:, -3:] = False
    outage_features[:, -3:] = 0.0
    outage_scores = _predict(
        model, outage_features.reshape_as(features), outage_valid, device,
    ).numpy()

    for name, scores, definition in (
        ("coordinate_jitter_sigma_0_01", jitter_scores, "Gaussian noise on valid normalized x/y"),
        ("random_valid_keypoint_dropout_10pct", dropped_scores, "10% fixed-seed valid-keypoint removal"),
        ("terminal_pose_outage_3_frames", outage_scores, "last three pose frames fully missing"),
    ):
        scenarios[name] = {
            "definition": definition,
            "metrics": binary_metrics(labels, scores),
            "decision_flips_vs_unperturbed": int(((scores >= 0.5) != (baseline >= 0.5)).sum()),
            "mean_absolute_probability_delta": float(np.mean(np.abs(scores - baseline))),
            "max_absolute_probability_delta": float(np.max(np.abs(scores - baseline))),
        }
    return scenarios


def _validate_registry(registry: dict, checkpoint: Path, windows: Path, config: Path) -> None:
    if registry.get("schema_version") != "racfall.frozen-baseline.v1":
        raise ValueError("Unsupported frozen baseline registry")
    if registry.get("status") != "frozen" or registry.get("selection", {}).get("seed") != 29:
        raise ValueError("Phase 3B requires frozen seed 29")
    expected_checkpoint = registry["checkpoint"]["sha256"]
    expected_windows = registry["preprocessing"]["artifact_sha256"]
    expected_config = registry["training_configuration"]["sha256"]
    actual = {
        "checkpoint": sha256_file(checkpoint), "windows": sha256_file(windows),
        "config": sha256_file(config),
    }
    expected = {
        "checkpoint": expected_checkpoint, "windows": expected_windows, "config": expected_config,
    }
    if actual != expected:
        raise ValueError(f"Frozen artifact checksum mismatch: expected={expected}, actual={actual}")


def analyze_phase3b(
    windows: str | Path, checkpoint: str | Path, registry: str | Path,
    phase3a_config: str | Path, output_dir: str | Path, *, device: str = "mps",
    repeat_inference: int = 20, robustness_seed: int = 20261108,
) -> dict:
    torch = __import__("torch")
    np = __import__("numpy")
    windows_path = Path(windows).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    registry_path = Path(registry).resolve()
    config_path = Path(phase3a_config).resolve()
    destination = Path(output_dir).resolve()
    for path in (windows_path, checkpoint_path, registry_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required but unavailable")
    frozen = json.loads(registry_path.read_text(encoding="utf-8"))
    _validate_registry(frozen, checkpoint_path, windows_path, config_path)

    with np.load(windows_path, allow_pickle=False) as data:
        if str(data["schema_version"]) != frozen["preprocessing"]["schema_version"]:
            raise ValueError("Preprocessing schema differs from frozen registry")
        if str(data["label_semantics"]) != "ground_state_at_causal_window_endpoint":
            raise ValueError("Phase 3B accepts ground-state endpoint labels only")
        validation = np.flatnonzero(data["split"] == "val")
        validate_validation_scope(data["subject_id"][validation].tolist())
        arrays = {name: data[name][validation] for name in (
            "features", "valid", "labels", "timestamp_s", "subject_id", "video_id",
            "start_frame", "end_frame", "track_coverage", "source_track_id",
        )}

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint_payload.get("schema_version") != "racfall.phase3a-checkpoint.v1" or
            checkpoint_payload.get("seed") != 29 or checkpoint_payload.get("epoch") != 6 or
            checkpoint_payload.get("windows_sha256") != sha256_file(windows_path)):
        raise ValueError("Checkpoint metadata does not match frozen selection")
    model_config = frozen["model"]
    model = CausalResidualTCN(
        input_channels=model_config["input_channels"], channels=model_config["channels"],
        dilations=tuple(model_config["dilations"]), kernel_size=model_config["kernel_size"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval().to(device)
    features = torch.from_numpy(arrays["features"])
    valid = torch.from_numpy(arrays["valid"])
    scores = _predict(model, features, valid, device).numpy()
    threshold = float(model_config["decision_threshold"])
    predictions = (scores >= threshold).astype(np.int64)

    rows = []
    for index in range(len(scores)):
        label, prediction = int(arrays["labels"][index]), int(predictions[index])
        diagnostic = window_diagnostics(
            arrays["features"][index], arrays["valid"][index], arrays["source_track_id"][index],
        )
        error_type = "correct" if label == prediction else ("false_positive" if label == 0 else "false_negative")
        rows.append({
            "subject_id": str(arrays["subject_id"][index]),
            "video_id": str(arrays["video_id"][index]),
            "activity": activity_from_video_id(
                str(arrays["video_id"][index]), str(arrays["subject_id"][index]),
            ),
            "start_frame": int(arrays["start_frame"][index]),
            "end_frame": int(arrays["end_frame"][index]),
            "endpoint_timestamp_s": float(arrays["timestamp_s"][index, -1]),
            "label": label, "prediction": prediction,
            "probability_ground_state_1": float(scores[index]),
            "correct": bool(label == prediction), "error_type": error_type,
            "track_coverage": float(arrays["track_coverage"][index]),
            **diagnostic,
        })

    destination.mkdir(parents=True, exist_ok=True)
    report_dir = destination / "reports"
    visualization_dir = destination / "visualizations"
    _write_csv(report_dir / "validation_windows.csv", rows)
    errors = [row for row in rows if not row["correct"]]
    atomic_json(report_dir / "validation_errors.json", {
        "schema_version": SCHEMA_VERSION, "errors": errors, "clusters": error_clusters(rows),
    })
    sequence_reports = {}
    for video_id in sorted({row["video_id"] for row in rows}):
        sequence_rows = [row for row in rows if row["video_id"] == video_id]
        visualization_dir.mkdir(parents=True, exist_ok=True)
        svg_path = visualization_dir / f"{video_id}.svg"
        svg_path.write_text(_sequence_svg(video_id, sequence_rows, threshold), encoding="utf-8")
        sequence_reports[video_id] = {
            "subject_id": sequence_rows[0]["subject_id"],
            "activity": sequence_rows[0]["activity"], "windows": len(sequence_rows),
            "false_positives": sum(row["error_type"] == "false_positive" for row in sequence_rows),
            "false_negatives": sum(row["error_type"] == "false_negative" for row in sequence_rows),
            "diagnostics": aggregate_diagnostics(sequence_rows),
            "visualization": str(svg_path),
        }

    metrics = binary_metrics(arrays["labels"], scores, threshold=threshold)
    metrics.update(subject_metrics(
        arrays["labels"], scores, arrays["subject_id"], threshold=threshold,
    ))
    grouped = {
        "correct": aggregate_diagnostics([row for row in rows if row["correct"]]),
        "incorrect": aggregate_diagnostics(errors),
        "false_positive": aggregate_diagnostics([
            row for row in rows if row["error_type"] == "false_positive"
        ]),
        "false_negative": aggregate_diagnostics([
            row for row in rows if row["error_type"] == "false_negative"
        ]),
        "subject3_sit_down_correct": aggregate_diagnostics([
            row for row in rows if row["subject_id"] == "Subject.3" and
            row["activity"] == "Sit down" and row["correct"]
        ]),
        "subject3_sit_down_false_positive": aggregate_diagnostics([
            row for row in rows if row["subject_id"] == "Subject.3" and
            row["activity"] == "Sit down" and row["error_type"] == "false_positive"
        ]),
    }
    consistency = inference_consistency(model, features, valid, device, repeat_inference)
    robustness = robustness_scenarios(
        model, features, valid, arrays["labels"], device, robustness_seed,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": "validation-only causal ground-state failure analysis; not event detection",
        "frozen_baseline": frozen,
        "artifact_verification": {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "windows_sha256": sha256_file(windows_path),
            "phase3a_config_sha256": sha256_file(config_path),
            "checkpoint_metadata_seed": int(checkpoint_payload["seed"]),
            "checkpoint_metadata_epoch": int(checkpoint_payload["epoch"]),
        },
        "scope": {
            "windows": len(rows), "subjects": sorted(set(arrays["subject_id"].tolist())),
            "sealed_subjects_accessed": [], "gmdcsa24_accessed": False,
        },
        "metrics": metrics,
        "error_counts": {
            "false_positive": sum(row["error_type"] == "false_positive" for row in rows),
            "false_negative": sum(row["error_type"] == "false_negative" for row in rows),
            "error_clusters": len(error_clusters(rows)),
        },
        "diagnostic_comparison": grouped,
        "sequences": sequence_reports,
        "inference_consistency": consistency,
        "robustness": {
            "purpose": "fixed sensitivity analysis only; no threshold or model tuning",
            "seed": robustness_seed, "scenarios": robustness,
        },
        "outputs": {
            "window_report": str(report_dir / "validation_windows.csv"),
            "error_report": str(report_dir / "validation_errors.json"),
            "visualization_count": len(sequence_reports),
        },
    }
    atomic_json(destination / "registry.json", {
        "schema_version": "racfall.frozen-baseline-registration.v1",
        "baseline": frozen, "verification": result["artifact_verification"],
        "original_checkpoint_preserved": True,
    })
    atomic_json(destination / "analysis.json", result)
    result["outputs"]["analysis_sha256"] = sha256_file(destination / "analysis.json")
    result["outputs"]["registry_sha256"] = sha256_file(destination / "registry.json")
    return result
