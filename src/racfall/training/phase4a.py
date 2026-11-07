"""Deterministic quality-aware verification over the frozen Pose-TCN."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .metrics import binary_metrics
from .models import CausalResidualTCN
from .phase3b import (
    _predict,
    _validate_registry,
    activity_from_video_id,
    atomic_json,
    sha256_file,
    validate_validation_scope,
    window_diagnostics,
)


SCHEMA_VERSION = "racfall.phase4a-results.v1"
STATES = {"NORMAL", "CANDIDATE", "UNCERTAIN", "ALERT", "COOLDOWN"}
VARIANTS = (
    "frozen_tcn", "fixed_verification", "quality_aware_verification",
    "risk_adaptive_verification",
)


@dataclass(frozen=True)
class Observation:
    probability: float
    probability_trend: float
    movement_velocity: float
    mean_visible_confidence: float
    missing_keypoint_rate: float
    fully_missing_frames: int
    track_coverage: float
    track_switches: int


def validate_config(config: dict) -> None:
    if config.get("schema_version") != "racfall.phase4a-config.v1":
        raise ValueError("Unsupported Phase 4A configuration")
    if config.get("status") != "preregistered_before_phase4a_validation_run":
        raise ValueError("Phase 4A configuration is not preregistered")
    if tuple(config.get("variants", {}).keys()) != VARIANTS:
        raise ValueError("Phase 4A requires exactly four ordered variants")
    if not 0 < config["release_threshold"] < config["candidate_threshold"] < 1:
        raise ValueError("Invalid probability thresholds")
    if config["cooldown_windows"] < 0 or config["expected_stride_frames"] <= 0:
        raise ValueError("Invalid temporal configuration")
    weights = config["quality_score_weights"]
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("Quality score weights must sum to one")


def quality_assessment(observation: Observation, config: dict) -> dict:
    gate = config["quality_gate"]
    reasons = []
    if observation.track_coverage < gate["minimum_track_coverage"]:
        reasons.append("low_track_coverage")
    if observation.missing_keypoint_rate > gate["maximum_missing_keypoint_rate"]:
        reasons.append("high_missing_keypoint_rate")
    if observation.mean_visible_confidence < gate["minimum_mean_visible_confidence"]:
        reasons.append("low_pose_confidence")
    if observation.fully_missing_frames > gate["maximum_fully_missing_frames"]:
        reasons.append("too_many_fully_missing_frames")
    if observation.track_switches > gate["maximum_track_switches"]:
        reasons.append("identity_discontinuity")
    weights = config["quality_score_weights"]
    continuity = max(0.0, 1.0 - observation.track_switches / 3.0)
    score = (
        weights["track_coverage"] * observation.track_coverage
        + weights["observed_keypoint_fraction"] * (1.0 - observation.missing_keypoint_rate)
        + weights["mean_visible_confidence"] * observation.mean_visible_confidence
        + weights["identity_continuity"] * continuity
    )
    return {"low_quality": bool(reasons), "reasons": reasons, "quality_score": float(score)}


def risk_score(observation: Observation, config: dict) -> float:
    adaptive = config["variants"]["risk_adaptive_verification"]
    weights = adaptive["risk_weights"]
    trend = min(1.0, max(0.0, observation.probability_trend) / adaptive["trend_scale"])
    velocity = min(1.0, max(0.0, observation.movement_velocity) / adaptive["velocity_scale"])
    return float(
        weights["tcn_probability"] * observation.probability
        + weights["positive_probability_trend"] * trend
        + weights["movement_velocity"] * velocity
    )


class VerificationMachine:
    """Causal state machine. It has no label or annotation input."""

    def __init__(self, variant: str, config: dict):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Phase 4A variant: {variant}")
        validate_config(config)
        self.variant = variant
        self.config = config
        self.candidate_streak = 0
        self.alert_active = False
        self.cooldown_remaining = 0

    def reset(self) -> None:
        self.candidate_streak = 0
        self.alert_active = False
        self.cooldown_remaining = 0

    def _required_confirmation(self, observation: Observation, quality_score: float) -> tuple[int, float]:
        if self.variant == "fixed_verification":
            return int(self.config["variants"][self.variant]["confirmation_windows"]), 0.0
        if self.variant == "quality_aware_verification":
            return int(self.config["variants"][self.variant]["confirmation_windows"]), 0.0
        score = risk_score(observation, self.config)
        adaptive = self.config["variants"]["risk_adaptive_verification"]
        windows = adaptive["confirmation_windows"]
        if score >= adaptive["high_risk_threshold"] and quality_score >= adaptive["high_quality_threshold"]:
            return int(windows["high_risk_high_quality"]), score
        if score >= adaptive["standard_risk_threshold"]:
            return int(windows["standard_risk"]), score
        return int(windows["borderline_risk"]), score

    def step(self, observation: Observation) -> dict:
        probability = observation.probability
        threshold = self.config["candidate_threshold"]
        release = self.config["release_threshold"]
        quality = quality_assessment(observation, self.config)

        if self.variant == "frozen_tcn":
            state = "ALERT" if probability >= threshold else "NORMAL"
            return {"state": state, "quality": quality, "risk_score": probability,
                    "required_confirmation_windows": 1, "candidate_streak": int(state == "ALERT")}

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            state = "UNCERTAIN" if quality["low_quality"] else "COOLDOWN"
            return {"state": state, "quality": quality, "risk_score": 0.0,
                    "required_confirmation_windows": 0, "candidate_streak": 0}

        always_gate = self.variant in {"quality_aware_verification", "risk_adaptive_verification"}
        fixed_negative_gate = self.variant == "fixed_verification" and probability < threshold
        if quality["low_quality"] and (always_gate or fixed_negative_gate):
            self.candidate_streak = 0
            self.alert_active = False
            return {"state": "UNCERTAIN", "quality": quality, "risk_score": 0.0,
                    "required_confirmation_windows": 0, "candidate_streak": 0}

        if self.alert_active:
            if probability >= release:
                return {"state": "ALERT", "quality": quality, "risk_score": probability,
                        "required_confirmation_windows": 0, "candidate_streak": self.candidate_streak}
            self.alert_active = False
            self.candidate_streak = 0
            self.cooldown_remaining = max(0, int(self.config["cooldown_windows"]) - 1)
            return {"state": "COOLDOWN", "quality": quality, "risk_score": probability,
                    "required_confirmation_windows": 0, "candidate_streak": 0}

        if probability >= threshold:
            self.candidate_streak += 1
            required, score = self._required_confirmation(observation, quality["quality_score"])
            if self.candidate_streak >= required:
                self.alert_active = True
                return {"state": "ALERT", "quality": quality, "risk_score": score,
                        "required_confirmation_windows": required,
                        "candidate_streak": self.candidate_streak}
            return {"state": "CANDIDATE", "quality": quality, "risk_score": score,
                    "required_confirmation_windows": required,
                    "candidate_streak": self.candidate_streak}

        self.candidate_streak = 0
        return {"state": "NORMAL", "quality": quality, "risk_score": probability,
                "required_confirmation_windows": 0, "candidate_streak": 0}


def _division(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_states(labels, states: list[str]) -> dict:
    np = __import__("numpy")
    labels = np.asarray(labels, dtype=np.int64)
    states_array = np.asarray(states)
    if len(labels) == 0 or len(labels) != len(states) or not set(states).issubset(STATES):
        raise ValueError("Labels/states must be non-empty and aligned")
    alert_predictions = (states_array == "ALERT").astype(np.int64)
    all_sample = binary_metrics(labels, alert_predictions, threshold=0.5)
    tn, fp = all_sample["confusion_matrix"][0]
    fn, tp = all_sample["confusion_matrix"][1]
    all_sample.update({
        "false_positives": fp, "false_negatives": fn,
        "precision_class_1": _division(tp, tp + fp),
    })
    uncertain = states_array == "UNCERTAIN"
    retained = ~uncertain
    uncertain_by_label = {
        "0": int(((labels == 0) & uncertain).sum()),
        "1": int(((labels == 1) & uncertain).sum()),
    }
    conditional = None
    if retained.any():
        conditional = binary_metrics(labels[retained], alert_predictions[retained], threshold=0.5)
        ctn, cfp = conditional["confusion_matrix"][0]
        cfn, ctp = conditional["confusion_matrix"][1]
        conditional.update({
            "false_positives": cfp, "false_negatives": cfn,
            "precision_class_1": _division(ctp, ctp + cfp),
        })
    strict_predictions = alert_predictions.copy()
    strict_predictions[uncertain] = 1 - labels[uncertain]
    strict = binary_metrics(labels, strict_predictions, threshold=0.5)
    stn, sfp = strict["confusion_matrix"][0]
    sfn, stp = strict["confusion_matrix"][1]
    strict.update({
        "false_positive_equivalents": sfp, "false_negative_equivalents": sfn,
        "precision_class_1": _division(stp, stp + sfp),
        "definition": "UNCERTAIN is counted as an error against its true class",
    })
    return {
        "all_sample_alert_decision": all_sample,
        "strict_abstention_as_error": strict,
        "conditional_non_abstained": conditional,
        "state_counts": dict(sorted(Counter(states).items())),
        "uncertain_by_label": uncertain_by_label,
        "uncertainty_rate": float(uncertain.mean()),
        "coverage": float(retained.mean()),
    }


def _delay_summary(rows: list[dict], state_key: str) -> dict:
    np = __import__("numpy")
    delays = []
    missed_runs = 0
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["end_frame"])
        run_start = None
        for index, row in enumerate(values + [{"label": 0}]):
            if row["label"] == 1 and run_start is None:
                run_start = index
            if row["label"] == 0 and run_start is not None:
                alert_index = next(
                    (position for position in range(run_start, index)
                     if values[position][state_key] == "ALERT"), None,
                )
                if alert_index is None:
                    missed_runs += 1
                else:
                    delays.append(alert_index - run_start)
                run_start = None
    return {
        "definition": "windows from first verified ground-state-positive endpoint to first ALERT in that positive run; not fall-event delay",
        "observed_positive_runs": len(delays) + missed_runs,
        "confirmed_runs": len(delays), "missed_runs": missed_runs,
        "delays_windows": delays,
        "mean_windows": float(np.mean(delays)) if delays else None,
        "median_windows": float(np.median(delays)) if delays else None,
        "maximum_windows": int(max(delays)) if delays else None,
    }


def _candidate_delay_summary(rows: list[dict], state_key: str) -> dict:
    delays = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["end_frame"])
        candidate_start = None
        for index, row in enumerate(values):
            state = row[state_key]
            if state == "CANDIDATE" and candidate_start is None:
                candidate_start = index
            elif state == "ALERT" and candidate_start is not None:
                delays.append(index - candidate_start)
                candidate_start = None
            elif state in {"NORMAL", "UNCERTAIN", "COOLDOWN"}:
                candidate_start = None
    return {
        "definition": "windows from CANDIDATE entry to ALERT; label-independent",
        "confirmed_candidate_episodes": len(delays), "delays_windows": delays,
        "mean_windows": sum(delays) / len(delays) if delays else None,
        "maximum_windows": max(delays) if delays else None,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_rows(windows: Path, checkpoint: Path, registry: dict, device: str) -> tuple[list[dict], dict]:
    torch = __import__("torch")
    np = __import__("numpy")
    with np.load(windows, allow_pickle=False) as data:
        if str(data["schema_version"]) != registry["preprocessing"]["schema_version"]:
            raise ValueError("Preprocessing schema differs from frozen registry")
        if str(data["label_semantics"]) != "ground_state_at_causal_window_endpoint":
            raise ValueError("Phase 4A requires ground-state endpoint labels")
        indices = np.flatnonzero(data["split"] == "val")
        validate_validation_scope(data["subject_id"][indices].tolist())
        arrays = {name: data[name][indices] for name in (
            "features", "valid", "labels", "timestamp_s", "subject_id", "video_id",
            "start_frame", "end_frame", "track_coverage", "source_track_id",
        )}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("schema_version") != "racfall.phase3a-checkpoint.v1" or
            payload.get("seed") != 29 or payload.get("epoch") != 6 or
            payload.get("windows_sha256") != sha256_file(windows)):
        raise ValueError("Checkpoint metadata does not match frozen selection")
    model_config = registry["model"]
    model = CausalResidualTCN(
        input_channels=model_config["input_channels"], channels=model_config["channels"],
        dilations=tuple(model_config["dilations"]), kernel_size=model_config["kernel_size"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    scores = _predict(
        model, torch.from_numpy(arrays["features"]), torch.from_numpy(arrays["valid"]), device,
    ).numpy()
    rows = []
    for index, probability in enumerate(scores):
        diagnostics = window_diagnostics(
            arrays["features"][index], arrays["valid"][index], arrays["source_track_id"][index],
        )
        rows.append({
            "subject_id": str(arrays["subject_id"][index]),
            "video_id": str(arrays["video_id"][index]),
            "activity": activity_from_video_id(
                str(arrays["video_id"][index]), str(arrays["subject_id"][index]),
            ),
            "start_frame": int(arrays["start_frame"][index]),
            "end_frame": int(arrays["end_frame"][index]),
            "endpoint_timestamp_s": float(arrays["timestamp_s"][index, -1]),
            "label": int(arrays["labels"][index]),
            "probability_ground_state_1": float(probability),
            "track_coverage": float(arrays["track_coverage"][index]),
            **diagnostics,
        })
    return rows, payload


def _apply_variant(rows: list[dict], variant: str, config: dict) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    state_key = f"state_{variant}"
    for values in grouped.values():
        values.sort(key=lambda row: row["end_frame"])
        machine = VerificationMachine(variant, config)
        previous = None
        for row in values:
            contiguous = (
                previous is not None and
                row["start_frame"] - previous["start_frame"] == config["expected_stride_frames"]
            )
            if not contiguous:
                machine.reset()
            trend = (
                row["probability_ground_state_1"] - previous["probability_ground_state_1"]
                if contiguous else 0.0
            )
            observation = Observation(
                probability=row["probability_ground_state_1"], probability_trend=trend,
                movement_velocity=row["mean_velocity_normalized_per_frame"],
                mean_visible_confidence=row["mean_visible_confidence"],
                missing_keypoint_rate=row["missing_keypoint_rate"],
                fully_missing_frames=row["fully_missing_frames"],
                track_coverage=row["track_coverage"], track_switches=row["track_switches"],
            )
            outcome = machine.step(observation)
            row[state_key] = outcome["state"]
            row[f"quality_score_{variant}"] = outcome["quality"]["quality_score"]
            row[f"quality_reasons_{variant}"] = ";".join(outcome["quality"]["reasons"])
            row[f"risk_score_{variant}"] = outcome["risk_score"]
            row[f"required_confirmation_{variant}"] = outcome["required_confirmation_windows"]
            row[f"candidate_streak_{variant}"] = outcome["candidate_streak"]
            row[f"probability_trend_{variant}"] = trend
            previous = row


def _subset_report(rows: list[dict], variant: str, predicate) -> dict:
    selected = [row for row in rows if predicate(row)]
    state_key = f"state_{variant}"
    return {
        "windows": len(selected),
        "metrics": evaluate_states(
            [row["label"] for row in selected], [row[state_key] for row in selected],
        ),
        "timeline": [{
            "frames": [row["start_frame"], row["end_frame"]],
            "label": row["label"], "probability": row["probability_ground_state_1"],
            "state": row[state_key], "quality_reasons": row[f"quality_reasons_{variant}"],
        } for row in selected],
    }


def run_phase4a(
    windows: str | Path, checkpoint: str | Path, frozen_registry: str | Path,
    phase3a_config: str | Path, verifier_config: str | Path, output_dir: str | Path,
    *, device: str = "mps",
) -> dict:
    torch = __import__("torch")
    windows_path = Path(windows).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    registry_path = Path(frozen_registry).resolve()
    phase3a_path = Path(phase3a_config).resolve()
    verifier_path = Path(verifier_config).resolve()
    destination = Path(output_dir).resolve()
    for path in (windows_path, checkpoint_path, registry_path, phase3a_path, verifier_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required but unavailable")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    config = json.loads(verifier_path.read_text(encoding="utf-8"))
    validate_config(config)
    _validate_registry(registry, checkpoint_path, windows_path, phase3a_path)
    if config["frozen_baseline_id"] != registry["baseline_id"]:
        raise ValueError("Verifier config references a different frozen baseline")
    rows, checkpoint_payload = _load_rows(windows_path, checkpoint_path, registry, device)
    for variant in VARIANTS:
        _apply_variant(rows, variant, config)

    labels = [row["label"] for row in rows]
    variants = {}
    for variant in VARIANTS:
        state_key = f"state_{variant}"
        variants[variant] = {
            "metrics": evaluate_states(labels, [row[state_key] for row in rows]),
            "ground_state_confirmation_delay": _delay_summary(rows, state_key),
            "candidate_confirmation_delay": _candidate_delay_summary(rows, state_key),
            "subject3_sit_down": _subset_report(
                rows, variant,
                lambda row: row["subject_id"] == "Subject.3" and row["activity"] == "Sit down",
            ),
            "subject7_fall_forward": _subset_report(
                rows, variant,
                lambda row: row["subject_id"] == "Subject.7" and row["activity"] == "Fall forward",
            ),
        }

    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "validation_windows.csv"
    _write_csv(report_path, rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": "quality-aware causal ground-state verification; not fall-event detection",
        "preregistered_config": config,
        "artifact_verification": {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "windows_sha256": sha256_file(windows_path),
            "phase3a_config_sha256": sha256_file(phase3a_path),
            "verifier_config_sha256": sha256_file(verifier_path),
            "checkpoint_seed": int(checkpoint_payload["seed"]),
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "checkpoint_unchanged": True,
        },
        "scope": {
            "windows": len(rows), "subjects": ["Subject.3", "Subject.7"],
            "sealed_subjects_accessed": [], "gmdcsa24_accessed": False,
            "raw_images_accessed": False,
        },
        "variants": variants,
        "interpretation_guardrails": {
            "uncertain_in_all_sample_metrics": "ALERT=positive; all other states=negative; uncertain positives therefore remain false negatives",
            "strict_metrics": "every UNCERTAIN is additionally counted as an error against its known validation class",
            "conditional_metrics": "reported only beside explicit coverage and uncertainty counts",
            "delay": "ground-state endpoint confirmation in windows, not fall-event or real alert delay",
        },
        "outputs": {"window_report": str(report_path)},
    }
    atomic_json(destination / "results.json", result)
    result["outputs"]["results_sha256"] = sha256_file(destination / "results.json")
    return result
