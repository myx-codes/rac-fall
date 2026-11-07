from pathlib import Path

import pytest

from racfall.training.phase4a import (
    Observation,
    VerificationMachine,
    evaluate_states,
    quality_assessment,
    risk_score,
)


@pytest.fixture
def config():
    import json
    return json.loads(Path("configs/phase4a_quality_verifier.json").read_text())


def observation(probability=0.1, **values):
    defaults = {
        "probability_trend": 0.0,
        "movement_velocity": 0.1,
        "mean_visible_confidence": 0.9,
        "missing_keypoint_rate": 0.1,
        "fully_missing_frames": 0,
        "track_coverage": 1.0,
        "track_switches": 0,
    }
    defaults.update(values)
    return Observation(probability=probability, **defaults)


def test_quality_gate_detects_tracking_loss_and_missing_keypoints(config):
    tracking = quality_assessment(observation(track_coverage=0.5), config)
    missing = quality_assessment(observation(
        missing_keypoint_rate=0.6, fully_missing_frames=10,
    ), config)
    assert tracking["low_quality"]
    assert "low_track_coverage" in tracking["reasons"]
    assert missing["low_quality"]
    assert {"high_missing_keypoint_rate", "too_many_fully_missing_frames"} <= set(missing["reasons"])
    identity = quality_assessment(observation(track_switches=3), config)
    assert "identity_discontinuity" in identity["reasons"]


def test_low_quality_is_uncertain_not_normal(config):
    low_quality = observation(probability=0.1, track_coverage=0.5)
    for variant in ("fixed_verification", "quality_aware_verification", "risk_adaptive_verification"):
        assert VerificationMachine(variant, config).step(low_quality)["state"] == "UNCERTAIN"


def test_fixed_state_transitions_and_cooldown(config):
    machine = VerificationMachine("fixed_verification", config)
    assert machine.step(observation(0.1))["state"] == "NORMAL"
    assert machine.step(observation(0.7))["state"] == "CANDIDATE"
    assert machine.step(observation(0.8))["state"] == "ALERT"
    assert machine.step(observation(0.6))["state"] == "ALERT"
    assert machine.step(observation(0.1))["state"] == "COOLDOWN"
    assert machine.step(observation(0.1))["state"] == "COOLDOWN"
    assert machine.step(observation(0.1))["state"] == "NORMAL"


def test_probability_fluctuation_resets_candidate(config):
    machine = VerificationMachine("fixed_verification", config)
    assert machine.step(observation(0.6))["state"] == "CANDIDATE"
    assert machine.step(observation(0.4))["state"] == "NORMAL"
    assert machine.step(observation(0.7))["state"] == "CANDIDATE"


def test_quality_aware_positive_low_quality_abstains(config):
    machine = VerificationMachine("quality_aware_verification", config)
    result = machine.step(observation(0.95, missing_keypoint_rate=0.7))
    assert result["state"] == "UNCERTAIN"


def test_risk_adaptive_high_risk_high_quality_can_confirm_immediately(config):
    obs = observation(0.95, probability_trend=0.3, movement_velocity=0.3)
    assert risk_score(obs, config) >= 0.8
    result = VerificationMachine("risk_adaptive_verification", config).step(obs)
    assert result["state"] == "ALERT"
    assert result["required_confirmation_windows"] == 1


def test_uncertain_is_visible_in_all_conditional_and_strict_metrics():
    result = evaluate_states([0, 1, 1], ["UNCERTAIN", "UNCERTAIN", "ALERT"])
    assert result["uncertain_by_label"] == {"0": 1, "1": 1}
    assert result["coverage"] == pytest.approx(1 / 3)
    assert result["all_sample_alert_decision"]["false_negatives"] == 1
    assert result["strict_abstention_as_error"]["false_positive_equivalents"] == 1
    assert result["strict_abstention_as_error"]["false_negative_equivalents"] == 1
    assert result["conditional_non_abstained"]["n"] == 1
