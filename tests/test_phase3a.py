from types import SimpleNamespace

import numpy as np
import pytest

from racfall.training.metrics import (
    average_precision, binary_metrics, subject_block_bootstrap, subject_metrics,
)
from racfall.training.models import CausalResidualTCN, StaticPoseBaseline, pose_mask_features
from racfall.training.phase3a import TrainingConfig


def test_causal_tcn_shape_and_future_isolation():
    torch = __import__("torch")
    torch.manual_seed(4)
    model = CausalResidualTCN(channels=8, dilations=(1, 2), dropout=0.0).eval()
    features = torch.randn(2, 12, 51)
    valid = torch.ones(2, 12, 17, dtype=torch.bool)
    changed = features.clone()
    changed[:, 8:] += 100.0
    original_steps = model.forward_sequence(features, valid)
    changed_steps = model.forward_sequence(changed, valid)
    assert tuple(model(features, valid).shape) == (2,)
    assert torch.allclose(original_steps[:, :8], changed_steps[:, :8], atol=1e-6)
    assert not torch.allclose(original_steps[:, -1], changed_steps[:, -1])


def test_static_baseline_uses_endpoint_and_explicit_mask():
    torch = __import__("torch")
    model = StaticPoseBaseline()
    features = torch.zeros(3, 30, 51)
    valid = torch.zeros(3, 30, 17, dtype=torch.bool)
    assert pose_mask_features(features, valid).shape == (3, 30, 68)
    before = model(features, valid)
    features[:, 0] = 99
    assert torch.allclose(before, model(features, valid))
    features[:, -1, 0] = 1
    assert not torch.allclose(before, model(features, valid))


def test_binary_metrics_and_subject_clustering():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.8, 0.7, 0.9]
    result = binary_metrics(labels, scores)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["recall_class_0"] == pytest.approx(0.5)
    assert result["recall_class_1"] == pytest.approx(1.0)
    assert average_precision(labels, scores) == pytest.approx(5 / 6)
    assert average_precision(labels, [0.0] * 4) == pytest.approx(0.5)
    clustered = subject_metrics(labels, scores, ["S1", "S1", "S2", "S2"])
    assert set(clustered["per_subject"]) == {"S1", "S2"}
    bootstrap = subject_block_bootstrap(
        labels, scores, ["S1", "S1", "S2", "S2"], samples=20, seed=2,
    )
    assert bootstrap["samples"] == 20
    assert "overlapping windows" in bootstrap["method"]


def test_training_config_rejects_invalid_values():
    TrainingConfig().validate()
    with pytest.raises(ValueError):
        TrainingConfig(seeds=(1, 1)).validate()
    with pytest.raises(ValueError):
        TrainingConfig(device="cuda").validate()
