"""Dependency-light binary and subject-cluster metrics."""
from __future__ import annotations

from collections import defaultdict


def _division(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def average_precision(labels, scores) -> float:
    np = __import__("numpy")
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    ranked_scores = scores[order]
    true_positive = np.cumsum(ranked == 1)
    false_positive = np.cumsum(ranked == 0)
    # Evaluate at distinct score thresholds so ties cannot inherit arbitrary
    # input ordering (the majority baseline intentionally has one tied score).
    threshold_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(ranked_scores) - 1]
    tp = true_positive[threshold_ends]
    fp = false_positive[threshold_ends]
    precision = tp / (tp + fp)
    recall = tp / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def binary_metrics(labels, scores, *, threshold: float = 0.5) -> dict:
    np = __import__("numpy")
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("labels and scores must be non-empty matching vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("binary finite labels/scores are required")
    predictions = (scores >= threshold).astype(np.int64)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    recall_0 = _division(tn, tn + fp)
    recall_1 = _division(tp, tp + fn)
    precision_0 = _division(tn, tn + fn)
    precision_1 = _division(tp, tp + fp)
    f1_0 = _division(2 * precision_0 * recall_0, precision_0 + recall_0)
    f1_1 = _division(2 * precision_1 * recall_1, precision_1 + recall_1)
    return {
        "n": len(labels),
        "threshold": threshold,
        "accuracy": _division(tp + tn, len(labels)),
        "balanced_accuracy": (recall_0 + recall_1) / 2,
        "macro_f1": (f1_0 + f1_1) / 2,
        "recall_class_0": recall_0,
        "recall_class_1": recall_1,
        "auprc": average_precision(labels, scores),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def subject_metrics(labels, scores, subjects, *, threshold: float = 0.5) -> dict:
    grouped = defaultdict(lambda: ([], []))
    for label, score, subject in zip(labels, scores, subjects, strict=True):
        grouped[str(subject)][0].append(int(label))
        grouped[str(subject)][1].append(float(score))
    per_subject = {
        subject: binary_metrics(values[0], values[1], threshold=threshold)
        for subject, values in sorted(grouped.items())
    }
    averaged_names = (
        "accuracy", "balanced_accuracy", "macro_f1", "recall_class_0",
        "recall_class_1", "auprc",
    )
    macro = {
        name: sum(row[name] for row in per_subject.values()) / len(per_subject)
        for name in averaged_names
    }
    return {"per_subject": per_subject, "subject_macro": macro}


def subject_block_bootstrap(
    labels, scores, subjects, *, threshold: float = 0.5,
    samples: int = 2000, seed: int = 20261107,
) -> dict:
    """Bootstrap subjects, never individual overlapping windows."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    np = __import__("numpy")
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("At least two subjects are required for clustered bootstrap")
    rng = np.random.default_rng(seed)
    names = ("accuracy", "balanced_accuracy", "macro_f1", "auprc")
    values = {name: [] for name in names}
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sampled_labels = []
        sampled_scores = []
        for subject in selected:
            mask = subjects == subject
            sampled_labels.extend(labels[mask].tolist())
            sampled_scores.extend(scores[mask].tolist())
        row = binary_metrics(sampled_labels, sampled_scores, threshold=threshold)
        for name in names:
            values[name].append(row[name])
    result = {
        "method": "subject-block bootstrap; overlapping windows are never sampled independently",
        "subjects": unique.tolist(),
        "samples": samples,
        "seed": seed,
        "warning": "Only two validation subjects; intervals are descriptive and low-resolution.",
    }
    result.update({
        name: {
            "lower_95": float(np.quantile(values[name], 0.025)),
            "upper_95": float(np.quantile(values[name], 0.975)),
        }
        for name in names
    })
    return result
