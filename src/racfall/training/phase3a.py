"""Reproducible Phase 3A ground-state baseline experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import time

from .metrics import binary_metrics, subject_block_bootstrap, subject_metrics
from .models import CausalResidualTCN, StaticPoseBaseline


SCHEMA_VERSION = "racfall.phase3a-results.v1"


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    channels: int = 64
    dilations: tuple[int, ...] = (1, 2, 4)
    kernel_size: int = 3
    dropout: float = 0.2
    threshold: float = 0.5
    device: str = "mps"
    bootstrap_samples: int = 2000

    def validate(self) -> None:
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds) or
                self.epochs <= 0 or self.patience <= 0 or self.batch_size <= 0 or
                self.learning_rate <= 0 or self.weight_decay < 0 or self.channels <= 0 or
                not self.dilations or self.kernel_size <= 0 or not 0 <= self.dropout < 1 or
                not 0 < self.threshold < 1 or self.bootstrap_samples <= 0):
            raise ValueError("Invalid Phase 3A training configuration")
        if self.device not in {"mps", "cpu"}:
            raise ValueError("device must be mps or cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _synchronize(torch, device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def _loaders(windows: Path, config: TrainingConfig, seed: int):
    torch = __import__("torch")
    from ..data.torch_data import make_dataloader
    generator = torch.Generator().manual_seed(seed)
    train = make_dataloader(
        windows, split="train", batch_size=config.batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )
    val = make_dataloader(
        windows, split="val", batch_size=config.batch_size, shuffle=False, num_workers=0,
    )
    return train, val


def _evaluate(model, loader, device: str, threshold: float) -> tuple[dict, list[dict]]:
    torch = __import__("torch")
    model.eval()
    labels = []
    scores = []
    subjects = []
    failures = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            valid = batch["valid"].to(device)
            logits = model(features, valid)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            batch_labels = batch["label"].cpu().tolist()
            labels.extend(batch_labels)
            scores.extend(probabilities)
            subjects.extend(batch["subject_id"])
            for index, (label, score) in enumerate(zip(batch_labels, probabilities, strict=True)):
                prediction = int(score >= threshold)
                if prediction != int(label):
                    failures.append({
                        "subject_id": batch["subject_id"][index],
                        "video_id": batch["video_id"][index],
                        "start_frame": int(batch["start_frame"][index]),
                        "end_frame": int(batch["end_frame"][index]),
                        "label": int(label), "prediction": prediction,
                        "probability_ground_state_1": float(score),
                    })
    metrics = binary_metrics(labels, scores, threshold=threshold)
    metrics.update(subject_metrics(labels, scores, subjects, threshold=threshold))
    return metrics, failures


def _benchmark(model, loader, device: str, *, repeats: int = 100) -> dict:
    torch = __import__("torch")
    np = __import__("numpy")
    batch = next(iter(loader))
    model.eval()

    def measure(features, valid, count: int) -> dict:
        with torch.inference_mode():
            for _ in range(10):
                model(features, valid)
            _synchronize(torch, device)
            samples = []
            for _ in range(count):
                started = time.perf_counter()
                model(features, valid)
                _synchronize(torch, device)
                samples.append((time.perf_counter() - started) * 1000)
        return {
            "batch_size": int(features.shape[0]),
            "iterations": count,
            "median_batch_ms": float(np.median(samples)),
            "p95_batch_ms": float(np.quantile(samples, 0.95)),
            "median_per_window_ms": float(np.median(samples) / features.shape[0]),
        }

    full_features = batch["features"].to(device)
    full_valid = batch["valid"].to(device)
    return {
        "batch_1": measure(full_features[:1], full_valid[:1], repeats),
        "batch_validation_loader": measure(full_features, full_valid, repeats),
    }


def _train_one(
    name: str, model, windows: Path, output_dir: Path, config: TrainingConfig, seed: int,
) -> dict:
    torch = __import__("torch")
    set_seed(seed, torch)
    device = torch.device(config.device)
    model = model.to(device)
    train_loader, val_loader = _loaders(windows, config, seed)
    train_labels = train_loader.dataset.data["labels"][train_loader.dataset.indices]
    negatives = int((train_labels == 0).sum())
    positives = int((train_labels == 1).sum())
    pos_weight_value = negatives / positives
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    curve = []
    checkpoint_path = output_dir / "checkpoints" / f"{name}-seed-{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            valid = batch["valid"].to(device)
            labels = batch["label"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features, valid), labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(labels)
            train_count += len(labels)
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                features = batch["features"].to(device)
                valid = batch["valid"].to(device)
                labels = batch["label"].float().to(device)
                loss = criterion(model(features, valid), labels)
                val_loss_sum += float(loss.cpu()) * len(labels)
                val_count += len(labels)
        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({
                "schema_version": "racfall.phase3a-checkpoint.v1",
                "model_name": name, "seed": seed, "epoch": epoch,
                "val_loss": val_loss, "state_dict": state,
                "config": asdict(config), "windows_sha256": _sha256(windows),
                "label_semantics": "ground_state_at_causal_window_endpoint",
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    _synchronize(torch, config.device)
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    metrics, failures = _evaluate(model, val_loader, config.device, config.threshold)
    bootstrap = subject_block_bootstrap(
        [int(val_loader.dataset.data["labels"][i]) for i in val_loader.dataset.indices],
        _predict_scores(model, val_loader, config.device),
        [str(val_loader.dataset.data["subject_id"][i]) for i in val_loader.dataset.indices],
        threshold=config.threshold, samples=config.bootstrap_samples, seed=seed + 1000,
    )
    latency = _benchmark(model, val_loader, config.device)
    curve_path = output_dir / "curves" / f"{name}-seed-{seed}.json"
    _atomic_json(curve_path, {"model": name, "seed": seed, "epochs": curve})
    failure_path = output_dir / "failures" / f"{name}-seed-{seed}.json"
    _atomic_json(failure_path, {"model": name, "seed": seed, "failures": failures})
    train_loader.dataset.close()
    val_loader.dataset.close()
    return {
        "model": name, "seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs_ran": len(curve), "best_epoch": best_epoch, "best_val_loss": best_loss,
        "early_stopped": len(curve) < config.epochs, "training_elapsed_s": elapsed,
        "pos_weight": pos_weight_value, "metrics": metrics, "subject_block_bootstrap": bootstrap,
        "inference_latency": latency, "failure_count": len(failures),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "curve": str(curve_path), "failures": str(failure_path),
    }


def _predict_scores(model, loader, device: str) -> list[float]:
    torch = __import__("torch")
    scores = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            scores.extend(torch.sigmoid(model(
                batch["features"].to(device), batch["valid"].to(device),
            )).cpu().tolist())
    return scores


def _aggregate(runs: list[dict]) -> dict:
    np = __import__("numpy")
    names = (
        "accuracy", "balanced_accuracy", "macro_f1", "recall_class_0",
        "recall_class_1", "auprc",
    )
    return {
        name: {
            "mean": float(np.mean([run["metrics"][name] for run in runs])),
            "std": float(np.std([run["metrics"][name] for run in runs], ddof=0)),
        }
        for name in names
    }


def run_phase3a(
    windows: str | Path, output_dir: str | Path, *, config: TrainingConfig = TrainingConfig(),
) -> dict:
    config.validate()
    torch = __import__("torch")
    np = __import__("numpy")
    windows_path = Path(windows).resolve()
    destination = Path(output_dir).resolve()
    if not windows_path.is_file():
        raise FileNotFoundError(windows_path)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required but unavailable")
    with np.load(windows_path, allow_pickle=False) as data:
        if str(data["label_semantics"]) != "ground_state_at_causal_window_endpoint":
            raise ValueError("Phase 3A requires causal ground-state endpoint labels")
        if (set(data["split"].tolist()) != {"train", "val"} or
                {"Subject.9", "Subject.10"} & set(data["subject_id"].tolist()) or
                (data["labels"] < 0).any()):
            raise ValueError("Only unsealed binary train/validation windows are allowed")
        train_mask = data["split"] == "train"
        val_mask = data["split"] == "val"
        expected_train = {"Subject.1", "Subject.2", "Subject.4", "Subject.5", "Subject.6", "Subject.8"}
        expected_val = {"Subject.3", "Subject.7"}
        if (set(data["dataset"].tolist()) != {"CAUCAFall"} or
                set(data["subject_id"][train_mask].tolist()) != expected_train or
                set(data["subject_id"][val_mask].tolist()) != expected_val):
            raise ValueError("Phase 3A frozen train/validation subject split does not match")
        train_labels = data["labels"][train_mask]
        val_labels = data["labels"][val_mask]
        val_subjects = data["subject_id"][val_mask].tolist()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "config.json", {
        "schema_version": SCHEMA_VERSION, "config": asdict(config),
        "windows": str(windows_path), "windows_sha256": _sha256(windows_path),
        "train_windows": int(len(train_labels)), "val_windows": int(len(val_labels)),
        "train_class_counts": {"0": int((train_labels == 0).sum()), "1": int((train_labels == 1).sum())},
        "val_class_counts": {"0": int((val_labels == 0).sum()), "1": int((val_labels == 1).sum())},
    })
    majority_class = int((train_labels == 1).sum() > (train_labels == 0).sum())
    majority_scores = [float(majority_class)] * len(val_labels)
    majority = binary_metrics(val_labels, majority_scores, threshold=config.threshold)
    majority.update(subject_metrics(val_labels, majority_scores, val_subjects, threshold=config.threshold))
    majority["subject_block_bootstrap"] = subject_block_bootstrap(
        val_labels, majority_scores, val_subjects, threshold=config.threshold,
        samples=config.bootstrap_samples, seed=20261107,
    )
    tcn_runs = []
    static_runs = []
    experiment_started = time.perf_counter()
    for seed in config.seeds:
        set_seed(seed, torch)
        tcn_runs.append(_train_one(
            "causal_tcn", CausalResidualTCN(
                channels=config.channels, dilations=config.dilations,
                kernel_size=config.kernel_size, dropout=config.dropout,
            ), windows_path, destination, config, seed,
        ))
        set_seed(seed, torch)
        static_runs.append(_train_one(
            "static_pose", StaticPoseBaseline(), windows_path, destination, config, seed,
        ))
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": "causal ground-state frame classification; not fall-event detection",
        "windows": str(windows_path), "windows_sha256": _sha256(windows_path),
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
        },
        "dataset": {
            "train_windows": len(train_labels), "val_windows": len(val_labels),
            "train_class_counts": {"0": int((train_labels == 0).sum()), "1": int((train_labels == 1).sum())},
            "val_class_counts": {"0": int((val_labels == 0).sum()), "1": int((val_labels == 1).sum())},
            "validation_subjects": sorted(set(val_subjects)),
            "sealed_subjects_accessed": [], "gmdcsa_accessed": False,
        },
        "majority_class": majority_class, "majority_baseline": majority,
        "causal_tcn": {"runs": tcn_runs, "aggregate": _aggregate(tcn_runs)},
        "static_pose": {"runs": static_runs, "aggregate": _aggregate(static_runs)},
        "experiment_elapsed_s": time.perf_counter() - experiment_started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    _atomic_json(destination / "results.json", result)
    return result
