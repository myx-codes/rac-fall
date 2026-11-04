"""PyTorch Dataset/DataLoader interfaces for generated pose windows."""
from __future__ import annotations

from pathlib import Path


class PoseWindowDataset:
    def __init__(self, path: str | Path, *, split: str | None = None, include_unknown: bool = False):
        try:
            self.np = __import__("numpy")
            self.torch = __import__("torch")
        except ImportError as exc:
            raise RuntimeError("Install NumPy and PyTorch to load pose windows") from exc
        self.data = self.np.load(Path(path), allow_pickle=False)
        if str(self.data["schema_version"]) not in {"racfall.windows.v1", "racfall.windows.v2"}:
            raise ValueError("Unsupported pose-window schema")
        selected = self.np.ones(len(self.data["labels"]), dtype=self.np.bool_)
        if split is not None:
            selected &= self.data["split"] == split
        if not include_unknown:
            selected &= self.data["labels"] >= 0
        self.indices = self.np.flatnonzero(selected)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        index = int(self.indices[item])
        return {
            "features": self.torch.from_numpy(self.data["features"][index]),
            "valid": self.torch.from_numpy(self.data["valid"][index]),
            "label": self.torch.tensor(int(self.data["labels"][index]), dtype=self.torch.long),
            "event_label": self.torch.tensor(
                int(self.data["event_labels"][index]) if "event_labels" in self.data.files else -1,
                dtype=self.torch.long,
            ),
            "timestamp_s": self.torch.from_numpy(self.data["timestamp_s"][index]),
            "video_id": str(self.data["video_id"][index]),
            "subject_id": str(self.data["subject_id"][index]),
            "track_id": int(self.data["track_id"][index]),
            "split": str(self.data["split"][index]),
        }

    def close(self) -> None:
        self.data.close()


def make_dataloader(path: str | Path, *, split: str, batch_size: int = 32,
                    shuffle: bool | None = None, num_workers: int = 0,
                    include_unknown: bool = False):
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid DataLoader configuration")
    torch = __import__("torch")
    dataset = PoseWindowDataset(path, split=split, include_unknown=include_unknown)
    if len(dataset) == 0:
        dataset.close()
        raise ValueError(
            f"No eligible windows for split={split!r}; verify ground-state labels and split assignment"
        )
    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
    )
