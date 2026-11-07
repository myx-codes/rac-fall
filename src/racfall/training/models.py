"""Lightweight causal models for normalized COCO pose windows."""
from __future__ import annotations


def _torch_modules():
    torch = __import__("torch")
    return torch, torch.nn, torch.nn.functional


torch, nn, F = _torch_modules()


class CausalConv1d(nn.Module):
    """A Conv1d whose output at t never reads input after t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        if kernel_size <= 0 or dilation <= 0:
            raise ValueError("kernel_size and dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0,
        )

    def forward(self, values):
        return self.conv(F.pad(values, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, kernel_size: int = 3,
        dilation: int = 1, dropout: float = 0.2,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values):
        residual = self.residual(values)
        hidden = self.dropout(F.relu(self.conv1(values)))
        hidden = self.dropout(F.relu(self.conv2(hidden)))
        return F.relu(hidden + residual)


def pose_mask_features(features, valid):
    """Append explicit 17-keypoint validity channels to 51 pose channels."""
    if features.ndim != 3 or valid.ndim != 3:
        raise ValueError("features and valid must be [batch, time, channels]")
    if features.shape[:2] != valid.shape[:2] or features.shape[-1] != 51 or valid.shape[-1] != 17:
        raise ValueError("Expected features [B,T,51] and valid [B,T,17]")
    return torch.cat((features.float(), valid.float()), dim=-1)


class CausalResidualTCN(nn.Module):
    def __init__(
        self, *, input_channels: int = 68, channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4), kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if input_channels <= 0 or channels <= 0 or not dilations:
            raise ValueError("input_channels, channels and dilations must be non-empty/positive")
        blocks = []
        current = input_channels
        for dilation in dilations:
            blocks.append(CausalResidualBlock(
                current, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout,
            ))
            current = channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)
        self.input_channels = input_channels
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward_sequence(self, features, valid):
        values = pose_mask_features(features, valid).transpose(1, 2)
        if values.shape[1] != self.input_channels:
            raise ValueError("Unexpected model input channel count")
        return self.head(self.network(values)).squeeze(1)

    def forward(self, features, valid):
        return self.forward_sequence(features, valid)[:, -1]


class StaticPoseBaseline(nn.Module):
    """Linear endpoint-only baseline; it has no access to earlier or future poses."""

    def __init__(self, input_channels: int = 68):
        super().__init__()
        self.classifier = nn.Linear(input_channels, 1)

    def forward(self, features, valid):
        endpoint = pose_mask_features(features, valid)[:, -1]
        return self.classifier(endpoint).squeeze(-1)
