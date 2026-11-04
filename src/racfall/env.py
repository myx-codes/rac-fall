"""Lightweight environment diagnostics without requiring vision packages."""
from __future__ import annotations

import importlib.util
import json
import platform
import sys


def diagnostic() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "ultralytics_installed": importlib.util.find_spec("ultralytics") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "apple_mps_available": False,
    }
    if info["torch_installed"]:
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["apple_mps_available"] = bool(torch.backends.mps.is_available())
            info["mps_built"] = bool(torch.backends.mps.is_built())
        except (ImportError, OSError, AttributeError, RuntimeError) as exc:
            info["torch_diagnostic_error"] = str(exc)
    return info


def print_diagnostic():
    print(json.dumps(diagnostic(), indent=2))
