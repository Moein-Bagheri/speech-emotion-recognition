"""Small reusable helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_directory(path: Path) -> Path:
    """Create *path* (including parents) and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON in a stable, human-readable representation."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu`` or ``cuda`` and fail clearly when needed."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device."
        )
    return torch.device(requested)
