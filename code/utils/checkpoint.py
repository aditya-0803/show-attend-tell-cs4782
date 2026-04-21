"""Minimal checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: Any = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
