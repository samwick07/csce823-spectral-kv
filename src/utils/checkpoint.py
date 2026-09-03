"""Checkpoint save/load utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict,
    path: str | Path,
    epoch: int = 0,
    step: int = 0,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state (optional).
        config: Configuration dict.
        path: Output directory path.
        epoch: Current epoch.
        step: Current step.

    Returns:
        Path to the saved checkpoint.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": config,
        "epoch": epoch,
        "step": step,
    }

    ckpt_path = path / f"checkpoint_step{step}.pt"
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # Also save config as JSON for easy inspection
    config_path = path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file or directory.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        device: Device to map tensors to.

    Returns:
        Checkpoint metadata dict.
    """
    path = Path(path)

    if path.is_dir():
        # Find latest checkpoint
        checkpoints = sorted(path.glob("checkpoint_step*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {path}")
        path = checkpoints[-1]

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint: {path} (epoch={checkpoint.get('epoch')}, step={checkpoint.get('step')})")

    return checkpoint
