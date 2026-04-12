import os
import torch
from pathlib import Path
from typing import Optional

from config import cfg
from logger import get_logger

log = get_logger("checkpoint")


def save_checkpoint(model, optimizer, step: int, tag: Optional[str] = None):
    """Save model and optimizer state."""
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }

    # Save as latest
    latest_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    torch.save(state, latest_path)
    log.info(f"Checkpoint saved: {latest_path} (step {step})")

    # Save as numbered checkpoint
    if tag:
        tagged_path = os.path.join(cfg.checkpoint_dir, f"{tag}.pt")
    else:
        tagged_path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")
    torch.save(state, tagged_path)
    log.info(f"Checkpoint saved: {tagged_path}")


def load_checkpoint(model, optimizer=None, path: Optional[str] = None) -> int:
    """Load model (and optionally optimizer) state. Returns step number."""
    if path is None:
        path = os.path.join(cfg.checkpoint_dir, "latest.pt")

    if not os.path.exists(path):
        log.info(f"No checkpoint found at {path}")
        return 0

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    step = state.get("step", 0)
    log.info(f"Checkpoint loaded from {path} (step {step})")
    return step


def save_best_model(model):
    """Overwrite best model weights."""
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    log.info(f"Best model saved: {path}")


def load_best_model(model) -> bool:
    """Load best model weights. Returns True if loaded."""
    path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    log.info(f"Best model loaded from {path}")
    return True
