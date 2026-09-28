"""Generic checkpoint discovery and completion-state helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from .config import Cfg
from .runtime import logger, safe_torch_load


def training_checkpoint_dir(cfg: Cfg, variant: str) -> str:
    return os.path.join(cfg.output_dir, "checkpoints", variant)


def mark_student_training_complete(
    checkpoint_dir: str,
    objective: str,
    epoch: int,
    best_score: float,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    marker = {
        "objective": objective,
        "epoch": epoch,
        "best_score": best_score,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(checkpoint_dir, "training_complete.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
    logger.info("Marked %s training complete: %s", objective, path)


def student_training_complete(checkpoint_dir: str) -> bool:
    if os.path.exists(os.path.join(checkpoint_dir, "training_complete.json")):
        return True
    latest_metadata = os.path.join(checkpoint_dir, "latest", "student_metadata.json")
    if not os.path.exists(latest_metadata):
        return False
    try:
        with open(latest_metadata, "r", encoding="utf-8") as handle:
            return bool(json.load(handle).get("completed", False))
    except (OSError, ValueError, TypeError):
        return False


def find_resume_checkpoint_dir(checkpoint_dir: str) -> str | None:
    latest = os.path.join(checkpoint_dir, "latest")
    if os.path.exists(os.path.join(latest, "student_metadata.json")):
        return latest
    if os.path.exists(os.path.join(checkpoint_dir, "student_metadata.json")):
        return checkpoint_dir
    return None


def resolve_student_checkpoint_model_dir(
    checkpoint_dir: str,
    prefer_latest: bool = False,
) -> str | None:
    latest = os.path.join(checkpoint_dir, "latest")
    candidates = [latest, checkpoint_dir] if prefer_latest else [checkpoint_dir, latest]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "student_metadata.json")):
            return candidate
        if os.path.exists(os.path.join(candidate, "modules.json")) or os.path.exists(
            os.path.join(candidate, "config_sentence_transformers.json")
        ):
            return candidate
    return None


def load_training_state_payload(
    checkpoint_dir: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    metadata_path = os.path.join(checkpoint_dir, "student_metadata.json")
    payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    if os.path.exists(state_path):
        loaded = safe_torch_load(state_path, map_location="cpu")
        if isinstance(loaded, dict):
            payload = loaded
            metadata = dict(loaded.get("metadata", {}))
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = {**metadata, **json.load(handle)}
    return payload, metadata
