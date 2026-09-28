"""Runtime dependencies, logging, reproducibility, and optimizer helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import Cfg

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

try:
    from transformers import Adafactor, get_linear_schedule_with_warmup
except ImportError as exc:  # pragma: no cover
    Adafactor = None  # type: ignore[assignment]
    get_linear_schedule_with_warmup = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger("rcd_retrieval")


def require_runtime_dependencies() -> None:
    missing = []
    if SentenceTransformer is None:
        missing.append(f"sentence-transformers ({_SENTENCE_TRANSFORMERS_IMPORT_ERROR})")
    if get_linear_schedule_with_warmup is None:
        missing.append(f"transformers ({_TRANSFORMERS_IMPORT_ERROR})")
    if missing:
        raise ImportError(
            "Missing required runtime dependencies:\n  - "
            + "\n  - ".join(missing)
            + "\nInstall the retrieval stack with `pip install -e .`."
        )


def cuda_total_memory_gb(cfg: Cfg) -> Optional[float]:
    if not cfg.device.startswith("cuda") or not torch.cuda.is_available():
        return None
    device_idx = torch.cuda.current_device()
    if ":" in cfg.device:
        try:
            device_idx = int(cfg.device.split(":", 1)[1])
        except ValueError:
            device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    return float(props.total_memory) / (1024**3)


def choose_optimizer_name(cfg: Cfg) -> str:
    requested = cfg.optimizer.strip().lower()
    if requested not in {"auto", "adamw", "adafactor"}:
        raise ValueError("--optimizer must be auto, adamw, or adafactor")
    if requested != "auto":
        return requested
    total_gb = cuda_total_memory_gb(cfg)
    if total_gb is not None and total_gb <= float(cfg.auto_adafactor_gpu_gb):
        logger.warning(
            "Detected %.2fGB CUDA device; using Adafactor to reduce optimizer-state memory. "
            "Pass --optimizer adamw to force AdamW.",
            total_gb,
        )
        return "adafactor"
    return "adamw"


def build_optimizer(
    param_groups: List[Dict[str, Any]], cfg: Cfg, context: str
) -> torch.optim.Optimizer:
    name = choose_optimizer_name(cfg)
    if name == "adafactor":
        if Adafactor is None:
            raise ImportError(
                "Adafactor optimizer requested but transformers.Adafactor is unavailable."
            )
        logger.info(
            "%s: optimizer=Adafactor | lr=%s | weight_decay=%s | "
            "scale_parameter=False | relative_step=False",
            context,
            cfg.lr,
            cfg.weight_decay,
        )
        return Adafactor(
            param_groups,
            lr=cfg.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    logger.info("%s: optimizer=AdamW | lr=%s | foreach=False | fused=False", context, cfg.lr)
    try:
        return torch.optim.AdamW(param_groups, lr=cfg.lr, foreach=False, fused=False)
    except TypeError:
        return torch.optim.AdamW(param_groups, lr=cfg.lr)


def setup_logging(cfg: Cfg) -> str:
    log_path = cfg.log_path
    if not log_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(cfg.output_dir, "logs", f"run_{timestamp}.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return log_path


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 96)
    logger.info(title)
    logger.info("=" * 96)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_cuda_cache_if_needed(cfg: Cfg) -> None:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_torch_load(
    path: str | Path, map_location: str | torch.device = "cpu", weights_only: bool = True
) -> Any:
    """Use PyTorch's safer loader when available, with a trusted-checkpoint fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if not weights_only:
            raise
        logger.warning(
            "torch.load(..., weights_only=True) failed for %s (%s). Falling back to "
            "weights_only=False; only do this for trusted local checkpoints.",
            path,
            exc,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def move_features_to_device(features: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in features.items()
    }
