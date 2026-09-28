"""Small experiment helpers shared by distillation objectives."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)


def mean_or_zero(total: float, count: int) -> float:
    return float(total) / float(count) if count > 0 else 0.0


def write_candidate_metadata(output_dir: str, metadata: Mapping[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "candidate_metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True)
    logger.info("Saved candidate metadata: %s", path)


def assert_unlabeled_training_contract(
    *,
    candidate_metadata: Mapping[str, Any],
    examples: Sequence[Any],
    training_qrels_by_qid_idx: Mapping[str, Mapping[int, float]] | None,
) -> None:
    for key in ("qrels_used_in_training", "qrel_positive_injection", "qrel_loss_active"):
        if candidate_metadata.get(key, False):
            raise AssertionError(f"{key} must be False in unlabeled student training")
    if training_qrels_by_qid_idx is not None:
        raise AssertionError("Unlabeled student training must not receive qrels")
    for example in examples:
        if any(float(label) != 0.0 for label in example.labels):
            raise AssertionError(f"Training example {example.qid} contains non-zero labels")


def assert_zero_labels(labels: torch.Tensor) -> None:
    if labels.detach().abs().sum().item() != 0.0:
        raise AssertionError("Unlabeled student loss received non-zero labels")


def log_run_header(cfg: Any, log_path: str, experiment_logger: logging.Logger, script: str) -> None:
    experiment_logger.info("")
    experiment_logger.info("=" * 96)
    experiment_logger.info("timestamp: %s", datetime.now().isoformat(timespec="seconds"))
    experiment_logger.info("RUN HEADER")
    experiment_logger.info("=" * 96)
    experiment_logger.info("script: %s", Path(script).name)
    experiment_logger.info("command: %s", " ".join(sys.argv))
    experiment_logger.info("log_path: %s", log_path)
    experiment_logger.info("config:\n%s", json.dumps(asdict(cfg), indent=2, sort_keys=True))
    experiment_logger.info(
        "dataset_format=%s | source_dataset_format=%s | dataset_name=%s",
        cfg.dataset_format,
        cfg.source_dataset_format or "(none)",
        cfg.dataset_name or "(none)",
    )
    experiment_logger.info(
        "candidate_source=frozen_base_retriever | candidate_pool_k=%s | ks=%s | eval_k=%s",
        cfg.candidate_pool_k,
        list(cfg.ks),
        cfg.eval_k,
    )
    experiment_logger.info(
        "mode=%s | variants=%s | dry_run=%s",
        getattr(cfg, "mode", "(none)"),
        list(getattr(cfg, "variants", ())),
        bool(getattr(cfg, "dry_run", False)),
    )
    experiment_logger.info(
        "teacher_training: loss=%s | negatives=%s | metric_selection=%s",
        getattr(cfg, "teacher_loss", "(none)"),
        getattr(cfg, "teacher_train_num_negatives", "(none)"),
        getattr(cfg, "teacher_metric_select", "(none)"),
    )
    experiment_logger.info("python: %s", sys.version.replace(os.linesep, " "))
    experiment_logger.info("platform: %s", platform.platform())
    experiment_logger.info(
        "torch: %s | cuda_available=%s", torch.__version__, torch.cuda.is_available()
    )
    try:
        import sentence_transformers
        import transformers

        experiment_logger.info(
            "sentence_transformers: %s | transformers: %s",
            getattr(sentence_transformers, "__version__", "unknown"),
            getattr(transformers, "__version__", "unknown"),
        )
    except ImportError:  # pragma: no cover
        experiment_logger.info("sentence_transformers/transformers versions unavailable")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        experiment_logger.info(
            "cuda_device_count=%s | cuda_current_device=%s | cuda_device_name=%s",
            torch.cuda.device_count(),
            device,
            torch.cuda.get_device_name(device),
        )
    experiment_logger.info(
        "memory_guard: l40s_safe_mode=%s | batch_size=%s | micro_batch_size=%s | "
        "train_candidate_cap=%s | eval_batch_size=%s | corpus_encode_batch_size=%s | "
        "gradient_checkpointing=%s | amp=%s | PYTORCH_CUDA_ALLOC_CONF=%s",
        cfg.l40s_safe_mode,
        cfg.batch_size,
        cfg.micro_batch_size,
        min(cfg.candidate_pool_k, cfg.max_train_candidates_per_query),
        cfg.eval_batch_size,
        cfg.corpus_encode_batch_size,
        cfg.gradient_checkpointing,
        cfg.amp,
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=False,
        )
        if root.returncode == 0:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
            )
            status = subprocess.run(
                ["git", "status", "--short"], text=True, capture_output=True, check=False
            )
            experiment_logger.info("git_root: %s", root.stdout.strip())
            experiment_logger.info("git_head: %s", head.stdout.strip())
            experiment_logger.info("git_status:\n%s", status.stdout.strip() or "(clean)")
        else:
            experiment_logger.info("git: current directory is not a git repository")
    except OSError as exc:  # pragma: no cover
        experiment_logger.warning("git summary unavailable: %s", exc)
