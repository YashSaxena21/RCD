#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EmbedDistill baseline under the same qrel-free student-training contract as RCD.

It reuses the existing implementation's:
- dataset loading, preparation, and split handling
- ComLQ and LegalBench-RAG suite orchestration
- frozen base-retriever embedding caches
- adapter-teacher checkpoint loading
- trainable student encoder
- validation / final evaluation helpers
- checkpoint-resume layout and multi-seed aggregation

Scientific intent:
- isolate supervision design while keeping the same student, teacher,
  candidates, retrieval scoring, and evaluation stack
- compare final-embedding matching against teacher-induced correction matching

Primary training objective:
- teacher targets: q_mod_T and d_mod_T only
- optional teacher-score KL distillation over the same valid candidate set
- no qrels in student candidate construction or student losses
- no delta supervision
- the projection head is training-only and not used at retrieval inference time

Examples:
  rcd-train embeddistill --help
  rcd-train embeddistill --run_loss_tests
  rcd-train embeddistill --dry_run \
      --dataset_format comlq --dataset_dir datasets/comlq/dataset \
      --teacher_checkpoint_path /path/to/imrnn_teacher.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import core as base
from . import experiment_utils, provenance
from .data_utils import build_eval_examples
from .legalbench import DATASETS as LEGALBENCH_RAG_DATASETS
from .objectives import ranking_kl_per_example

try:
    from imrnns.checkpoints import sanitize_legacy_state_dict
except ImportError:  # pragma: no cover
    sanitize_legacy_state_dict = None  # type: ignore[assignment]

logger = base.logger
assert_unsupervised_training_contract = experiment_utils.assert_unlabeled_training_contract
assert_zero_labels = experiment_utils.assert_zero_labels
mean_or_zero = experiment_utils.mean_or_zero
write_candidate_metadata = experiment_utils.write_candidate_metadata

EMBED_DISTANCE_L2 = "l2"
EMBED_DISTANCE_COSINE = "cosine"
EMBED_DISTANCE_MSE = "mse"
EMBEDDISTILL_VARIANT_RANK_EMBED = "rank_embed"
EMBEDDISTILL_VARIANT_EMBED_ONLY = "embed_only"
OBJECTIVE_EMBEDDISTILL = "embeddistill_style"
SYSTEM_EMBEDDISTILL = "EMBEDDISTILL"
SYSTEM_EMBEDDISTILL_EMB_ONLY = "EMBEDDISTILL_EMB_ONLY"
USED_TEACHER_TARGET_FIELDS: Tuple[str, ...] = ("q_mod_T", "d_mod_T", "teacher_scores")


@dataclass
class EmbedDistillCfg(base.Cfg):
    output_dir: str = "runs/embeddistill"
    auto_train_teacher_if_missing: bool = False
    variants: Tuple[str, ...] = (SYSTEM_EMBEDDISTILL,)
    embeddistill_variant: str = EMBEDDISTILL_VARIANT_RANK_EMBED
    embed_distance: str = EMBED_DISTANCE_L2
    embed_query_weight: float = 1.0
    embed_document_weight: float = 1.0
    embed_loss_weight: float = 1.0
    projection_bias: bool = True
    teacher_checkpoint_template: str = ""
    allow_teacher_metadata_mismatch: bool = False
    run_loss_tests: bool = False


@dataclass
class TeacherFinalTargets:
    teacher_query_embedding: torch.Tensor
    teacher_document_embeddings: torch.Tensor
    teacher_scores: torch.Tensor
    teacher_output_dim: int
    used_fields: Tuple[str, ...] = USED_TEACHER_TARGET_FIELDS


@dataclass
class EmbedAlignmentStats:
    total: torch.Tensor
    query_loss_per_query: torch.Tensor
    document_loss_per_query: torch.Tensor
    combined_loss_per_query: torch.Tensor
    valid_query_mask: torch.Tensor
    valid_query_count: torch.Tensor
    valid_document_count_total: torch.Tensor
    query_alignment_cos_sum: torch.Tensor
    document_alignment_cos_sum: torch.Tensor
    teacher_query_norm_sum: torch.Tensor
    teacher_document_norm_sum: torch.Tensor
    student_query_norm_sum: torch.Tensor
    student_document_norm_sum: torch.Tensor


@dataclass
class EpochMeters:
    total_loss_sum: float = 0.0
    score_kd_loss_sum: float = 0.0
    query_embed_loss_sum: float = 0.0
    document_embed_loss_sum: float = 0.0
    combined_embed_loss_sum: float = 0.0
    query_alignment_cos_sum: float = 0.0
    document_alignment_cos_sum: float = 0.0
    teacher_query_norm_sum: float = 0.0
    teacher_document_norm_sum: float = 0.0
    student_query_norm_sum: float = 0.0
    student_document_norm_sum: float = 0.0
    student_query_drift_sum: float = 0.0
    valid_query_count: int = 0
    valid_document_count: int = 0
    skipped_outer_batches: int = 0

    def update(self, loss_info: Dict[str, torch.Tensor]) -> None:
        self.total_loss_sum += float(loss_info["total_loss_sum"].detach().cpu())
        self.score_kd_loss_sum += float(loss_info["score_kd_loss_sum"].detach().cpu())
        self.query_embed_loss_sum += float(loss_info["query_embed_loss_sum"].detach().cpu())
        self.document_embed_loss_sum += float(loss_info["document_embed_loss_sum"].detach().cpu())
        self.combined_embed_loss_sum += float(loss_info["combined_embed_loss_sum"].detach().cpu())
        self.query_alignment_cos_sum += float(loss_info["query_alignment_cos_sum"].detach().cpu())
        self.document_alignment_cos_sum += float(
            loss_info["document_alignment_cos_sum"].detach().cpu()
        )
        self.teacher_query_norm_sum += float(loss_info["teacher_query_norm_sum"].detach().cpu())
        self.teacher_document_norm_sum += float(
            loss_info["teacher_document_norm_sum"].detach().cpu()
        )
        self.student_query_norm_sum += float(loss_info["student_query_norm_sum"].detach().cpu())
        self.student_document_norm_sum += float(
            loss_info["student_document_norm_sum"].detach().cpu()
        )
        self.student_query_drift_sum += float(loss_info["student_query_drift_sum"].detach().cpu())
        self.valid_query_count += int(loss_info["valid_query_count"].detach().cpu())
        self.valid_document_count += int(loss_info["valid_document_count"].detach().cpu())


class SharedLinearProjection(nn.Module):
    """Training-only shared linear projection into the teacher scoring space."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.bias = bool(bias)
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=self.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


class NoDeltaAccessSignals(dict):
    """Test helper that raises if delta teacher fields are accessed."""

    def __getitem__(self, key: str) -> Any:
        if key in {"delta_q_T", "delta_d_T"}:
            raise AssertionError(f"delta field access is forbidden in EmbedDistill baseline: {key}")
        return super().__getitem__(key)


def system_name_for_cfg(cfg: EmbedDistillCfg) -> str:
    return (
        SYSTEM_EMBEDDISTILL
        if cfg.embeddistill_variant == EMBEDDISTILL_VARIANT_RANK_EMBED
        else SYSTEM_EMBEDDISTILL_EMB_ONLY
    )


def sqrt_l2_param_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        total += float(param.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def projection_parameter_norm(projection: SharedLinearProjection) -> float:
    return sqrt_l2_param_norm(list(projection.parameters()))


def normalize_key(text: Any) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def inferred_dataset_identity(cfg: EmbedDistillCfg) -> str:
    if cfg.dataset_name:
        return str(cfg.dataset_name)
    dataset_dir = Path(cfg.dataset_dir)
    dataset_name = dataset_dir.name
    if (
        dataset_name.lower() in {"dataset", "data", "prepared", "prepared_data"}
        and dataset_dir.parent.name
    ):
        dataset_name = dataset_dir.parent.name
    if dataset_name:
        return dataset_name
    dataset_format = cfg.source_dataset_format or cfg.dataset_format
    return str(dataset_format)


def student_encoder_label(model_name: str) -> str:
    lowered = str(model_name).lower()
    if "e5" in lowered:
        return "e5"
    return normalize_key(Path(str(model_name)).name)


def canonical_encoder_label(value: Any) -> str:
    """Normalize known encoder aliases without weakening model compatibility."""
    normalized = normalize_key(value)
    qwen3_embedding_aliases = {
        "qwen3embedding",
        "qwen3embedding06b",
        "qwenqwen3embedding06b",
    }
    if normalized in qwen3_embedding_aliases:
        return "qwen3embedding06b"
    return normalized


def current_dataset_formats(cfg: EmbedDistillCfg) -> set[str]:
    values = {normalize_key(cfg.dataset_format)}
    if cfg.source_dataset_format:
        values.add(normalize_key(cfg.source_dataset_format))
    return {value for value in values if value}


def resolve_teacher_spec_for_cfg(cfg: EmbedDistillCfg) -> str:
    template = str(cfg.teacher_checkpoint_template).strip()
    raw = str(cfg.teacher_checkpoint_path).strip()
    if template and raw:
        raise ValueError(
            "Provide either --teacher_checkpoint_path or --teacher_checkpoint_template, not both."
        )
    if template:
        dataset = inferred_dataset_identity(cfg)
        try:
            resolved = template.format(dataset=dataset, seed=cfg.seed)
        except KeyError as exc:
            raise ValueError(
                f"--teacher_checkpoint_template contains unsupported placeholder {exc!s}. "
                "Supported placeholders are {dataset} and {seed}."
            ) from exc
        if not resolved.strip():
            raise ValueError("Resolved --teacher_checkpoint_template to an empty path.")
        logger.info(
            f"Resolved teacher checkpoint template: dataset={dataset} | seed={cfg.seed} | path={resolved}"
        )
        return resolved
    if raw:
        return raw
    raise ValueError(
        "This baseline requires an explicit teacher specification. Pass either "
        "--teacher_checkpoint_path or --teacher_checkpoint_template."
    )


def load_teacher_checkpoint_metadata(
    path: str | Path,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    payload = base.safe_torch_load(path, map_location="cpu", weights_only=False)
    metadata: Dict[str, Any]
    state_dict: Dict[str, torch.Tensor]
    if isinstance(payload, dict) and "model_state" in payload:
        metadata = dict(payload.get("metadata", {}))
        state_dict = dict(payload["model_state"])
    elif isinstance(payload, dict) and "state_dict" in payload:
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
        state_dict = dict(payload["state_dict"])
    elif isinstance(payload, dict):
        metadata = {}
        state_dict = dict(payload)
    else:
        raise TypeError(f"Unsupported teacher checkpoint format: {path}")
    if sanitize_legacy_state_dict is not None:
        state_dict = sanitize_legacy_state_dict(state_dict)
    return metadata, state_dict


def assert_teacher_metadata_compatible(
    cfg: EmbedDistillCfg,
    metadata: Mapping[str, Any],
    teacher_output_dim: int,
    teacher_path: str | Path,
) -> None:
    mismatches: List[str] = []
    dataset_now = normalize_key(inferred_dataset_identity(cfg))
    dataset_meta_values = [metadata.get("dataset_name"), metadata.get("dataset")]
    dataset_meta_norm = [
        normalize_key(value) for value in dataset_meta_values if value not in (None, "")
    ]
    if dataset_now and dataset_meta_norm and dataset_now not in dataset_meta_norm:
        mismatches.append(
            f"dataset mismatch: current={inferred_dataset_identity(cfg)!r} vs metadata={dataset_meta_values}"
        )

    if metadata.get("seed") not in (None, ""):
        try:
            meta_seed = int(metadata["seed"])
            if meta_seed != int(cfg.seed):
                mismatches.append(f"seed mismatch: current={cfg.seed} vs metadata={meta_seed}")
        except Exception:
            mismatches.append(f"seed metadata is not parseable as int: {metadata.get('seed')!r}")

    meta_encoder_model = str(metadata.get("encoder_model_name", "")).strip()
    if meta_encoder_model and normalize_key(meta_encoder_model) != normalize_key(cfg.model_name):
        mismatches.append(
            f"encoder_model_name mismatch: current={cfg.model_name!r} vs metadata={meta_encoder_model!r}"
        )
    else:
        meta_encoder = str(
            metadata.get("normalized_encoder") or metadata.get("encoder") or ""
        ).strip()
        if meta_encoder and canonical_encoder_label(meta_encoder) != canonical_encoder_label(
            student_encoder_label(cfg.model_name)
        ):
            mismatches.append(
                f"encoder mismatch: current={student_encoder_label(cfg.model_name)!r} vs metadata={meta_encoder!r}"
            )

    meta_output_dim = metadata.get("model_config", {}).get("output_dim", metadata.get("output_dim"))
    if meta_output_dim not in (None, ""):
        try:
            if int(meta_output_dim) != int(teacher_output_dim):
                mismatches.append(
                    f"teacher output dim mismatch: current={teacher_output_dim} vs metadata={meta_output_dim}"
                )
        except Exception:
            mismatches.append(
                f"teacher output dim metadata is not parseable as int: {meta_output_dim!r}"
            )

    meta_dataset_format = str(metadata.get("dataset_format", "")).strip()
    if meta_dataset_format:
        expected_formats = current_dataset_formats(cfg)
        if normalize_key(meta_dataset_format) not in expected_formats:
            mismatches.append(
                f"dataset_format mismatch: current={sorted(expected_formats)} vs metadata={meta_dataset_format!r}"
            )

    meta_dataset_dir = str(metadata.get("dataset_dir", "")).strip()
    if meta_dataset_dir and not dataset_meta_norm and not meta_dataset_format:
        current_dir_name = normalize_key(Path(cfg.dataset_dir).name)
        meta_dir_name = normalize_key(Path(meta_dataset_dir).name)
        if current_dir_name and meta_dir_name and current_dir_name != meta_dir_name:
            mismatches.append(
                f"dataset_dir basename mismatch: current={cfg.dataset_dir!r} vs metadata={meta_dataset_dir!r}"
            )

    if mismatches:
        message = (
            f"Teacher checkpoint metadata mismatch for {teacher_path}:\n- "
            + "\n- ".join(mismatches)
            + "\nPass --allow_teacher_metadata_mismatch only if you have verified this is intentional."
        )
        if cfg.allow_teacher_metadata_mismatch:
            logger.warning(message)
        else:
            raise ValueError(message)


def log_objective_summary(cfg: EmbedDistillCfg, projection_output_dim: int) -> None:
    base.log_section("OBJECTIVE SUMMARY")
    logger.info("objective: EmbedDistill-style final-embedding matching")
    logger.info("teacher_targets: q_mod_T and d_mod_T")
    logger.info(
        f"score_distillation: {str(cfg.embeddistill_variant == EMBEDDISTILL_VARIANT_RANK_EMBED).lower()}"
    )
    logger.info(f"distance: {cfg.embed_distance}")
    logger.info(f"query_weight: {cfg.embed_query_weight}")
    logger.info(f"document_weight: {cfg.embed_document_weight}")
    logger.info(f"embedding_loss_weight: {cfg.embed_loss_weight}")
    logger.info("projection_type: shared linear")
    logger.info(f"projection_input_dim: {cfg.embedding_dim}")
    logger.info(f"projection_output_dim: {projection_output_dim}")
    logger.info("qrels_used_for_student_loss: false")
    logger.info("delta_supervision: false")
    logger.info("query_generation: false")
    logger.info("teacher_document_inheritance: false")
    logger.info("student_inference_only: true")


def validate_embeddistill_cfg(cfg: EmbedDistillCfg) -> None:
    if cfg.mode not in {base.MODE_UNSUPERVISED, base.MODE_EVAL_ONLY}:
        raise ValueError("EmbedDistill baseline supports only UNSUPERVISED or EVAL_ONLY mode.")
    if cfg.embeddistill_variant not in {
        EMBEDDISTILL_VARIANT_RANK_EMBED,
        EMBEDDISTILL_VARIANT_EMBED_ONLY,
    }:
        raise ValueError("--embeddistill_variant must be rank_embed or embed_only")
    if cfg.embed_distance not in {EMBED_DISTANCE_L2, EMBED_DISTANCE_COSINE, EMBED_DISTANCE_MSE}:
        raise ValueError("--embed_distance must be l2, cosine, or mse")
    if cfg.dataset_format not in {"auto", "standard", "comlq", "legalbench_rag"}:
        raise ValueError("--dataset_format must be auto, standard, comlq, or legalbench_rag")
    if cfg.query_type_filter not in {
        "all",
        "negation",
        "conjunction",
        "union",
        "projection",
        "custom",
    }:
        raise ValueError(
            "--query_type_filter must be all, negation, conjunction, union, projection, or custom"
        )
    if cfg.optimizer not in {"auto", "adamw", "adafactor"}:
        raise ValueError("--optimizer must be auto, adamw, or adafactor")
    if cfg.teacher_loss != "listnet":
        raise ValueError("EmbedDistill requires the ListNet IMRNNS teacher")
    if cfg.teacher_label_temp <= 0:
        raise ValueError("--teacher_label_temp must be positive")
    if cfg.teacher_max_positives < 0 or cfg.teacher_max_candidates < 0:
        raise ValueError(
            "--teacher_max_positives and --teacher_max_candidates must be non-negative"
        )
    for name in ("teacher_select_recall_k", "teacher_select_ndcg_k", "teacher_select_mrr_k"):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if cfg.embed_query_weight < 0 or cfg.embed_document_weight < 0 or cfg.embed_loss_weight < 0:
        raise ValueError("EmbedDistill weights must be non-negative")
    if not cfg.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if cfg.batch_size < 1 or cfg.micro_batch_size < 1:
        raise ValueError("--batch_size and --micro_batch_size must be positive")
    if cfg.candidate_pool_k < 1 or cfg.max_train_candidates_per_query < 1:
        raise ValueError("--candidate_pool_k and --max_train_candidates_per_query must be positive")
    if cfg.eval_batch_size < 1 or cfg.corpus_encode_batch_size < 1:
        raise ValueError("--eval_batch_size and --corpus_encode_batch_size must be positive")
    if cfg.legalbench_chunk_strategy not in {"naive", "rcts"}:
        raise ValueError("--legalbench_chunk_strategy must be naive or rcts")
    if cfg.legalbench_chunk_size < 1:
        raise ValueError("--legalbench_chunk_size must be positive")
    if cfg.teacher_train_epochs < 1 or cfg.teacher_train_batch_size < 1:
        raise ValueError("--teacher_train_epochs and --teacher_train_batch_size must be positive")
    if cfg.teacher_train_num_negatives < 1:
        raise ValueError("--teacher_train_num_negatives must be positive")
    if cfg.teacher_train_negative_pool < cfg.teacher_train_num_negatives:
        logger.warning(
            f"teacher_train_negative_pool={cfg.teacher_train_negative_pool} is smaller than "
            f"teacher_train_num_negatives={cfg.teacher_train_num_negatives}; increasing pool."
        )
        cfg.teacher_train_negative_pool = cfg.teacher_train_num_negatives
    if cfg.eval_k <= 0:
        cfg.eval_k = 50 if 50 in cfg.ks else (10 if 10 in cfg.ks else max(cfg.ks))
    if cfg.feedback_k < max(cfg.ks):
        logger.warning(
            f"feedback_k={cfg.feedback_k} is smaller than max ks={max(cfg.ks)}; reranking will use max ks"
        )
    if cfg.auto_train_teacher_if_missing:
        logger.warning(
            "Teacher auto-training has been explicitly enabled. By default this baseline fails if the requested "
            "teacher checkpoint is missing to preserve comparability."
        )
    if cfg.l40s_safe_mode:
        if cfg.batch_size > 1:
            logger.warning(f"L40S safe mode reducing batch_size from {cfg.batch_size} to 1")
            cfg.batch_size = 1
        if cfg.micro_batch_size > 1:
            logger.warning(
                f"L40S safe mode reducing micro_batch_size from {cfg.micro_batch_size} to 1"
            )
            cfg.micro_batch_size = 1
        if cfg.max_train_candidates_per_query > 32:
            logger.warning(
                f"L40S safe mode reducing max_train_candidates_per_query from {cfg.max_train_candidates_per_query} to 32"
            )
            cfg.max_train_candidates_per_query = 32
        if cfg.corpus_encode_batch_size > 64:
            logger.warning(
                f"L40S safe mode reducing corpus_encode_batch_size from {cfg.corpus_encode_batch_size} to 64"
            )
            cfg.corpus_encode_batch_size = 64
        if cfg.eval_batch_size > 32:
            logger.warning(
                f"L40S safe mode reducing eval_batch_size from {cfg.eval_batch_size} to 32"
            )
            cfg.eval_batch_size = 32
        logger.info(
            "L40S safe mode active: batch_size=1, micro_batch_size=1, "
            f"max_train_candidates_per_query={cfg.max_train_candidates_per_query}, AMP={cfg.amp}, "
            f"gradient_checkpointing={cfg.gradient_checkpointing}, "
            f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}"
        )
    cfg.variants = (system_name_for_cfg(cfg),)


def augment_candidate_metadata(
    cfg: EmbedDistillCfg,
    candidate_metadata: Dict[str, Any],
    examples: Sequence[base.TrainExample],
    passages: Sequence[base.PassageRecord],
) -> Dict[str, Any]:
    out = provenance.augment_candidate_metadata(candidate_metadata, examples, passages)
    out.update(
        {
            "objective": OBJECTIVE_EMBEDDISTILL,
            "embeddistill_variant": cfg.embeddistill_variant,
            "embed_distance": cfg.embed_distance,
            "embed_query_weight": cfg.embed_query_weight,
            "embed_document_weight": cfg.embed_document_weight,
            "embed_loss_weight": cfg.embed_loss_weight,
            "candidate_source": "frozen_base_retriever",
            "dynamic_refresh": False,
            "max_train_candidates_per_query": int(cfg.max_train_candidates_per_query),
            "candidate_pool_k": int(cfg.candidate_pool_k),
        }
    )
    return out


def safe_masked_tensor(
    values: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0
) -> torch.Tensor:
    fill = torch.full_like(values, float(fill_value), dtype=torch.float32)
    return torch.where(mask, values.to(dtype=torch.float32), fill)


def select_final_teacher_targets(signals: Mapping[str, torch.Tensor]) -> TeacherFinalTargets:
    q_mod = signals["q_mod_T"]
    d_mod = signals["d_mod_T"]
    teacher_scores = signals["teacher_scores"]
    if q_mod.shape[-1] != d_mod.shape[-1]:
        raise ValueError(
            f"Teacher query/document target dimensions must match, got q_mod_T={tuple(q_mod.shape)} and d_mod_T={tuple(d_mod.shape)}"
        )
    return TeacherFinalTargets(
        teacher_query_embedding=q_mod.detach().float(),
        teacher_document_embeddings=d_mod.detach().float(),
        teacher_scores=teacher_scores.detach().float(),
        teacher_output_dim=int(q_mod.shape[-1]),
    )


@torch.no_grad()
def extract_final_teacher_embeddings_and_scores(
    teacher: base.TeacherAdapter,
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    mask: torch.Tensor,
) -> TeacherFinalTargets:
    signals = teacher.extract_modulation_signals(query_embeddings, document_embeddings, mask)
    targets = select_final_teacher_targets(signals)
    if targets.used_fields != USED_TEACHER_TARGET_FIELDS:
        raise AssertionError(f"Unexpected teacher target fields: {targets.used_fields}")
    return targets


def embedding_distance_per_vector(
    student: torch.Tensor, teacher: torch.Tensor, distance: str
) -> torch.Tensor:
    if distance == EMBED_DISTANCE_L2:
        return torch.linalg.vector_norm(student - teacher, ord=2, dim=-1)
    if distance == EMBED_DISTANCE_COSINE:
        return 1.0 - F.cosine_similarity(student, teacher, dim=-1, eps=1e-8)
    if distance == EMBED_DISTANCE_MSE:
        return (student - teacher).pow(2).mean(dim=-1)
    raise ValueError(f"Unknown embed distance: {distance}")


def embeddistill_alignment_per_example(
    projected_student_query: torch.Tensor,
    projected_student_documents: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_documents: torch.Tensor,
    document_mask: torch.Tensor,
    distance: str = EMBED_DISTANCE_L2,
    query_weight: float = 1.0,
    document_weight: float = 1.0,
) -> EmbedAlignmentStats:
    doc_mask = document_mask.to(device=projected_student_documents.device, dtype=torch.bool)
    student_q = projected_student_query.to(dtype=torch.float32)
    student_d = projected_student_documents.to(dtype=torch.float32)
    teacher_q = teacher_query.detach().to(device=student_q.device, dtype=torch.float32)
    teacher_d = teacher_documents.detach().to(device=student_d.device, dtype=torch.float32)

    if student_q.shape != teacher_q.shape:
        raise ValueError(
            f"Projected student query shape {tuple(student_q.shape)} must match teacher query shape {tuple(teacher_q.shape)}"
        )
    if student_d.shape != teacher_d.shape:
        raise ValueError(
            f"Projected student doc shape {tuple(student_d.shape)} must match teacher doc shape {tuple(teacher_d.shape)}"
        )
    if teacher_q.shape[-1] != teacher_d.shape[-1]:
        raise ValueError(
            f"Teacher query/document dims must match, got {teacher_q.shape[-1]} and {teacher_d.shape[-1]}"
        )

    valid_doc_count = doc_mask.sum(dim=-1)
    valid_query_mask = valid_doc_count >= 1
    safe_teacher_d = safe_masked_tensor(teacher_d, doc_mask.unsqueeze(-1), fill_value=0.0)
    safe_student_d = safe_masked_tensor(student_d, doc_mask.unsqueeze(-1), fill_value=0.0)

    query_loss = embedding_distance_per_vector(student_q, teacher_q, distance)
    doc_distance = embedding_distance_per_vector(safe_student_d, safe_teacher_d, distance)
    doc_loss = (doc_distance * doc_mask.float()).sum(dim=-1) / valid_doc_count.float().clamp_min(
        1.0
    )
    combined = float(query_weight) * query_loss + float(document_weight) * doc_loss

    zero_q = torch.zeros_like(query_loss)
    zero_d = torch.zeros_like(doc_loss)
    zero_c = torch.zeros_like(combined)
    query_loss = torch.where(valid_query_mask, query_loss, zero_q)
    doc_loss = torch.where(valid_query_mask, doc_loss, zero_d)
    combined = torch.where(valid_query_mask, combined, zero_c)

    if valid_query_mask.any():
        total = combined[valid_query_mask].mean()
    else:
        total = student_q.sum() * 0.0

    query_alignment_cos = F.cosine_similarity(student_q, teacher_q, dim=-1, eps=1e-8)
    doc_alignment_cos = F.cosine_similarity(safe_student_d, safe_teacher_d, dim=-1, eps=1e-8)
    teacher_query_norm = teacher_q.norm(dim=-1)
    teacher_doc_norm = safe_teacher_d.norm(dim=-1)
    student_query_norm = student_q.norm(dim=-1)
    student_doc_norm = safe_student_d.norm(dim=-1)

    return EmbedAlignmentStats(
        total=total,
        query_loss_per_query=query_loss,
        document_loss_per_query=doc_loss,
        combined_loss_per_query=combined,
        valid_query_mask=valid_query_mask,
        valid_query_count=valid_query_mask.sum().to(dtype=torch.float32),
        valid_document_count_total=doc_mask.sum().to(dtype=torch.float32),
        query_alignment_cos_sum=torch.where(
            valid_query_mask, query_alignment_cos, torch.zeros_like(query_alignment_cos)
        ).sum(),
        document_alignment_cos_sum=torch.where(
            doc_mask, doc_alignment_cos, torch.zeros_like(doc_alignment_cos)
        ).sum(),
        teacher_query_norm_sum=torch.where(
            valid_query_mask, teacher_query_norm, torch.zeros_like(teacher_query_norm)
        ).sum(),
        teacher_document_norm_sum=torch.where(
            doc_mask, teacher_doc_norm, torch.zeros_like(teacher_doc_norm)
        ).sum(),
        student_query_norm_sum=torch.where(
            valid_query_mask, student_query_norm, torch.zeros_like(student_query_norm)
        ).sum(),
        student_document_norm_sum=torch.where(
            doc_mask, student_doc_norm, torch.zeros_like(student_doc_norm)
        ).sum(),
    )


def compute_embeddistill_loss(
    cfg: EmbedDistillCfg,
    student_scores: torch.Tensor,
    projected_student_query: torch.Tensor,
    projected_student_documents: torch.Tensor,
    teacher_targets: TeacherFinalTargets,
    labels: torch.Tensor,
    mask: torch.Tensor,
    student_query_embeddings: torch.Tensor,
    frozen_query_embeddings: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    assert_zero_labels(labels)
    doc_mask = mask.to(device=student_scores.device, dtype=torch.bool)
    alignment = embeddistill_alignment_per_example(
        projected_student_query=projected_student_query,
        projected_student_documents=projected_student_documents,
        teacher_query=teacher_targets.teacher_query_embedding,
        teacher_documents=teacher_targets.teacher_document_embeddings,
        document_mask=doc_mask,
        distance=cfg.embed_distance,
        query_weight=cfg.embed_query_weight,
        document_weight=cfg.embed_document_weight,
    )

    score_loss_per_query = torch.zeros_like(alignment.combined_loss_per_query)
    if cfg.embeddistill_variant == EMBEDDISTILL_VARIANT_RANK_EMBED:
        raw_score_loss = ranking_kl_per_example(
            student_scores.to(dtype=torch.float32),
            teacher_targets.teacher_scores.to(device=student_scores.device, dtype=torch.float32),
            doc_mask,
            cfg.tau,
        )
        score_valid_mask = doc_mask.sum(dim=-1) >= 2
        score_loss_per_query = torch.where(
            score_valid_mask,
            raw_score_loss.to(dtype=torch.float32),
            torch.zeros_like(raw_score_loss.to(dtype=torch.float32)),
        )

    embed_component = float(cfg.embed_loss_weight) * alignment.combined_loss_per_query
    total_per_query = embed_component + score_loss_per_query
    if alignment.valid_query_mask.any():
        total = total_per_query[alignment.valid_query_mask].mean()
    else:
        total = student_scores.to(dtype=torch.float32).sum() * 0.0

    q_s = student_query_embeddings.to(dtype=torch.float32)
    base_q = frozen_query_embeddings.to(device=q_s.device, dtype=torch.float32)
    student_query_drift = 1.0 - F.cosine_similarity(q_s, base_q, dim=-1, eps=1e-8)
    student_query_drift_sum = torch.where(
        alignment.valid_query_mask, student_query_drift, torch.zeros_like(student_query_drift)
    ).sum()

    return {
        "total": total,
        "total_loss_sum": total_per_query[alignment.valid_query_mask].sum()
        if alignment.valid_query_mask.any()
        else total * 0.0,
        "score_kd_loss_sum": score_loss_per_query[alignment.valid_query_mask].sum()
        if alignment.valid_query_mask.any()
        else total * 0.0,
        "query_embed_loss_sum": alignment.query_loss_per_query[alignment.valid_query_mask].sum()
        if alignment.valid_query_mask.any()
        else total * 0.0,
        "document_embed_loss_sum": alignment.document_loss_per_query[
            alignment.valid_query_mask
        ].sum()
        if alignment.valid_query_mask.any()
        else total * 0.0,
        "combined_embed_loss_sum": embed_component[alignment.valid_query_mask].sum()
        if alignment.valid_query_mask.any()
        else total * 0.0,
        "query_alignment_cos_sum": alignment.query_alignment_cos_sum,
        "document_alignment_cos_sum": alignment.document_alignment_cos_sum,
        "teacher_query_norm_sum": alignment.teacher_query_norm_sum,
        "teacher_document_norm_sum": alignment.teacher_document_norm_sum,
        "student_query_norm_sum": alignment.student_query_norm_sum,
        "student_document_norm_sum": alignment.student_document_norm_sum,
        "student_query_drift_sum": student_query_drift_sum,
        "valid_query_count": alignment.valid_query_count,
        "valid_document_count": alignment.valid_document_count_total,
    }


def save_embeddistill_checkpoint(
    student: base.TrainableRetriever,
    projection: SharedLinearProjection,
    checkpoint_dir: str,
    cfg: EmbedDistillCfg,
    system_name: str,
    epoch: int,
    best_score: float,
    best_metrics: Dict[int, Dict[str, float]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    teacher_checkpoint_fingerprint_sha1: str,
    teacher_output_dim: int,
    teacher_metadata: Mapping[str, Any],
    patience_ctr: int = 0,
    completed: bool = False,
    checkpoint_kind: str = "best",
    candidate_metadata: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    student.save(checkpoint_dir)
    candidate_metadata = candidate_metadata or {}
    metadata = {
        "checkpoint_format": "dense-retriever-student-embeddistill-v2",
        "checkpoint_kind": checkpoint_kind,
        "variant": system_name,
        "objective": OBJECTIVE_EMBEDDISTILL,
        "embeddistill_variant": cfg.embeddistill_variant,
        "embedding_distance": cfg.embed_distance,
        "embedding_weights": {
            "query": cfg.embed_query_weight,
            "document": cfg.embed_document_weight,
            "embedding_loss_weight": cfg.embed_loss_weight,
        },
        "teacher_target_fields": list(USED_TEACHER_TARGET_FIELDS),
        "score_distillation_enabled": cfg.embeddistill_variant == EMBEDDISTILL_VARIANT_RANK_EMBED,
        "teacher_checkpoint_path": cfg.teacher_checkpoint_path,
        "teacher_checkpoint_fingerprint_sha1": teacher_checkpoint_fingerprint_sha1,
        "teacher_metadata": dict(teacher_metadata),
        "teacher_output_dim": int(teacher_output_dim),
        "projection": {
            "architecture": "shared_linear",
            "input_dim": int(cfg.embedding_dim),
            "output_dim": int(teacher_output_dim),
            "bias": bool(projection.bias),
            "training_only": True,
        },
        "delta_supervision": False,
        "query_generation": False,
        "teacher_document_inheritance": False,
        "qrels_used_in_training": False,
        "qrel_positive_injection": False,
        "qrel_loss_active": False,
        "candidate_metadata": candidate_metadata,
        "dataset_name": cfg.dataset_name,
        "dataset_format": cfg.source_dataset_format or cfg.dataset_format,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_score": best_score,
        "best_val_metrics": best_metrics,
        "patience_ctr": patience_ctr,
        "completed": completed,
        "student_model_name": cfg.model_name,
        "config": asdict(cfg),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "adaptation_note": (
            "EmbedDistill-style adaptation: IMRNN teacher document targets are query-conditioned final modulated "
            "representations rather than standalone corpus encoder embeddings."
        ),
    }
    metadata = provenance.add_run_contract(
        metadata,
        cfg=cfg,
        objective=OBJECTIVE_EMBEDDISTILL,
        variant=system_name,
        candidate_metadata=candidate_metadata,
        teacher_checkpoint_path=cfg.teacher_checkpoint_path,
    )
    with open(
        os.path.join(checkpoint_dir, "student_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "projection_head_state": projection.state_dict(),
            "metadata": metadata,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    logger.info(
        f"Saved {checkpoint_kind} {system_name} checkpoint to {checkpoint_dir} | "
        f"epoch={epoch} | best_score={best_score:.6f} | patience={patience_ctr} | completed={completed}"
    )


def save_latest_embeddistill_checkpoint(
    student: base.TrainableRetriever,
    projection: SharedLinearProjection,
    checkpoint_dir: str,
    cfg: EmbedDistillCfg,
    system_name: str,
    epoch: int,
    best_score: float,
    best_metrics: Dict[int, Dict[str, float]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    teacher_checkpoint_fingerprint_sha1: str,
    teacher_output_dim: int,
    teacher_metadata: Mapping[str, Any],
    patience_ctr: int,
    completed: bool = False,
    candidate_metadata: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
) -> None:
    latest_dir = os.path.join(checkpoint_dir, "latest")
    save_embeddistill_checkpoint(
        student=student,
        projection=projection,
        checkpoint_dir=latest_dir,
        cfg=cfg,
        system_name=system_name,
        epoch=epoch,
        best_score=best_score,
        best_metrics=best_metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher_checkpoint_fingerprint_sha1=teacher_checkpoint_fingerprint_sha1,
        teacher_output_dim=teacher_output_dim,
        teacher_metadata=teacher_metadata,
        patience_ctr=patience_ctr,
        completed=completed,
        checkpoint_kind="latest",
        candidate_metadata=candidate_metadata,
        scaler=scaler,
    )


def probe_teacher_target_space(
    cfg: EmbedDistillCfg,
    examples: Sequence[base.TrainExample],
    passages: Sequence[base.PassageRecord],
    teacher: base.TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
) -> TeacherFinalTargets:
    chosen: Optional[base.TrainExample] = None
    for example in examples:
        if len(example.candidate_doc_idxs) >= 1:
            chosen = example
            break
    if chosen is None:
        raise ValueError(
            "Could not find a training example with at least one valid candidate to probe teacher target space"
        )
    batch = base.collate_student_candidates([chosen], list(passages))
    safe_doc_idxs = batch["doc_idxs"].clone()
    safe_doc_idxs[safe_doc_idxs < 0] = 0
    q_raw = raw_query_embs_cpu[batch["q_idx"]].to(cfg.device)
    d_raw = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
    targets = extract_final_teacher_embeddings_and_scores(teacher, q_raw, d_raw, batch["mask"])
    if (
        cfg.signal_projection_output_dim
        and int(cfg.signal_projection_output_dim) != targets.teacher_output_dim
    ):
        logger.warning(
            f"Ignoring cfg.signal_projection_output_dim={cfg.signal_projection_output_dim}; "
            f"actual teacher target dim is {targets.teacher_output_dim}."
        )
    return targets


def write_teacher_embedding_debug_sample(
    cfg: EmbedDistillCfg,
    path: str,
    examples: Sequence[base.TrainExample],
    passages: Sequence[base.PassageRecord],
    teacher: base.TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
) -> None:
    sample = list(examples[: max(0, cfg.debug_teacher_samples)])
    if not sample:
        return
    batch = base.collate_student_candidates(sample, list(passages))
    safe_doc_idxs = batch["doc_idxs"].clone()
    safe_doc_idxs[safe_doc_idxs < 0] = 0
    q_raw = raw_query_embs_cpu[batch["q_idx"]].to(cfg.device)
    d_raw = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
    targets = extract_final_teacher_embeddings_and_scores(teacher, q_raw, d_raw, batch["mask"])
    scores = targets.teacher_scores.cpu()
    rows: List[Dict[str, Any]] = []
    for i, example in enumerate(sample):
        valid = torch.where(batch["mask"][i])[0].tolist()
        rows.append(
            {
                "qid": example.qid,
                "query_text": example.query_text,
                "candidate_ids": [passages[int(batch["doc_idxs"][i, j])].pid for j in valid],
                "teacher_scores": [float(scores[i, j]) for j in valid],
                "teacher_query_norm": float(targets.teacher_query_embedding[i].norm().cpu()),
                "teacher_document_norms": [
                    float(targets.teacher_document_embeddings[i, j].norm().cpu()) for j in valid
                ],
                "teacher_output_dim": int(targets.teacher_output_dim),
            }
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info(f"Saved teacher embedding debug sample: {path}")


def build_micro_slices(
    batch: Dict[str, Any], micro_batch_size: int
) -> List[Tuple[Dict[str, Any], int]]:
    slices: List[Tuple[Dict[str, Any], int]] = []
    outer_batch_size = len(batch["query_texts"])
    micro = micro_batch_size if micro_batch_size > 0 else outer_batch_size
    for start in range(0, outer_batch_size, micro):
        end = min(start + micro, outer_batch_size)
        sub = base.slice_training_batch(batch, start, end)
        valid_queries = int((sub["mask"].sum(dim=-1) >= 1).sum().item())
        slices.append((sub, valid_queries))
    return slices


def choose_dry_run_examples(
    examples: Sequence[base.TrainExample], batch_size: int
) -> List[base.TrainExample]:
    selected = [example for example in examples if len(example.candidate_doc_idxs) >= 1]
    if not selected:
        raise ValueError(
            "Dry run requires at least one training example with one or more valid candidates"
        )
    return selected[: max(1, batch_size)]


def dry_run_embeddistill(
    cfg: EmbedDistillCfg,
    examples: Sequence[base.TrainExample],
    passages: Sequence[base.PassageRecord],
    teacher: base.TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    candidate_metadata: Dict[str, Any],
    teacher_output_dim: int,
) -> Dict[str, Any]:
    base.log_section("DRY RUN")
    assert_unsupervised_training_contract(
        candidate_metadata=candidate_metadata,
        examples=examples,
        training_qrels_by_qid_idx=None,
    )
    batch_examples = choose_dry_run_examples(examples, cfg.batch_size)
    batch = base.collate_student_candidates(batch_examples, list(passages))
    labels = batch["labels"].to(cfg.device)
    mask = batch["mask"].to(cfg.device)
    assert_zero_labels(labels)

    safe_doc_idxs = batch["doc_idxs"].clone()
    safe_doc_idxs[safe_doc_idxs < 0] = 0
    base_q = raw_query_embs_cpu[batch["q_idx"]].to(cfg.device)
    base_docs = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
    teacher_targets = extract_final_teacher_embeddings_and_scores(
        teacher, base_q, base_docs, batch["mask"]
    )
    if teacher_targets.teacher_output_dim != teacher_output_dim:
        raise AssertionError(
            f"Teacher output dim changed between probe and dry run: {teacher_output_dim} vs {teacher_targets.teacher_output_dim}"
        )
    if teacher_targets.used_fields != USED_TEACHER_TARGET_FIELDS:
        raise AssertionError(
            f"Unexpected teacher target fields in dry run: {teacher_targets.used_fields}"
        )

    student = base.TrainableRetriever(cfg.model_name, cfg).to(cfg.device)
    projection = SharedLinearProjection(
        cfg.embedding_dim, teacher_output_dim, bias=cfg.projection_bias
    ).to(cfg.device)
    params = list(student.parameters()) + list(projection.parameters())
    optimizer_groups = [
        {"params": list(student.parameters()), "weight_decay": cfg.weight_decay},
        {"params": list(projection.parameters()), "weight_decay": cfg.projection_head_weight_decay},
    ]
    optimizer = base.build_optimizer(optimizer_groups, cfg, context="dry_run_embeddistill")
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    student.train()
    projection.train()
    optimizer.zero_grad(set_to_none=True)
    flat_doc_texts = [text for row in batch["candidate_texts"] for text in row]
    bsz, docs_per_query = batch["doc_idxs"].shape
    with torch.amp.autocast("cuda", enabled=use_amp):
        q_s = student.encode_query_texts_train(batch["query_texts"])
        d_s = student.encode_doc_texts_train(flat_doc_texts).view(bsz, docs_per_query, -1)
        s_scores = torch.einsum("bd,bkd->bk", q_s, d_s)
    projected_q = projection(q_s.to(dtype=torch.float32))
    projected_d = projection(d_s.to(dtype=torch.float32))
    loss_info = compute_embeddistill_loss(
        cfg=cfg,
        student_scores=s_scores,
        projected_student_query=projected_q,
        projected_student_documents=projected_d,
        teacher_targets=teacher_targets,
        labels=labels,
        mask=mask,
        student_query_embeddings=q_s,
        frozen_query_embeddings=base_q,
    )
    scaler.scale(loss_info["total"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
    base.assert_teacher_frozen(teacher)

    if not any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in student.parameters()
        if param.requires_grad
    ):
        raise AssertionError("Dry run expected finite student gradients")
    if not any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in projection.parameters()
        if param.requires_grad
    ):
        raise AssertionError("Dry run expected finite projection gradients")

    dry_output = {
        "system_name": system_name_for_cfg(cfg),
        "objective": OBJECTIVE_EMBEDDISTILL,
        "variant": cfg.embeddistill_variant,
        "distance": cfg.embed_distance,
        "candidate_index_fingerprint_sha1": candidate_metadata.get(
            "candidate_index_fingerprint_sha1", ""
        ),
        "candidate_id_fingerprint_sha1": candidate_metadata.get(
            "candidate_id_fingerprint_sha1", ""
        ),
        "corpus_order_fingerprint_sha1": candidate_metadata.get(
            "corpus_order_fingerprint_sha1", ""
        ),
        "teacher_target_fields": list(teacher_targets.used_fields),
        "qrels_used_in_training": False,
        "qrel_positive_injection": False,
        "qrel_loss_active": False,
        "valid_query_count": int(loss_info["valid_query_count"].detach().cpu()),
        "valid_document_count": int(loss_info["valid_document_count"].detach().cpu()),
        "loss": float(loss_info["total"].detach().cpu()),
        "score_kd_loss_mean": mean_or_zero(
            float(loss_info["score_kd_loss_sum"].detach().cpu()),
            int(loss_info["valid_query_count"].detach().cpu()),
        ),
        "query_embed_loss_mean": mean_or_zero(
            float(loss_info["query_embed_loss_sum"].detach().cpu()),
            int(loss_info["valid_query_count"].detach().cpu()),
        ),
        "document_embed_loss_mean": mean_or_zero(
            float(loss_info["document_embed_loss_sum"].detach().cpu()),
            int(loss_info["valid_query_count"].detach().cpu()),
        ),
        "combined_embed_loss_mean": mean_or_zero(
            float(loss_info["combined_embed_loss_sum"].detach().cpu()),
            int(loss_info["valid_query_count"].detach().cpu()),
        ),
        "student_query_drift_mean": mean_or_zero(
            float(loss_info["student_query_drift_sum"].detach().cpu()),
            int(loss_info["valid_query_count"].detach().cpu()),
        ),
        "projection_norm": projection_parameter_norm(projection),
        "gradient_norm": float(grad_norm.detach().cpu())
        if torch.is_tensor(grad_norm)
        else float(grad_norm),
        "query_batch_shape": list(q_s.shape),
        "doc_batch_shape": list(d_s.shape),
        "projected_query_shape": list(projected_q.shape),
        "projected_doc_shape": list(projected_d.shape),
        "teacher_query_shape": list(teacher_targets.teacher_query_embedding.shape),
        "teacher_doc_shape": list(teacher_targets.teacher_document_embeddings.shape),
    }
    logger.info(f"DRY_RUN | {json.dumps(dry_output, sort_keys=True)}")
    del (
        student,
        projection,
        optimizer,
        q_s,
        d_s,
        s_scores,
        projected_q,
        projected_d,
        loss_info,
        teacher_targets,
    )
    base.clear_cuda_cache_if_needed(cfg)
    return dry_output


def train_embeddistill_student(
    cfg: EmbedDistillCfg,
    initial_examples: List[base.TrainExample],
    candidate_metadata: Dict[str, Any],
    val_examples: List[base.EvalExample],
    passages: List[base.PassageRecord],
    query_by_id: Dict[str, base.QueryRecord],
    query_id_to_index: Dict[str, int],
    train_qids: Sequence[str],
    eval_qrels_by_qid_idx: Dict[str, Dict[int, float]],
    teacher: base.TeacherAdapter,
    raw_index: base.SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    teacher_output_dim: int,
    teacher_checkpoint_fingerprint_sha1: str,
    teacher_metadata: Mapping[str, Any],
) -> base.TrainableRetriever:
    system_name = system_name_for_cfg(cfg)
    base.log_section(f"TRAIN OBJECTIVE | {system_name} | seed={cfg.seed}")
    log_objective_summary(cfg, teacher_output_dim)
    assert_unsupervised_training_contract(
        candidate_metadata=candidate_metadata,
        examples=initial_examples,
        training_qrels_by_qid_idx=None,
    )

    base.set_seed(cfg.seed)
    ckpt_dir = base.training_checkpoint_dir(cfg, system_name)
    run_contract = provenance.build_run_contract(
        cfg=cfg,
        objective=OBJECTIVE_EMBEDDISTILL,
        variant=system_name,
        candidate_metadata=candidate_metadata,
        teacher_checkpoint_path=cfg.teacher_checkpoint_path,
    )
    contract_candidate_metadata = dict(candidate_metadata)
    resume_payload: Dict[str, Any] = {}
    resume_metadata: Dict[str, Any] = {}
    if cfg.resume and base.student_training_complete(ckpt_dir):
        provenance.assert_resume_contract(ckpt_dir, run_contract)
        logger.info(
            f"{system_name} has training_complete marker; loading best checkpoint from {ckpt_dir}"
        )
        loaded = base.load_student_checkpoint(ckpt_dir, cfg)
        if loaded is not None:
            return loaded
    resume_dir = base.find_resume_checkpoint_dir(ckpt_dir) if cfg.resume else None
    if resume_dir:
        provenance.assert_resume_contract(resume_dir, run_contract)
        resume_payload, resume_metadata = base.load_training_state_payload(resume_dir)
        if resume_metadata.get("completed", False):
            logger.info(
                f"{system_name} is marked complete; loading best checkpoint from {ckpt_dir}"
            )
            loaded = base.load_student_checkpoint(ckpt_dir, cfg)
            if loaded is not None:
                return loaded
        logger.info(f"Resuming {system_name} from checkpoint directory: {resume_dir}")
        student = base.TrainableRetriever(resume_dir, cfg).to(cfg.device)
    else:
        student = base.TrainableRetriever(cfg.model_name, cfg).to(cfg.device)

    projection = SharedLinearProjection(
        cfg.embedding_dim, teacher_output_dim, bias=cfg.projection_bias
    ).to(cfg.device)
    projection_meta = resume_metadata.get("projection", {}) if resume_metadata else {}
    if projection_meta:
        if int(projection_meta.get("output_dim", teacher_output_dim)) != int(teacher_output_dim):
            raise ValueError(
                f"Resume projection output dim mismatch: checkpoint={projection_meta.get('output_dim')} vs current={teacher_output_dim}"
            )
        if bool(projection_meta.get("bias", cfg.projection_bias)) != bool(cfg.projection_bias):
            raise ValueError(
                f"Resume projection bias mismatch: checkpoint={projection_meta.get('bias')} vs current={cfg.projection_bias}"
            )
    if resume_payload.get("projection_head_state"):
        try:
            projection.load_state_dict(resume_payload["projection_head_state"])
            logger.info(f"Restored projection head for {system_name} from {resume_dir}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not restore projection head for {system_name}: {exc}"
            ) from exc

    student_params = list(student.parameters())
    projection_params = list(projection.parameters())
    optimizer_groups = [
        {"params": student_params, "weight_decay": cfg.weight_decay},
        {"params": projection_params, "weight_decay": cfg.projection_head_weight_decay},
    ]
    optimizer = base.build_optimizer(optimizer_groups, cfg, context=system_name)
    total_steps = cfg.epochs * max(1, math.ceil(len(initial_examples) / max(1, cfg.batch_size)))
    scheduler = base.get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    if resume_payload.get("optimizer_state"):
        try:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            logger.info(f"Restored optimizer state for {system_name}")
        except Exception as exc:
            logger.warning(f"Could not restore optimizer state for {system_name}: {exc}")
    if resume_payload.get("scheduler_state"):
        try:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
            logger.info(f"Restored scheduler state for {system_name}")
        except Exception as exc:
            logger.warning(f"Could not restore scheduler state for {system_name}: {exc}")
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_payload.get("scaler_state"):
        try:
            scaler.load_state_dict(resume_payload["scaler_state"])
            logger.info(f"Restored AMP GradScaler state for {system_name}")
        except Exception as exc:
            logger.warning(f"Could not restore AMP GradScaler state for {system_name}: {exc}")

    examples = list(initial_examples)
    resume_epoch = int(resume_metadata.get("epoch", 0)) if resume_metadata else 0
    start_epoch = resume_epoch + 1
    best_score = float(resume_metadata.get("best_score", -1.0e9)) if resume_metadata else -1.0e9
    best_metrics: Dict[int, Dict[str, float]] = (
        resume_metadata.get("best_val_metrics", {}) if resume_metadata else {}
    )
    patience_ctr = int(resume_metadata.get("patience_ctr", 0)) if resume_metadata else 0
    if start_epoch > cfg.epochs:
        logger.info(
            f"{system_name} already reached cfg.epochs={cfg.epochs}; loading best checkpoint"
        )
        loaded = base.load_student_checkpoint(ckpt_dir, cfg)
        if loaded is not None:
            return loaded
    logger.info(
        f"{system_name} training resume state: start_epoch={start_epoch} | best_score={best_score:.6f} | "
        f"patience_ctr={patience_ctr} | checkpoint_path={ckpt_dir}"
    )

    last_epoch = resume_epoch
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        dataset = base.StudentCandidateDataset(examples, passages)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda batch: base.collate_student_candidates(batch, passages),
        )
        student.train()
        projection.train()
        meters = EpochMeters()

        for batch in loader:
            micro_slices = build_micro_slices(batch, cfg.micro_batch_size)
            outer_valid_query_count = sum(valid for _, valid in micro_slices)
            if outer_valid_query_count <= 0:
                meters.skipped_outer_batches += 1
                logger.warning(
                    "Skipping optimizer step because the outer batch contained no valid training signal"
                )
                continue

            optimizer.zero_grad(set_to_none=True)
            for micro, valid_queries_in_micro in micro_slices:
                if valid_queries_in_micro <= 0:
                    continue
                labels = micro["labels"].to(cfg.device)
                mask = micro["mask"].to(cfg.device)
                assert_zero_labels(labels)
                flat_doc_texts = [text for row in micro["candidate_texts"] for text in row]
                bsz, docs_per_query = micro["doc_idxs"].shape
                weight = float(valid_queries_in_micro) / float(max(1, outer_valid_query_count))

                safe_doc_idxs = micro["doc_idxs"].clone()
                safe_doc_idxs[safe_doc_idxs < 0] = 0
                base_q = raw_query_embs_cpu[micro["q_idx"]].to(cfg.device)
                base_docs = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
                teacher_targets = extract_final_teacher_embeddings_and_scores(
                    teacher, base_q, base_docs, micro["mask"]
                )
                if teacher_targets.teacher_output_dim != teacher_output_dim:
                    raise AssertionError(
                        f"Teacher target dim changed during training: expected {teacher_output_dim}, got {teacher_targets.teacher_output_dim}"
                    )

                with torch.amp.autocast("cuda", enabled=use_amp):
                    q_s = student.encode_query_texts_train(micro["query_texts"])
                    flat_d_s = student.encode_doc_texts_train(flat_doc_texts)
                    d_s = flat_d_s.view(bsz, docs_per_query, -1)
                    if q_s.shape[-1] != cfg.embedding_dim or d_s.shape[-1] != cfg.embedding_dim:
                        raise AssertionError("Student output dimension mismatch")
                    s_scores = torch.einsum("bd,bkd->bk", q_s, d_s)
                projected_q = projection(q_s.to(dtype=torch.float32))
                projected_d = projection(d_s.to(dtype=torch.float32))
                loss_info = compute_embeddistill_loss(
                    cfg=cfg,
                    student_scores=s_scores,
                    projected_student_query=projected_q,
                    projected_student_documents=projected_d,
                    teacher_targets=teacher_targets,
                    labels=labels,
                    mask=mask,
                    student_query_embeddings=q_s,
                    frozen_query_embeddings=base_q,
                )
                if int(loss_info["valid_query_count"].detach().cpu()) <= 0:
                    continue
                scaler.scale(loss_info["total"] * weight).backward()
                meters.update(loss_info)
                del (
                    q_s,
                    flat_d_s,
                    d_s,
                    s_scores,
                    projected_q,
                    projected_d,
                    teacher_targets,
                    loss_info,
                )

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_params + projection_params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            base.assert_teacher_frozen(teacher)

        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.lr
        mean_total = mean_or_zero(meters.total_loss_sum, meters.valid_query_count)
        mean_score_kd = mean_or_zero(meters.score_kd_loss_sum, meters.valid_query_count)
        mean_query_embed = mean_or_zero(meters.query_embed_loss_sum, meters.valid_query_count)
        mean_document_embed = mean_or_zero(meters.document_embed_loss_sum, meters.valid_query_count)
        mean_combined_embed = mean_or_zero(meters.combined_embed_loss_sum, meters.valid_query_count)
        mean_query_alignment_cos = mean_or_zero(
            meters.query_alignment_cos_sum, meters.valid_query_count
        )
        mean_document_alignment_cos = mean_or_zero(
            meters.document_alignment_cos_sum, meters.valid_document_count
        )
        mean_teacher_query_norm = mean_or_zero(
            meters.teacher_query_norm_sum, meters.valid_query_count
        )
        mean_teacher_document_norm = mean_or_zero(
            meters.teacher_document_norm_sum, meters.valid_document_count
        )
        mean_student_query_norm = mean_or_zero(
            meters.student_query_norm_sum, meters.valid_query_count
        )
        mean_student_document_norm = mean_or_zero(
            meters.student_document_norm_sum, meters.valid_document_count
        )
        mean_student_drift = mean_or_zero(meters.student_query_drift_sum, meters.valid_query_count)
        logger.info(
            f"Epoch {epoch} | {system_name} | total_loss={mean_total:.6f} | score_kd={mean_score_kd:.6f} | "
            f"query_embed={mean_query_embed:.6f} | document_embed={mean_document_embed:.6f} | "
            f"combined_embed={mean_combined_embed:.6f} | query_align_cos={mean_query_alignment_cos:.6f} | "
            f"document_align_cos={mean_document_alignment_cos:.6f} | teacher_query_norm={mean_teacher_query_norm:.6f} | "
            f"teacher_document_norm={mean_teacher_document_norm:.6f} | projected_student_query_norm={mean_student_query_norm:.6f} | "
            f"projected_student_document_norm={mean_student_document_norm:.6f} | projection_norm={projection_parameter_norm(projection):.6f} | "
            f"student_drift={mean_student_drift:.6f} | queries={meters.valid_query_count} | "
            f"valid_documents={meters.valid_document_count} | skipped_outer_batches={meters.skipped_outer_batches} | lr={lr:.8f}"
        )

        base.clear_cuda_cache_if_needed(cfg)
        student.eval()
        val_doc_embs = base.encode_corpus_with_student(student, passages, cfg)
        val_index = base.SearchIndex(val_doc_embs)
        val_metrics = base.evaluate_student(
            student, val_index, val_examples, eval_qrels_by_qid_idx, cfg
        )
        base.log_metrics_table(f"{system_name} | VAL | EPOCH {epoch}", val_metrics, cfg.ks)
        score = base.validation_score(val_metrics, cfg)
        logger.info(f"Epoch {epoch} | {system_name} | validation_selection_score={score:.6f}")
        del val_index, val_doc_embs
        base.clear_cuda_cache_if_needed(cfg)

        if score > best_score + cfg.min_delta:
            best_score = score
            best_metrics = val_metrics
            patience_ctr = 0
            save_embeddistill_checkpoint(
                student=student,
                projection=projection,
                checkpoint_dir=ckpt_dir,
                cfg=cfg,
                system_name=system_name,
                epoch=epoch,
                best_score=best_score,
                best_metrics=best_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                teacher_checkpoint_fingerprint_sha1=teacher_checkpoint_fingerprint_sha1,
                teacher_output_dim=teacher_output_dim,
                teacher_metadata=teacher_metadata,
                patience_ctr=patience_ctr,
                completed=False,
                checkpoint_kind="best",
                candidate_metadata=contract_candidate_metadata,
                scaler=scaler,
            )
        else:
            patience_ctr += 1
            logger.info(f"No improvement for {system_name}. Patience {patience_ctr}/{cfg.patience}")

        save_latest_embeddistill_checkpoint(
            student=student,
            projection=projection,
            checkpoint_dir=ckpt_dir,
            cfg=cfg,
            system_name=system_name,
            epoch=epoch,
            best_score=best_score,
            best_metrics=best_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            teacher_checkpoint_fingerprint_sha1=teacher_checkpoint_fingerprint_sha1,
            teacher_output_dim=teacher_output_dim,
            teacher_metadata=teacher_metadata,
            patience_ctr=patience_ctr,
            completed=False,
            candidate_metadata=contract_candidate_metadata,
            scaler=scaler,
        )
        if patience_ctr >= cfg.patience:
            logger.info(f"Early stopping {system_name}")
            break

    if last_epoch > 0:
        save_latest_embeddistill_checkpoint(
            student=student,
            projection=projection,
            checkpoint_dir=ckpt_dir,
            cfg=cfg,
            system_name=system_name,
            epoch=last_epoch,
            best_score=best_score,
            best_metrics=best_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            teacher_checkpoint_fingerprint_sha1=teacher_checkpoint_fingerprint_sha1,
            teacher_output_dim=teacher_output_dim,
            teacher_metadata=teacher_metadata,
            patience_ctr=patience_ctr,
            completed=True,
            candidate_metadata=contract_candidate_metadata,
            scaler=scaler,
        )
        base.mark_student_training_complete(ckpt_dir, system_name, last_epoch, best_score)
    best_student = base.load_student_checkpoint(ckpt_dir, cfg)
    if best_student is None:
        return student
    return best_student


def run_one_seed(cfg: EmbedDistillCfg) -> Dict[str, Any]:
    log_path = base.setup_logging(cfg)
    base.set_seed(cfg.seed)
    experiment_utils.log_run_header(cfg, log_path, logger, __file__)
    base.require_runtime_dependencies()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg.teacher_checkpoint_path = resolve_teacher_spec_for_cfg(cfg)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, sort_keys=True)

    corpus_path = os.path.join(cfg.dataset_dir, cfg.corpus_file)
    queries_path = os.path.join(cfg.dataset_dir, cfg.queries_file)
    qrels_dir = os.path.join(cfg.dataset_dir, cfg.qrels_dir)
    passages = base.load_passages(corpus_path)
    queries = base.load_queries(queries_path)
    split_ids, qrels, split_source = base.make_split_qids(cfg, qrels_dir, queries)
    base.validate_references(queries, passages, qrels)
    base.log_schema_summary(cfg, passages, queries, qrels, split_ids, split_source)
    logger.info(
        "Qrels are used only to recover benchmark split membership and for validation/test evaluation. "
        "Relevance scores and relevant-document identities are not used for student candidate construction or training losses."
    )
    if not split_ids["val"]:
        raise ValueError("No validation queries remain after filtering/splitting")

    passage_id_to_index = {p.pid: i for i, p in enumerate(passages)}
    query_id_to_index = {q.qid: i for i, q in enumerate(queries)}
    query_by_id = {q.qid: q for q in queries}
    qrels_by_qid_idx = base.build_qrels_by_qid_idx(qrels, passage_id_to_index)
    val_examples = build_eval_examples(
        split_ids["val"], query_by_id, query_id_to_index, qrels, base.EvalExample
    )
    test_examples = build_eval_examples(
        split_ids["test"], query_by_id, query_id_to_index, qrels, base.EvalExample
    )
    base.log_section("FROZEN BASE-RETRIEVER CACHE SUMMARY")
    frozen_encoder = base.SentenceTransformer(cfg.model_name, device=cfg.device)
    raw_doc_embs_cpu = base.build_or_load_frozen_corpus_embeddings(cfg, passages, frozen_encoder)
    raw_query_embs_cpu = base.build_or_load_frozen_query_embeddings(cfg, queries, frozen_encoder)
    del frozen_encoder
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if (
        raw_doc_embs_cpu.shape[1] != cfg.embedding_dim
        or raw_query_embs_cpu.shape[1] != cfg.embedding_dim
    ):
        raise AssertionError("Frozen base-retriever cache dimension mismatch")
    raw_index = base.SearchIndex(raw_doc_embs_cpu)

    teacher_checkpoint_path = base.ensure_teacher_checkpoint(
        cfg,
        split_ids,
        qrels_by_qid_idx,
        query_id_to_index,
        raw_index,
        raw_doc_embs_cpu,
        raw_query_embs_cpu,
        split_source,
    )
    cfg.teacher_checkpoint_path = str(teacher_checkpoint_path)
    teacher_metadata_raw, teacher_state_dict = load_teacher_checkpoint_metadata(
        teacher_checkpoint_path
    )
    teacher_output_dim_from_state = int(
        base.infer_teacher_config_from_state(teacher_state_dict, cfg)[1]
    )
    teacher_checkpoint_fingerprint_sha1 = provenance.file_sha1(teacher_checkpoint_path)
    assert_teacher_metadata_compatible(
        cfg, teacher_metadata_raw, teacher_output_dim_from_state, teacher_checkpoint_path
    )
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, sort_keys=True)

    teacher = base.TeacherAdapter(cfg)
    base.assert_teacher_frozen(teacher)
    teacher_output_dim = (
        teacher.model.project.weight.shape[0]
        if hasattr(teacher.model, "project") and hasattr(teacher.model.project, "weight")
        else teacher_output_dim_from_state
    )
    assert_teacher_metadata_compatible(
        cfg, teacher.metadata, int(teacher_output_dim), teacher_checkpoint_path
    )

    if cfg.dry_run:
        quality = {"dry_run": True, "teacher_quality_gate_skipped": True}
        logger.info("Dry run: skipping full teacher quality gate.")
    else:
        quality = base.teacher_quality_gate(
            cfg,
            teacher,
            raw_index,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
            val_examples,
            qrels_by_qid_idx,
        )

    students: Dict[str, base.TrainableRetriever] = {}
    train_qids = list(split_ids["train"])
    if cfg.train_queries_file:
        with open(cfg.train_queries_file, "r", encoding="utf-8") as handle:
            requested = [line.strip() for line in handle if line.strip()]
        train_qids = [qid for qid in requested if qid in query_id_to_index]
        logger.info(
            f"Using train query list from {cfg.train_queries_file}: {len(train_qids)} queries"
        )
    if cfg.mode != base.MODE_EVAL_ONLY and not train_qids:
        raise ValueError("No train queries available")
    if cfg.dry_run and cfg.mode == base.MODE_EVAL_ONLY:
        raise ValueError("--dry_run requires UNSUPERVISED mode")

    candidate_metadata: Dict[str, Any] = {}
    system_name = system_name_for_cfg(cfg)
    if cfg.mode != base.MODE_EVAL_ONLY:
        build_result = base.build_unsupervised_train_examples(
            cfg,
            train_qids,
            query_by_id,
            query_id_to_index,
            raw_index,
            raw_query_embs_cpu,
            passages=passages,
        )
        base_examples = build_result.examples
        candidate_metadata = augment_candidate_metadata(
            cfg, build_result.metadata, base_examples, passages
        )
        assert_unsupervised_training_contract(
            candidate_metadata=candidate_metadata,
            examples=base_examples,
            training_qrels_by_qid_idx=None,
        )
        write_candidate_metadata(cfg.output_dir, candidate_metadata)
        write_teacher_embedding_debug_sample(
            cfg,
            os.path.join(cfg.output_dir, "teacher_embedding_debug_sample.json"),
            base_examples,
            passages,
            teacher,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
        )
        probe_targets = probe_teacher_target_space(
            cfg, base_examples, passages, teacher, raw_doc_embs_cpu, raw_query_embs_cpu
        )
        teacher_output_dim = int(probe_targets.teacher_output_dim)
        log_objective_summary(cfg, teacher_output_dim)
        if cfg.dry_run:
            dry_output = dry_run_embeddistill(
                cfg,
                base_examples,
                passages,
                teacher,
                raw_doc_embs_cpu,
                raw_query_embs_cpu,
                candidate_metadata,
                teacher_output_dim,
            )
            payload = {
                "seed": cfg.seed,
                "mode": cfg.mode,
                "dataset_format": cfg.source_dataset_format or cfg.dataset_format,
                "dataset_name": cfg.dataset_name,
                "selected_variants": [system_name],
                "objective": OBJECTIVE_EMBEDDISTILL,
                "variant": cfg.embeddistill_variant,
                "dry_run": True,
                "quality_gate": quality,
                "candidate_metadata": candidate_metadata,
                "teacher_checkpoint_fingerprint_sha1": teacher_checkpoint_fingerprint_sha1,
                "teacher_output_dim": teacher_output_dim,
                "dry_run_variants": {system_name: dry_output},
                "results": {},
            }
            with open(
                os.path.join(cfg.output_dir, "final_results.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            base.write_csv_rows(os.path.join(cfg.output_dir, "final_results.csv"), [])
            return payload
        students[system_name] = train_embeddistill_student(
            cfg,
            base_examples,
            candidate_metadata,
            val_examples,
            passages,
            query_by_id,
            query_id_to_index,
            train_qids,
            qrels_by_qid_idx,
            teacher,
            raw_index,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
            teacher_output_dim,
            teacher_checkpoint_fingerprint_sha1,
            teacher.metadata,
        )
    elif cfg.mode == base.MODE_EVAL_ONLY:
        ckpt = base.training_checkpoint_dir(cfg, system_name)
        loaded = base.load_student_checkpoint(ckpt, cfg)
        if loaded is not None:
            students[system_name] = loaded

    results = base.evaluate_systems(
        cfg,
        teacher,
        raw_index,
        raw_doc_embs_cpu,
        raw_query_embs_cpu,
        passages,
        val_examples,
        test_examples,
        qrels_by_qid_idx,
        students,
    )
    payload = {
        "seed": cfg.seed,
        "mode": cfg.mode,
        "dataset_format": cfg.source_dataset_format or cfg.dataset_format,
        "dataset_name": cfg.dataset_name,
        "selected_variants": [system_name],
        "objective": OBJECTIVE_EMBEDDISTILL,
        "variant": cfg.embeddistill_variant,
        "dry_run": False,
        "quality_gate": quality,
        "candidate_metadata": candidate_metadata,
        "teacher_checkpoint_fingerprint_sha1": teacher_checkpoint_fingerprint_sha1,
        "teacher_output_dim": teacher_output_dim,
        "results": results,
    }
    with open(os.path.join(cfg.output_dir, "final_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    base.write_csv_rows(
        os.path.join(cfg.output_dir, "final_results.csv"), base.flatten_results(results)
    )
    base.write_comlq_test_breakdown_results(cfg, results)
    return payload


def main_embeddistill(cfg: EmbedDistillCfg) -> None:
    base.run_experiment(
        cfg,
        validate_embeddistill_cfg,
        run_one_seed,
        lambda current, path: experiment_utils.log_run_header(current, path, logger, __file__),
    )


def run_embeddistill_loss_tests() -> None:
    # 1. identical projected-student and teacher embeddings give zero L2 loss
    q_s = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    d_s = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    stats = embeddistill_alignment_per_example(
        q_s, d_s, q_s.clone(), d_s.clone(), mask, distance=EMBED_DISTANCE_L2
    )
    assert abs(float(stats.total.detach().cpu())) < 1e-8, (
        "Identical embeddings should give zero L2 loss"
    )

    # 2. L2 uses Euclidean norm, not squared L2 and not dimension-averaged MSE
    q_teacher = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    q_student = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    d_teacher = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    d_student = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    l2_stats = embeddistill_alignment_per_example(
        q_student,
        d_student,
        q_teacher,
        d_teacher,
        torch.tensor([[True]]),
        distance=EMBED_DISTANCE_L2,
    )
    assert abs(float(l2_stats.query_loss_per_query[0].detach().cpu()) - 5.0) < 1e-6, (
        "L2 must use Euclidean norm"
    )

    # 3. Query loss is calculated once per query
    q_teacher = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    q_student = torch.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
    d_teacher = torch.zeros((2, 2, 2), dtype=torch.float32)
    d_student = torch.zeros((2, 2, 2), dtype=torch.float32)
    stats_query = embeddistill_alignment_per_example(
        q_student,
        d_student,
        q_teacher,
        d_teacher,
        torch.tensor([[True, True], [True, True]]),
        distance=EMBED_DISTANCE_L2,
    )
    assert abs(float(stats_query.query_loss_per_query[0].detach().cpu()) - 5.0) < 1e-6
    assert abs(float(stats_query.query_loss_per_query[1].detach().cpu()) - 10.0) < 1e-6

    # 4. Document loss is averaged within each query
    d_teacher = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    d_student = torch.tensor([[[3.0, 4.0], [0.0, 8.0]]], dtype=torch.float32)
    stats_doc = embeddistill_alignment_per_example(
        torch.zeros((1, 2)),
        d_student,
        torch.zeros((1, 2)),
        d_teacher,
        mask,
        distance=EMBED_DISTANCE_L2,
    )
    expected_doc = (5.0 + 8.0) / 2.0
    assert abs(float(stats_doc.document_loss_per_query[0].detach().cpu()) - expected_doc) < 1e-6

    # 5. Queries receive equal weight despite different candidate counts
    q_teacher = torch.zeros((2, 2), dtype=torch.float32)
    q_student = torch.zeros((2, 2), dtype=torch.float32)
    d_teacher = torch.zeros((2, 3, 2), dtype=torch.float32)
    d_student = torch.tensor(
        [[[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]], [[6.0, 8.0], [0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    mask_eq = torch.tensor([[True, False, False], [True, True, False]], dtype=torch.bool)
    stats_eq = embeddistill_alignment_per_example(
        q_student, d_student, q_teacher, d_teacher, mask_eq, distance=EMBED_DISTANCE_L2
    )
    expected_total = (5.0 + (10.0 + 0.0) / 2.0) / 2.0
    assert abs(float(stats_eq.total.detach().cpu()) - expected_total) < 1e-6, (
        "Queries must be equally weighted"
    )

    # 6-7. Padded documents have no effect, even if padded teacher values change
    padded_mask = torch.tensor([[True, False]], dtype=torch.bool)
    d_student = torch.tensor([[[1.0, 0.0], [9.0, 9.0]]], dtype=torch.float32)
    d_teacher_a = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    d_teacher_b = torch.tensor([[[0.0, 0.0], [99.0, -50.0]]], dtype=torch.float32)
    stats_pad_a = embeddistill_alignment_per_example(
        torch.zeros((1, 2)),
        d_student,
        torch.zeros((1, 2)),
        d_teacher_a,
        padded_mask,
        distance=EMBED_DISTANCE_L2,
    )
    stats_pad_b = embeddistill_alignment_per_example(
        torch.zeros((1, 2)),
        d_student,
        torch.zeros((1, 2)),
        d_teacher_b,
        padded_mask,
        distance=EMBED_DISTANCE_L2,
    )
    assert (
        abs(float(stats_pad_a.total.detach().cpu()) - float(stats_pad_b.total.detach().cpu()))
        < 1e-8
    ), "Padded docs must not affect loss"

    # 8. Teacher tensors remain detached
    q_student = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    d_student = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32, requires_grad=True)
    q_teacher = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    d_teacher = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    det_stats = embeddistill_alignment_per_example(
        q_student,
        d_student,
        q_teacher,
        d_teacher,
        torch.tensor([[True]]),
        distance=EMBED_DISTANCE_L2,
    )
    det_stats.total.backward()
    assert q_teacher.grad is None and d_teacher.grad is None, "Teacher tensors must remain detached"

    # 9. Student and projection parameters receive gradients
    projection = SharedLinearProjection(2, 2, bias=True)
    q_input = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    d_input = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32, requires_grad=True)
    q_proj = projection(q_input)
    d_proj = projection(d_input)
    grad_stats = embeddistill_alignment_per_example(
        q_proj,
        d_proj,
        torch.zeros((1, 2)),
        torch.zeros((1, 1, 2)),
        torch.tensor([[True]]),
        distance=EMBED_DISTANCE_L2,
    )
    grad_stats.total.backward()
    assert q_input.grad is not None and torch.isfinite(q_input.grad).all().item(), (
        "Student query input should get gradients"
    )
    assert d_input.grad is not None and torch.isfinite(d_input.grad).all().item(), (
        "Student doc input should get gradients"
    )
    assert (
        projection.linear.weight.grad is not None
        and torch.isfinite(projection.linear.weight.grad).all().item()
    ), "Projection weights should get gradients"

    # 10. A single-candidate query still receives embedding supervision
    single_stats = embeddistill_alignment_per_example(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[[0.0, 2.0]]], dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2), dtype=torch.float32),
        torch.tensor([[True]]),
        distance=EMBED_DISTANCE_L2,
    )
    assert int(single_stats.valid_query_count.detach().cpu()) == 1, (
        "Single-candidate query should have valid embedding supervision"
    )
    assert float(single_stats.total.detach().cpu()) > 0.0, (
        "Single-candidate query should yield non-zero embedding loss"
    )

    # 11-12. rank_embed includes ranking loss, embed_only excludes it completely
    cfg_rank = EmbedDistillCfg(
        embeddistill_variant=EMBEDDISTILL_VARIANT_RANK_EMBED, embed_distance=EMBED_DISTANCE_L2
    )
    cfg_embed = EmbedDistillCfg(
        embeddistill_variant=EMBEDDISTILL_VARIANT_EMBED_ONLY, embed_distance=EMBED_DISTANCE_L2
    )
    labels_zero = torch.zeros((1, 2), dtype=torch.float32)
    teacher_targets = TeacherFinalTargets(
        teacher_query_embedding=torch.zeros((1, 2), dtype=torch.float32),
        teacher_document_embeddings=torch.zeros((1, 2, 2), dtype=torch.float32),
        teacher_scores=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        teacher_output_dim=2,
    )
    student_scores = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    projected_q = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    projected_d = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)
    common = dict(
        student_scores=student_scores,
        projected_student_query=projected_q,
        projected_student_documents=projected_d,
        teacher_targets=teacher_targets,
        labels=labels_zero,
        mask=torch.tensor([[True, True]]),
        student_query_embeddings=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        frozen_query_embeddings=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
    )
    rank_loss = compute_embeddistill_loss(cfg_rank, **common)
    embed_loss = compute_embeddistill_loss(cfg_embed, **common)
    assert float(rank_loss["score_kd_loss_sum"].detach().cpu()) > 0.0, (
        "rank_embed should include ranking loss"
    )
    assert abs(float(embed_loss["score_kd_loss_sum"].detach().cpu())) < 1e-8, (
        "embed_only should exclude ranking loss"
    )
    assert float(rank_loss["total"].detach().cpu()) > float(embed_loss["total"].detach().cpu()), (
        "rank_embed total should exceed embed_only when score KD is non-zero"
    )

    # 13. No frozen-E5 subtraction is performed in the objective
    frozen_a = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    frozen_b = torch.tensor([[9.0, -4.0]], dtype=torch.float32)
    loss_a = compute_embeddistill_loss(cfg_rank, **{**common, "frozen_query_embeddings": frozen_a})
    loss_b = compute_embeddistill_loss(cfg_rank, **{**common, "frozen_query_embeddings": frozen_b})
    assert (
        abs(float(loss_a["total"].detach().cpu()) - float(loss_b["total"].detach().cpu())) < 1e-8
    ), "Objective must not depend on frozen-E5 subtraction"

    # 14. The loss does not access delta_q_T or delta_d_T
    protected = NoDeltaAccessSignals(
        q_mod_T=torch.zeros((1, 2), dtype=torch.float32),
        d_mod_T=torch.zeros((1, 1, 2), dtype=torch.float32),
        teacher_scores=torch.zeros((1, 1), dtype=torch.float32),
        delta_q_T=torch.ones((1, 2), dtype=torch.float32),
        delta_d_T=torch.ones((1, 1, 2), dtype=torch.float32),
    )
    selected = select_final_teacher_targets(protected)
    assert selected.teacher_output_dim == 2, (
        "Teacher target selector should work without touching delta fields"
    )

    # 15-16. Nonzero qrel labels raise AssertionError, and the test does not self-pass
    caught_nonzero = False
    try:
        assert_zero_labels(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    except AssertionError:
        caught_nonzero = True
    assert caught_nonzero, "Non-zero qrel labels must raise AssertionError"

    # 17. Projection save and reload reproduce identical outputs
    projection = SharedLinearProjection(2, 3, bias=True)
    sample_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "proj.pt")
        torch.save(projection.state_dict(), path)
        reloaded = SharedLinearProjection(2, 3, bias=True)
        reloaded.load_state_dict(torch.load(path, map_location="cpu"))
        out_a = projection(sample_input)
        out_b = reloaded(sample_input)
        assert torch.allclose(out_a, out_b), "Projection reload should reproduce identical outputs"

    # 18. Losses remain finite with AMP-compatible inputs
    half_stats = embeddistill_alignment_per_example(
        torch.tensor([[1.0, 0.0]], dtype=torch.float16),
        torch.tensor([[[0.0, 1.0]]], dtype=torch.float16),
        torch.zeros((1, 2), dtype=torch.float16),
        torch.zeros((1, 1, 2), dtype=torch.float16),
        torch.tensor([[True]]),
        distance=EMBED_DISTANCE_L2,
    )
    assert torch.isfinite(half_stats.total).item(), (
        "Loss must remain finite for AMP-compatible inputs"
    )

    # 19. Query and document teacher dimensions are checked
    mismatch_caught = False
    try:
        embeddistill_alignment_per_example(
            torch.zeros((1, 2), dtype=torch.float32),
            torch.zeros((1, 1, 3), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.float32),
            torch.zeros((1, 1, 3), dtype=torch.float32),
            torch.tensor([[True]]),
        )
    except ValueError:
        mismatch_caught = True
    assert mismatch_caught, "Teacher/student dimension mismatch should raise ValueError"

    # 20. A dataset/teacher metadata mismatch fails clearly
    mismatch_cfg = EmbedDistillCfg(
        dataset_name="contractnli",
        dataset_format="legalbench_rag",
        source_dataset_format="legalbench_rag",
        teacher_checkpoint_path="dummy.pt",
    )
    mismatch_failed = False
    try:
        assert_teacher_metadata_compatible(
            mismatch_cfg,
            {"dataset": "privacy_qa", "encoder": "e5", "model_config": {"output_dim": 256}},
            256,
            "dummy.pt",
        )
    except ValueError as exc:
        mismatch_failed = "dataset mismatch" in str(exc)
    assert mismatch_failed, "Teacher metadata mismatch should fail clearly"

    # 21. Historical and canonical Qwen3-Embedding-0.6B aliases are equivalent.
    qwen_cfg = EmbedDistillCfg(
        dataset_name="scifact",
        dataset_format="comlq",
        source_dataset_format="beir",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        seed=42,
        teacher_checkpoint_path="dummy.pt",
    )
    assert_teacher_metadata_compatible(
        qwen_cfg,
        {
            "dataset": "scifact",
            "seed": 42,
            "encoder_model_name": "Qwen/Qwen3-Embedding-0.6B",
            "normalized_encoder": "qwen3_embedding",
            "model_config": {"output_dim": 1024},
        },
        1024,
        "dummy.pt",
    )

    logger.info("EmbedDistill synthetic loss tests passed")


def parse_embeddistill_args() -> EmbedDistillCfg:
    parser = argparse.ArgumentParser(
        description="EmbedDistill-style baseline for dense-retriever post-training from frozen teacher targets"
    )
    parser.add_argument(
        "--dataset_format",
        choices=["auto", "standard", "comlq", "legalbench_rag"],
        default="auto",
    )
    parser.add_argument("--dataset_dir", default="datasets/comlq/dataset")
    parser.add_argument("--corpus_file", default="corpus.jsonl")
    parser.add_argument("--queries_file", default="queries.jsonl")
    parser.add_argument("--qrels_dir", default="qrels")
    parser.add_argument("--train_queries_file", default="")
    parser.add_argument(
        "--legalbench_rag_root",
        default="",
        help="Root containing LegalBench-RAG corpus/ and benchmarks/. Defaults to --dataset_dir when dataset_format=legalbench_rag.",
    )
    parser.add_argument(
        "--legalbench_rag_datasets",
        nargs="+",
        default=list(LEGALBENCH_RAG_DATASETS),
        help="LegalBench-RAG datasets to run: privacy_qa contractnli.",
    )
    parser.add_argument("--legalbench_prepared_dir", default="")
    parser.add_argument("--legalbench_extract_dir", default="")
    parser.add_argument("--legalbench_chunk_size", type=int, default=500)
    parser.add_argument("--legalbench_chunk_strategy", choices=["naive", "rcts"], default="naive")
    parser.add_argument(
        "--query_type_filter",
        choices=["all", "negation", "conjunction", "union", "projection", "custom"],
        default="all",
    )
    parser.add_argument("--query_types", default="")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--use_binary_relevance", action="store_true")
    parser.add_argument("--no_report_comlq_slices", action="store_true")

    parser.add_argument(
        "--teacher_checkpoint_path",
        default="",
        help="Frozen original IMRNN teacher checkpoint path for a single dataset run.",
    )
    parser.add_argument(
        "--teacher_checkpoint_template",
        default="",
        help="Teacher checkpoint template for suite runs, supporting {dataset} and {seed} placeholders.",
    )
    parser.add_argument("--allow_teacher_metadata_mismatch", action="store_true")
    parser.add_argument("--teacher_output_dim", type=int)
    parser.add_argument("--teacher_hidden_dim", type=int)
    parser.add_argument("--teacher_dropout", type=float, default=0.1)
    parser.add_argument(
        "--auto_train_teacher_if_missing",
        action="store_true",
        help="Explicit opt-in: if the requested teacher checkpoint is missing, train the original IMRNN teacher for compatibility.",
    )
    parser.add_argument("--teacher_train_epochs", type=int, default=10)
    parser.add_argument("--teacher_train_batch_size", type=int, default=32)
    parser.add_argument("--teacher_train_lr", type=float, default=1e-4)
    parser.add_argument("--teacher_train_weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--teacher_train_num_negatives",
        "--teacher_num_negatives",
        dest="teacher_train_num_negatives",
        type=int,
        default=20,
    )
    parser.add_argument("--teacher_train_negative_pool", type=int, default=200)
    parser.add_argument("--teacher_label_temp", type=float, default=1.0)
    parser.add_argument("--teacher_max_positives", type=int, default=0)
    parser.add_argument("--teacher_max_candidates", type=int, default=0)
    parser.add_argument(
        "--teacher_metric_select", dest="teacher_metric_select", action="store_true", default=True
    )
    parser.add_argument(
        "--no_teacher_metric_select", dest="teacher_metric_select", action="store_false"
    )
    parser.add_argument("--teacher_select_recall_k", type=int, default=50)
    parser.add_argument("--teacher_select_ndcg_k", type=int, default=10)
    parser.add_argument("--teacher_select_mrr_k", type=int, default=10)
    parser.add_argument("--teacher_recall_w", type=float, default=1.0)
    parser.add_argument("--teacher_ndcg_w", type=float, default=0.25)
    parser.add_argument("--teacher_mrr_w", type=float, default=0.10)
    parser.add_argument("--allow_weak_teacher", action="store_true")

    parser.add_argument("--model_name", default="intfloat/e5-large-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--student_max_seq_length", type=int, default=0)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    parser.add_argument(
        "--mode",
        choices=[base.MODE_UNSUPERVISED, base.MODE_EVAL_ONLY],
        default=base.MODE_UNSUPERVISED,
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_loss_tests", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--corpus_encode_batch_size", type=int, default=64)
    parser.add_argument("--optimizer", choices=["auto", "adamw", "adafactor"], default="auto")
    parser.add_argument("--auto_adafactor_gpu_gb", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    parser.add_argument(
        "--embeddistill_variant",
        choices=[EMBEDDISTILL_VARIANT_RANK_EMBED, EMBEDDISTILL_VARIANT_EMBED_ONLY],
        default=EMBEDDISTILL_VARIANT_RANK_EMBED,
    )
    parser.add_argument(
        "--embed_distance",
        choices=[EMBED_DISTANCE_L2, EMBED_DISTANCE_COSINE, EMBED_DISTANCE_MSE],
        default=EMBED_DISTANCE_L2,
    )
    parser.add_argument("--embed_query_weight", type=float, default=1.0)
    parser.add_argument("--embed_document_weight", type=float, default=1.0)
    parser.add_argument("--embed_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--projection_bias", dest="projection_bias", action="store_true", default=True
    )
    parser.add_argument("--no_projection_bias", dest="projection_bias", action="store_false")
    parser.add_argument("--tau", type=float, default=1.0)

    parser.add_argument("--candidate_pool_k", type=int, default=64)
    parser.add_argument("--max_train_candidates_per_query", type=int, default=32)
    parser.add_argument("--feedback_k", type=int, default=100)
    parser.add_argument("--disable_l40s_safe_mode", action="store_true")

    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--eval_k", type=int, default=64)
    parser.add_argument("--selection_recall_w", type=float, default=1.0)
    parser.add_argument("--selection_ndcg_w", type=float, default=0.25)
    parser.add_argument("--selection_mrr_w", type=float, default=0.10)

    parser.add_argument("--frozen_corpus_emb_path", default="base_corpus_embeddings.pt")
    parser.add_argument("--frozen_query_emb_path", default="base_query_embeddings.pt")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--projection_head_weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", default="runs/embeddistill")
    parser.add_argument("--log_path", default="")
    parser.add_argument("--debug_teacher_samples", type=int, default=5)
    args = parser.parse_args()

    cfg = EmbedDistillCfg(
        dataset_format=args.dataset_format,
        dataset_dir=args.dataset_dir,
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        qrels_dir=args.qrels_dir,
        train_queries_file=args.train_queries_file,
        legalbench_rag_root=args.legalbench_rag_root,
        legalbench_rag_datasets=tuple(args.legalbench_rag_datasets),
        legalbench_prepared_dir=args.legalbench_prepared_dir,
        legalbench_extract_dir=args.legalbench_extract_dir,
        legalbench_chunk_size=args.legalbench_chunk_size,
        legalbench_chunk_strategy=args.legalbench_chunk_strategy,
        query_type_filter=args.query_type_filter,
        query_types=args.query_types,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_graded_relevance=not args.use_binary_relevance,
        report_comlq_slices=not args.no_report_comlq_slices,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        teacher_checkpoint_template=args.teacher_checkpoint_template,
        allow_teacher_metadata_mismatch=args.allow_teacher_metadata_mismatch,
        teacher_output_dim=args.teacher_output_dim,
        teacher_hidden_dim=args.teacher_hidden_dim,
        teacher_dropout=args.teacher_dropout,
        auto_train_teacher_if_missing=args.auto_train_teacher_if_missing,
        teacher_train_epochs=args.teacher_train_epochs,
        teacher_train_batch_size=args.teacher_train_batch_size,
        teacher_train_lr=args.teacher_train_lr,
        teacher_train_weight_decay=args.teacher_train_weight_decay,
        teacher_train_num_negatives=args.teacher_train_num_negatives,
        teacher_train_negative_pool=args.teacher_train_negative_pool,
        teacher_label_temp=args.teacher_label_temp,
        teacher_max_positives=args.teacher_max_positives,
        teacher_max_candidates=args.teacher_max_candidates,
        teacher_metric_select=args.teacher_metric_select,
        teacher_select_recall_k=args.teacher_select_recall_k,
        teacher_select_ndcg_k=args.teacher_select_ndcg_k,
        teacher_select_mrr_k=args.teacher_select_mrr_k,
        teacher_recall_w=args.teacher_recall_w,
        teacher_ndcg_w=args.teacher_ndcg_w,
        teacher_mrr_w=args.teacher_mrr_w,
        allow_weak_teacher=args.allow_weak_teacher,
        model_name=args.model_name,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        embedding_dim=args.embedding_dim,
        device=args.device,
        amp=not args.no_amp,
        student_max_seq_length=args.student_max_seq_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        mode=base.MODE_EVAL_ONLY if args.eval_only else args.mode,
        resume=args.resume,
        dry_run=args.dry_run,
        run_loss_tests=args.run_loss_tests,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
        corpus_encode_batch_size=args.corpus_encode_batch_size,
        optimizer=args.optimizer,
        auto_adafactor_gpu_gb=args.auto_adafactor_gpu_gb,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        embeddistill_variant=args.embeddistill_variant,
        embed_distance=args.embed_distance,
        embed_query_weight=args.embed_query_weight,
        embed_document_weight=args.embed_document_weight,
        embed_loss_weight=args.embed_loss_weight,
        projection_bias=args.projection_bias,
        tau=args.tau,
        candidate_pool_k=args.candidate_pool_k,
        max_train_candidates_per_query=args.max_train_candidates_per_query,
        feedback_k=args.feedback_k,
        l40s_safe_mode=not args.disable_l40s_safe_mode,
        ks=tuple(args.ks),
        eval_k=args.eval_k,
        selection_recall_w=args.selection_recall_w,
        selection_ndcg_w=args.selection_ndcg_w,
        selection_mrr_w=args.selection_mrr_w,
        frozen_corpus_emb_path=args.frozen_corpus_emb_path,
        frozen_query_emb_path=args.frozen_query_emb_path,
        force_rebuild_embedding_cache=args.force_rebuild_cache,
        projection_head_weight_decay=args.projection_head_weight_decay,
        output_dir=args.output_dir,
        log_path=args.log_path,
        debug_teacher_samples=args.debug_teacher_samples,
    )
    cfg.variants = (system_name_for_cfg(cfg),)
    return cfg


def main() -> None:
    cfg = parse_embeddistill_args()
    validate_embeddistill_cfg(cfg)
    if cfg.run_loss_tests:
        log_path = base.setup_logging(cfg)
        experiment_utils.log_run_header(cfg, log_path, logger, __file__)
        run_embeddistill_loss_tests()
        return
    main_embeddistill(cfg)


if __name__ == "__main__":
    main()
