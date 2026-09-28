#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core training and evaluation pipeline for Representation Correction Distillation.

The student remains a plain bi-encoder at inference time:
  query/document text -> encoder -> normalized embedding -> cosine score.

The selected frozen teacher is used only during training/evaluation to produce:
  1. candidate-list scores for teacher-ranking supervision
  2. internal projected-space final representations and modulation signals:
       q_mod_T, d_mod_T
       delta_q_T = q_mod_T - q_base_T
       delta_d_T = d_mod_T - d_base_T

Student training is qrel-free: qrels are used only to define benchmark splits,
select validation checkpoints, and compute final retrieval metrics.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import experiment_utils, provenance
from .candidates import build_unsupervised_train_examples
from .checkpoints import (
    find_resume_checkpoint_dir,
    load_training_state_payload,
    mark_student_training_complete,
    resolve_student_checkpoint_model_dir,
    student_training_complete,
    training_checkpoint_dir,
)
from .comlq import build_slices
from .config import (
    MODE_EVAL_ONLY,
    MODE_UNSUPERVISED,
    VARIANT_CORRECTION_ONLY,
    VARIANT_E5_BASE,
    VARIANT_FULL_RCD,
    VARIANT_IMRNN_TEACHER,
    VARIANT_RANKING_ONLY,
    Cfg,
)
from .data_utils import SearchIndex, build_eval_examples
from .dataset_io import (
    build_qrels_by_qid_idx,
    infer_dataset_format,
    load_passages,
    load_queries,
    log_schema_summary,
    make_split_qids,
    validate_references,
)
from .embedding_cache import (
    build_or_load_frozen_corpus_embeddings,
    build_or_load_frozen_query_embeddings,
)
from .evaluation import (
    encode_corpus_with_student,
    evaluate_base_retriever,
    evaluate_base_with_teacher,
    evaluate_student,
    log_metrics_table,
    validation_score,
)
from .legalbench import (
    ensure_data_root as ensure_legalbench_rag_data_root,
)
from .legalbench import (
    normalize_dataset_name as normalize_legalbench_dataset_name,
)
from .legalbench import (
    prepare_retrieval_dataset as prepare_legalbench_retrieval_dataset,
)
from .objectives import (
    assert_teacher_frozen,
    compute_variant_loss,
    projection_head_norm,
    representation_alignment_target,
    teacher_distribution_diagnostics,
    variant_uses_modulation,
    variant_uses_rank,
)
from .records import EvalExample, PassageRecord, TrainExample
from .results import (
    aggregate_seed_results,
    flatten_results,
    write_comlq_test_breakdown_results,
    write_csv_rows,
    write_legalbench_suite_results,
)
from .runtime import (
    SentenceTransformer,
    build_optimizer,
    clear_cuda_cache_if_needed,
    get_linear_schedule_with_warmup,
    log_section,
    logger,
    move_features_to_device,
    require_runtime_dependencies,
    set_seed,
    setup_logging,
)
from .training_data import (
    StudentCandidateDataset,
    collate_student_candidates,
    slice_training_batch,
)

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError("This script requires numpy. Install it with `pip install numpy`.") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires PyTorch. Install it from https://pytorch.org/get-started/locally/."
    ) from exc


TRAINABLE_VARIANTS = {
    VARIANT_RANKING_ONLY,
    VARIANT_CORRECTION_ONLY,
    VARIANT_FULL_RCD,
}

BASELINE_VARIANTS = {VARIANT_E5_BASE, VARIANT_IMRNN_TEACHER}


def log_run_header(cfg: Cfg, log_path: str) -> None:
    experiment_utils.log_run_header(cfg, log_path, logger, __file__)


def infer_teacher_config_from_state(
    state_dict: Dict[str, torch.Tensor], cfg: Cfg
) -> Tuple[int, int, int, float]:
    projector_weight = state_dict.get("projector.weight")
    if projector_weight is None:
        raise ValueError(
            "Teacher checkpoint does not contain projector.weight; cannot infer IMRNN dimensions"
        )
    output_dim = int(cfg.teacher_output_dim or projector_weight.shape[0])
    input_dim = int(projector_weight.shape[1])
    hidden_key = "query_hypernet.hypernet.0.weight"
    hidden_weight = state_dict.get(hidden_key)
    hidden_dim = int(
        cfg.teacher_hidden_dim or (hidden_weight.shape[0] if hidden_weight is not None else 128)
    )
    return input_dim, output_dim, hidden_dim, float(cfg.teacher_dropout)


def resolve_teacher_checkpoint_path(cfg: Cfg) -> Path:
    raw_path = str(cfg.teacher_checkpoint_path).strip()
    if raw_path.lower() in {"", "auto", "__auto_train_teacher__", "auto_train"}:
        raise FileNotFoundError(
            "Teacher checkpoint auto-training was requested for this dataset/seed."
        )
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path
    raise FileNotFoundError(f"Teacher checkpoint not found: {path}")


def default_trained_teacher_checkpoint_path(cfg: Cfg) -> Path:
    dataset_slug = Path(cfg.dataset_dir).name or "dataset"
    encoder_slug = "qwen3" if "qwen" in cfg.model_name.lower() else "e5"
    fraction = float(getattr(cfg, "_teacher_fraction", 1.0))
    fraction_tag = f"f{int(round(fraction * 100)):03d}"
    filename = (
        f"imrnns-{encoder_slug}-{dataset_slug}-{fraction_tag}-"
        f"{cfg.teacher_loss}-seed-{cfg.seed}.pt"
    )
    return Path(cfg.output_dir) / "teacher" / filename




def metric_at_or_zero(metrics: Dict[int, Dict[str, float]], k: int, metric_name: str) -> float:
    if not metrics:
        return 0.0
    key = int(k)
    if key not in metrics:
        key = max(metrics.keys())
    return float(metrics.get(key, {}).get(metric_name, 0.0))



def train_teacher_checkpoint(*args: Any, **kwargs: Any) -> Path:
    raise RuntimeError(
        "No teacher-training backend is installed. Run through `rcd-train`, which installs "
        "the IMRNNS or Search-Adaptor backend before starting the experiment."
    )


def ensure_teacher_checkpoint(
    cfg: Cfg,
    split_ids: Dict[str, List[str]],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    query_id_to_index: Dict[str, int],
    raw_index: SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    split_source: str,
) -> Path:
    try:
        path = resolve_teacher_checkpoint_path(cfg)
        cfg.teacher_checkpoint_path = str(path)
        return path
    except FileNotFoundError as exc:
        if not cfg.auto_train_teacher_if_missing:
            raise
        if cfg.mode == MODE_EVAL_ONLY:
            raise FileNotFoundError(
                f"{exc}\nmode=EVAL_ONLY cannot auto-train the missing teacher checkpoint."
            ) from exc
        raw_path = str(cfg.teacher_checkpoint_path).strip()
        use_default_path = (
            raw_path.lower() in {"auto", "__auto_train_teacher__", "auto_train"} or not raw_path
        )
        checkpoint_path = (
            default_trained_teacher_checkpoint_path(cfg) if use_default_path else Path(raw_path)
        )
        if not use_default_path and not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        logger.warning(f"{exc}\nAuto-training IMRNN teacher checkpoint at: {checkpoint_path}")
        trained_path = train_teacher_checkpoint(
            cfg,
            checkpoint_path,
            split_ids,
            qrels_by_qid_idx,
            query_id_to_index,
            raw_index,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
            split_source,
        )
        cfg.teacher_checkpoint_path = str(trained_path)
        return trained_path


class TeacherAdapter:
    def __init__(self, cfg: Cfg):
        del cfg
        raise RuntimeError(
            "No teacher backend is installed. Run through `rcd-train` so the requested backend "
            "is installed before the experiment starts."
        )


class TrainableRetriever(nn.Module):
    def __init__(self, model_name_or_path: str, cfg: Cfg):
        super().__init__()
        require_runtime_dependencies()
        self.cfg = cfg
        self.encoder = SentenceTransformer(model_name_or_path, device=cfg.device)
        self._configure_encoder_memory()
        self.encoder.to(cfg.device)

    def _format(self, texts: List[str], prefix: str) -> List[str]:
        return [f"{prefix}{text}" if prefix else text for text in texts]

    def _configure_encoder_memory(self) -> None:
        first_module = (
            self.encoder._first_module() if hasattr(self.encoder, "_first_module") else None
        )
        if (
            self.cfg.student_max_seq_length
            and first_module is not None
            and hasattr(first_module, "max_seq_length")
        ):
            old_len = getattr(first_module, "max_seq_length")
            setattr(first_module, "max_seq_length", int(self.cfg.student_max_seq_length))
            logger.info(
                f"Student max_seq_length set to {self.cfg.student_max_seq_length} (was {old_len})"
            )
        auto_model = getattr(first_module, "auto_model", None)
        if (
            self.cfg.gradient_checkpointing
            and auto_model is not None
            and hasattr(auto_model, "gradient_checkpointing_enable")
        ):
            auto_model.gradient_checkpointing_enable()
            if hasattr(auto_model, "config") and hasattr(auto_model.config, "use_cache"):
                auto_model.config.use_cache = False
            logger.info("Enabled student encoder gradient checkpointing to reduce training memory")

    def encode_query_texts_train(self, texts: List[str]) -> torch.Tensor:
        return self._encode_train(self._format(texts, self.cfg.query_prefix))

    def encode_doc_texts_train(self, texts: List[str]) -> torch.Tensor:
        return self._encode_train(self._format(texts, self.cfg.passage_prefix))

    def _encode_train(self, texts: List[str]) -> torch.Tensor:
        features = self.encoder.tokenize(texts)
        features = move_features_to_device(features, self.cfg.device)
        output = self.encoder(features)
        return F.normalize(output["sentence_embedding"], p=2, dim=-1)

    @torch.no_grad()
    def encode_query_texts_eval(self, texts: List[str], batch_size: int) -> torch.Tensor:
        self.encoder.eval()
        emb = self.encoder.encode(
            self._format(texts, self.cfg.query_prefix),
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.cfg.device,
            normalize_embeddings=True,
        )
        return emb.detach().float().cpu()

    @torch.no_grad()
    def encode_doc_texts_eval(self, texts: List[str], batch_size: int) -> torch.Tensor:
        self.encoder.eval()
        emb = self.encoder.encode(
            self._format(texts, self.cfg.passage_prefix),
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.cfg.device,
            normalize_embeddings=True,
        )
        return emb.detach().float().cpu()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.encoder.save(path)


class SupervisionProjectionHead(nn.Module):
    """Base-retriever correction vector -> teacher supervision space.

    This head is used only for modulation-signal losses. Retrieval scoring always
    uses the student's native normalized embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(delta))


def save_student_checkpoint(
    student: TrainableRetriever,
    projection_head: Optional[SupervisionProjectionHead],
    checkpoint_dir: str,
    cfg: Cfg,
    variant: str,
    epoch: int,
    best_score: float,
    best_metrics: Dict[int, Dict[str, float]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
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
        "checkpoint_format": "dense-retriever-student-rcd-v3",
        "checkpoint_kind": checkpoint_kind,
        "variant": variant,
        "representation_alignment_target": representation_alignment_target(variant),
        "mode": cfg.mode,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_score": best_score,
        "best_val_metrics": best_metrics,
        "patience_ctr": patience_ctr,
        "completed": completed,
        "qrels_used_in_training": bool(candidate_metadata.get("qrels_used_in_training", False)),
        "qrel_positive_injection": bool(candidate_metadata.get("qrel_positive_injection", False)),
        "candidate_metadata": candidate_metadata,
        "supervision_projection": {
            "input_dim": cfg.embedding_dim,
            "output_dim": cfg.signal_projection_output_dim,
            "architecture": "linear_layer_norm",
        }
        if projection_head is not None
        else None,
        "teacher_checkpoint_path": cfg.teacher_checkpoint_path,
        "config": asdict(cfg),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata = provenance.add_run_contract(
        metadata,
        cfg=cfg,
        objective="rcd",
        variant=variant,
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
            "projection_head_state": projection_head.state_dict()
            if projection_head is not None
            else None,
            "metadata": metadata,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    logger.info(
        f"Saved {checkpoint_kind} {variant} student checkpoint to {checkpoint_dir} | "
        f"epoch={epoch} | best_score={best_score:.6f} | patience={patience_ctr} | completed={completed}"
    )


def save_latest_student_checkpoint(
    student: TrainableRetriever,
    projection_head: Optional[SupervisionProjectionHead],
    checkpoint_dir: str,
    cfg: Cfg,
    variant: str,
    epoch: int,
    best_score: float,
    best_metrics: Dict[int, Dict[str, float]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    patience_ctr: int,
    completed: bool = False,
    candidate_metadata: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
) -> None:
    latest_dir = os.path.join(checkpoint_dir, "latest")
    save_student_checkpoint(
        student=student,
        projection_head=projection_head,
        checkpoint_dir=latest_dir,
        cfg=cfg,
        variant=variant,
        epoch=epoch,
        best_score=best_score,
        best_metrics=best_metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        patience_ctr=patience_ctr,
        completed=completed,
        checkpoint_kind="latest",
        candidate_metadata=candidate_metadata,
        scaler=scaler,
    )


def load_student_checkpoint(checkpoint_dir: str, cfg: Cfg) -> Optional[TrainableRetriever]:
    model_dir = resolve_student_checkpoint_model_dir(checkpoint_dir)
    if model_dir is None:
        logger.warning(f"Student checkpoint directory missing; skipping: {checkpoint_dir}")
        return None
    logger.info(f"Loading student checkpoint: {model_dir}")
    student = TrainableRetriever(model_dir, cfg).to(cfg.device)
    metadata_path = os.path.join(model_dir, "student_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        logger.info(
            f"student_checkpoint_metadata:\n{json.dumps(metadata, indent=2, sort_keys=True, default=str)}"
        )
    return student


def canonical_variant(name: str) -> str:
    value = name.strip().upper()
    aliases = {
        "BASE": VARIANT_E5_BASE,
        "TEACHER": VARIANT_IMRNN_TEACHER,
        "RANKING": VARIANT_RANKING_ONLY,
        "CORRECTION": VARIANT_CORRECTION_ONLY,
        "RCD": VARIANT_FULL_RCD,
    }
    value = aliases.get(value, value)
    valid = {
        VARIANT_E5_BASE,
        VARIANT_IMRNN_TEACHER,
        VARIANT_RANKING_ONLY,
        VARIANT_CORRECTION_ONLY,
        VARIANT_FULL_RCD,
    }
    if value not in valid:
        raise ValueError(f"Unknown variant {name!r}. Valid variants: {sorted(valid)}")
    return value


def validate_mode_and_variants(cfg: Cfg) -> None:
    cfg.variants = tuple(canonical_variant(variant) for variant in cfg.variants)
    requested = set(cfg.variants)
    if cfg.mode == MODE_UNSUPERVISED:
        pass
    elif cfg.mode == MODE_EVAL_ONLY:
        pass
    else:
        raise ValueError("--mode must be UNSUPERVISED or EVAL_ONLY")

    unknown_trainless = requested - TRAINABLE_VARIANTS - BASELINE_VARIANTS
    if unknown_trainless:
        raise ValueError(f"Unsupported variants requested: {sorted(unknown_trainless)}")
    logger.info("variant_selection: selected_variants=%s", list(cfg.variants))


def make_seed_cfg(cfg: Cfg, seed: int) -> Cfg:
    seed_output = os.path.join(cfg.output_dir, f"seed_{seed}")
    raw_teacher_path = str(cfg.teacher_checkpoint_path).strip()
    template = str(getattr(cfg, "teacher_checkpoint_template", "")).strip()
    if not template and ("{seed}" in raw_teacher_path or "{dataset}" in raw_teacher_path):
        template = raw_teacher_path
    dataset = cfg.dataset_name or Path(cfg.dataset_dir).name or "dataset"
    if template:
        try:
            teacher_path = template.format(dataset=dataset, seed=int(seed))
        except KeyError as exc:
            raise ValueError(
                f"Unsupported teacher checkpoint placeholder {exc!s}; use {{dataset}} and/or {{seed}}."
            ) from exc
        if not teacher_path.strip():
            raise ValueError("Teacher checkpoint template resolved to an empty path")
    else:
        default_path = str(Cfg.__dataclass_fields__["teacher_checkpoint_path"].default)
        auto_requested = raw_teacher_path.lower() in {
            "",
            "auto_train",
            "__auto_train_teacher__",
        }
        if cfg.auto_train_teacher_if_missing and (
            auto_requested or raw_teacher_path == default_path
        ):
            teacher_path = "__auto_train_teacher__"
        elif len(set(int(value) for value in cfg.seeds)) > 1 and raw_teacher_path.lower() != "auto":
            raise ValueError(
                "A single explicit teacher checkpoint cannot be reused across multiple seeds. "
                "Pass --teacher_checkpoint_template with {seed}, or use an auto-trained teacher."
            )
        else:
            teacher_path = cfg.teacher_checkpoint_path
    return replace(
        cfg,
        seed=int(seed),
        output_dir=seed_output,
        log_path="",
        frozen_corpus_emb_path=os.path.join(seed_output, "cache", "frozen_base_corpus.pt"),
        frozen_query_emb_path=os.path.join(seed_output, "cache", "frozen_base_queries.pt"),
        teacher_checkpoint_path=teacher_path,
        teacher_checkpoint_template="",
    )


def validate_unsup_cfg(cfg: Cfg) -> None:
    validate_mode_and_variants(cfg)
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
        raise ValueError("IMRNN teacher training requires listnet")
    if cfg.teacher_label_temp <= 0:
        raise ValueError("--teacher_label_temp must be positive")
    if cfg.teacher_max_positives < 0:
        raise ValueError("--teacher_max_positives must be non-negative")
    if cfg.teacher_max_candidates < 0:
        raise ValueError("--teacher_max_candidates must be non-negative")
    for name in ("teacher_select_recall_k", "teacher_select_ndcg_k", "teacher_select_mrr_k"):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not cfg.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if not cfg.variants:
        raise ValueError("--variants must contain at least one variant")
    if cfg.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if cfg.micro_batch_size < 1:
        raise ValueError("--micro_batch_size must be positive")
    if cfg.candidate_pool_k < 1:
        raise ValueError("--candidate_pool_k must be positive")
    if cfg.max_train_candidates_per_query < 1:
        raise ValueError("--max_train_candidates_per_query must be positive")
    if cfg.eval_batch_size < 1 or cfg.corpus_encode_batch_size < 1:
        raise ValueError("--eval_batch_size and --corpus_encode_batch_size must be positive")
    if cfg.legalbench_chunk_strategy not in {"naive", "rcts"}:
        raise ValueError("--legalbench_chunk_strategy must be naive or rcts")
    if cfg.legalbench_chunk_size < 1:
        raise ValueError("--legalbench_chunk_size must be positive")
    if cfg.teacher_train_epochs < 1:
        raise ValueError("--teacher_train_epochs must be positive")
    if cfg.teacher_train_batch_size < 1:
        raise ValueError("--teacher_train_batch_size must be positive")
    if cfg.teacher_train_num_negatives < 1:
        raise ValueError("--teacher_train_num_negatives must be positive")
    if cfg.teacher_train_negative_pool < cfg.teacher_train_num_negatives:
        logger.warning(
            f"teacher_train_negative_pool={cfg.teacher_train_negative_pool} is smaller than "
            f"teacher_train_num_negatives={cfg.teacher_train_num_negatives}; increasing pool."
        )
        cfg.teacher_train_negative_pool = cfg.teacher_train_num_negatives
    if cfg.eval_k <= 0:
        if 50 in cfg.ks:
            cfg.eval_k = 50
        else:
            cfg.eval_k = 10 if 10 in cfg.ks else max(cfg.ks)
    if cfg.feedback_k < max(cfg.ks):
        logger.warning(
            f"feedback_k={cfg.feedback_k} is smaller than max ks={max(cfg.ks)}; reranking will use max ks"
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
            f"gradient_checkpointing={cfg.gradient_checkpointing}, PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}"
        )


def log_variant_plan(cfg: Cfg, variant: str) -> None:
    logger.info(
        "variant_plan: "
        f"variant={variant} | uses_rank={variant_uses_rank(variant)} | "
        f"uses_correction={variant_uses_modulation(variant)} | "
        f"representation_alignment_target={representation_alignment_target(variant)} | "
        f"uses_qrels=False | checkpoint_path={training_checkpoint_dir(cfg, variant)}"
    )


def train_variant(
    variant: str,
    cfg: Cfg,
    initial_examples: List[TrainExample],
    candidate_metadata: Dict[str, Any],
    val_examples: List[EvalExample],
    passages: List[PassageRecord],
    training_qrels_by_qid_idx: Optional[Dict[str, Dict[int, float]]],
    eval_qrels_by_qid_idx: Dict[str, Dict[int, float]],
    teacher: TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
) -> TrainableRetriever:
    log_section(f"TRAIN VARIANT | {variant} | seed={cfg.seed}")
    log_variant_plan(cfg, variant)
    if cfg.mode == MODE_UNSUPERVISED:
        if training_qrels_by_qid_idx is not None:
            raise AssertionError("UNSUPERVISED train_variant received training qrels")
        if candidate_metadata.get("qrels_used_in_training", False):
            raise AssertionError("UNSUPERVISED candidates were built with qrels")
        for example in initial_examples:
            if any(float(label) != 0.0 for label in example.labels):
                raise AssertionError(f"UNSUPERVISED example {example.qid} has nonzero labels")
    set_seed(cfg.seed)
    ckpt_dir = training_checkpoint_dir(cfg, variant)
    run_contract = provenance.build_run_contract(
        cfg=cfg,
        objective="rcd",
        variant=variant,
        candidate_metadata=candidate_metadata,
        teacher_checkpoint_path=cfg.teacher_checkpoint_path,
    )
    contract_candidate_metadata = dict(candidate_metadata)
    resume_payload: Dict[str, Any] = {}
    resume_metadata: Dict[str, Any] = {}
    if cfg.resume and student_training_complete(ckpt_dir):
        provenance.assert_resume_contract(ckpt_dir, run_contract)
        logger.info(
            f"{variant} has training_complete marker; loading best checkpoint from {ckpt_dir}"
        )
        loaded = load_student_checkpoint(ckpt_dir, cfg)
        if loaded is not None:
            return loaded
    resume_dir = find_resume_checkpoint_dir(ckpt_dir) if cfg.resume else None
    if resume_dir:
        provenance.assert_resume_contract(resume_dir, run_contract)
        resume_payload, resume_metadata = load_training_state_payload(resume_dir)
        if resume_metadata.get("completed", False):
            logger.info(f"{variant} is marked complete; loading best checkpoint from {ckpt_dir}")
            loaded = load_student_checkpoint(ckpt_dir, cfg)
            if loaded is not None:
                return loaded
        logger.info(f"Resuming {variant} from checkpoint directory: {resume_dir}")
        student = TrainableRetriever(resume_dir, cfg).to(cfg.device)
    else:
        student = TrainableRetriever(cfg.model_name, cfg).to(cfg.device)
    projection_head: Optional[SupervisionProjectionHead] = None
    if variant_uses_modulation(variant):
        projection_head = SupervisionProjectionHead(
            cfg.embedding_dim,
            cfg.signal_projection_output_dim,
        ).to(cfg.device)
        if resume_payload.get("projection_head_state"):
            try:
                projection_head.load_state_dict(resume_payload["projection_head_state"])
                logger.info(f"Restored projection head for {variant} from {resume_dir}")
            except Exception as exc:
                logger.warning(f"Could not restore projection head for {variant}: {exc}")

    student_params = list(student.parameters())
    trainable_proj = (
        [p for p in projection_head.parameters() if p.requires_grad]
        if projection_head is not None
        else []
    )
    params = student_params + trainable_proj
    optimizer_groups = [{"params": student_params, "weight_decay": cfg.weight_decay}]
    if trainable_proj:
        optimizer_groups.append(
            {"params": trainable_proj, "weight_decay": cfg.projection_head_weight_decay}
        )
    optimizer = build_optimizer(optimizer_groups, cfg, context=f"student_variant_{variant}")
    total_steps = cfg.epochs * max(1, math.ceil(len(initial_examples) / max(1, cfg.batch_size)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    if resume_payload.get("optimizer_state"):
        try:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            logger.info(f"Restored optimizer state for {variant}")
        except Exception as exc:
            logger.warning(f"Could not restore optimizer state for {variant}: {exc}")
    if resume_payload.get("scheduler_state"):
        try:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
            logger.info(f"Restored scheduler state for {variant}")
        except Exception as exc:
            logger.warning(f"Could not restore scheduler state for {variant}: {exc}")
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_payload.get("scaler_state"):
        try:
            scaler.load_state_dict(resume_payload["scaler_state"])
            logger.info(f"Restored AMP GradScaler state for {variant}")
        except Exception as exc:
            logger.warning(f"Could not restore AMP GradScaler state for {variant}: {exc}")

    examples = initial_examples
    resume_epoch = int(resume_metadata.get("epoch", 0)) if resume_metadata else 0
    start_epoch = resume_epoch + 1
    best_score = float(resume_metadata.get("best_score", -1e9)) if resume_metadata else -1e9
    best_metrics: Dict[int, Dict[str, float]] = (
        resume_metadata.get("best_val_metrics", {}) if resume_metadata else {}
    )
    patience_ctr = int(resume_metadata.get("patience_ctr", 0)) if resume_metadata else 0
    if start_epoch > cfg.epochs:
        logger.info(f"{variant} already reached cfg.epochs={cfg.epochs}; loading best checkpoint")
        loaded = load_student_checkpoint(ckpt_dir, cfg)
        if loaded is not None:
            return loaded
    logger.info(
        f"{variant} training resume state: start_epoch={start_epoch} | best_score={best_score:.6f} | "
        f"patience_ctr={patience_ctr} | checkpoint_path={ckpt_dir}"
    )
    last_epoch = resume_epoch
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        dataset = StudentCandidateDataset(examples, passages)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda batch: collate_student_candidates(batch, passages),
        )
        student.train()
        if projection_head is not None:
            projection_head.train()

        meters: Dict[str, List[float]] = defaultdict(list)
        for batch in loader:
            minimum_candidates = 1 if variant_uses_modulation(variant) else 2
            outer_valid_query_count = int(
                (batch["mask"].sum(dim=-1) >= minimum_candidates).sum().item()
            )
            if outer_valid_query_count <= 0:
                logger.warning(
                    "Skipping optimizer step because this batch has no valid objective signal"
                )
                continue
            optimizer.zero_grad(set_to_none=True)
            outer_batch_size = len(batch["query_texts"])
            micro_batch = cfg.micro_batch_size if cfg.micro_batch_size > 0 else outer_batch_size
            for micro_start in range(0, outer_batch_size, micro_batch):
                micro_end = min(micro_start + micro_batch, outer_batch_size)
                micro = slice_training_batch(batch, micro_start, micro_end)
                labels = micro["labels"].to(cfg.device)
                mask = micro["mask"].to(cfg.device)
                valid_queries_in_micro = int((mask.sum(dim=-1) >= minimum_candidates).sum().item())
                if valid_queries_in_micro <= 0:
                    continue
                flat_doc_texts = [text for row in micro["candidate_texts"] for text in row]
                bsz, docs_per_query = micro["doc_idxs"].shape
                weight = float(valid_queries_in_micro) / float(outer_valid_query_count)
                safe_doc_idxs = micro["doc_idxs"].clone()
                safe_doc_idxs[safe_doc_idxs < 0] = 0
                base_q = raw_query_embs_cpu[micro["q_idx"]].to(cfg.device)
                base_docs = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
                teacher_signals = teacher.extract_modulation_signals(
                    base_q, base_docs, micro["mask"]
                )

                with torch.amp.autocast("cuda", enabled=use_amp):
                    q_s = student.encode_query_texts_train(micro["query_texts"])
                    flat_d_s = student.encode_doc_texts_train(flat_doc_texts)
                    d_s = flat_d_s.view(bsz, docs_per_query, -1)
                    if q_s.shape[-1] != cfg.embedding_dim or d_s.shape[-1] != cfg.embedding_dim:
                        raise AssertionError("Student output dimension mismatch")
                    s_scores = torch.einsum("bd,bkd->bk", q_s, d_s)
                    loss_dict = compute_variant_loss(
                        variant=variant,
                        cfg=cfg,
                        student_scores=s_scores,
                        teacher_signals=teacher_signals,
                        student_embeddings={"query": q_s, "docs": d_s},
                        frozen_base_embeddings={"query": base_q, "docs": base_docs},
                        projection_head=projection_head,
                        labels=labels,
                        mask=mask,
                        mode=cfg.mode,
                    )
                    loss = loss_dict["total"]

                scaler.scale(loss * weight).backward()
                meters["loss"].append(float(loss_dict["total"].detach().cpu()))
                meters["rank"].append(float(loss_dict["rank"].detach().cpu()))
                meters["correction"].append(float(loss_dict["correction"].detach().cpu()))
                meters["correction_q"].append(float(loss_dict["correction_q"].detach().cpu()))
                meters["correction_d"].append(float(loss_dict["correction_d"].detach().cpu()))
                meters["entropy"].append(float(loss_dict["teacher_entropy"].detach().cpu()))
                meters["student_query_drift"].append(
                    float(loss_dict["student_query_drift"].detach().cpu())
                )
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            assert_teacher_frozen(teacher)

        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.lr
        logger.info(
            f"Epoch {epoch} | {variant} | loss={np.mean(meters['loss']):.6f} | "
            f"L_ranking={np.mean(meters['rank']):.6f} | L_correction={np.mean(meters['correction']):.6f} | "
            f"L_correction_q={np.mean(meters['correction_q']):.6f} | "
            f"L_correction_d={np.mean(meters['correction_d']):.6f} | "
            f"teacher_entropy={np.mean(meters['entropy']):.4f} | "
            f"proj_norm={projection_head_norm(projection_head):.4f} | "
            f"student_drift={np.mean(meters['student_query_drift']):.4f} | lr={lr:.8f}"
        )

        clear_cuda_cache_if_needed(cfg)
        student.eval()
        val_doc_embs = encode_corpus_with_student(student, passages, cfg)
        val_index = SearchIndex(val_doc_embs)
        val_metrics = evaluate_student(student, val_index, val_examples, eval_qrels_by_qid_idx, cfg)
        log_metrics_table(f"{variant} | VAL | EPOCH {epoch}", val_metrics, cfg.ks)
        score = validation_score(val_metrics, cfg)
        del val_index, val_doc_embs
        clear_cuda_cache_if_needed(cfg)
        if score > best_score + cfg.min_delta:
            best_score = score
            best_metrics = val_metrics
            patience_ctr = 0
            save_student_checkpoint(
                student,
                projection_head,
                ckpt_dir,
                cfg,
                variant,
                epoch,
                best_score,
                best_metrics,
                optimizer,
                scheduler,
                patience_ctr=patience_ctr,
                completed=False,
                checkpoint_kind="best",
                candidate_metadata=contract_candidate_metadata,
                scaler=scaler,
            )
        else:
            patience_ctr += 1
            logger.info(f"No improvement for {variant}. Patience {patience_ctr}/{cfg.patience}")
        save_latest_student_checkpoint(
            student,
            projection_head,
            ckpt_dir,
            cfg,
            variant,
            epoch,
            best_score,
            best_metrics,
            optimizer,
            scheduler,
            patience_ctr,
            completed=False,
            candidate_metadata=contract_candidate_metadata,
            scaler=scaler,
        )
        if patience_ctr >= cfg.patience:
            logger.info(f"Early stopping {variant}")
            break

    if last_epoch > 0:
        save_latest_student_checkpoint(
            student,
            projection_head,
            ckpt_dir,
            cfg,
            variant,
            last_epoch,
            best_score,
            best_metrics,
            optimizer,
            scheduler,
            patience_ctr,
            completed=True,
            candidate_metadata=contract_candidate_metadata,
            scaler=scaler,
        )
        mark_student_training_complete(ckpt_dir, variant, last_epoch, best_score)
    best_student = load_student_checkpoint(ckpt_dir, cfg)
    if best_student is None:
        return student
    return best_student


def evaluate_systems(
    cfg: Cfg,
    teacher: TeacherAdapter,
    raw_index: SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    passages: List[PassageRecord],
    val_examples: List[EvalExample],
    test_examples: List[EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    students: Dict[str, TrainableRetriever],
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    slices = (
        build_slices(val_examples, test_examples)
        if cfg.report_comlq_slices
        else {"VAL": val_examples, "TEST": test_examples}
    )
    results: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = defaultdict(dict)

    for slice_name, examples in slices.items():
        if not examples:
            logger.warning("Skipping %s | %s: no examples", VARIANT_E5_BASE, slice_name)
            continue
        metrics = evaluate_base_retriever(
            raw_index, raw_query_embs_cpu, examples, qrels_by_qid_idx, cfg
        )
        results[VARIANT_E5_BASE][slice_name] = metrics
        log_metrics_table(f"{VARIANT_E5_BASE} | {slice_name}", metrics, cfg.ks)
        metrics_t = evaluate_base_with_teacher(
            raw_index,
            teacher,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
            examples,
            qrels_by_qid_idx,
            cfg,
        )
        results[VARIANT_IMRNN_TEACHER][slice_name] = metrics_t
        log_metrics_table(f"{VARIANT_IMRNN_TEACHER} | {slice_name}", metrics_t, cfg.ks)

    for variant, student in students.items():
        student.eval()
        doc_embs = encode_corpus_with_student(student, passages, cfg)
        index = SearchIndex(doc_embs)
        for slice_name, examples in slices.items():
            if not examples:
                continue
            metrics = evaluate_student(student, index, examples, qrels_by_qid_idx, cfg)
            results[variant][slice_name] = metrics
            log_metrics_table(f"{variant} | {slice_name}", metrics, cfg.ks)
        del index, doc_embs
        clear_cuda_cache_if_needed(cfg)
    return dict(results)


def teacher_quality_gate(
    cfg: Cfg,
    teacher: TeacherAdapter,
    raw_index: SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    val_examples: List[EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    log_section("TEACHER QUALITY GATE")
    base_metrics = evaluate_base_retriever(
        raw_index, raw_query_embs_cpu, val_examples, qrels_by_qid_idx, cfg
    )
    teacher_metrics = evaluate_base_with_teacher(
        raw_index,
        teacher,
        raw_doc_embs_cpu,
        raw_query_embs_cpu,
        val_examples,
        qrels_by_qid_idx,
        cfg,
    )
    log_metrics_table(f"{VARIANT_E5_BASE} | VAL | TEACHER GATE", base_metrics, cfg.ks)
    log_metrics_table(f"{VARIANT_IMRNN_TEACHER} | VAL | TEACHER GATE", teacher_metrics, cfg.ks)
    base_score = validation_score(base_metrics, cfg)
    teacher_score = validation_score(teacher_metrics, cfg)
    weak = teacher_score < base_score
    logger.info(
        f"teacher_quality_score={teacher_score:.6f} | base_score={base_score:.6f} | weak_teacher={weak}"
    )
    if weak:
        logger.warning(
            "WEAK TEACHER: %s is worse than %s on the primary validation score.",
            VARIANT_IMRNN_TEACHER,
            VARIANT_E5_BASE,
        )
        if not cfg.allow_weak_teacher:
            raise RuntimeError(
                "Stopping because teacher is weak. Pass --allow_weak_teacher to continue."
            )
    return {"base_score": base_score, "teacher_score": teacher_score, "weak_teacher": weak}


def write_teacher_signal_debug_sample(
    cfg: Cfg,
    path: str,
    examples: List[TrainExample],
    passages: List[PassageRecord],
    teacher: TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
) -> None:
    sample = examples[: max(0, cfg.debug_teacher_samples)]
    if not sample:
        return
    batch = collate_student_candidates(sample, passages)
    mask = batch["mask"]
    safe_doc_idxs = batch["doc_idxs"].clone()
    safe_doc_idxs[safe_doc_idxs < 0] = 0
    q_raw = raw_query_embs_cpu[batch["q_idx"]]
    d_raw = raw_doc_embs_cpu[safe_doc_idxs]
    signals = teacher.extract_modulation_signals(q_raw, d_raw, mask)
    scores = signals["teacher_scores"].cpu()
    p_t, entropy, margin = teacher_distribution_diagnostics(scores, mask, cfg.tau)
    rows = []
    for i, ex in enumerate(sample):
        valid = torch.where(mask[i])[0].tolist()
        rows.append(
            {
                "qid": ex.qid,
                "query_text": ex.query_text,
                "candidate_ids": [passages[int(batch["doc_idxs"][i, j])].pid for j in valid],
                "teacher_scores": [float(scores[i, j]) for j in valid],
                "teacher_probabilities": [float(p_t[i, j]) for j in valid],
                "teacher_entropy_norm": float(entropy[i]),
                "teacher_margin": float(margin[i]),
                "delta_q_T_norm": float(signals["delta_q_T"][i].norm().cpu()),
                "delta_d_T_norms": [float(signals["delta_d_T"][i, j].norm().cpu()) for j in valid],
            }
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info(f"Saved teacher signal debug sample: {path}")


def dry_run_selected_variants(
    cfg: Cfg,
    variants: Sequence[str],
    examples: List[TrainExample],
    passages: List[PassageRecord],
    teacher: TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    candidate_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    log_section("DRY RUN")
    if not examples:
        raise ValueError("Dry run requires at least one training example")
    logger.info(
        "dry_run_candidate_rules: "
        f"qrels_used_in_training={candidate_metadata.get('qrels_used_in_training', False)} | "
        f"qrel_positive_injection={candidate_metadata.get('qrel_positive_injection', False)} | "
        f"qrel_loss_active={candidate_metadata.get('qrel_loss_active', False)}"
    )
    if cfg.mode == MODE_UNSUPERVISED:
        assert not candidate_metadata.get("qrels_used_in_training", False)
        assert not candidate_metadata.get("qrel_positive_injection", False)
        assert not candidate_metadata.get("qrel_loss_active", False)

    batch_examples = examples[: max(1, min(len(examples), cfg.batch_size))]
    batch = collate_student_candidates(batch_examples, passages)
    mask = batch["mask"].to(cfg.device)
    labels = batch["labels"].to(cfg.device)
    safe_doc_idxs = batch["doc_idxs"].clone()
    safe_doc_idxs[safe_doc_idxs < 0] = 0
    base_q = raw_query_embs_cpu[batch["q_idx"]].to(cfg.device)
    base_docs = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
    flat_doc_texts = [text for row in batch["candidate_texts"] for text in row]
    bsz, docs_per_query = batch["doc_idxs"].shape

    output: Dict[str, Any] = {}
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    for variant in variants:
        if variant not in TRAINABLE_VARIANTS:
            continue
        log_variant_plan(cfg, variant)
        student = TrainableRetriever(cfg.model_name, cfg).to(cfg.device)
        projection_head: Optional[SupervisionProjectionHead] = None
        if variant_uses_modulation(variant):
            projection_head = SupervisionProjectionHead(
                cfg.embedding_dim,
                cfg.signal_projection_output_dim,
            ).to(cfg.device)
        student.train()
        if projection_head is not None:
            projection_head.train()
        with torch.no_grad():
            teacher_signals = teacher.extract_modulation_signals(base_q, base_docs, batch["mask"])
        with torch.amp.autocast("cuda", enabled=use_amp):
            q_s = student.encode_query_texts_train(batch["query_texts"])
            d_s = student.encode_doc_texts_train(flat_doc_texts).view(bsz, docs_per_query, -1)
            scores = torch.einsum("bd,bkd->bk", q_s, d_s)
            loss_dict = compute_variant_loss(
                variant=variant,
                cfg=cfg,
                student_scores=scores,
                teacher_signals=teacher_signals,
                student_embeddings={"query": q_s, "docs": d_s},
                frozen_base_embeddings={"query": base_q, "docs": base_docs},
                projection_head=projection_head,
                labels=labels,
                mask=mask,
                mode=cfg.mode,
            )
        loss_dict["total"].backward()
        student_gradients = [
            parameter.grad for parameter in student.parameters() if parameter.grad is not None
        ]
        if not student_gradients or not all(
            torch.isfinite(gradient).all().item() for gradient in student_gradients
        ):
            raise AssertionError(f"DRY_RUN {variant}: student gradients are missing or non-finite")
        if projection_head is not None:
            projection_gradients = [
                parameter.grad
                for parameter in projection_head.parameters()
                if parameter.grad is not None
            ]
            if not projection_gradients or not all(
                torch.isfinite(gradient).all().item() for gradient in projection_gradients
            ):
                raise AssertionError(
                    f"DRY_RUN {variant}: projection gradients are missing or non-finite"
                )
        loss_summary = {
            key: float(value.detach().cpu())
            for key, value in loss_dict.items()
            if torch.is_tensor(value) and value.ndim == 0
        }
        shape_summary = {
            "query_batch": list(q_s.shape),
            "doc_batch": list(d_s.shape),
            "student_scores": list(scores.shape),
            "teacher_delta_q": list(teacher_signals["delta_q_T"].shape),
            "teacher_delta_d": list(teacher_signals["delta_d_T"].shape),
        }
        logger.info(f"DRY_RUN | {variant} | shapes={shape_summary}")
        logger.info(f"DRY_RUN | {variant} | losses={json.dumps(loss_summary, sort_keys=True)}")
        output[variant] = {
            "shapes": shape_summary,
            "losses": loss_summary,
            "backward_verified": True,
        }
        assert_teacher_frozen(teacher)
        del student, projection_head, q_s, d_s, scores, loss_dict
        clear_cuda_cache_if_needed(cfg)
    return output


def run_one_seed(cfg: Cfg) -> Dict[str, Any]:
    log_path = setup_logging(cfg)
    set_seed(cfg.seed)
    log_run_header(cfg, log_path)
    logger.info(
        "startup_variant_summary: mode=%s | selected_variants=%s",
        cfg.mode,
        list(cfg.variants),
    )
    require_runtime_dependencies()
    assert cfg.mode in {MODE_UNSUPERVISED, MODE_EVAL_ONLY}
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, sort_keys=True)

    corpus_path = os.path.join(cfg.dataset_dir, cfg.corpus_file)
    queries_path = os.path.join(cfg.dataset_dir, cfg.queries_file)
    qrels_dir = os.path.join(cfg.dataset_dir, cfg.qrels_dir)
    passages = load_passages(corpus_path)
    queries = load_queries(queries_path)
    split_ids, qrels, split_source = make_split_qids(cfg, qrels_dir, queries)
    validate_references(queries, passages, qrels)
    log_schema_summary(cfg, passages, queries, qrels, split_ids, split_source)
    logger.info(
        "Unsupervised training does not use qrels. Validation qrels are used only for model selection and evaluation."
        if cfg.mode == MODE_UNSUPERVISED
        else f"mode={cfg.mode}"
    )
    if not split_ids["val"]:
        raise ValueError("No validation queries remain after filtering/splitting")

    passage_id_to_index = {p.pid: i for i, p in enumerate(passages)}
    query_id_to_index = {q.qid: i for i, q in enumerate(queries)}
    query_by_id = {q.qid: q for q in queries}
    qrels_by_qid_idx = build_qrels_by_qid_idx(qrels, passage_id_to_index)
    val_examples = build_eval_examples(
        split_ids["val"], query_by_id, query_id_to_index, qrels, EvalExample
    )
    test_examples = build_eval_examples(
        split_ids["test"], query_by_id, query_id_to_index, qrels, EvalExample
    )
    log_section("FROZEN BASE-RETRIEVER CACHE SUMMARY")
    frozen_encoder = SentenceTransformer(cfg.model_name, device=cfg.device)
    raw_doc_embs_cpu = build_or_load_frozen_corpus_embeddings(cfg, passages, frozen_encoder)
    raw_query_embs_cpu = build_or_load_frozen_query_embeddings(cfg, queries, frozen_encoder)
    del frozen_encoder
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if (
        raw_doc_embs_cpu.shape[1] != cfg.embedding_dim
        or raw_query_embs_cpu.shape[1] != cfg.embedding_dim
    ):
        raise AssertionError("Frozen base-retriever cache dimension mismatch")
    raw_index = SearchIndex(raw_doc_embs_cpu)

    ensure_teacher_checkpoint(
        cfg,
        split_ids,
        qrels_by_qid_idx,
        query_id_to_index,
        raw_index,
        raw_doc_embs_cpu,
        raw_query_embs_cpu,
        split_source,
    )
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, sort_keys=True)

    teacher = TeacherAdapter(cfg)
    assert_teacher_frozen(teacher)
    if cfg.dry_run:
        quality = {"dry_run": True, "teacher_quality_gate_skipped": True}
        logger.info("Dry run: skipping full teacher quality gate.")
    else:
        quality = teacher_quality_gate(
            cfg,
            teacher,
            raw_index,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
            val_examples,
            qrels_by_qid_idx,
        )

    students: Dict[str, TrainableRetriever] = {}
    trainable_variants = [v for v in cfg.variants if v in TRAINABLE_VARIANTS]
    logger.info(f"trainable_variants_for_this_seed: {trainable_variants}")
    train_qids = list(split_ids["train"])
    if cfg.train_queries_file:
        with open(cfg.train_queries_file, "r", encoding="utf-8") as handle:
            requested = [line.strip() for line in handle if line.strip()]
        train_qids = [qid for qid in requested if qid in query_id_to_index]
        logger.info(
            f"Using train query list from {cfg.train_queries_file}: {len(train_qids)} queries"
        )
    if cfg.mode != MODE_EVAL_ONLY and not train_qids:
        raise ValueError("No train queries available")
    if cfg.dry_run and cfg.mode == MODE_EVAL_ONLY:
        raise ValueError("--dry_run requires UNSUPERVISED mode")

    candidate_metadata: Dict[str, Any] = {}
    if cfg.mode != MODE_EVAL_ONLY and trainable_variants:
        if cfg.mode != MODE_UNSUPERVISED:
            raise AssertionError(f"Unexpected training mode: {cfg.mode}")
        build_result = build_unsupervised_train_examples(
            cfg,
            train_qids,
            query_by_id,
            query_id_to_index,
            raw_index,
            raw_query_embs_cpu,
            passages=passages,
        )
        assert not build_result.metadata["qrels_used_in_training"]
        assert not build_result.metadata["qrel_positive_injection"]
        assert not build_result.metadata["qrel_loss_active"]
        training_qrels = None
        base_examples = build_result.examples
        candidate_metadata = build_result.metadata
        write_teacher_signal_debug_sample(
            cfg,
            os.path.join(cfg.output_dir, "teacher_signal_debug_sample.json"),
            base_examples,
            passages,
            teacher,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
        )
        if cfg.dry_run:
            dry_output = dry_run_selected_variants(
                cfg,
                trainable_variants,
                base_examples,
                passages,
                teacher,
                raw_doc_embs_cpu,
                raw_query_embs_cpu,
                candidate_metadata,
            )
            payload = {
                "seed": cfg.seed,
                "mode": cfg.mode,
                "dataset_format": cfg.source_dataset_format or cfg.dataset_format,
                "dataset_name": cfg.dataset_name,
                "selected_variants": list(cfg.variants),
                "dry_run": True,
                "quality_gate": quality,
                "candidate_metadata": candidate_metadata,
                "dry_run_variants": dry_output,
                "results": {},
            }
            with open(
                os.path.join(cfg.output_dir, "final_results.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            write_csv_rows(os.path.join(cfg.output_dir, "final_results.csv"), [])
            return payload
        for variant in trainable_variants:
            students[variant] = train_variant(
                variant,
                cfg,
                base_examples,
                candidate_metadata,
                val_examples,
                passages,
                training_qrels,
                qrels_by_qid_idx,
                teacher,
                raw_doc_embs_cpu,
                raw_query_embs_cpu,
            )
    elif cfg.mode == MODE_EVAL_ONLY:
        for variant in trainable_variants:
            ckpt = training_checkpoint_dir(cfg, variant)
            loaded = load_student_checkpoint(ckpt, cfg)
            if loaded is not None:
                students[variant] = loaded

    results = evaluate_systems(
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
        "selected_variants": list(cfg.variants),
        "dry_run": False,
        "quality_gate": quality,
        "candidate_metadata": candidate_metadata,
        "results": results,
    }
    with open(os.path.join(cfg.output_dir, "final_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_csv_rows(os.path.join(cfg.output_dir, "final_results.csv"), flatten_results(results))
    write_comlq_test_breakdown_results(cfg, results)
    return payload


def _suite_teacher_path(cfg: Cfg, dataset_count: int) -> str:
    raw = str(cfg.teacher_checkpoint_path).strip()
    template = str(getattr(cfg, "teacher_checkpoint_template", "")).strip()
    has_template = bool(template or "{dataset}" in raw or "{seed}" in raw)
    default_path = str(Cfg.__dataclass_fields__["teacher_checkpoint_path"].default)
    auto_values = {"", "auto", "auto_train", "__auto_train_teacher__"}
    if has_template:
        return cfg.teacher_checkpoint_path
    if cfg.auto_train_teacher_if_missing and (raw.lower() in auto_values or raw == default_path):
        return "__auto_train_teacher__"
    if dataset_count > 1:
        raise ValueError(
            "A multi-dataset suite cannot reuse one teacher checkpoint. Pass "
            "--teacher_checkpoint_template with {dataset} and {seed} placeholders, or enable "
            "dataset-specific automatic teacher training."
        )
    return cfg.teacher_checkpoint_path


def run_legalbench_rag_suite(
    cfg: Cfg,
    seed_runner: Any,
    header_logger: Any,
) -> None:
    suite_log_path = setup_logging(cfg)
    header_logger(cfg, suite_log_path)
    log_section("LEGALBENCH-RAG SUITE")
    if tuple(cfg.ks) == (1, 5, 10, 20, 50, 100):
        cfg.ks = (1, 2, 4, 8, 16, 32, 64)
        if cfg.eval_k == 50:
            cfg.eval_k = 64
        logger.info("LegalBench-RAG defaults replaced with ks=[1,2,4,8,16,32,64].")
    dataset_names = tuple(
        normalize_legalbench_dataset_name(name) for name in cfg.legalbench_rag_datasets
    )
    resolved_root = ensure_legalbench_rag_data_root(cfg)
    logger.info(
        "legalbench_rag_root=%s | datasets=%s | chunk_strategy=%s | chunk_size=%s",
        resolved_root,
        list(dataset_names),
        cfg.legalbench_chunk_strategy,
        cfg.legalbench_chunk_size,
    )
    teacher_path = _suite_teacher_path(cfg, len(dataset_names))
    suite_output_dir = cfg.output_dir
    prepared_root = cfg.legalbench_prepared_dir or os.path.join(
        suite_output_dir, "legalbench_rag_prepared"
    )
    dataset_payloads: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_name in dataset_names:
        dataset_output_dir = os.path.join(suite_output_dir, dataset_name)
        prepared_dir = os.path.join(prepared_root, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        metadata = prepare_legalbench_retrieval_dataset(
            cfg,
            dataset_name,
            prepared_dir,
            include_paper_note=True,
            progress_logger=logger,
        )
        child_cfg = replace(
            cfg,
            dataset_format="standard",
            source_dataset_format="legalbench_rag",
            dataset_name=dataset_name,
            dataset_dir=prepared_dir,
            corpus_file="corpus.jsonl",
            queries_file="queries.jsonl",
            qrels_dir="qrels",
            query_type_filter="all",
            query_types="",
            report_comlq_slices=False,
            output_dir=dataset_output_dir,
            log_path="",
            teacher_checkpoint_path=teacher_path,
            legalbench_metadata_path=os.path.join(prepared_dir, "legalbench_metadata.json"),
        )
        logger.info(
            "Running LegalBench-RAG dataset=%s | passages=%s | queries=%s",
            dataset_name,
            metadata["passage_count"],
            metadata["query_count"],
        )
        per_seed = []
        for seed in child_cfg.seeds:
            per_seed.append(seed_runner(make_seed_cfg(child_cfg, int(seed))))
            setup_logging(replace(cfg, log_path=suite_log_path))
            logger.info("Completed LegalBench-RAG dataset=%s | seed=%s", dataset_name, seed)
        aggregate_seed_results(dataset_output_dir, per_seed)
        dataset_payloads[dataset_name] = per_seed
        setup_logging(replace(cfg, log_path=suite_log_path))
    write_legalbench_suite_results(suite_output_dir, dataset_payloads)
    print(f"Saved LegalBench-RAG dataset-wise results under {suite_output_dir}", flush=True)



def run_experiment(
    cfg: Cfg,
    validator: Any,
    seed_runner: Any,
    header_logger: Any,
) -> None:
    cfg.dataset_format = infer_dataset_format(cfg)
    validator(cfg)
    if cfg.dataset_format == "legalbench_rag":
        run_legalbench_rag_suite(cfg, seed_runner, header_logger)
        return
    parent_output = cfg.output_dir
    per_seed = []
    for seed in cfg.seeds:
        seed_cfg = make_seed_cfg(cfg, int(seed))
        per_seed.append(seed_runner(seed_cfg))
    aggregate_seed_results(parent_output, per_seed)
    print(f"Saved aggregate results under {parent_output}", flush=True)


def main_unsup(cfg: Cfg) -> None:
    run_experiment(cfg, validate_unsup_cfg, run_one_seed, log_run_header)
