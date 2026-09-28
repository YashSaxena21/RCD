#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Margin-MSE baseline under the same qrel-free student-training contract as RCD.

It reuses the existing implementation's:
- dataset loading and split handling
- ComLQ and LegalBench-RAG dataset preparation
- frozen base-retriever embedding caches
- adapter-teacher checkpoint loading
- trainable student encoder
- validation / final evaluation helpers
- checkpoint-resume layout and multi-seed aggregation

Scientific intent:
- isolate the supervision objective
- compare modulation-delta distillation against a conventional
  score-margin distillation baseline from the same frozen IMRNN teacher

Primary training objective:
- Margin-MSE with pair strategy `top_vs_rest`
- teacher supervision uses candidate scores only
- no qrels in student training
- no modulation supervision

Examples:
  rcd-train margin-mse --help
  rcd-train margin-mse --run_loss_tests
  rcd-train margin-mse --dry_run \
      --dataset_format comlq --dataset_dir datasets/comlq/dataset \
      --teacher_checkpoint_path /path/to/imrnn_teacher.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import core as base
from . import experiment_utils, provenance
from .data_utils import build_eval_examples
from .legalbench import DATASETS as LEGALBENCH_RAG_DATASETS

logger = base.logger
assert_unsupervised_training_contract = experiment_utils.assert_unlabeled_training_contract
assert_zero_labels = experiment_utils.assert_zero_labels
mean_or_zero = experiment_utils.mean_or_zero
write_candidate_metadata = experiment_utils.write_candidate_metadata

MARGIN_PAIR_TOP_VS_REST = "top_vs_rest"
MARGIN_PAIR_ALL_PAIRS = "all_pairs"
OBJECTIVE_MARGIN_MSE = "margin_mse"
SYSTEM_MARGIN_MSE = "MARGIN_MSE"
SYSTEM_MARGIN_MSE_ALL_PAIRS = "MARGIN_MSE_ALL_PAIRS"


@dataclass
class MarginMseCfg(base.Cfg):
    output_dir: str = "runs/margin_mse"
    auto_train_teacher_if_missing: bool = False
    variants: Tuple[str, ...] = (SYSTEM_MARGIN_MSE,)
    margin_pair_strategy: str = MARGIN_PAIR_TOP_VS_REST
    run_loss_tests: bool = False


@dataclass
class MarginMSEStats:
    total: torch.Tensor
    per_query_loss: torch.Tensor
    valid_query_mask: torch.Tensor
    excluded_query_mask: torch.Tensor
    pair_count_per_query: torch.Tensor
    total_pair_count: torch.Tensor
    valid_query_count: torch.Tensor
    excluded_query_count: torch.Tensor
    teacher_margin_sum: torch.Tensor
    teacher_margin_sq_sum: torch.Tensor
    student_margin_sum: torch.Tensor
    abs_margin_error_sum: torch.Tensor
    sq_margin_error_sum: torch.Tensor
    top_indices: torch.Tensor


@dataclass
class EpochMeters:
    loss_sum: float = 0.0
    valid_query_count: int = 0
    excluded_query_count: int = 0
    pair_count: int = 0
    teacher_margin_sum: float = 0.0
    teacher_margin_sq_sum: float = 0.0
    student_margin_sum: float = 0.0
    abs_margin_error_sum: float = 0.0
    sq_margin_error_sum: float = 0.0
    student_query_drift_sum: float = 0.0
    student_query_drift_count: int = 0

    def update(self, loss_info: Dict[str, torch.Tensor], batch_query_count: int) -> None:
        self.loss_sum += float(loss_info["per_query_loss_sum"].detach().cpu())
        self.valid_query_count += int(loss_info["valid_query_count"].detach().cpu())
        self.excluded_query_count += int(loss_info["excluded_query_count"].detach().cpu())
        self.pair_count += int(loss_info["pair_count_total"].detach().cpu())
        self.teacher_margin_sum += float(loss_info["teacher_margin_sum"].detach().cpu())
        self.teacher_margin_sq_sum += float(loss_info["teacher_margin_sq_sum"].detach().cpu())
        self.student_margin_sum += float(loss_info["student_margin_sum"].detach().cpu())
        self.abs_margin_error_sum += float(loss_info["abs_margin_error_sum"].detach().cpu())
        self.sq_margin_error_sum += float(loss_info["sq_margin_error_sum"].detach().cpu())
        self.student_query_drift_sum += float(
            loss_info["student_query_drift_mean"].detach().cpu()
        ) * float(batch_query_count)
        self.student_query_drift_count += int(batch_query_count)


def system_name_for_cfg(cfg: MarginMseCfg) -> str:
    return (
        SYSTEM_MARGIN_MSE_ALL_PAIRS
        if cfg.margin_pair_strategy == MARGIN_PAIR_ALL_PAIRS
        else SYSTEM_MARGIN_MSE
    )


def log_objective_summary(cfg: MarginMseCfg) -> None:
    base.log_section("OBJECTIVE SUMMARY")
    logger.info("objective: Margin-MSE")
    logger.info(f"pair_strategy: {cfg.margin_pair_strategy}")
    logger.info("teacher_signal: candidate scores only")
    logger.info("qrels_used_in_student_training: false")
    logger.info("modulation_supervision: false")
    logger.info("student_inference_only: true")


def validate_margin_cfg(cfg: MarginMseCfg) -> None:
    if cfg.mode not in {base.MODE_UNSUPERVISED, base.MODE_EVAL_ONLY}:
        raise ValueError("Margin-MSE baseline supports only UNSUPERVISED or EVAL_ONLY mode.")
    if cfg.margin_pair_strategy not in {MARGIN_PAIR_TOP_VS_REST, MARGIN_PAIR_ALL_PAIRS}:
        raise ValueError("--margin_pair_strategy must be top_vs_rest or all_pairs")
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
        raise ValueError("Margin-MSE requires the ListNet IMRNNS teacher")
    if cfg.teacher_label_temp <= 0:
        raise ValueError("--teacher_label_temp must be positive")
    if cfg.teacher_max_positives < 0 or cfg.teacher_max_candidates < 0:
        raise ValueError(
            "--teacher_max_positives and --teacher_max_candidates must be non-negative"
        )
    for name in ("teacher_select_recall_k", "teacher_select_ndcg_k", "teacher_select_mrr_k"):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
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
    cfg: MarginMseCfg,
    candidate_metadata: Dict[str, Any],
    examples: Sequence[base.TrainExample],
    passages: Optional[Sequence[base.PassageRecord]] = None,
) -> Dict[str, Any]:
    out = provenance.augment_candidate_metadata(candidate_metadata, examples, passages)
    out.update(
        {
            "objective": OBJECTIVE_MARGIN_MSE,
            "margin_pair_strategy": cfg.margin_pair_strategy,
            "candidate_source": "frozen_base_retriever",
            "dynamic_refresh": False,
            "max_train_candidates_per_query": int(cfg.max_train_candidates_per_query),
            "candidate_pool_k": int(cfg.candidate_pool_k),
        }
    )
    return out


def _safe_masked_scores(
    scores: torch.Tensor, mask: torch.Tensor, fill_value: float
) -> torch.Tensor:
    fill = torch.full_like(scores, float(fill_value), dtype=torch.float32)
    return torch.where(mask, scores.to(dtype=torch.float32), fill)


def margin_mse_per_example(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    pair_strategy: str = MARGIN_PAIR_TOP_VS_REST,
) -> MarginMSEStats:
    """Per-query Margin-MSE with strict masking and detached teacher scores."""
    if pair_strategy not in {MARGIN_PAIR_TOP_VS_REST, MARGIN_PAIR_ALL_PAIRS}:
        raise ValueError(f"Unknown pair_strategy={pair_strategy!r}")

    mask = mask.to(device=student_scores.device, dtype=torch.bool)
    student_f = student_scores.to(dtype=torch.float32)
    teacher_f = teacher_scores.detach().to(device=student_scores.device, dtype=torch.float32)
    valid_count = mask.sum(dim=-1)
    valid_query_mask = valid_count >= 2
    excluded_query_mask = ~valid_query_mask

    safe_teacher = _safe_masked_scores(teacher_f, mask, fill_value=-1.0e6)
    top_indices = safe_teacher.argmax(dim=-1)

    if pair_strategy == MARGIN_PAIR_TOP_VS_REST:
        top_mask = F.one_hot(top_indices, num_classes=teacher_f.shape[1]).to(
            dtype=torch.bool, device=mask.device
        )
        pair_mask = mask & (~top_mask) & valid_query_mask.unsqueeze(-1)

        top_teacher = teacher_f.gather(1, top_indices.unsqueeze(1)).squeeze(1)
        top_student = student_f.gather(1, top_indices.unsqueeze(1)).squeeze(1)
        teacher_margin = top_teacher.unsqueeze(1) - teacher_f
        student_margin = top_student.unsqueeze(1) - student_f
        margin_error = student_margin - teacher_margin

        teacher_margin_masked = torch.where(
            pair_mask, teacher_margin, torch.zeros_like(teacher_margin)
        )
        student_margin_masked = torch.where(
            pair_mask, student_margin, torch.zeros_like(student_margin)
        )
        abs_error_masked = torch.where(
            pair_mask, margin_error.abs(), torch.zeros_like(margin_error)
        )
        sq_error_masked = torch.where(
            pair_mask, margin_error.pow(2), torch.zeros_like(margin_error)
        )

        pair_count_per_query = pair_mask.sum(dim=-1).to(dtype=torch.float32)
        per_query_loss = sq_error_masked.sum(dim=-1) / pair_count_per_query.clamp_min(1.0)
    else:
        k = teacher_f.shape[1]
        tri_mask = torch.triu(torch.ones((k, k), dtype=torch.bool, device=mask.device), diagonal=1)
        pair_mask = (
            mask.unsqueeze(2)
            & mask.unsqueeze(1)
            & tri_mask.unsqueeze(0)
            & valid_query_mask.view(-1, 1, 1)
        )

        teacher_margin = teacher_f.unsqueeze(2) - teacher_f.unsqueeze(1)
        student_margin = student_f.unsqueeze(2) - student_f.unsqueeze(1)
        margin_error = student_margin - teacher_margin

        teacher_margin_masked = torch.where(
            pair_mask, teacher_margin, torch.zeros_like(teacher_margin)
        )
        student_margin_masked = torch.where(
            pair_mask, student_margin, torch.zeros_like(student_margin)
        )
        abs_error_masked = torch.where(
            pair_mask, margin_error.abs(), torch.zeros_like(margin_error)
        )
        sq_error_masked = torch.where(
            pair_mask, margin_error.pow(2), torch.zeros_like(margin_error)
        )

        pair_count_per_query = pair_mask.sum(dim=(1, 2)).to(dtype=torch.float32)
        per_query_loss = sq_error_masked.sum(dim=(1, 2)) / pair_count_per_query.clamp_min(1.0)

    per_query_loss = torch.where(valid_query_mask, per_query_loss, torch.zeros_like(per_query_loss))
    valid_query_count = valid_query_mask.sum().to(dtype=torch.float32)
    if valid_query_mask.any():
        total = per_query_loss[valid_query_mask].mean()
    else:
        total = student_f.sum() * 0.0

    teacher_margin_sum = teacher_margin_masked.sum()
    teacher_margin_sq_sum = teacher_margin_masked.pow(2).sum()
    student_margin_sum = student_margin_masked.sum()
    abs_margin_error_sum = abs_error_masked.sum()
    sq_margin_error_sum = sq_error_masked.sum()
    total_pair_count = pair_count_per_query.sum()

    return MarginMSEStats(
        total=total,
        per_query_loss=per_query_loss,
        valid_query_mask=valid_query_mask,
        excluded_query_mask=excluded_query_mask,
        pair_count_per_query=pair_count_per_query,
        total_pair_count=total_pair_count,
        valid_query_count=valid_query_count,
        excluded_query_count=excluded_query_mask.sum().to(dtype=torch.float32),
        teacher_margin_sum=teacher_margin_sum,
        teacher_margin_sq_sum=teacher_margin_sq_sum,
        student_margin_sum=student_margin_sum,
        abs_margin_error_sum=abs_margin_error_sum,
        sq_margin_error_sum=sq_margin_error_sum,
        top_indices=top_indices,
    )


def compute_margin_mse_loss(
    cfg: MarginMseCfg,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    student_query_embeddings: torch.Tensor,
    frozen_query_embeddings: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    stats = margin_mse_per_example(
        student_scores=student_scores,
        teacher_scores=teacher_scores,
        mask=mask,
        pair_strategy=cfg.margin_pair_strategy,
    )
    q_s = student_query_embeddings.to(dtype=torch.float32)
    base_q = frozen_query_embeddings.to(device=q_s.device, dtype=torch.float32)
    student_query_drift = 1.0 - F.cosine_similarity(q_s, base_q, dim=-1)
    student_query_drift_mean = (
        student_query_drift[stats.valid_query_mask].mean()
        if stats.valid_query_mask.any()
        else q_s.sum() * 0.0
    )
    return {
        "total": stats.total,
        "per_query_loss_sum": stats.per_query_loss[stats.valid_query_mask].sum()
        if stats.valid_query_mask.any()
        else stats.total * 0.0,
        "valid_query_count": stats.valid_query_count,
        "excluded_query_count": stats.excluded_query_count,
        "pair_count_total": stats.total_pair_count,
        "teacher_margin_sum": stats.teacher_margin_sum,
        "teacher_margin_sq_sum": stats.teacher_margin_sq_sum,
        "student_margin_sum": stats.student_margin_sum,
        "abs_margin_error_sum": stats.abs_margin_error_sum,
        "sq_margin_error_sum": stats.sq_margin_error_sum,
        "student_query_drift_mean": student_query_drift_mean,
        "top_indices": stats.top_indices,
    }


def std_from_sums(sum_value: float, sq_sum: float, count: int) -> float:
    if count <= 0:
        return 0.0
    mean = sum_value / float(count)
    var = max(0.0, (sq_sum / float(count)) - (mean * mean))
    return math.sqrt(var)


def save_margin_checkpoint(
    student: base.TrainableRetriever,
    checkpoint_dir: str,
    cfg: MarginMseCfg,
    system_name: str,
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
        "checkpoint_format": "dense-retriever-student-margin-mse-v2",
        "checkpoint_kind": checkpoint_kind,
        "variant": system_name,
        "objective": OBJECTIVE_MARGIN_MSE,
        "pair_strategy": cfg.margin_pair_strategy,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_score": best_score,
        "best_val_metrics": best_metrics,
        "patience_ctr": patience_ctr,
        "completed": completed,
        "qrels_used_in_training": False,
        "qrel_positive_injection": False,
        "qrel_loss_active": False,
        "modulation_supervision": False,
        "student_model_name": cfg.model_name,
        "teacher_checkpoint_path": cfg.teacher_checkpoint_path,
        "candidate_set_fingerprint_sha1": candidate_metadata.get(
            "candidate_set_fingerprint_sha1", ""
        ),
        "candidate_metadata": candidate_metadata,
        "config": asdict(cfg),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata = provenance.add_run_contract(
        metadata,
        cfg=cfg,
        objective=OBJECTIVE_MARGIN_MSE,
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
            "metadata": metadata,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    logger.info(
        f"Saved {checkpoint_kind} {system_name} checkpoint to {checkpoint_dir} | "
        f"epoch={epoch} | best_score={best_score:.6f} | patience={patience_ctr} | completed={completed}"
    )


def save_latest_margin_checkpoint(
    student: base.TrainableRetriever,
    checkpoint_dir: str,
    cfg: MarginMseCfg,
    system_name: str,
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
    save_margin_checkpoint(
        student=student,
        checkpoint_dir=latest_dir,
        cfg=cfg,
        system_name=system_name,
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


def write_teacher_score_debug_sample(
    cfg: MarginMseCfg,
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
    q_raw = raw_query_embs_cpu[batch["q_idx"]]
    d_raw = raw_doc_embs_cpu[safe_doc_idxs]
    scores = teacher.score_batch(q_raw, d_raw, batch["mask"]).cpu()
    rows: List[Dict[str, Any]] = []
    for i, example in enumerate(sample):
        valid = torch.where(batch["mask"][i])[0].tolist()
        rows.append(
            {
                "qid": example.qid,
                "query_text": example.query_text,
                "candidate_ids": [passages[int(batch["doc_idxs"][i, j])].pid for j in valid],
                "teacher_scores": [float(scores[i, j]) for j in valid],
            }
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info(f"Saved teacher score debug sample: {path}")


def choose_dry_run_examples(
    examples: Sequence[base.TrainExample], batch_size: int
) -> List[base.TrainExample]:
    selected = [example for example in examples if len(example.candidate_doc_idxs) >= 2]
    if not selected:
        raise ValueError(
            "Dry run requires at least one training example with two or more valid candidates"
        )
    return selected[: max(1, batch_size)]


def dry_run_margin_mse(
    cfg: MarginMseCfg,
    examples: Sequence[base.TrainExample],
    passages: Sequence[base.PassageRecord],
    teacher: base.TeacherAdapter,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    candidate_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    base.log_section("DRY RUN")
    assert_unsupervised_training_contract(
        candidate_metadata=candidate_metadata,
        examples=examples,
        training_qrels_by_qid_idx=None,
    )
    batch_examples = choose_dry_run_examples(examples, cfg.batch_size)
    batch = base.collate_student_candidates(batch_examples, list(passages))
    mask = batch["mask"].to(cfg.device)
    labels = batch["labels"].to(cfg.device)
    assert_zero_labels(labels)
    safe_doc_idxs = batch["doc_idxs"].clone()
    safe_doc_idxs[safe_doc_idxs < 0] = 0
    base_q = raw_query_embs_cpu[batch["q_idx"]].to(cfg.device)
    base_docs = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
    flat_doc_texts = [text for row in batch["candidate_texts"] for text in row]
    bsz, docs_per_query = batch["doc_idxs"].shape

    student = base.TrainableRetriever(cfg.model_name, cfg).to(cfg.device)
    params = list(student.parameters())
    optimizer = base.build_optimizer(
        [{"params": params, "weight_decay": cfg.weight_decay}], cfg, context="dry_run_margin_mse"
    )
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    student.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_scores = teacher.score_batch(base_q, base_docs, batch["mask"])
    with torch.amp.autocast("cuda", enabled=use_amp):
        q_s = student.encode_query_texts_train(batch["query_texts"])
        d_s = student.encode_doc_texts_train(flat_doc_texts).view(bsz, docs_per_query, -1)
        s_scores = torch.einsum("bd,bkd->bk", q_s, d_s)
    loss_info = compute_margin_mse_loss(
        cfg=cfg,
        student_scores=s_scores,
        teacher_scores=teacher_scores,
        mask=mask,
        student_query_embeddings=q_s,
        frozen_query_embeddings=base_q,
    )
    scaler.scale(loss_info["total"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
    base.assert_teacher_frozen(teacher)
    dry_output = {
        "system_name": system_name_for_cfg(cfg),
        "objective": OBJECTIVE_MARGIN_MSE,
        "pair_strategy": cfg.margin_pair_strategy,
        "candidate_set_fingerprint_sha1": candidate_metadata.get(
            "candidate_set_fingerprint_sha1", ""
        ),
        "qrels_used_in_training": False,
        "qrel_positive_injection": False,
        "qrel_loss_active": False,
        "valid_query_count": int(loss_info["valid_query_count"].detach().cpu()),
        "excluded_query_count": int(loss_info["excluded_query_count"].detach().cpu()),
        "pair_count_total": int(loss_info["pair_count_total"].detach().cpu()),
        "loss": float(loss_info["total"].detach().cpu()),
        "student_query_drift_mean": float(loss_info["student_query_drift_mean"].detach().cpu()),
        "teacher_margin_sum": float(loss_info["teacher_margin_sum"].detach().cpu()),
        "student_margin_sum": float(loss_info["student_margin_sum"].detach().cpu()),
        "mean_abs_margin_error": mean_or_zero(
            float(loss_info["abs_margin_error_sum"].detach().cpu()),
            int(loss_info["pair_count_total"].detach().cpu()),
        ),
        "mean_sq_margin_error": mean_or_zero(
            float(loss_info["sq_margin_error_sum"].detach().cpu()),
            int(loss_info["pair_count_total"].detach().cpu()),
        ),
        "gradient_norm": float(grad_norm.detach().cpu())
        if torch.is_tensor(grad_norm)
        else float(grad_norm),
        "query_batch_shape": list(q_s.shape),
        "doc_batch_shape": list(d_s.shape),
        "student_scores_shape": list(s_scores.shape),
        "teacher_scores_shape": list(teacher_scores.shape),
    }
    logger.info(f"DRY_RUN | {json.dumps(dry_output, sort_keys=True)}")
    del student, optimizer, q_s, d_s, s_scores, teacher_scores, loss_info
    base.clear_cuda_cache_if_needed(cfg)
    return dry_output


def train_margin_mse_student(
    cfg: MarginMseCfg,
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
) -> base.TrainableRetriever:
    system_name = system_name_for_cfg(cfg)
    base.log_section(f"TRAIN OBJECTIVE | {system_name} | seed={cfg.seed}")
    log_objective_summary(cfg)
    assert_unsupervised_training_contract(
        candidate_metadata=candidate_metadata,
        examples=initial_examples,
        training_qrels_by_qid_idx=None,
    )

    base.set_seed(cfg.seed)
    ckpt_dir = base.training_checkpoint_dir(cfg, system_name)
    run_contract = provenance.build_run_contract(
        cfg=cfg,
        objective=OBJECTIVE_MARGIN_MSE,
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

    student_params = list(student.parameters())
    optimizer = base.build_optimizer(
        [{"params": student_params, "weight_decay": cfg.weight_decay}], cfg, context=system_name
    )
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
        meters = EpochMeters()

        for batch in loader:
            outer_valid_query_count = int((batch["mask"].sum(dim=-1) >= 2).sum().item())
            if outer_valid_query_count <= 0:
                meters.excluded_query_count += len(batch["query_texts"])
                logger.warning(
                    "Skipping optimizer step because the outer batch has no query with two candidates"
                )
                continue
            optimizer.zero_grad(set_to_none=True)
            outer_batch_size = len(batch["query_texts"])
            micro_batch = cfg.micro_batch_size if cfg.micro_batch_size > 0 else outer_batch_size
            for micro_start in range(0, outer_batch_size, micro_batch):
                micro_end = min(micro_start + micro_batch, outer_batch_size)
                micro = base.slice_training_batch(batch, micro_start, micro_end)
                labels = micro["labels"].to(cfg.device)
                mask = micro["mask"].to(cfg.device)
                valid_queries_in_micro = int((mask.sum(dim=-1) >= 2).sum().item())
                if valid_queries_in_micro <= 0:
                    meters.excluded_query_count += len(micro["query_texts"])
                    continue
                assert_zero_labels(labels)
                flat_doc_texts = [text for row in micro["candidate_texts"] for text in row]
                bsz, docs_per_query = micro["doc_idxs"].shape
                weight = float(valid_queries_in_micro) / float(outer_valid_query_count)
                safe_doc_idxs = micro["doc_idxs"].clone()
                safe_doc_idxs[safe_doc_idxs < 0] = 0
                base_q = raw_query_embs_cpu[micro["q_idx"]].to(cfg.device)
                base_docs = raw_doc_embs_cpu[safe_doc_idxs].to(cfg.device)
                with torch.no_grad():
                    teacher_scores = teacher.score_batch(base_q, base_docs, micro["mask"])

                with torch.amp.autocast("cuda", enabled=use_amp):
                    q_s = student.encode_query_texts_train(micro["query_texts"])
                    flat_d_s = student.encode_doc_texts_train(flat_doc_texts)
                    d_s = flat_d_s.view(bsz, docs_per_query, -1)
                    if q_s.shape[-1] != cfg.embedding_dim or d_s.shape[-1] != cfg.embedding_dim:
                        raise AssertionError("Student output dimension mismatch")
                    s_scores = torch.einsum("bd,bkd->bk", q_s, d_s)
                loss_info = compute_margin_mse_loss(
                    cfg=cfg,
                    student_scores=s_scores,
                    teacher_scores=teacher_scores,
                    mask=mask,
                    student_query_embeddings=q_s,
                    frozen_query_embeddings=base_q,
                )
                scaler.scale(loss_info["total"] * weight).backward()
                meters.update(loss_info, batch_query_count=valid_queries_in_micro)
                del q_s, flat_d_s, d_s, s_scores, teacher_scores, loss_info

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            base.assert_teacher_frozen(teacher)

        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.lr
        mean_loss = mean_or_zero(meters.loss_sum, meters.valid_query_count)
        mean_teacher_margin = mean_or_zero(meters.teacher_margin_sum, meters.pair_count)
        std_teacher_margin = std_from_sums(
            meters.teacher_margin_sum, meters.teacher_margin_sq_sum, meters.pair_count
        )
        mean_student_margin = mean_or_zero(meters.student_margin_sum, meters.pair_count)
        mean_abs_margin_error = mean_or_zero(meters.abs_margin_error_sum, meters.pair_count)
        mean_sq_margin_error = mean_or_zero(meters.sq_margin_error_sum, meters.pair_count)
        mean_student_drift = mean_or_zero(
            meters.student_query_drift_sum, meters.student_query_drift_count
        )
        logger.info(
            f"Epoch {epoch} | {system_name} | loss={mean_loss:.6f} | "
            f"valid_queries={meters.valid_query_count} | excluded_queries={meters.excluded_query_count} | "
            f"pairs={meters.pair_count} | teacher_margin_mean={mean_teacher_margin:.6f} | "
            f"teacher_margin_std={std_teacher_margin:.6f} | student_margin_mean={mean_student_margin:.6f} | "
            f"abs_margin_error={mean_abs_margin_error:.6f} | sq_margin_error={mean_sq_margin_error:.6f} | "
            f"student_drift={mean_student_drift:.6f} | lr={lr:.8f}"
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
            save_margin_checkpoint(
                student=student,
                checkpoint_dir=ckpt_dir,
                cfg=cfg,
                system_name=system_name,
                epoch=epoch,
                best_score=best_score,
                best_metrics=best_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                patience_ctr=patience_ctr,
                completed=False,
                checkpoint_kind="best",
                candidate_metadata=contract_candidate_metadata,
                scaler=scaler,
            )
        else:
            patience_ctr += 1
            logger.info(f"No improvement for {system_name}. Patience {patience_ctr}/{cfg.patience}")

        save_latest_margin_checkpoint(
            student=student,
            checkpoint_dir=ckpt_dir,
            cfg=cfg,
            system_name=system_name,
            epoch=epoch,
            best_score=best_score,
            best_metrics=best_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            patience_ctr=patience_ctr,
            completed=False,
            candidate_metadata=contract_candidate_metadata,
            scaler=scaler,
        )
        if patience_ctr >= cfg.patience:
            logger.info(f"Early stopping {system_name}")
            break

    if last_epoch > 0:
        save_latest_margin_checkpoint(
            student=student,
            checkpoint_dir=ckpt_dir,
            cfg=cfg,
            system_name=system_name,
            epoch=last_epoch,
            best_score=best_score,
            best_metrics=best_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
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


def run_one_seed(cfg: MarginMseCfg) -> Dict[str, Any]:
    log_path = base.setup_logging(cfg)
    base.set_seed(cfg.seed)
    experiment_utils.log_run_header(cfg, log_path, logger, __file__)
    log_objective_summary(cfg)
    base.require_runtime_dependencies()
    os.makedirs(cfg.output_dir, exist_ok=True)
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
        "Unsupervised Margin-MSE training does not use qrels. Validation qrels are used only for model selection and evaluation."
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

    base.ensure_teacher_checkpoint(
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

    teacher = base.TeacherAdapter(cfg)
    base.assert_teacher_frozen(teacher)
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
        write_teacher_score_debug_sample(
            cfg,
            os.path.join(cfg.output_dir, "teacher_score_debug_sample.json"),
            base_examples,
            passages,
            teacher,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
        )
        if cfg.dry_run:
            dry_output = dry_run_margin_mse(
                cfg,
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
                "selected_variants": [system_name],
                "objective": OBJECTIVE_MARGIN_MSE,
                "pair_strategy": cfg.margin_pair_strategy,
                "dry_run": True,
                "quality_gate": quality,
                "candidate_metadata": candidate_metadata,
                "dry_run_variants": {system_name: dry_output},
                "results": {},
            }
            with open(
                os.path.join(cfg.output_dir, "final_results.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            base.write_csv_rows(os.path.join(cfg.output_dir, "final_results.csv"), [])
            return payload
        students[system_name] = train_margin_mse_student(
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
        "objective": OBJECTIVE_MARGIN_MSE,
        "pair_strategy": cfg.margin_pair_strategy,
        "dry_run": False,
        "quality_gate": quality,
        "candidate_metadata": candidate_metadata,
        "results": results,
    }
    with open(os.path.join(cfg.output_dir, "final_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    base.write_csv_rows(
        os.path.join(cfg.output_dir, "final_results.csv"), base.flatten_results(results)
    )
    base.write_comlq_test_breakdown_results(cfg, results)
    return payload


def main_margin(cfg: MarginMseCfg) -> None:
    base.run_experiment(
        cfg,
        validate_margin_cfg,
        run_one_seed,
        lambda current, path: experiment_utils.log_run_header(current, path, logger, __file__),
    )


def run_margin_mse_loss_tests() -> None:
    mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    teacher = torch.tensor([[0.2, 0.9, 0.4]], dtype=torch.float32)
    student_same = teacher.clone().requires_grad_(True)
    stats = margin_mse_per_example(
        student_same, teacher, mask, pair_strategy=MARGIN_PAIR_TOP_VS_REST
    )
    assert abs(float(stats.total.detach().cpu())) < 1e-8, "identical scores should give zero loss"

    shifted = (teacher + 7.5).clone().requires_grad_(True)
    stats_shift = margin_mse_per_example(
        shifted, teacher, mask, pair_strategy=MARGIN_PAIR_TOP_VS_REST
    )
    assert abs(float(stats_shift.total.detach().cpu())) < 1e-8, (
        "constant score shift should preserve margins"
    )

    padded_teacher = torch.tensor([[0.2, 0.9, 0.4, -999.0]], dtype=torch.float32)
    padded_student = torch.tensor([[0.2, 0.9, 0.4, 3.7]], dtype=torch.float32).requires_grad_(True)
    padded_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    stats_padded = margin_mse_per_example(
        padded_student, padded_teacher, padded_mask, pair_strategy=MARGIN_PAIR_TOP_VS_REST
    )
    assert abs(float(stats_padded.total.detach().cpu())) < 1e-8, (
        "padded candidates must have no effect"
    )

    top_idx = int(stats.top_indices[0].detach().cpu())
    assert top_idx == 1, f"teacher-top candidate should be index 1, got {top_idx}"

    one_valid_mask = torch.tensor([[True, False, False]], dtype=torch.bool)
    one_valid_stats = margin_mse_per_example(
        torch.tensor([[0.7, 0.0, 0.0]], dtype=torch.float32, requires_grad=True),
        torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float32),
        one_valid_mask,
        pair_strategy=MARGIN_PAIR_TOP_VS_REST,
    )
    assert int(one_valid_stats.valid_query_count.detach().cpu()) == 0, (
        "single-candidate query must be excluded"
    )
    assert int(one_valid_stats.excluded_query_count.detach().cpu()) == 1, (
        "single-candidate query exclusion count mismatch"
    )

    all_pairs_stats = margin_mse_per_example(
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, requires_grad=True),
        torch.tensor([[0.3, 0.2, 0.1]], dtype=torch.float32),
        mask,
        pair_strategy=MARGIN_PAIR_ALL_PAIRS,
    )
    assert int(all_pairs_stats.total_pair_count.detach().cpu()) == 3, (
        "all_pairs must count each unordered pair exactly once"
    )

    grad_student = torch.tensor([[0.3, 0.1, -0.2]], dtype=torch.float32, requires_grad=True)
    grad_teacher = torch.tensor([[0.5, -0.1, -0.4]], dtype=torch.float32, requires_grad=True)
    grad_stats = margin_mse_per_example(
        grad_student, grad_teacher, mask, pair_strategy=MARGIN_PAIR_TOP_VS_REST
    )
    grad_stats.total.backward()
    assert grad_student.grad is not None, "student scores should receive gradients"
    assert torch.isfinite(grad_student.grad).all().item(), "student gradients must be finite"
    assert grad_teacher.grad is None, "teacher scores must remain detached"

    rejected_nonzero_labels = False
    try:
        assert_zero_labels(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    except AssertionError:
        rejected_nonzero_labels = True
    assert rejected_nonzero_labels, "assert_zero_labels should reject non-zero labels"

    half_student = torch.tensor([[0.3, 0.1, -0.2]], dtype=torch.float16, requires_grad=True)
    half_teacher = torch.tensor([[0.5, -0.1, -0.4]], dtype=torch.float16)
    half_stats = margin_mse_per_example(
        half_student, half_teacher, mask, pair_strategy=MARGIN_PAIR_TOP_VS_REST
    )
    assert torch.isfinite(half_stats.total).item(), (
        "loss must stay finite for AMP-compatible inputs"
    )

    logger.info("Margin-MSE synthetic loss tests passed")


def parse_margin_args() -> MarginMseCfg:
    parser = argparse.ArgumentParser(
        description="Margin-MSE distillation from frozen teacher scores"
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
        help="Frozen IMRNNS teacher checkpoint path for a single-dataset run.",
    )
    parser.add_argument(
        "--teacher_checkpoint_template",
        default="",
        help="Per-run checkpoint template supporting {dataset} and {seed}.",
    )
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

    parser.add_argument("--candidate_pool_k", type=int, default=64)
    parser.add_argument("--max_train_candidates_per_query", type=int, default=32)
    parser.add_argument("--feedback_k", type=int, default=100)
    parser.add_argument("--disable_l40s_safe_mode", action="store_true")

    parser.add_argument(
        "--margin_pair_strategy",
        choices=[MARGIN_PAIR_TOP_VS_REST, MARGIN_PAIR_ALL_PAIRS],
        default=MARGIN_PAIR_TOP_VS_REST,
    )

    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--eval_k", type=int, default=64)
    parser.add_argument("--selection_recall_w", type=float, default=1.0)
    parser.add_argument("--selection_ndcg_w", type=float, default=0.25)
    parser.add_argument("--selection_mrr_w", type=float, default=0.10)

    parser.add_argument("--frozen_corpus_emb_path", default="base_corpus_embeddings.pt")
    parser.add_argument("--frozen_query_emb_path", default="base_query_embeddings.pt")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--output_dir", default="runs/margin_mse")
    parser.add_argument("--log_path", default="")
    parser.add_argument("--debug_teacher_samples", type=int, default=5)
    args = parser.parse_args()

    cfg = MarginMseCfg(
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
        output_dir=args.output_dir,
        log_path=args.log_path,
        debug_teacher_samples=args.debug_teacher_samples,
        margin_pair_strategy=args.margin_pair_strategy,
    )
    cfg.variants = (system_name_for_cfg(cfg),)
    return cfg


def main() -> None:
    cfg = parse_margin_args()
    validate_margin_cfg(cfg)
    if cfg.run_loss_tests:
        log_path = base.setup_logging(cfg)
        experiment_utils.log_run_header(cfg, log_path, logger, __file__)
        run_margin_mse_loss_tests()
        return
    main_margin(cfg)


if __name__ == "__main__":
    main()
