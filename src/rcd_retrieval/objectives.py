"""Representation Correction Distillation objectives and diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .config import (
    MODE_UNSUPERVISED,
    VARIANT_CORRECTION_ONLY,
    VARIANT_FULL_RCD,
    VARIANT_RANKING_ONLY,
    Cfg,
)


def variant_uses_rank(variant: str) -> bool:
    return variant in {VARIANT_RANKING_ONLY, VARIANT_FULL_RCD}


def variant_uses_modulation(variant: str) -> bool:
    return variant in {VARIANT_CORRECTION_ONLY, VARIANT_FULL_RCD}


def representation_alignment_target(variant: str) -> str:
    return "delta_vectors" if variant_uses_modulation(variant) else "none"


def teacher_distribution_diagnostics(
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    negative_infinity = torch.finfo(teacher_scores.dtype).min
    temperature = max(float(tau), 1e-6)
    masked_scores = teacher_scores.masked_fill(~mask, negative_infinity)
    probabilities = F.softmax(masked_scores / temperature, dim=-1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    entropy = -(probabilities * log_probabilities).masked_fill(~mask, 0.0).sum(dim=-1)
    denominator = mask.float().sum(dim=-1).clamp_min(2.0).log()
    normalized_entropy = (entropy / denominator.clamp_min(1e-6)).clamp(0.0, 1.0)
    top_two = masked_scores.topk(k=min(2, masked_scores.shape[1]), dim=-1).values
    margin = (
        torch.zeros_like(normalized_entropy)
        if top_two.shape[1] == 1
        else top_two[:, 0] - top_two[:, 1]
    )
    return probabilities, normalized_entropy, margin


def ranking_kl_per_example(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    negative_infinity = torch.finfo(student_scores.dtype).min
    temperature = max(float(tau), 1e-6)
    teacher_probabilities = F.softmax(
        teacher_scores.masked_fill(~mask, negative_infinity) / temperature,
        dim=-1,
    )
    teacher_log_probabilities = torch.log(teacher_probabilities.clamp_min(1e-12))
    student_log_probabilities = F.log_softmax(
        student_scores.masked_fill(~mask, negative_infinity) / temperature,
        dim=-1,
    )
    return (
        teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)
    ).masked_fill(~mask, 0.0).sum(dim=-1)


def delta_loss_per_example(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    eps: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target_norm = target_delta.norm(dim=-1)
    nonzero = target_norm >= float(eps)
    cosine_loss = 1.0 - F.cosine_similarity(
        predicted_delta,
        target_delta,
        dim=-1,
        eps=max(float(eps), 1e-8),
    )
    zero_loss = predicted_delta.pow(2).mean(dim=-1)
    loss = torch.where(nonzero, cosine_loss, zero_loss)
    if mask is None:
        return loss
    valid = mask.to(loss.device).bool()
    return loss.masked_fill(~valid, 0.0).sum(dim=-1) / valid.float().sum(dim=-1).clamp_min(1.0)


def representation_alignment_tensors(
    student_query: torch.Tensor,
    student_documents: torch.Tensor,
    frozen_query: torch.Tensor,
    frozen_documents: torch.Tensor,
    teacher_signals: dict[str, torch.Tensor],
    projection_head: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project student correction vectors and return detached teacher targets."""
    student_query_delta = student_query - frozen_query
    student_document_delta = student_documents - frozen_documents
    teacher_query_delta = teacher_signals["delta_q_T"].to(
        device=student_query.device,
        dtype=student_query.dtype,
    )
    teacher_document_delta = teacher_signals["delta_d_T"].to(
        device=student_documents.device,
        dtype=student_documents.dtype,
    )
    return (
        projection_head(student_query_delta),
        projection_head(student_document_delta),
        teacher_query_delta.detach(),
        teacher_document_delta.detach(),
    )


def projection_head_norm(projection_head: Any | None) -> float:
    if projection_head is None:
        return 0.0
    squared_norm = sum(
        float(parameter.detach().pow(2).sum().cpu())
        for parameter in projection_head.parameters()
    )
    return math.sqrt(squared_norm)


def assert_teacher_frozen(teacher: Any, debug: bool = True) -> None:
    for name, parameter in teacher.model.named_parameters():
        if parameter.requires_grad:
            raise AssertionError(f"Teacher parameter unexpectedly trainable: {name}")
        if debug and parameter.grad is not None:
            raise AssertionError(f"Teacher parameter received gradient: {name}")


def compute_variant_loss(
    variant: str,
    cfg: Cfg,
    student_scores: torch.Tensor,
    teacher_signals: dict[str, torch.Tensor],
    student_embeddings: dict[str, torch.Tensor],
    frozen_base_embeddings: dict[str, torch.Tensor],
    projection_head: Any | None,
    labels: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Compute one public RCD variant and its scalar diagnostics."""
    student_query = student_embeddings["query"]
    student_documents = student_embeddings["docs"]
    frozen_query = frozen_base_embeddings["query"].to(
        device=student_query.device,
        dtype=student_query.dtype,
    )
    frozen_documents = frozen_base_embeddings["docs"].to(
        device=student_documents.device,
        dtype=student_documents.dtype,
    )
    mask = mask.to(student_scores.device).bool()
    labels = labels.to(student_scores.device)
    zero_per_query = torch.zeros(
        (student_scores.shape[0],),
        device=student_scores.device,
        dtype=student_scores.dtype,
    )

    teacher_scores = teacher_signals["teacher_scores"].to(
        student_scores.device,
        dtype=student_scores.dtype,
    )
    _, normalized_entropy, _ = teacher_distribution_diagnostics(
        teacher_scores,
        mask,
        cfg.tau,
    )
    ranking_loss = (
        ranking_kl_per_example(student_scores, teacher_scores, mask, cfg.tau)
        if variant_uses_rank(variant)
        else zero_per_query
    )
    query_correction_loss = zero_per_query
    document_correction_loss = zero_per_query
    if variant_uses_modulation(variant):
        if projection_head is None:
            raise AssertionError(f"{variant} requires a supervision projection head")
        student_query_target, student_document_targets, teacher_query_target, teacher_document_targets = (
            representation_alignment_tensors(
                student_query,
                student_documents,
                frozen_query,
                frozen_documents,
                teacher_signals,
                projection_head,
            )
        )
        if (
            student_query_target.shape[-1] != cfg.signal_projection_output_dim
            or student_document_targets.shape[-1] != cfg.signal_projection_output_dim
        ):
            raise AssertionError("Projection-head output dimension mismatch")
        if (
            student_query_target.shape != teacher_query_target.shape
            or student_document_targets.shape != teacher_document_targets.shape
        ):
            raise AssertionError(
                f"{variant} representation-alignment shape mismatch: "
                f"student_q={tuple(student_query_target.shape)} "
                f"teacher_q={tuple(teacher_query_target.shape)} "
                f"student_d={tuple(student_document_targets.shape)} "
                f"teacher_d={tuple(teacher_document_targets.shape)}"
            )
        query_correction_loss = delta_loss_per_example(
            student_query_target,
            teacher_query_target,
            cfg.delta_eps,
        )
        document_correction_loss = delta_loss_per_example(
            student_document_targets,
            teacher_document_targets,
            cfg.delta_eps,
            mask,
        )

    correction_loss = (
        cfg.beta_q * query_correction_loss + cfg.beta_d * document_correction_loss
    )
    objective = zero_per_query
    if variant_uses_rank(variant):
        objective = objective + ranking_loss
    if variant_uses_modulation(variant):
        objective = objective + cfg.alpha * correction_loss

    minimum_candidates = 1 if variant_uses_modulation(variant) else 2
    valid_queries = mask.sum(dim=-1) >= minimum_candidates
    if not valid_queries.any():
        raise AssertionError(f"{variant} loss received no query with a valid objective signal")
    if mode != MODE_UNSUPERVISED:
        raise AssertionError(f"RCD student loss requires UNSUPERVISED mode, got {mode!r}")
    if labels.detach().abs().sum().item() != 0:
        raise AssertionError("RCD student loss received nonzero labels")

    return {
        "total": objective[valid_queries].mean(),
        "rank": ranking_loss[valid_queries].mean(),
        "correction": correction_loss[valid_queries].mean(),
        "correction_q": query_correction_loss[valid_queries].mean(),
        "correction_d": document_correction_loss[valid_queries].mean(),
        "teacher_entropy": normalized_entropy[valid_queries].mean(),
        "student_query_drift": (
            1.0 - F.cosine_similarity(student_query, frozen_query, dim=-1)
        )[valid_queries].mean(),
    }
