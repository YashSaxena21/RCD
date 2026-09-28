"""Shared retrieval encoding, metrics, and validation selection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .config import Cfg
from .data_utils import SearchIndex
from .metrics import legalbench_character_metrics, retrieval_metrics
from .records import EvalExample, PassageRecord
from .runtime import log_section, logger


@torch.no_grad()
def encode_corpus_with_student(
    student: Any,
    passages: list[PassageRecord],
    cfg: Cfg,
) -> torch.Tensor:
    log_section(f"ENCODING CORPUS WITH STUDENT | {len(passages)} passages")
    chunks = []
    for start in range(0, len(passages), cfg.corpus_encode_batch_size):
        chunk = passages[start : start + cfg.corpus_encode_batch_size]
        chunks.append(
            student.encode_doc_texts_eval(
                [passage.text for passage in chunk], cfg.corpus_encode_batch_size
            )
        )
        logger.info(
            "  student encoded corpus %s/%s",
            min(start + len(chunk), len(passages)),
            len(passages),
        )
    return torch.cat(chunks, dim=0).contiguous()


@torch.no_grad()
def encode_eval_queries_with_student(
    student: Any,
    examples: list[EvalExample],
    cfg: Cfg,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(examples), cfg.eval_batch_size):
        chunk = examples[start : start + cfg.eval_batch_size]
        chunks.append(
            student.encode_query_texts_eval(
                [example.query_text for example in chunk], cfg.eval_batch_size
            )
        )
    if chunks:
        return torch.cat(chunks, dim=0).contiguous()
    return torch.empty((0, cfg.embedding_dim))


def add_legalbench_char_metrics(
    metrics: dict[int, dict[str, float]],
    indices: np.ndarray,
    examples: list[EvalExample],
    cfg: Cfg,
) -> dict[int, dict[str, float]]:
    character_metrics = legalbench_character_metrics(
        indices, examples, cfg.legalbench_metadata_path or None, cfg.ks
    )
    for k, values in character_metrics.items():
        if k in metrics:
            metrics[k].update(values)
    return metrics


def log_metrics_table(
    title: str,
    metrics: dict[int, dict[str, float]],
    ks: Sequence[int],
) -> None:
    logger.info("")
    logger.info("--- %s ---", title)
    header = "k |     recall |        mrr |       ndcg |        map |  precision |       n"
    logger.info(header)
    logger.info("-" * len(header))
    for k in sorted(set(int(value) for value in ks)):
        row = metrics[k]
        logger.info(
            "%2d | %.4f     | %.4f    | %.4f    | %.4f    | %.4f     | %.0f",
            k,
            row["recall"],
            row["mrr"],
            row["ndcg"],
            row["map"],
            row["precision"],
            row["n"],
        )
        if "legalbench_char_recall" in row or "legalbench_char_precision" in row:
            logger.info(
                "     legalbench_char_recall=%.4f | legalbench_char_precision=%.4f",
                row.get("legalbench_char_recall", 0.0),
                row.get("legalbench_char_precision", 0.0),
            )


def validation_score(metrics: dict[int, dict[str, float]], cfg: Cfg) -> float:
    eval_k = cfg.eval_k if cfg.eval_k in metrics else max(metrics)
    row = metrics[eval_k]
    return (
        cfg.selection_recall_w * row["recall"]
        + cfg.selection_ndcg_w * row["ndcg"]
        + cfg.selection_mrr_w * row["mrr"]
    )


@torch.no_grad()
def evaluate_base_retriever(
    index: SearchIndex,
    query_embeddings: torch.Tensor,
    examples: list[EvalExample],
    qrels_by_qid_idx: dict[str, dict[int, float]],
    cfg: Cfg,
) -> dict[int, dict[str, float]]:
    query_indices = [example.q_idx for example in examples]
    queries = query_embeddings[query_indices].numpy().astype(np.float32)
    indices = index.search(queries, max(cfg.ks))
    metrics = retrieval_metrics(indices, examples, qrels_by_qid_idx, cfg.ks)
    return add_legalbench_char_metrics(metrics, indices, examples, cfg)


@torch.no_grad()
def evaluate_base_with_teacher(
    index: SearchIndex,
    teacher: Any,
    document_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    examples: list[EvalExample],
    qrels_by_qid_idx: dict[str, dict[int, float]],
    cfg: Cfg,
) -> dict[int, dict[str, float]]:
    query_indices = [example.q_idx for example in examples]
    queries = query_embeddings[query_indices].numpy().astype(np.float32)
    candidate_indices = index.search(queries, max(cfg.feedback_k, max(cfg.ks)))
    ranked = teacher.rerank_candidate_indices(
        query_embeddings,
        document_embeddings,
        query_indices,
        candidate_indices,
        max(cfg.ks),
    )
    metrics = retrieval_metrics(ranked, examples, qrels_by_qid_idx, cfg.ks)
    return add_legalbench_char_metrics(metrics, ranked, examples, cfg)


@torch.no_grad()
def evaluate_student(
    student: Any,
    student_document_index: SearchIndex,
    examples: list[EvalExample],
    qrels_by_qid_idx: dict[str, dict[int, float]],
    cfg: Cfg,
) -> dict[int, dict[str, float]]:
    query_embeddings = encode_eval_queries_with_student(student, examples, cfg)
    indices = student_document_index.search(
        query_embeddings.numpy().astype(np.float32), max(cfg.ks)
    )
    metrics = retrieval_metrics(indices, examples, qrels_by_qid_idx, cfg.ks)
    return add_legalbench_char_metrics(metrics, indices, examples, cfg)
