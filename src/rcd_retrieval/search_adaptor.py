#!/usr/bin/env python3
"""Search-Adaptor teacher backend for Representation Correction Distillation.

The backend reuses RCD data preparation, frozen-retriever candidates, student
objectives, evaluation, checkpointing, and result aggregation. It replaces the
IMRNNS teacher with a Search-Adaptor teacher.

By default one invocation runs ComLQ, PrivacyQA, and ContractNLI with
30%, 50%, and 100% deterministic nested subsets of the available teacher-training
queries::

    rcd-train rcd --teacher search-adaptor

Base-pipeline arguments remain available, for example::

    rcd-train rcd --teacher search-adaptor \
      --dataset_dir datasets/comlq/dataset \
      --legalbench_rag_root /path/to/legalbench-rag \
      --variants RANKING_ONLY CORRECTION_ONLY FULL_RCD \
      --seeds 42 --ks 1 2 4 8 16 32 64

Scientific contract
-------------------
Search-Adaptor itself is supervised: its ranking and prediction losses use the
teacher-training split's qrels. The subsequent student post-training remains
qrel-free: frozen-retriever candidates are not positive-injected, all student labels are
zero, and only frozen Search-Adaptor scores/deltas supervise the student. These
two stages are recorded separately in every checkpoint and result file.

Paper: Yoon et al., "Search-Adaptor: Embedding Customization for Information
Retrieval", ACL 2024, https://aclanthology.org/2024.acl-long.661/

The paper specifies MLPs but does not report their depth, hidden width, activation,
or initialization. This script therefore exposes those details and records them as
an implementation assumption. All specified parts of the paper are implemented:
shared residual adaptation, cosine retrieval, weighted pairwise logistic ranking,
L1 recovery, relevance-weighted L1 query prediction, Adam, 2,000 iterations,
batch size 128, patience 125, learning rate 1e-3, a 10:1 negative-pair sampling
ratio, and validation tuning over the published alpha/beta grids.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core as pipeline
from . import provenance
from .evaluation import add_legalbench_char_metrics
from .legalbench import (
    has_archive_pair as legalbench_has_archive_pair,
)
from .legalbench import (
    has_data_directories as legalbench_has_data_directories,
)
from .legalbench import (
    prepare_retrieval_dataset as prepare_legalbench_retrieval_dataset,
)
from .runtime import (
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR,
    _TRANSFORMERS_IMPORT_ERROR,
    SentenceTransformer,
    get_linear_schedule_with_warmup,
)
from .sampling import deterministic_fraction_qids

PAPER_URL = "https://aclanthology.org/2024.acl-long.661/"
SEARCH_ADAPTOR = "SEARCH_ADAPTOR"
SEARCH_ADAPTOR_DATASETS = ("comlq", "privacy_qa", "contractnli")
LEGALBENCH_DATASETS = {"privacy_qa", "contractnli"}
_ORIGINAL_LOG_RUN_HEADER = pipeline.log_run_header


@dataclass(frozen=True)
class SearchAdaptorOptions:
    datasets: Tuple[str, ...] = SEARCH_ADAPTOR_DATASETS
    train_fractions: Tuple[float, ...] = (0.10, 0.30, 0.50, 1.00)
    fraction_seed: int = 42
    batch_size: int = 128
    max_iterations: int = 2000
    patience: int = 125
    eval_every: int = 25
    save_every: int = 25
    learning_rate: float = 1e-3
    negative_pair_ratio: int = 10
    alpha_grid: Tuple[float, ...] = (0.0, 0.1, 1.0)
    beta_grid: Tuple[float, ...] = (0.0, 0.01, 0.1)
    fixed_alpha: float = 0.1
    fixed_beta: float = 0.01
    grid_search: bool = True
    hidden_dim: int = 0
    num_layers: int = 2
    activation: str = "relu"
    inference_batch_size: int = 8192
    fail_on_weak_teacher: bool = False
    run_tests: bool = False


ACTIVE_OPTIONS = SearchAdaptorOptions()


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported Search-Adaptor activation: {name}")


def make_mlp(input_dim: int, hidden_dim: int, num_layers: int, activation: str) -> nn.Sequential:
    """Build the explicitly documented MLP assumption used for f and p."""
    if input_dim < 1 or hidden_dim < 1 or num_layers < 1:
        raise ValueError("Search-Adaptor MLP dimensions and layer count must be positive")
    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, input_dim))
    layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), _activation(activation)]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), _activation(activation)])
    layers.append(nn.Linear(hidden_dim, input_dim))
    return nn.Sequential(*layers)


class SearchAdaptorModel(nn.Module):
    """Shared residual adapter f and training-only query predictor p."""

    def __init__(
        self, embedding_dim: int, hidden_dim: int, num_layers: int = 2, activation: str = "relu"
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation_name = str(activation)
        self.adapter = make_mlp(
            self.embedding_dim, self.hidden_dim, self.num_layers, self.activation_name
        )
        self.query_predictor = make_mlp(
            self.embedding_dim, self.hidden_dim, self.num_layers, self.activation_name
        )

    def adapt(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Search-Adaptor expected {self.embedding_dim}-d embeddings, got {embeddings.shape[-1]}"
            )
        return embeddings + self.adapter(embeddings)

    def score(
        self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adapted_queries = self.adapt(query_embeddings.float())
        adapted_documents = self.adapt(document_embeddings.float())
        scores = torch.einsum(
            "bd,bkd->bk",
            F.normalize(adapted_queries, p=2, dim=-1),
            F.normalize(adapted_documents, p=2, dim=-1),
        )
        return adapted_queries, adapted_documents, scores


def search_adaptor_ranking_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Paper Sec. 4.2: sum over all valid ordered pairs where y_j > y_k."""
    if scores.shape != relevance.shape or scores.shape != mask.shape:
        raise ValueError(
            "Search-Adaptor ranking tensors must have identical [batch, candidates] shapes"
        )
    scores_f = scores.float()
    relevance_f = relevance.detach().float()
    valid = mask.bool()
    # Build one candidate-pair matrix at a time. This is algebraically identical
    # to [B,K,K] vectorization but avoids quadratic memory growth across B.
    total = scores_f.sum() * 0.0
    pair_count = 0
    for row in range(scores_f.shape[0]):
        row_scores = scores_f[row].masked_select(valid[row])
        row_relevance = relevance_f[row].masked_select(valid[row])
        if row_scores.numel() < 2:
            continue
        relevance_difference = row_relevance.unsqueeze(1) - row_relevance.unsqueeze(0)
        pair_mask = relevance_difference > 0.0
        if not pair_mask.any():
            continue
        # Axis 0 is j and axis 1 is k, hence s_k - s_j below.
        score_difference = row_scores.unsqueeze(0) - row_scores.unsqueeze(1)
        total = (
            total
            + (relevance_difference.clamp_min(0.0) * F.softplus(score_difference))
            .masked_select(pair_mask)
            .sum()
        )
        pair_count += int(pair_mask.sum().item())
    if pair_count == 0:
        return scores_f.sum() * 0.0, 0
    return total, pair_count


def search_adaptor_recovery_loss(
    base_queries: torch.Tensor,
    adapted_queries: torch.Tensor,
    base_documents: torch.Tensor,
    adapted_documents: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Paper Sec. 4.3: mean vector L1 distance for queries plus corpus."""
    query_l1 = (adapted_queries.float() - base_queries.detach().float()).abs().sum(dim=-1).mean()
    valid = mask.bool()
    document_l1 = (adapted_documents.float() - base_documents.detach().float()).abs().sum(dim=-1)
    if not valid.any():
        return query_l1
    return query_l1 + document_l1.masked_select(valid).mean()


def search_adaptor_prediction_loss(
    adapted_queries: torch.Tensor,
    predicted_queries_from_documents: torch.Tensor,
    relevance: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Paper Sec. 4.3: relevance-weighted L1 query prediction."""
    weights = relevance.detach().float().clamp_min(0.0) * mask.bool().float()
    denominator = weights.sum()
    if float(denominator.item()) <= 0.0:
        return predicted_queries_from_documents.float().sum() * 0.0
    distances = (
        (adapted_queries.float().unsqueeze(1) - predicted_queries_from_documents.float())
        .abs()
        .sum(dim=-1)
    )
    return (weights * distances).sum() / denominator


def search_adaptor_objective(
    model: SearchAdaptorModel,
    base_queries: torch.Tensor,
    base_documents: torch.Tensor,
    relevance: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    beta: float,
) -> Dict[str, Any]:
    """Compute L_rank + alpha*L_rec + beta*L_pred in float32."""
    adapted_queries, adapted_documents, scores = model.score(base_queries, base_documents)
    scores = scores.masked_fill(~mask.bool(), 0.0)
    ranking, pair_count = search_adaptor_ranking_loss(scores, relevance, mask)
    recovery = search_adaptor_recovery_loss(
        base_queries, adapted_queries, base_documents, adapted_documents, mask
    )
    predicted_queries = model.query_predictor(adapted_documents.float())
    prediction = search_adaptor_prediction_loss(adapted_queries, predicted_queries, relevance, mask)
    total = ranking + float(alpha) * recovery + float(beta) * prediction
    return {
        "total": total,
        "ranking": ranking,
        "recovery": recovery,
        "prediction": prediction,
        "scores": scores,
        "adapted_queries": adapted_queries,
        "adapted_documents": adapted_documents,
        "ranking_pair_count": pair_count,
    }


def _next_query_batch(
    ordered_qids: List[str],
    cursor: int,
    batch_size: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], int]:
    if not ordered_qids:
        raise ValueError("Search-Adaptor has no teacher-training queries")
    batch: List[str] = []
    order = list(ordered_qids)
    while len(batch) < min(batch_size, len(order)):
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        take = min(min(batch_size, len(order)) - len(batch), len(order) - cursor)
        batch.extend(order[cursor : cursor + take])
        cursor += take
    return batch, order, cursor


def _sample_negative_indices(
    corpus_size: int,
    excluded: set[int],
    count: int,
    rng: random.Random,
) -> List[int]:
    available = corpus_size - len(excluded)
    target = min(max(0, int(count)), max(0, available))
    if target == 0:
        return []
    if available <= max(4096, target * 4):
        candidates = [idx for idx in range(corpus_size) if idx not in excluded]
        return rng.sample(candidates, target)
    sampled: List[int] = []
    seen = set(excluded)
    while len(sampled) < target:
        idx = rng.randrange(corpus_size)
        if idx in seen:
            continue
        seen.add(idx)
        sampled.append(idx)
    return sampled


def build_search_adaptor_batch(
    qids: Sequence[str],
    query_id_to_index: Mapping[str, int],
    qrels_by_qid_idx: Mapping[str, Mapping[int, float]],
    raw_query_embs_cpu: torch.Tensor,
    raw_doc_embs_cpu: torch.Tensor,
    negative_pair_ratio: int,
    use_graded_relevance: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    """Build per-query positive-plus-random-negative candidate sets.

    Every positive is retained. For each positive query-document pair, the
    sampler draws ``negative_pair_ratio`` zero-relevance documents, matching the
    paper's reported negative-pair subsampling ratio.
    """
    rows: List[Tuple[str, int, List[int], List[float]]] = []
    corpus_size = int(raw_doc_embs_cpu.shape[0])
    for qid in qids:
        if qid not in query_id_to_index:
            continue
        positive_items = [
            (int(doc_idx), float(label))
            for doc_idx, label in qrels_by_qid_idx.get(qid, {}).items()
            if float(label) > 0.0
        ]
        positive_items.sort(key=lambda item: (-item[1], item[0]))
        if not positive_items:
            continue
        positive_ids = [doc_idx for doc_idx, _ in positive_items]
        labels = [label if use_graded_relevance else 1.0 for _, label in positive_items]
        negatives = _sample_negative_indices(
            corpus_size,
            set(positive_ids),
            int(negative_pair_ratio) * len(positive_ids),
            rng,
        )
        if not negatives and len(set(labels)) < 2:
            continue
        rows.append(
            (
                str(qid),
                int(query_id_to_index[qid]),
                positive_ids + negatives,
                labels + [0.0] * len(negatives),
            )
        )
    if not rows:
        raise ValueError("Search-Adaptor batch has no valid positive/negative relevance pairs")
    max_candidates = max(len(row[2]) for row in rows)
    embedding_dim = int(raw_doc_embs_cpu.shape[1])
    query_embeddings = torch.stack([raw_query_embs_cpu[row[1]].float() for row in rows])
    documents = torch.zeros((len(rows), max_candidates, embedding_dim), dtype=torch.float32)
    relevance = torch.zeros((len(rows), max_candidates), dtype=torch.float32)
    mask = torch.zeros((len(rows), max_candidates), dtype=torch.bool)
    candidate_indices = torch.full((len(rows), max_candidates), -1, dtype=torch.long)
    for row_idx, (_, _, doc_indices, labels) in enumerate(rows):
        length = len(doc_indices)
        documents[row_idx, :length] = raw_doc_embs_cpu[doc_indices].float()
        relevance[row_idx, :length] = torch.tensor(labels, dtype=torch.float32)
        mask[row_idx, :length] = True
        candidate_indices[row_idx, :length] = torch.tensor(doc_indices, dtype=torch.long)
    return {
        "qids": [row[0] for row in rows],
        "query_embeddings": query_embeddings,
        "documents": documents,
        "relevance": relevance,
        "mask": mask,
        "candidate_indices": candidate_indices,
    }


@torch.no_grad()
def adapt_embeddings_in_chunks(
    model: SearchAdaptorModel,
    embeddings_cpu: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, len(embeddings_cpu), max(1, int(batch_size))):
        chunk = embeddings_cpu[start : start + batch_size].to(device).float()
        chunks.append(model.adapt(chunk).detach().float().cpu())
    if was_training:
        model.train()
    if not chunks:
        return torch.empty((0, model.embedding_dim), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate_search_adaptor_model(
    model: SearchAdaptorModel,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    examples: Sequence[pipeline.EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    cfg: pipeline.Cfg,
    ks: Sequence[int],
    adapted_doc_embeddings: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, Dict[str, float]], torch.Tensor, np.ndarray]:
    if not examples:
        return {}, torch.empty(0), np.empty((0, 0), dtype=np.int64)
    options = ACTIVE_OPTIONS
    adapted_docs = adapted_doc_embeddings
    if adapted_docs is None:
        adapted_docs = adapt_embeddings_in_chunks(
            model, raw_doc_embs_cpu, cfg.device, options.inference_batch_size
        )
    index = pipeline.SearchIndex(adapted_docs)
    query_indices = [example.q_idx for example in examples]
    adapted_queries = adapt_embeddings_in_chunks(
        model,
        raw_query_embs_cpu[query_indices],
        cfg.device,
        options.inference_batch_size,
    )
    ranked = index.search(
        adapted_queries.numpy().astype(np.float32), max(int(k) for k in ks)
    )
    metrics = pipeline.eval_metrics_multi_k_from_inds(ranked, list(examples), qrels_by_qid_idx, ks)
    metrics = add_legalbench_char_metrics(metrics, ranked, list(examples), cfg)
    return metrics, adapted_docs, ranked


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _trial_slug(alpha: float, beta: float) -> str:
    def clean(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return f"alpha_{clean(alpha)}__beta_{clean(beta)}"


def _save_search_trial_state(
    path: Path,
    model: SearchAdaptorModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_step: int,
    best_score: float,
    order: Sequence[str],
    cursor: int,
    rng: random.Random,
    history: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "step": int(step),
            "best_step": int(best_step),
            "best_score": float(best_score),
            "query_order": list(order),
            "query_cursor": int(cursor),
            "python_rng_state": rng.getstate(),
            "history": list(history),
            "metadata": metadata,
        },
        path,
    )


def _teacher_metadata_base(
    cfg: pipeline.Cfg,
    fraction: float,
    full_train_qids: Sequence[str],
    selected_train_qids: Sequence[str],
    split_source: str,
) -> Dict[str, Any]:
    options = ACTIVE_OPTIONS
    hidden_dim = int(options.hidden_dim or cfg.embedding_dim)
    return {
        "checkpoint_format": "search-adaptor-paper-objective-v1",
        "paper": PAPER_URL,
        "dataset_name": cfg.dataset_name or Path(cfg.dataset_dir).name,
        "dataset_dir": str(Path(cfg.dataset_dir).resolve()),
        "dataset_format": cfg.source_dataset_format or cfg.dataset_format,
        "seed": int(cfg.seed),
        "encoder_model_name": cfg.model_name,
        "embedding_dim": int(cfg.embedding_dim),
        "teacher_training_fraction": float(fraction),
        "fraction_selection_seed": int(getattr(cfg, "_fraction_selection_seed", cfg.seed)),
        "teacher_training_uses_qrels": True,
        "student_training_uses_qrels": False,
        "dry_run_teacher": bool(cfg.dry_run),
        "split_source": split_source,
        "full_teacher_train_query_count": len(full_train_qids),
        "selected_teacher_train_query_count": len(selected_train_qids),
        "full_ordered_train_qids_sha1": provenance.sha1_jsonable(
            deterministic_fraction_qids(
                full_train_qids,
                1.0,
                int(getattr(cfg, "_fraction_selection_seed", cfg.seed)),
            )[0]
        ),
        "selected_teacher_train_qids_sha1": provenance.sha1_jsonable(list(selected_train_qids)),
        "selected_teacher_train_qids": list(selected_train_qids),
        "model_config": {
            "embedding_dim": int(cfg.embedding_dim),
            "hidden_dim": hidden_dim,
            "num_layers": int(options.num_layers),
            "activation": options.activation,
            "shared_query_document_adapter": True,
            "residual_skip_connection": True,
            "query_predictor_training_only": True,
            "normalization_before_adapter": False,
            "cosine_similarity_after_adapter": True,
            "dropout": 0.0,
            "layer_norm": False,
            "architecture_disclosure": (
                "The paper specifies MLP adapter/predictor modules but does not report depth, hidden width, "
                "activation, or initialization. These values are explicit implementation assumptions."
            ),
        },
        "paper_fixed_hyperparameters": {
            "batch_size": int(options.batch_size),
            "maximum_training_iterations": int(options.max_iterations),
            "early_stopping_patience_iterations": int(options.patience),
            "learning_rate": float(options.learning_rate),
            "optimizer": "Adam",
            "negative_pair_subsampling_ratio": int(options.negative_pair_ratio),
            "alpha_grid": list(options.alpha_grid),
            "beta_grid": list(options.beta_grid),
        },
        "objective": {
            "ranking": "sum I(y_j>y_k)*(y_j-y_k)*softplus(s_k-s_j)",
            "recovery": "mean_q ||q_hat-q||_1 + mean_d ||d_hat-d||_1",
            "prediction": "sum y_qd*||q_hat-p(d_hat)||_1 / sum y_qd",
            "total": "L_rank + alpha*L_rec + beta*L_pred",
        },
        "controlled_pipeline_deviations": {
            "benchmark_train_validation_test_splits_reused": True,
            "paper_internal_80_20_training_split_not_created": True,
            "reason": "Preserve exactly the same benchmark splits used by the IMRNN comparison.",
        },
    }


def _assert_trial_metadata_matches(
    expected: Mapping[str, Any], saved: Mapping[str, Any], path: Path
) -> None:
    keys = (
        "checkpoint_format",
        "dataset_name",
        "dataset_dir",
        "dataset_format",
        "seed",
        "encoder_model_name",
        "embedding_dim",
        "teacher_training_fraction",
        "dry_run_teacher",
        "selected_teacher_train_qids_sha1",
        "model_config",
        "paper_fixed_hyperparameters",
        "alpha",
        "beta",
    )
    mismatches = [
        f"{key}: saved={saved.get(key)!r} expected={expected.get(key)!r}"
        for key in keys
        if saved.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"Incompatible Search-Adaptor trial checkpoint {path}:\n- " + "\n- ".join(mismatches)
        )


def _train_one_search_adaptor_trial(
    cfg: pipeline.Cfg,
    trial_dir: Path,
    alpha: float,
    beta: float,
    train_qids: Sequence[str],
    val_examples: Sequence[pipeline.EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    query_id_to_index: Dict[str, int],
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    common_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    options = ACTIVE_OPTIONS
    trial_dir.mkdir(parents=True, exist_ok=True)
    best_path = trial_dir / "best.pt"
    latest_path = trial_dir / "latest.pt"
    complete_path = trial_dir / "complete.json"
    trial_metadata = dict(common_metadata)
    trial_metadata.update({"alpha": float(alpha), "beta": float(beta), "trial_seed": int(cfg.seed)})
    if cfg.resume and complete_path.exists() and best_path.exists():
        best_payload = pipeline.safe_torch_load(best_path, map_location="cpu", weights_only=False)
        _assert_trial_metadata_matches(
            trial_metadata, dict(best_payload.get("metadata", {})), best_path
        )
        with complete_path.open("r", encoding="utf-8") as handle:
            completed = json.load(handle)
        pipeline.logger.info(
            f"Resuming Search-Adaptor grid: completed trial alpha={alpha} beta={beta} "
            f"best_nDCG@10={completed['best_validation_ndcg_10']:.6f}"
        )
        return completed

    # Use the same initialization and stochastic candidate stream for every
    # alpha/beta trial so regularizer selection is not confounded by sampling.
    trial_seed = int(cfg.seed)
    pipeline.set_seed(cfg.seed)
    hidden_dim = int(options.hidden_dim or cfg.embedding_dim)
    model = SearchAdaptorModel(
        cfg.embedding_dim, hidden_dim, options.num_layers, options.activation
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    rng = random.Random(trial_seed)
    query_order = list(train_qids)
    rng.shuffle(query_order)
    cursor = 0
    start_step = 0
    best_step = 0
    best_score = float("-inf")
    history: List[Dict[str, Any]] = []
    if cfg.resume and latest_path.exists():
        payload = pipeline.safe_torch_load(latest_path, map_location="cpu", weights_only=False)
        _assert_trial_metadata_matches(
            trial_metadata, dict(payload.get("metadata", {})), latest_path
        )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["step"])
        best_step = int(payload["best_step"])
        best_score = float(payload["best_score"])
        query_order = list(payload["query_order"])
        cursor = int(payload["query_cursor"])
        rng.setstate(payload["python_rng_state"])
        history = list(payload.get("history", []))
        pipeline.logger.info(
            f"Resuming Search-Adaptor trial alpha={alpha} beta={beta} at iteration {start_step}"
        )

    max_iterations = 1 if cfg.dry_run else int(options.max_iterations)
    eval_every = 1 if cfg.dry_run else max(1, int(options.eval_every))
    save_every = 1 if cfg.dry_run else max(1, int(options.save_every))
    eval_ks = (10,)

    if start_step == 0:
        val_metrics, _, _ = evaluate_search_adaptor_model(
            model,
            raw_doc_embs_cpu,
            raw_query_embs_cpu,
            val_examples,
            qrels_by_qid_idx,
            cfg,
            eval_ks,
        )
        best_score = pipeline.metric_at_or_zero(val_metrics, 10, "ndcg")
        torch.save(
            {
                "model_state": _cpu_state_dict(model),
                "metadata": {
                    **trial_metadata,
                    "best_step": 0,
                    "best_validation_ndcg_10": best_score,
                },
            },
            best_path,
        )
        pipeline.logger.info(
            f"Search-Adaptor alpha={alpha} beta={beta} initial validation nDCG@10={best_score:.6f}"
        )

    running = defaultdict(float)
    running_steps = 0
    stopped_early = False
    last_step = int(start_step)
    for step in range(start_step + 1, max_iterations + 1):
        last_step = int(step)
        batch_qids, query_order, cursor = _next_query_batch(
            query_order, cursor, options.batch_size, rng
        )
        batch = build_search_adaptor_batch(
            batch_qids,
            query_id_to_index,
            qrels_by_qid_idx,
            raw_query_embs_cpu,
            raw_doc_embs_cpu,
            options.negative_pair_ratio,
            cfg.use_graded_relevance,
            rng,
        )
        base_q = batch["query_embeddings"].to(cfg.device)
        base_docs = batch["documents"].to(cfg.device)
        relevance = batch["relevance"].to(cfg.device)
        mask = batch["mask"].to(cfg.device)
        optimizer.zero_grad(set_to_none=True)
        losses = search_adaptor_objective(
            model, base_q, base_docs, relevance, mask, alpha=alpha, beta=beta
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite Search-Adaptor loss at iteration {step}")
        if int(losses["ranking_pair_count"]) == 0:
            raise AssertionError("Search-Adaptor optimizer update has no valid ranking pair")
        losses["total"].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Non-finite Search-Adaptor gradient in {name} at iteration {step}"
                )
        optimizer.step()

        running["total"] += float(losses["total"].detach().cpu())
        running["ranking"] += float(losses["ranking"].detach().cpu())
        running["recovery"] += float(losses["recovery"].detach().cpu())
        running["prediction"] += float(losses["prediction"].detach().cpu())
        running["ranking_pairs"] += int(losses["ranking_pair_count"])
        running["queries"] += len(batch["qids"])
        running_steps += 1

        should_eval = step % eval_every == 0 or step == max_iterations
        if should_eval:
            val_metrics, _, _ = evaluate_search_adaptor_model(
                model,
                raw_doc_embs_cpu,
                raw_query_embs_cpu,
                val_examples,
                qrels_by_qid_idx,
                cfg,
                eval_ks,
            )
            validation_ndcg = pipeline.metric_at_or_zero(val_metrics, 10, "ndcg")
            improved = validation_ndcg > best_score + 1e-12
            if improved:
                best_score = validation_ndcg
                best_step = step
                torch.save(
                    {
                        "model_state": _cpu_state_dict(model),
                        "metadata": {
                            **trial_metadata,
                            "best_step": int(best_step),
                            "best_validation_ndcg_10": float(best_score),
                        },
                    },
                    best_path,
                )
            row = {
                "iteration": step,
                "alpha": float(alpha),
                "beta": float(beta),
                "loss": running["total"] / max(1, running_steps),
                "ranking_loss": running["ranking"] / max(1, running_steps),
                "recovery_loss": running["recovery"] / max(1, running_steps),
                "prediction_loss": running["prediction"] / max(1, running_steps),
                "ranking_pair_count": int(running["ranking_pairs"]),
                "query_count": int(running["queries"]),
                "validation_ndcg_10": float(validation_ndcg),
                "best_validation_ndcg_10": float(best_score),
                "best_iteration": int(best_step),
                "improved": bool(improved),
            }
            history.append(row)
            trial_dir.mkdir(parents=True, exist_ok=True)
            with (trial_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            pipeline.logger.info(
                f"Search-Adaptor iter={step}/{max_iterations} alpha={alpha:g} beta={beta:g} "
                f"loss={row['loss']:.6f} rank={row['ranking_loss']:.6f} "
                f"rec={row['recovery_loss']:.6f} pred={row['prediction_loss']:.6f} "
                f"val_nDCG@10={validation_ndcg:.6f} best={best_score:.6f}@{best_step}"
            )
            running = defaultdict(float)
            running_steps = 0

        if step % save_every == 0 or step == max_iterations:
            _save_search_trial_state(
                latest_path,
                model,
                optimizer,
                step,
                best_step,
                best_score,
                query_order,
                cursor,
                rng,
                history,
                trial_metadata,
            )
        if not cfg.dry_run and step - best_step >= int(options.patience):
            stopped_early = True
            pipeline.logger.info(
                f"Search-Adaptor early stopping at iteration {step}; no nDCG@10 improvement for "
                f"{step - best_step} iterations."
            )
            break

    result = {
        "alpha": float(alpha),
        "beta": float(beta),
        "best_validation_ndcg_10": float(best_score),
        "best_iteration": int(best_step),
        "last_iteration": last_step,
        "stopped_early": bool(stopped_early),
        "best_checkpoint": str(best_path),
    }
    trial_dir.mkdir(parents=True, exist_ok=True)
    with complete_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    del model, optimizer
    pipeline.clear_cuda_cache_if_needed(cfg)
    return result


def train_search_adaptor_checkpoint(
    cfg: pipeline.Cfg,
    checkpoint_path: Path,
    split_ids: Dict[str, List[str]],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    query_id_to_index: Dict[str, int],
    raw_index: pipeline.SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    split_source: str,
) -> Path:
    del raw_index
    options = ACTIVE_OPTIONS
    fraction = float(getattr(cfg, "_search_adaptor_fraction", 1.0))
    eligible_train_qids = [
        qid
        for qid in split_ids["train"]
        if qid in query_id_to_index
        and any(float(label) > 0.0 for label in qrels_by_qid_idx.get(qid, {}).values())
    ]
    fraction_seed = int(getattr(cfg, "_fraction_selection_seed", cfg.seed))
    full_order, selected_qids = deterministic_fraction_qids(
        eligible_train_qids, fraction, fraction_seed
    )
    if not selected_qids:
        raise ValueError("No qrel-bearing teacher-training queries remain for Search-Adaptor")
    val_qids = [
        qid for qid in split_ids["val"] if qid in query_id_to_index and qid in qrels_by_qid_idx
    ]
    val_examples = [
        pipeline.EvalExample(
            qid=qid, q_idx=query_id_to_index[qid], query_text="", query_type="", base_type=""
        )
        for qid in val_qids
    ]
    if not val_examples:
        raise ValueError(
            "Search-Adaptor requires validation queries for paper-style nDCG@10 selection"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fraction": fraction,
        "seed": cfg.seed,
        "fraction_selection_seed": fraction_seed,
        "full_ordered_train_qids": full_order,
        "selected_train_qids": selected_qids,
        "selected_train_qids_sha1": provenance.sha1_jsonable(selected_qids),
        "nested_prefix_selection": True,
    }
    with (checkpoint_path.parent / "teacher_training_subset.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    common_metadata = _teacher_metadata_base(
        cfg, fraction, eligible_train_qids, selected_qids, split_source
    )
    pipeline.log_section("TRAIN SEARCH-ADAPTOR TEACHER")
    pipeline.logger.warning(
        "Search-Adaptor teacher training is SUPERVISED and uses qrels. The downstream E5 student stage remains "
        "qrel-free and receives only frozen Search-Adaptor scores/deltas."
    )
    pipeline.logger.info(
        f"dataset={common_metadata['dataset_name']} seed={cfg.seed} fraction={fraction:.2f} "
        f"selected_teacher_queries={len(selected_qids)}/{len(eligible_train_qids)} "
        f"batch_size={options.batch_size} max_iterations={options.max_iterations} "
        f"patience={options.patience} lr={options.learning_rate} negative_ratio={options.negative_pair_ratio}"
    )

    if cfg.dry_run or not options.grid_search:
        grid = [(float(options.fixed_alpha), float(options.fixed_beta))]
    else:
        grid = [
            (float(alpha), float(beta))
            for alpha in options.alpha_grid
            for beta in options.beta_grid
        ]
    trial_results: List[Dict[str, Any]] = []
    trials_root = checkpoint_path.parent / "trials"
    for alpha, beta in grid:
        trial_results.append(
            _train_one_search_adaptor_trial(
                cfg,
                trials_root / _trial_slug(alpha, beta),
                alpha,
                beta,
                selected_qids,
                val_examples,
                qrels_by_qid_idx,
                query_id_to_index,
                raw_doc_embs_cpu,
                raw_query_embs_cpu,
                common_metadata,
            )
        )
    selected = max(
        trial_results,
        key=lambda row: (
            float(row["best_validation_ndcg_10"]),
            -float(row["alpha"]),
            -float(row["beta"]),
        ),
    )
    best_payload = pipeline.safe_torch_load(
        selected["best_checkpoint"], map_location="cpu", weights_only=False
    )
    selected_metadata = {
        **common_metadata,
        "selected_alpha": float(selected["alpha"]),
        "selected_beta": float(selected["beta"]),
        "best_validation_ndcg_10": float(selected["best_validation_ndcg_10"]),
        "best_iteration": int(selected["best_iteration"]),
        "regularizer_grid_search": bool(options.grid_search and not cfg.dry_run),
        "grid_results": trial_results,
        "selected_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(
        {"model_state": best_payload["model_state"], "metadata": selected_metadata},
        checkpoint_path,
    )
    with (checkpoint_path.parent / "search_adaptor_training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(selected_metadata, handle, indent=2, sort_keys=True)
    setattr(cfg, "_search_adaptor_teacher_metadata", selected_metadata)
    pipeline.logger.info(
        f"Saved selected Search-Adaptor checkpoint: {checkpoint_path} | "
        f"alpha={selected['alpha']} beta={selected['beta']} "
        f"val_nDCG@10={selected['best_validation_ndcg_10']:.6f}"
    )
    return checkpoint_path


def ensure_search_adaptor_checkpoint(
    cfg: pipeline.Cfg,
    split_ids: Dict[str, List[str]],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    query_id_to_index: Dict[str, int],
    raw_index: pipeline.SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    split_source: str,
) -> Path:
    path = Path(cfg.teacher_checkpoint_path).expanduser()
    if path.exists() and (cfg.resume or cfg.mode == pipeline.MODE_EVAL_ONLY):
        payload = pipeline.safe_torch_load(path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        eligible = [
            qid
            for qid in split_ids["train"]
            if qid in query_id_to_index
            and any(float(label) > 0.0 for label in qrels_by_qid_idx.get(qid, {}).values())
        ]
        fraction = float(getattr(cfg, "_search_adaptor_fraction", 1.0))
        fraction_seed = int(getattr(cfg, "_fraction_selection_seed", cfg.seed))
        full_order, selected = deterministic_fraction_qids(eligible, fraction, fraction_seed)
        expected_subset = {
            "fraction_selection_seed": fraction_seed,
            "full_teacher_train_query_count": len(eligible),
            "selected_teacher_train_query_count": len(selected),
            "full_ordered_train_qids_sha1": provenance.sha1_jsonable(full_order),
            "selected_teacher_train_qids_sha1": provenance.sha1_jsonable(selected),
        }
        _assert_teacher_metadata(cfg, metadata, path, expected_subset)
        setattr(cfg, "_search_adaptor_teacher_metadata", metadata)
        return path
    if cfg.mode == pipeline.MODE_EVAL_ONLY:
        raise FileNotFoundError(f"Search-Adaptor checkpoint required for eval-only mode: {path}")
    if not cfg.auto_train_teacher_if_missing:
        raise FileNotFoundError(f"Search-Adaptor checkpoint does not exist: {path}")
    return train_search_adaptor_checkpoint(
        cfg,
        path,
        split_ids,
        qrels_by_qid_idx,
        query_id_to_index,
        raw_index,
        raw_doc_embs_cpu,
        raw_query_embs_cpu,
        split_source,
    )


def _assert_teacher_metadata(
    cfg: pipeline.Cfg,
    metadata: Mapping[str, Any],
    path: Path,
    expected_subset: Mapping[str, Any] | None = None,
) -> None:
    expected_dataset = cfg.dataset_name or Path(cfg.dataset_dir).name
    mismatches = []
    if metadata.get("checkpoint_format") != "search-adaptor-paper-objective-v1":
        mismatches.append(f"checkpoint_format={metadata.get('checkpoint_format')!r}")
    if bool(metadata.get("dry_run_teacher", False)) != bool(cfg.dry_run):
        mismatches.append(
            f"dry_run_teacher current={bool(cfg.dry_run)} checkpoint={metadata.get('dry_run_teacher')!r}"
        )
    if str(metadata.get("dataset_name")) != str(expected_dataset):
        mismatches.append(
            f"dataset current={expected_dataset!r} checkpoint={metadata.get('dataset_name')!r}"
        )
    if int(metadata.get("seed", -1)) != int(cfg.seed):
        mismatches.append(f"seed current={cfg.seed} checkpoint={metadata.get('seed')!r}")
    if str(metadata.get("encoder_model_name")) != str(cfg.model_name):
        mismatches.append(
            f"encoder current={cfg.model_name!r} checkpoint={metadata.get('encoder_model_name')!r}"
        )
    expected_fraction = float(getattr(cfg, "_search_adaptor_fraction", 1.0))
    if not math.isclose(float(metadata.get("teacher_training_fraction", -1.0)), expected_fraction):
        mismatches.append(
            f"fraction current={expected_fraction} checkpoint={metadata.get('teacher_training_fraction')!r}"
        )
    model_cfg = metadata.get("model_config", {})
    expected_model_cfg = {
        "embedding_dim": int(cfg.embedding_dim),
        "hidden_dim": int(ACTIVE_OPTIONS.hidden_dim or cfg.embedding_dim),
        "num_layers": int(ACTIVE_OPTIONS.num_layers),
        "activation": ACTIVE_OPTIONS.activation,
    }
    for key, expected in expected_model_cfg.items():
        if model_cfg.get(key) != expected:
            mismatches.append(
                f"model_config.{key} current={expected!r} checkpoint={model_cfg.get(key)!r}"
            )
    paper_hparams = metadata.get("paper_fixed_hyperparameters", {})
    expected_hparams = {
        "batch_size": int(ACTIVE_OPTIONS.batch_size),
        "maximum_training_iterations": int(ACTIVE_OPTIONS.max_iterations),
        "early_stopping_patience_iterations": int(ACTIVE_OPTIONS.patience),
        "learning_rate": float(ACTIVE_OPTIONS.learning_rate),
        "optimizer": "Adam",
        "negative_pair_subsampling_ratio": int(ACTIVE_OPTIONS.negative_pair_ratio),
        "alpha_grid": list(ACTIVE_OPTIONS.alpha_grid),
        "beta_grid": list(ACTIVE_OPTIONS.beta_grid),
    }
    for key, expected in expected_hparams.items():
        if paper_hparams.get(key) != expected:
            mismatches.append(
                f"paper_fixed_hyperparameters.{key} current={expected!r} "
                f"checkpoint={paper_hparams.get(key)!r}"
            )
    if not ACTIVE_OPTIONS.grid_search:
        if not math.isclose(
            float(metadata.get("selected_alpha", -1.0)), ACTIVE_OPTIONS.fixed_alpha
        ):
            mismatches.append("selected alpha does not match --search_adaptor_fixed_alpha")
        if not math.isclose(float(metadata.get("selected_beta", -1.0)), ACTIVE_OPTIONS.fixed_beta):
            mismatches.append("selected beta does not match --search_adaptor_fixed_beta")
    if expected_subset is not None:
        for key, expected in expected_subset.items():
            if metadata.get(key) != expected:
                mismatches.append(f"{key} current={expected!r} checkpoint={metadata.get(key)!r}")
    if mismatches:
        raise ValueError(
            f"Incompatible Search-Adaptor checkpoint {path}:\n- " + "\n- ".join(mismatches)
        )


class SearchAdaptorTeacherWrapper:
    """Frozen teacher interface expected by the existing student pipeline."""

    def __init__(self, cfg: pipeline.Cfg) -> None:
        self.cfg = cfg
        self.device = cfg.device
        path = Path(cfg.teacher_checkpoint_path).expanduser()
        payload = pipeline.safe_torch_load(path, map_location="cpu", weights_only=False)
        self.metadata = dict(payload.get("metadata", {}))
        _assert_teacher_metadata(cfg, self.metadata, path)
        model_cfg = self.metadata["model_config"]
        self.model = SearchAdaptorModel(
            int(model_cfg["embedding_dim"]),
            int(model_cfg["hidden_dim"]),
            int(model_cfg["num_layers"]),
            str(model_cfg["activation"]),
        )
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._adapted_corpus_cache: Optional[torch.Tensor] = None
        setattr(cfg, "_search_adaptor_teacher_metadata", self.metadata)
        pipeline.log_section("FROZEN SEARCH-ADAPTOR TEACHER")
        pipeline.logger.info(f"teacher_checkpoint_path: {path}")
        pipeline.logger.info(
            f"teacher_training_fraction={self.metadata['teacher_training_fraction']} | "
            f"selected_alpha={self.metadata['selected_alpha']} | selected_beta={self.metadata['selected_beta']} | "
            f"validation_nDCG@10={self.metadata['best_validation_ndcg_10']:.6f}"
        )
        pipeline.logger.info(
            "teacher_training_uses_qrels=true | student_training_uses_qrels=false | "
            "inference_uses_shared_adapter=true | inference_uses_query_predictor=false"
        )

    @torch.no_grad()
    def score_batch(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        document_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = query_embeddings.to(self.device).float()
        docs = document_embeddings.to(self.device).float()
        _, _, scores = self.model.score(q, docs)
        if document_mask is not None:
            scores = scores.masked_fill(~document_mask.to(self.device).bool(), -1e9)
        return scores.detach()

    @torch.no_grad()
    def extract_modulation_signals(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        document_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        q_base = query_embeddings.to(self.device).float()
        d_base = document_embeddings.to(self.device).float()
        q_mod, d_mod, scores = self.model.score(q_base, d_base)
        if document_mask is None:
            mask = torch.ones(scores.shape, dtype=torch.bool, device=self.device)
        else:
            mask = document_mask.to(self.device).bool()
        scores = scores.masked_fill(~mask, -1e9)
        d_mod = d_mod.masked_fill(~mask.unsqueeze(-1), 0.0)
        d_base_safe = d_base.masked_fill(~mask.unsqueeze(-1), 0.0)
        return {
            "q_base_T": q_base.detach(),
            "q_mod_T": q_mod.detach(),
            "delta_q_T": (q_mod - q_base).detach(),
            "d_base_T": d_base_safe.detach(),
            "d_mod_T": d_mod.detach(),
            "delta_d_T": (d_mod - d_base_safe).masked_fill(~mask.unsqueeze(-1), 0.0).detach(),
            "teacher_scores": scores.detach(),
        }

    @torch.no_grad()
    def rerank_candidate_indices(
        self,
        query_embs_cpu: torch.Tensor,
        doc_embs_cpu: torch.Tensor,
        q_indices: List[int],
        candidate_inds: np.ndarray,
        max_k: int,
    ) -> np.ndarray:
        ranked_rows: List[np.ndarray] = []
        batch_size = max(1, min(int(self.cfg.eval_batch_size), 128))
        for start in range(0, len(q_indices), batch_size):
            q_chunk = q_indices[start : start + batch_size]
            candidates = candidate_inds[start : start + batch_size]
            valid_np = candidates >= 0
            safe = candidates.copy()
            safe[~valid_np] = 0
            q = query_embs_cpu[q_chunk].to(self.device).float()
            docs = doc_embs_cpu[torch.from_numpy(safe).long()].to(self.device).float()
            mask = torch.from_numpy(valid_np).to(self.device)
            scores = self.score_batch(q, docs, mask)
            order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
            ranked_rows.append(np.take_along_axis(candidates, order, axis=1)[:, :max_k])
        return np.vstack(ranked_rows) if ranked_rows else np.empty((0, 0), dtype=np.int64)

    @torch.no_grad()
    def adapted_corpus(self, raw_doc_embs_cpu: torch.Tensor) -> torch.Tensor:
        if self._adapted_corpus_cache is None:
            self._adapted_corpus_cache = adapt_embeddings_in_chunks(
                self.model,
                raw_doc_embs_cpu,
                self.device,
                ACTIVE_OPTIONS.inference_batch_size,
            )
        return self._adapted_corpus_cache


def evaluate_search_adaptor_teacher(
    teacher: SearchAdaptorTeacherWrapper,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    examples: Sequence[pipeline.EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    cfg: pipeline.Cfg,
) -> Dict[int, Dict[str, float]]:
    metrics, _, _ = evaluate_search_adaptor_model(
        teacher.model,
        raw_doc_embs_cpu,
        raw_query_embs_cpu,
        examples,
        qrels_by_qid_idx,
        cfg,
        cfg.ks,
        adapted_doc_embeddings=teacher.adapted_corpus(raw_doc_embs_cpu),
    )
    return metrics


def evaluate_search_adaptor_systems(
    cfg: pipeline.Cfg,
    teacher: SearchAdaptorTeacherWrapper,
    raw_index: pipeline.SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    passages: List[pipeline.PassageRecord],
    val_examples: List[pipeline.EvalExample],
    test_examples: List[pipeline.EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    students: Dict[str, pipeline.TrainableRetriever],
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    slices = (
        pipeline.build_slices(val_examples, test_examples)
        if cfg.report_comlq_slices
        else {"VAL": val_examples, "TEST": test_examples}
    )
    results: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = defaultdict(dict)
    teacher.adapted_corpus(raw_doc_embs_cpu)
    for slice_name, examples in slices.items():
        if not examples:
            continue
        base_metrics = pipeline.evaluate_base_retriever(
            raw_index, raw_query_embs_cpu, examples, qrels_by_qid_idx, cfg
        )
        results[pipeline.VARIANT_E5_BASE][slice_name] = base_metrics
        pipeline.log_metrics_table(
            f"{pipeline.VARIANT_E5_BASE} | {slice_name}", base_metrics, cfg.ks
        )
        teacher_metrics = evaluate_search_adaptor_teacher(
            teacher, raw_doc_embs_cpu, raw_query_embs_cpu, examples, qrels_by_qid_idx, cfg
        )
        results[SEARCH_ADAPTOR][slice_name] = teacher_metrics
        pipeline.log_metrics_table(f"{SEARCH_ADAPTOR} | {slice_name}", teacher_metrics, cfg.ks)

    for variant, student in students.items():
        student.eval()
        doc_embeddings = pipeline.encode_corpus_with_student(student, passages, cfg)
        student_index = pipeline.SearchIndex(doc_embeddings)
        for slice_name, examples in slices.items():
            if not examples:
                continue
            metrics = pipeline.evaluate_student(
                student, student_index, examples, qrels_by_qid_idx, cfg
            )
            results[variant][slice_name] = metrics
            pipeline.log_metrics_table(f"{variant} | {slice_name}", metrics, cfg.ks)
        del student_index, doc_embeddings
        pipeline.clear_cuda_cache_if_needed(cfg)
    return dict(results)


def search_adaptor_quality_gate(
    cfg: pipeline.Cfg,
    teacher: SearchAdaptorTeacherWrapper,
    raw_index: pipeline.SearchIndex,
    raw_doc_embs_cpu: torch.Tensor,
    raw_query_embs_cpu: torch.Tensor,
    val_examples: List[pipeline.EvalExample],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    pipeline.log_section("SEARCH-ADAPTOR QUALITY DIAGNOSTIC")
    base_metrics = pipeline.evaluate_base_retriever(
        raw_index, raw_query_embs_cpu, val_examples, qrels_by_qid_idx, cfg
    )
    teacher_metrics = evaluate_search_adaptor_teacher(
        teacher, raw_doc_embs_cpu, raw_query_embs_cpu, val_examples, qrels_by_qid_idx, cfg
    )
    base_score = pipeline.validation_score(base_metrics, cfg)
    teacher_score = pipeline.validation_score(teacher_metrics, cfg)
    weak = teacher_score < base_score
    pipeline.log_metrics_table(
        f"{pipeline.VARIANT_E5_BASE} | VAL | TEACHER DIAGNOSTIC", base_metrics, cfg.ks
    )
    pipeline.log_metrics_table("SEARCH_ADAPTOR | VAL | TEACHER DIAGNOSTIC", teacher_metrics, cfg.ks)
    pipeline.logger.info(
        f"search_adaptor_score={teacher_score:.6f} base_score={base_score:.6f} weak_teacher={weak}"
    )
    if weak and ACTIVE_OPTIONS.fail_on_weak_teacher:
        raise RuntimeError(
            "Search-Adaptor is weaker than the base retriever on validation and "
            "--fail_on_weak_search_adaptor is set"
        )
    if weak:
        pipeline.logger.warning(
            "Search-Adaptor is weaker than the frozen base retriever on this validation split. Continuing so low-data "
            "fraction results are not selectively censored."
        )
    return {
        "base_score": base_score,
        "teacher_score": teacher_score,
        "weak_teacher": weak,
        "gate_is_diagnostic_only": not ACTIVE_OPTIONS.fail_on_weak_teacher,
    }


def save_search_adaptor_student_checkpoint(
    student: pipeline.TrainableRetriever,
    projection_head: Optional[pipeline.SupervisionProjectionHead],
    checkpoint_dir: str,
    cfg: pipeline.Cfg,
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
    teacher_metadata = dict(getattr(cfg, "_search_adaptor_teacher_metadata", {}))
    metadata = {
        "checkpoint_format": "dense-retriever-student-rcd-search-adaptor-v2",
        "checkpoint_kind": checkpoint_kind,
        "variant": variant,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_score": best_score,
        "best_val_metrics": best_metrics,
        "patience_ctr": patience_ctr,
        "completed": completed,
        "teacher_type": SEARCH_ADAPTOR,
        "teacher_training_uses_qrels": True,
        "qrels_used_in_student_training": bool(
            candidate_metadata.get("qrels_used_in_training", False)
        ),
        "qrel_positive_injection": bool(candidate_metadata.get("qrel_positive_injection", False)),
        "qrel_loss_active": bool(candidate_metadata.get("qrel_loss_active", False)),
        "student_inference_requires_search_adaptor": False,
        "teacher_training_fraction": teacher_metadata.get("teacher_training_fraction"),
        "search_adaptor_teacher_metadata": teacher_metadata,
        "candidate_metadata": candidate_metadata,
        "supervision_projection": {
            "input_dim": cfg.embedding_dim,
            "output_dim": cfg.signal_projection_output_dim,
            "architecture": "linear_layer_norm",
            "training_only": True,
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
    if cfg.mode == pipeline.MODE_UNSUPERVISED:
        assert metadata["qrels_used_in_student_training"] is False
        assert metadata["qrel_positive_injection"] is False
        assert metadata["qrel_loss_active"] is False
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
    pipeline.logger.info(
        f"Saved {checkpoint_kind} {variant} Search-Adaptor-distilled student checkpoint to {checkpoint_dir}"
    )


def require_search_adaptor_runtime_dependencies() -> None:
    missing = []
    if SentenceTransformer is None:
        missing.append(f"sentence-transformers ({_SENTENCE_TRANSFORMERS_IMPORT_ERROR})")
    if get_linear_schedule_with_warmup is None:
        missing.append(f"transformers ({_TRANSFORMERS_IMPORT_ERROR})")
    if missing:
        raise ImportError("Missing required runtime dependencies:\n  - " + "\n  - ".join(missing))


def log_search_adaptor_run_header(cfg: pipeline.Cfg, log_path: str) -> None:
    _ORIGINAL_LOG_RUN_HEADER(cfg, log_path)
    pipeline.log_section("SEARCH-ADAPTOR EXPERIMENT CONTRACT")
    pipeline.logger.info(f"actual_entry_point: {Path(__file__).name}")
    pipeline.logger.info("teacher: Search-Adaptor (Yoon et al., ACL 2024)")
    pipeline.logger.info("teacher_training_uses_qrels: true")
    pipeline.logger.info("student_candidate_construction_uses_qrels: false")
    pipeline.logger.info("student_loss_uses_qrels: false")
    pipeline.logger.info("candidate_source: frozen_base_retriever")
    pipeline.logger.info("dynamic_refresh: false")
    pipeline.logger.info(
        f"teacher_training_fraction: {float(getattr(cfg, '_search_adaptor_fraction', 1.0)):.2f}"
    )
    pipeline.logger.info(
        "Search-Adaptor objective: weighted pairwise logistic ranking + alpha*L1 recovery + "
        "beta*relevance-weighted L1 query prediction"
    )
    pipeline.logger.info(
        "Search-Adaptor inference: one shared residual adapter over frozen base-retriever embeddings; "
        "query predictor discarded"
    )


def install_pipeline_hooks() -> None:
    """Replace only the teacher-specific extension points in the base pipeline."""
    pipeline.require_runtime_dependencies = require_search_adaptor_runtime_dependencies
    pipeline.train_teacher_checkpoint = train_search_adaptor_checkpoint
    pipeline.ensure_teacher_checkpoint = ensure_search_adaptor_checkpoint
    pipeline.TeacherAdapter = SearchAdaptorTeacherWrapper
    pipeline.evaluate_systems = evaluate_search_adaptor_systems
    pipeline.teacher_quality_gate = search_adaptor_quality_gate
    pipeline.save_student_checkpoint = save_search_adaptor_student_checkpoint
    pipeline.log_run_header = log_search_adaptor_run_header
    pipeline.VARIANT_IMRNN_TEACHER = SEARCH_ADAPTOR
    pipeline.BASELINE_VARIANTS = {pipeline.VARIANT_E5_BASE, SEARCH_ADAPTOR}


def _normalize_dataset_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key == "comlq":
        return key
    if key in LEGALBENCH_DATASETS:
        return pipeline.normalize_legalbench_dataset_name(key)
    return key


def _resolve_comlq_dir(path: str) -> str:
    root = Path(path).expanduser()
    candidates = (root, root / "dataset")
    for candidate in candidates:
        if (candidate / "corpus.jsonl").exists() and (candidate / "queries.jsonl").exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find ComLQ corpus.jsonl and queries.jsonl in {root} or {root / 'dataset'}"
    )


def _dataset_cfg(base_cfg: pipeline.Cfg, dataset_name: str, suite_output: Path) -> pipeline.Cfg:
    if dataset_name == "comlq":
        dataset_dir = _resolve_comlq_dir(base_cfg.dataset_dir)
        return replace(
            base_cfg,
            dataset_format="standard",
            source_dataset_format="comlq",
            dataset_name="comlq",
            dataset_dir=dataset_dir,
            corpus_file="corpus.jsonl",
            queries_file="queries.jsonl",
            qrels_dir="qrels",
            legalbench_metadata_path="",
            report_comlq_slices=base_cfg.report_comlq_slices,
            output_dir=str(suite_output / dataset_name),
            log_path="",
        )

    if dataset_name not in LEGALBENCH_DATASETS:
        root = Path(base_cfg.dataset_dir).expanduser()
        candidates = (root / dataset_name, root)
        dataset_dir = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "corpus.jsonl").is_file()
                and (candidate / "queries.jsonl").is_file()
                and (candidate / "qrels").is_dir()
            ),
            None,
        )
        if dataset_dir is None:
            raise FileNotFoundError(
                f"Could not find prepared standard dataset {dataset_name!r} under {root}"
            )
        return replace(
            base_cfg,
            dataset_format="comlq",
            source_dataset_format="standard",
            dataset_name=dataset_name,
            dataset_dir=str(dataset_dir),
            corpus_file="corpus.jsonl",
            queries_file="queries.jsonl",
            qrels_dir="qrels",
            query_type_filter="all",
            query_types="",
            legalbench_metadata_path="",
            report_comlq_slices=False,
            output_dir=str(suite_output / dataset_name),
            log_path="",
        )

    legalbench_cfg = base_cfg
    if not base_cfg.legalbench_rag_root:
        candidates = (
            Path("datasets/legalbench-rag"),
            Path("datasets/legalbench_rag"),
            Path("datasets/LegalBench-RAG"),
            Path("legalbench-rag"),
            Path("legalbench_rag"),
            Path.cwd(),
        )
        discovered = next(
            (
                candidate
                for candidate in candidates
                if legalbench_has_data_directories(candidate)
                or legalbench_has_archive_pair(candidate)
                or legalbench_has_data_directories(candidate / "data")
                or legalbench_has_archive_pair(candidate / "data")
            ),
            None,
        )
        if discovered is None:
            raise FileNotFoundError(
                "LegalBench-RAG data was not found in the standard local paths. Pass "
                "--legalbench_rag_root /path/containing/corpus-and-benchmarks."
            )
        legalbench_cfg = replace(base_cfg, legalbench_rag_root=str(discovered))
    prepared_root = (
        Path(legalbench_cfg.legalbench_prepared_dir)
        if legalbench_cfg.legalbench_prepared_dir
        else suite_output / "prepared_legalbench_rag"
    )
    prepared_dir = prepared_root / dataset_name
    metadata = prepare_legalbench_retrieval_dataset(
        legalbench_cfg,
        dataset_name,
        prepared_dir,
        include_paper_note=True,
        progress_logger=pipeline.logger,
    )
    pipeline.logger.info(
        f"Prepared LegalBench-RAG {dataset_name}: passages={metadata['passage_count']} "
        f"queries={metadata['query_count']} qrels={metadata['qrel_count']}"
    )
    return replace(
        base_cfg,
        dataset_format="standard",
        source_dataset_format="legalbench_rag",
        dataset_name=dataset_name,
        dataset_dir=str(prepared_dir),
        corpus_file="corpus.jsonl",
        queries_file="queries.jsonl",
        qrels_dir="qrels",
        query_type_filter="all",
        query_types="",
        legalbench_metadata_path=str(prepared_dir / "legalbench_metadata.json"),
        report_comlq_slices=False,
        output_dir=str(suite_output / dataset_name),
        log_path="",
    )


def _fraction_tag(fraction: float) -> str:
    return f"fraction_{int(round(100.0 * float(fraction))):03d}"


def _write_suite_results(
    output_dir: Path, payloads: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "search_adaptor_results_by_dataset_fraction.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payloads, handle, indent=2, sort_keys=True)
    rows: List[Dict[str, Any]] = []
    for dataset_name, fractions in payloads.items():
        for fraction_tag, per_seed in fractions.items():
            fraction = float(fraction_tag.split("_", 1)[1]) / 100.0
            for payload in per_seed:
                for row in pipeline.flatten_results(payload.get("results", {})):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "teacher_train_fraction": fraction,
                            "seed": payload.get("seed"),
                            **row,
                        }
                    )
    pipeline.write_csv_rows(
        str(output_dir / "search_adaptor_results_by_dataset_fraction.csv"), rows
    )

    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"dataset", "teacher_train_fraction", "seed", "system", "slice", "k"}
            and isinstance(value, (int, float))
        }
    )
    grouped: Dict[Tuple[str, float, str, str, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["dataset"]),
            float(row["teacher_train_fraction"]),
            str(row["system"]),
            str(row["slice"]),
            int(row["k"]),
        )
        for metric in metric_names:
            if metric in row:
                grouped[key][metric].append(float(row[metric]))
    aggregate_rows: List[Dict[str, Any]] = []
    for (dataset, fraction, system, slice_name, k), values_by_metric in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "teacher_train_fraction": fraction,
            "system": system,
            "slice": slice_name,
            "k": k,
        }
        for metric, values in values_by_metric.items():
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        aggregate_rows.append(row)
    pipeline.write_csv_rows(
        str(output_dir / "search_adaptor_results_by_dataset_fraction_aggregate.csv"), aggregate_rows
    )
    with (output_dir / "search_adaptor_results_by_dataset_fraction_aggregate.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate_rows, handle, indent=2, sort_keys=True)


def run_search_adaptor_suite(cfg: pipeline.Cfg) -> None:
    install_pipeline_hooks()
    cfg.dataset_format = "comlq"
    cfg.signal_projection_output_dim = int(cfg.embedding_dim)
    cfg.auto_train_teacher_if_missing = True
    pipeline.validate_unsup_cfg(cfg)
    suite_output = Path(cfg.output_dir)
    suite_output.mkdir(parents=True, exist_ok=True)
    payloads: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for raw_dataset_name in ACTIVE_OPTIONS.datasets:
        dataset_name = _normalize_dataset_name(raw_dataset_name)
        dataset_cfg = _dataset_cfg(cfg, dataset_name, suite_output)
        shared_cache = suite_output / dataset_name / "shared_cache"
        for fraction in ACTIVE_OPTIONS.train_fractions:
            fraction_dir = suite_output / dataset_name / _fraction_tag(fraction)
            per_seed: List[Dict[str, Any]] = []
            for seed in dataset_cfg.seeds:
                run_dir = fraction_dir / f"seed_{int(seed)}"
                seed_cfg = replace(
                    dataset_cfg,
                    seed=int(seed),
                    seeds=(int(seed),),
                    output_dir=str(run_dir),
                    log_path="",
                    teacher_checkpoint_path=str(run_dir / "teacher" / "search_adaptor_selected.pt"),
                    frozen_corpus_emb_path=str(shared_cache / "frozen_base_corpus.pt"),
                    frozen_query_emb_path=str(shared_cache / "frozen_base_queries.pt"),
                    signal_projection_output_dim=int(dataset_cfg.embedding_dim),
                )
                setattr(seed_cfg, "_search_adaptor_fraction", float(fraction))
                setattr(seed_cfg, "_fraction_selection_seed", int(ACTIVE_OPTIONS.fraction_seed))
                os.makedirs(seed_cfg.output_dir, exist_ok=True)
                run_metadata = {
                    "teacher": SEARCH_ADAPTOR,
                    "teacher_train_fraction": float(fraction),
                    "teacher_training_uses_qrels": True,
                    "student_training_uses_qrels": False,
                    "student_uses_full_training_query_split": True,
                    "candidate_source": "frozen_base_retriever",
                    "dynamic_refresh": False,
                    "search_adaptor_options": asdict(ACTIVE_OPTIONS),
                    "base_config": asdict(seed_cfg),
                }
                with open(
                    os.path.join(seed_cfg.output_dir, "search_adaptor_run_config.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(run_metadata, handle, indent=2, sort_keys=True)
                payload = pipeline.run_one_seed(seed_cfg)
                payload["search_adaptor"] = {
                    **run_metadata,
                    "teacher_metadata": getattr(seed_cfg, "_search_adaptor_teacher_metadata", {}),
                }
                with open(
                    os.path.join(seed_cfg.output_dir, "final_results.json"), "w", encoding="utf-8"
                ) as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                per_seed.append(payload)
            pipeline.aggregate_seed_results(str(fraction_dir), per_seed)
            payloads[dataset_name][_fraction_tag(fraction)] = per_seed
    _write_suite_results(suite_output, dict(payloads))
    print(f"Saved Search-Adaptor suite results under {suite_output}", flush=True)


def run_search_adaptor_tests() -> None:
    torch.manual_seed(7)
    scores = torch.tensor([[0.2, 0.1, -0.3]], requires_grad=True)
    labels = torch.tensor([[2.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, True]])
    ranking, pairs = search_adaptor_ranking_loss(scores, labels, mask)
    assert pairs == 3
    expected = (
        1.0 * F.softplus(scores[0, 1] - scores[0, 0])
        + 2.0 * F.softplus(scores[0, 2] - scores[0, 0])
        + 1.0 * F.softplus(scores[0, 2] - scores[0, 1])
    )
    assert torch.allclose(ranking, expected)
    ranking.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    padded_scores = torch.tensor([[0.2, 0.1, 9999.0]])
    padded_labels = torch.tensor([[1.0, 0.0, 9999.0]])
    padded_mask = torch.tensor([[True, True, False]])
    padded_loss, padded_pairs = search_adaptor_ranking_loss(
        padded_scores, padded_labels, padded_mask
    )
    reference_loss, reference_pairs = search_adaptor_ranking_loss(
        padded_scores[:, :2], padded_labels[:, :2], padded_mask[:, :2]
    )
    assert padded_pairs == reference_pairs == 1
    assert torch.allclose(padded_loss, reference_loss)

    base_q = torch.zeros((2, 3))
    adapted_q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    base_d = torch.zeros((2, 2, 3))
    adapted_d = torch.tensor(
        [[[1.0, 1.0, 0.0], [99.0, 99.0, 99.0]], [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]]
    )
    doc_mask = torch.tensor([[True, False], [True, True]])
    recovery = search_adaptor_recovery_loss(base_q, adapted_q, base_d, adapted_d, doc_mask)
    expected_recovery = torch.tensor((1.0 + 2.0) / 2.0 + (2.0 + 1.0 + 2.0) / 3.0)
    assert torch.allclose(recovery, expected_recovery)

    predicted = torch.zeros((1, 2, 2))
    adapted = torch.tensor([[1.0, 2.0]])
    relevance = torch.tensor([[2.0, 0.0]])
    prediction = search_adaptor_prediction_loss(
        adapted, predicted, relevance, torch.tensor([[True, True]])
    )
    assert torch.allclose(prediction, torch.tensor(3.0))

    ordered, selected_30 = deterministic_fraction_qids([f"q{i}" for i in range(10)], 0.3, 42)
    ordered_50, selected_50 = deterministic_fraction_qids([f"q{i}" for i in range(10)], 0.5, 42)
    ordered_100, selected_100 = deterministic_fraction_qids([f"q{i}" for i in range(10)], 1.0, 42)
    assert ordered == ordered_50 == ordered_100
    assert selected_30 == selected_50[: len(selected_30)]
    assert selected_50 == selected_100[: len(selected_50)]

    model = SearchAdaptorModel(4, 5, 2, "relu")
    assert model.adapter is model.adapter
    q = torch.randn(2, 4, requires_grad=True)
    docs = torch.randn(2, 3, 4, requires_grad=True)
    relevance = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mask = torch.ones((2, 3), dtype=torch.bool)
    result = search_adaptor_objective(model, q, docs, relevance, mask, 0.1, 0.01)
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert any(parameter.grad is not None for parameter in model.adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.query_predictor.parameters())

    print("Search-Adaptor focused tests passed", flush=True)


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--search_adaptor_datasets",
        nargs="+",
        default=list(SEARCH_ADAPTOR_DATASETS),
        help="Datasets to run: comlq privacy_qa contractnli.",
    )
    parser.add_argument(
        "--search_adaptor_train_fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.30, 0.50, 1.00],
    )
    parser.add_argument("--search_adaptor_fraction_seed", type=int, default=42)
    parser.add_argument("--search_adaptor_batch_size", type=int, default=128)
    parser.add_argument("--search_adaptor_max_iterations", type=int, default=2000)
    parser.add_argument("--search_adaptor_patience", type=int, default=125)
    parser.add_argument("--search_adaptor_eval_every", type=int, default=25)
    parser.add_argument("--search_adaptor_save_every", type=int, default=25)
    parser.add_argument("--search_adaptor_lr", type=float, default=1e-3)
    parser.add_argument("--search_adaptor_negative_pair_ratio", type=int, default=10)
    parser.add_argument(
        "--search_adaptor_alpha_grid", nargs="+", type=float, default=[0.0, 0.1, 1.0]
    )
    parser.add_argument(
        "--search_adaptor_beta_grid", nargs="+", type=float, default=[0.0, 0.01, 0.1]
    )
    parser.add_argument("--search_adaptor_fixed_alpha", type=float, default=0.1)
    parser.add_argument("--search_adaptor_fixed_beta", type=float, default=0.01)
    parser.add_argument("--no_search_adaptor_grid_search", action="store_true")
    parser.add_argument("--search_adaptor_hidden_dim", type=int, default=0)
    parser.add_argument("--search_adaptor_num_layers", type=int, default=2)
    parser.add_argument(
        "--search_adaptor_activation", choices=["relu", "gelu", "tanh"], default="relu"
    )
    parser.add_argument("--search_adaptor_inference_batch_size", type=int, default=8192)
    parser.add_argument("--fail_on_weak_search_adaptor", action="store_true")
    parser.add_argument("--run_search_adaptor_tests", action="store_true")
    return parser


@contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterable[None]:
    old = sys.argv
    sys.argv = [old[0], *argv]
    try:
        yield
    finally:
        sys.argv = old


def parse_args() -> Tuple[pipeline.Cfg, SearchAdaptorOptions]:
    custom_parser = _custom_parser()
    custom, remaining = custom_parser.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        print("\nSearch-Adaptor-specific arguments:\n")
        print(custom_parser.format_help())
        print(
            "Note: --teacher_* arguments shown below belong to the IMRNNS backend and are "
            "not used to configure Search-Adaptor. Use the --search_adaptor_* arguments above.\n"
        )
    with _temporary_argv(remaining):
        from .rcd_args import parse_args as parse_rcd_args

        cfg = parse_rcd_args()
    if "--output_dir" not in remaining:
        cfg.output_dir = "runs/rcd_search_adaptor"
    options = SearchAdaptorOptions(
        datasets=tuple(_normalize_dataset_name(name) for name in custom.search_adaptor_datasets),
        train_fractions=tuple(float(value) for value in custom.search_adaptor_train_fractions),
        fraction_seed=int(custom.search_adaptor_fraction_seed),
        batch_size=int(custom.search_adaptor_batch_size),
        max_iterations=int(custom.search_adaptor_max_iterations),
        patience=int(custom.search_adaptor_patience),
        eval_every=int(custom.search_adaptor_eval_every),
        save_every=int(custom.search_adaptor_save_every),
        learning_rate=float(custom.search_adaptor_lr),
        negative_pair_ratio=int(custom.search_adaptor_negative_pair_ratio),
        alpha_grid=tuple(float(value) for value in custom.search_adaptor_alpha_grid),
        beta_grid=tuple(float(value) for value in custom.search_adaptor_beta_grid),
        fixed_alpha=float(custom.search_adaptor_fixed_alpha),
        fixed_beta=float(custom.search_adaptor_fixed_beta),
        grid_search=not bool(custom.no_search_adaptor_grid_search),
        hidden_dim=int(custom.search_adaptor_hidden_dim),
        num_layers=int(custom.search_adaptor_num_layers),
        activation=str(custom.search_adaptor_activation),
        inference_batch_size=int(custom.search_adaptor_inference_batch_size),
        fail_on_weak_teacher=bool(custom.fail_on_weak_search_adaptor),
        run_tests=bool(custom.run_search_adaptor_tests),
    )
    if not options.datasets:
        raise ValueError("--search_adaptor_datasets cannot be empty")
    if not options.train_fractions:
        raise ValueError("--search_adaptor_train_fractions cannot be empty")
    if any(not 0.0 < value <= 1.0 for value in options.train_fractions):
        raise ValueError("Every Search-Adaptor train fraction must be in (0, 1]")
    if options.batch_size < 1 or options.max_iterations < 1 or options.patience < 1:
        raise ValueError("Search-Adaptor batch size, iterations, and patience must be positive")
    if options.negative_pair_ratio < 1:
        raise ValueError("Search-Adaptor negative-pair ratio must be positive")
    if options.learning_rate <= 0.0:
        raise ValueError("Search-Adaptor learning rate must be positive")
    if options.hidden_dim < 0 or options.num_layers < 1:
        raise ValueError(
            "Search-Adaptor hidden dimension must be 0 or positive; layers must be positive"
        )
    return cfg, options


def main() -> None:
    global ACTIVE_OPTIONS
    cfg, ACTIVE_OPTIONS = parse_args()
    if ACTIVE_OPTIONS.run_tests:
        run_search_adaptor_tests()
        return
    run_search_adaptor_suite(cfg)


if __name__ == "__main__":
    main()
