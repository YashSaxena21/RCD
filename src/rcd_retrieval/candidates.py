"""Qrel-free fixed candidate construction for student post-training."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

from . import provenance
from .config import MODE_UNSUPERVISED, Cfg
from .data_utils import SearchIndex
from .records import CandidateBuildResult, PassageRecord, QueryRecord, TrainExample
from .runtime import logger


def retrieve_training_candidate_lists(
    cfg: Cfg,
    train_qids: Sequence[str],
    query_id_to_index: dict[str, int],
    index: SearchIndex,
    query_embeddings: torch.Tensor,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Retrieve deterministic fixed candidates without reading qrels."""
    candidate_cap = min(cfg.candidate_pool_k, cfg.max_train_candidates_per_query)
    logger.info(
        "Building candidates | mode=%s | source=frozen_base_retriever | "
        "candidate_pool_k=%s | train_candidate_cap=%s | qrel_injection=false",
        cfg.mode,
        cfg.candidate_pool_k,
        candidate_cap,
    )
    qids = list(train_qids)
    raw_candidates: dict[str, list[int]] = {}
    for start in range(0, len(qids), 256):
        chunk = qids[start : start + 256]
        query_indices = [query_id_to_index[qid] for qid in chunk]
        queries = query_embeddings[query_indices].numpy().astype(np.float32)
        indices = index.search(queries, cfg.candidate_pool_k)
        for row, qid in enumerate(chunk):
            raw_candidates[qid] = [
                int(document_index)
                for document_index in indices[row].tolist()
                if int(document_index) >= 0
            ]

    candidate_lists = {}
    lengths: Counter[int] = Counter()
    for qid in qids:
        deduplicated = list(dict.fromkeys(raw_candidates.get(qid, ())))[:candidate_cap]
        if not deduplicated:
            continue
        candidate_lists[qid] = deduplicated
        lengths[len(deduplicated)] += 1

    metadata: dict[str, object] = {
        "candidate_source": "frozen_base_retriever",
        "candidate_pool_k": cfg.candidate_pool_k,
        "train_candidate_cap": candidate_cap,
        "candidate_length_distribution": dict(sorted(lengths.items())),
        "train_query_count": len(qids),
        "example_count": len(candidate_lists),
    }
    logger.info("Candidate length distribution: %s", metadata["candidate_length_distribution"])
    return candidate_lists, metadata


def build_unsupervised_train_examples(
    cfg: Cfg,
    train_qids: Sequence[str],
    query_by_id: dict[str, QueryRecord],
    query_id_to_index: dict[str, int],
    index: SearchIndex,
    query_embeddings: torch.Tensor,
    passages: list[PassageRecord] | None = None,
) -> CandidateBuildResult:
    candidate_lists, metadata = retrieve_training_candidate_lists(
        cfg,
        train_qids,
        query_id_to_index,
        index,
        query_embeddings,
    )
    examples = [
        TrainExample(
            qid=qid,
            q_idx=query_id_to_index[qid],
            query_text=query_by_id[qid].text,
            query_type=query_by_id[qid].query_type,
            base_type=query_by_id[qid].base_type,
            candidate_doc_idxs=document_indices,
            labels=[0.0] * len(document_indices),
        )
        for qid, document_indices in candidate_lists.items()
    ]
    complete_metadata = provenance.augment_candidate_metadata(
        {
            **metadata,
            "mode": MODE_UNSUPERVISED,
            "qrels_used_in_training": False,
            "qrel_positive_injection": False,
            "qrel_loss_active": False,
        },
        examples,
        passages,
    )
    logger.info(
        "Student candidate contract: qrels_used_in_training=false | "
        "qrel_positive_injection=false | qrel_loss_active=false"
    )
    return CandidateBuildResult(examples=examples, metadata=complete_metadata)
