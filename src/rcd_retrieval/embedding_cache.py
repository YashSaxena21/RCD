"""Persistent embeddings from the frozen base retriever."""

from __future__ import annotations

import os
from typing import Any

import torch

from .config import Cfg
from .records import PassageRecord, QueryRecord
from .runtime import logger, safe_torch_load


@torch.no_grad()
def encode_texts(
    encoder: Any,
    texts: list[str],
    prefix: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    formatted = [f"{prefix}{text}" if prefix else text for text in texts]
    embeddings = encoder.encode(
        formatted,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device,
        normalize_embeddings=True,
    )
    return embeddings.detach().float().cpu().contiguous()


def _ids_match(saved: dict[str, Any], key: str, expected: list[str]) -> bool:
    return key in saved and list(map(str, saved[key])) == list(map(str, expected))


def _metadata_mismatches(saved: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    metadata = saved.get("metadata", {}) if isinstance(saved, dict) else {}
    return [
        f"{key}: cache={metadata.get(key)!r} expected={expected_value!r}"
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    ]


def _load_valid_cache(
    path: str,
    id_key: str,
    expected_ids: list[str],
    expected_metadata: dict[str, Any],
) -> torch.Tensor | None:
    saved = safe_torch_load(path, map_location="cpu")
    ids_match = isinstance(saved, dict) and _ids_match(saved, id_key, expected_ids)
    mismatches = _metadata_mismatches(saved, expected_metadata)
    if ids_match and not mismatches:
        embeddings = saved["embeddings"].float().contiguous()
        logger.info(
            "%s cache valid | shape=%s | id_order=ok | metadata=ok",
            id_key.removesuffix("_ids").title(),
            tuple(embeddings.shape),
        )
        return embeddings
    logger.warning("Embedding cache invalid. id_order=%s; metadata_mismatches=%s", ids_match, mismatches)
    return None


def build_or_load_frozen_corpus_embeddings(
    cfg: Cfg,
    passages: list[PassageRecord],
    encoder: Any,
) -> torch.Tensor:
    passage_ids = [passage.pid for passage in passages]
    metadata = {
        "model_name": cfg.model_name,
        "prefix": cfg.passage_prefix,
        "embedding_dim": cfg.embedding_dim,
        "dataset_dir": cfg.dataset_dir,
        "count": len(passages),
    }
    path = cfg.frozen_corpus_emb_path
    if os.path.exists(path) and not cfg.force_rebuild_embedding_cache:
        logger.info("Loading frozen base-retriever corpus cache: %s", path)
        cached = _load_valid_cache(path, "passage_ids", passage_ids, metadata)
        if cached is not None:
            return cached

    logger.info(
        "Rebuilding frozen base-retriever corpus cache with %s: passages=%s",
        cfg.model_name,
        len(passages),
    )
    chunks = []
    for start in range(0, len(passages), cfg.corpus_encode_batch_size):
        chunk = passages[start : start + cfg.corpus_encode_batch_size]
        chunks.append(
            encode_texts(
                encoder,
                [passage.text for passage in chunk],
                cfg.passage_prefix,
                cfg.corpus_encode_batch_size,
                cfg.device,
            )
        )
        logger.info("  encoded corpus %s/%s", min(start + len(chunk), len(passages)), len(passages))
    embeddings = torch.cat(chunks, dim=0).contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"embeddings": embeddings, "passage_ids": passage_ids, "metadata": metadata},
        path,
    )
    logger.info("Saved frozen base-retriever corpus cache: %s | shape=%s", path, tuple(embeddings.shape))
    return embeddings


def build_or_load_frozen_query_embeddings(
    cfg: Cfg,
    queries: list[QueryRecord],
    encoder: Any,
) -> torch.Tensor:
    query_ids = [query.qid for query in queries]
    metadata = {
        "model_name": cfg.model_name,
        "prefix": cfg.query_prefix,
        "embedding_dim": cfg.embedding_dim,
        "dataset_dir": cfg.dataset_dir,
        "count": len(queries),
    }
    path = cfg.frozen_query_emb_path
    if os.path.exists(path) and not cfg.force_rebuild_embedding_cache:
        logger.info("Loading frozen base-retriever query cache: %s", path)
        cached = _load_valid_cache(path, "query_ids", query_ids, metadata)
        if cached is not None:
            return cached

    logger.info(
        "Rebuilding frozen base-retriever query cache with %s: queries=%s",
        cfg.model_name,
        len(queries),
    )
    chunks = []
    for start in range(0, len(queries), cfg.eval_batch_size):
        chunk = queries[start : start + cfg.eval_batch_size]
        chunks.append(
            encode_texts(
                encoder,
                [query.text for query in chunk],
                cfg.query_prefix,
                cfg.eval_batch_size,
                cfg.device,
            )
        )
    embeddings = torch.cat(chunks, dim=0).contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"embeddings": embeddings, "query_ids": query_ids, "metadata": metadata}, path)
    logger.info("Saved frozen base-retriever query cache: %s | shape=%s", path, tuple(embeddings.shape))
    return embeddings
