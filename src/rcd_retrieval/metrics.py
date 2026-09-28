"""Retrieval metrics shared by supervised and distillation experiments."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .legalbench import span_overlap


def _dcg(gains: Sequence[float]) -> float:
    return float(sum(gain / math.log2(rank + 2.0) for rank, gain in enumerate(gains)))


def retrieval_metrics(
    indices: np.ndarray,
    examples: Sequence[Any],
    qrels_by_qid_index: Mapping[str, Mapping[int, float]],
    ks: Sequence[int],
) -> dict[int, dict[str, float]]:
    cutoffs = sorted(set(int(k) for k in ks))
    totals = {
        k: {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "map": 0.0, "n": 0}
        for k in cutoffs
    }
    for row, example in enumerate(examples):
        relevance = qrels_by_qid_index[str(example.qid)]
        relevant_indices = set(relevance)
        ranked = [int(value) for value in indices[row].tolist()]
        for k in cutoffs:
            retrieved = ranked[:k]
            hits = [index for index in retrieved if index in relevant_indices]
            totals[k]["recall"] += len(hits) / max(1, len(relevant_indices))
            totals[k]["precision"] += len(hits) / float(k)
            totals[k]["mrr"] += next(
                (
                    1.0 / rank
                    for rank, index in enumerate(retrieved, start=1)
                    if index in relevant_indices
                ),
                0.0,
            )
            gains = [float(relevance.get(index, 0.0)) for index in retrieved]
            ideal = _dcg(sorted((float(value) for value in relevance.values()), reverse=True)[:k])
            totals[k]["ndcg"] += _dcg(gains) / ideal if ideal > 0.0 else 0.0
            hit_count = 0
            average_precision = 0.0
            for rank, index in enumerate(retrieved, start=1):
                if index in relevant_indices:
                    hit_count += 1
                    average_precision += hit_count / float(rank)
            totals[k]["map"] += average_precision / max(1, min(len(relevant_indices), k))
            totals[k]["n"] += 1

    return {
        k: {
            metric: float(value) / max(1, int(totals[k]["n"]))
            for metric, value in totals[k].items()
            if metric != "n"
        }
        | {"n": float(totals[k]["n"])}
        for k in cutoffs
    }


@lru_cache(maxsize=16)
def _load_metadata(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"LegalBench-RAG metadata must be a JSON object: {path}")
    return payload


def legalbench_character_metrics(
    indices: np.ndarray,
    examples: Sequence[Any],
    metadata_path: str | Path | None,
    ks: Sequence[int],
) -> dict[int, dict[str, float]]:
    if metadata_path is None or not Path(metadata_path).is_file():
        return {}
    metadata = _load_metadata(str(Path(metadata_path).resolve()))
    passage_ids = metadata.get("passage_id_order", [])
    passage_metadata = metadata.get("passages", {})
    query_metadata = metadata.get("queries", {})
    cutoffs = sorted(set(int(k) for k in ks))
    totals = {
        k: {"legalbench_char_recall": 0.0, "legalbench_char_precision": 0.0, "n": 0}
        for k in cutoffs
    }
    for row, example in enumerate(examples):
        snippets = query_metadata.get(str(example.qid), {}).get("snippets", [])
        gold_length = sum(max(0, int(item["span"][1]) - int(item["span"][0])) for item in snippets)
        ranked = [int(value) for value in indices[row].tolist()]
        for k in cutoffs:
            retrieved_length = 0
            overlap_length = 0
            for document_index in ranked[:k]:
                if document_index < 0 or document_index >= len(passage_ids):
                    continue
                passage = passage_metadata.get(passage_ids[document_index])
                if not passage:
                    continue
                retrieved_length += max(0, int(passage["span"][1]) - int(passage["span"][0]))
                overlap_length += sum(
                    span_overlap(passage["span"], snippet["span"])
                    for snippet in snippets
                    if passage["file_path"] == snippet["file_path"]
                )
            totals[k]["legalbench_char_recall"] += overlap_length / max(1, gold_length)
            totals[k]["legalbench_char_precision"] += overlap_length / max(1, retrieved_length)
            totals[k]["n"] += 1

    return {
        k: {
            "legalbench_char_recall": row["legalbench_char_recall"] / max(1, int(row["n"])),
            "legalbench_char_precision": row["legalbench_char_precision"] / max(1, int(row["n"])),
        }
        for k, row in totals.items()
    }
