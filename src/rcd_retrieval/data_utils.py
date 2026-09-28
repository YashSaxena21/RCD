"""Format-agnostic helpers shared by training and evaluation pipelines."""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def base_query_type(query_type: str) -> str:
    return str(query_type).split("_", 1)[0]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def merge_qrels(
    *qrel_dicts: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    merged: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for qrels in qrel_dicts:
        for qid, relevances in qrels.items():
            for document_id, score in relevances.items():
                merged[qid][document_id] = max(
                    float(score), merged[qid].get(document_id, float("-inf"))
                )
    return {qid: dict(relevances) for qid, relevances in merged.items()}


def build_eval_examples(
    qids: Sequence[str],
    query_by_id: Mapping[str, Any],
    query_id_to_index: Mapping[str, int],
    qrels: Mapping[str, Mapping[str, float]],
    example_type: type[Any],
) -> list[Any]:
    return [
        example_type(
            qid=qid,
            q_idx=query_id_to_index[qid],
            query_text=query_by_id[qid].text,
            query_type=query_by_id[qid].query_type,
            base_type=query_by_id[qid].base_type,
        )
        for qid in qids
        if qid in qrels
    ]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denominator, 1e-12)


class SearchIndex:
    """Exact inner-product search with a NumPy fallback when FAISS is unavailable."""

    def __init__(self, embeddings: Any):
        start = time.perf_counter()
        self.embeddings = normalize_rows(embeddings.detach().cpu().numpy().astype(np.float32))
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        self.build_time_sec = time.perf_counter() - start

    @property
    def ntotal(self) -> int:
        return int(self.embeddings.shape[0])

    def search(self, query_embeddings: np.ndarray, k: int) -> np.ndarray:
        if len(query_embeddings) == 0:
            return np.empty((0, 0), dtype=np.int64)
        queries = normalize_rows(query_embeddings.astype(np.float32, copy=False))
        k = min(int(k), self.ntotal)
        if self.index is not None:
            _, indices = self.index.search(queries, k)
            return indices.astype(np.int64)

        scores = queries @ self.embeddings.T
        if k >= scores.shape[1]:
            indices = np.argsort(-scores, axis=1)
        else:
            candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            candidate_scores = np.take_along_axis(scores, candidates, axis=1)
            order = np.argsort(-candidate_scores, axis=1)
            indices = np.take_along_axis(candidates, order, axis=1)
        return indices.astype(np.int64)

    def size_mb(self) -> float:
        if self.index is None or faiss is None:
            return self.embeddings.nbytes / (1024.0 * 1024.0)
        with tempfile.NamedTemporaryFile(suffix=".index") as handle:
            faiss.write_index(self.index, handle.name)
            return Path(handle.name).stat().st_size / (1024.0 * 1024.0)


def find_official_qrel_paths(
    qrels_dir: str | Path,
) -> dict[str, Path] | None:
    root = Path(qrels_dir)
    train = root / "train.tsv"
    validation = root / "val.tsv"
    development = root / "dev.tsv"
    test = root / "test.tsv"
    validation_like = validation if validation.exists() else development
    if train.exists() and validation_like.exists() and test.exists():
        return {"train": train, "val": validation_like, "test": test}
    return None
