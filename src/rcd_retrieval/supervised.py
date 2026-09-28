from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

try:
    import psutil
except Exception:
    psutil = None  # type: ignore

from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from .data_utils import (
    SearchIndex,
    base_query_type,
    build_eval_examples,
    find_official_qrel_paths,
    load_jsonl,
    merge_qrels,
)
from .legalbench import (
    has_archive_pair as legalbench_rag_has_zip_pair,
)
from .legalbench import (
    has_data_directories as legalbench_rag_has_dirs,
)
from .legalbench import (
    normalize_dataset_name as normalize_legalbench_dataset_name,
)
from .legalbench import (
    prepare_retrieval_dataset as prepare_legalbench_retrieval_dataset,
)
from .legalbench import (
    resolve_data_root as resolve_legalbench_rag_data_root,
)
from .metrics import (
    legalbench_character_metrics as legalbench_char_metrics_from_inds,
)
from .metrics import (
    retrieval_metrics as eval_metrics_multi_k_from_inds,
)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
NEGATION_BASE_TYPES = {"2in", "3in", "inp", "pin", "pni"}
CONJUNCTION_BASE_TYPES = {"2i", "3i", "pi", "ip"}
UNION_BASE_TYPES = {"2u", "up"}
PROJECTION_BASE_TYPES = {"1p", "2p", "3p", "pi", "ip", "up", "inp", "pin", "pni"}


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class QueryRecord:
    qid: str
    text: str
    query_type: str = ""
    base_type: str = ""


@dataclass
class SplitSpec:
    dataset_name: str
    dataset_dir: str
    qrels_source: str
    split_source: str
    seed: int
    val_ratio: float
    test_ratio: float
    query_type_filter: str
    query_types: str
    train_qids: List[str]
    val_qids: List[str]
    test_qids: List[str]


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_dir: Path
    qrels_dir: Path
    corpus: List[CorpusDoc]
    queries: List[QueryRecord]
    qrels: Dict[str, Dict[str, float]]
    split: SplitSpec
    legalbench_metadata_path: Optional[Path] = None

    @property
    def train_qids(self) -> List[str]:
        return self.split.train_qids

    @property
    def val_qids(self) -> List[str]:
        return self.split.val_qids

    @property
    def test_qids(self) -> List[str]:
        return self.split.test_qids


@dataclass(frozen=True)
class EvalExample:
    qid: str
    q_idx: int
    query_text: str
    query_type: str
    base_type: str


@dataclass
class ComputeTracker:
    precision_mode: str = "fp32"
    process: Any = field(
        default_factory=lambda: psutil.Process(os.getpid()) if psutil is not None else None
    )
    start_time: float = field(default_factory=time.perf_counter)
    segments: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    peak_cpu_rss_bytes: int = 0

    def __post_init__(self) -> None:
        self.update_cpu_peak()
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(device_idx):
                    torch.cuda.reset_peak_memory_stats()

    def update_cpu_peak(self) -> None:
        try:
            if self.process is None:
                return
            rss = int(self.process.memory_info().rss)
            self.peak_cpu_rss_bytes = max(self.peak_cpu_rss_bytes, rss)
        except Exception:
            pass

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.segments[name] = self.segments.get(name, 0.0) + (time.perf_counter() - start)
            self.update_cpu_peak()

    def add_time(self, name: str, seconds: float) -> None:
        self.segments[name] = self.segments.get(name, 0.0) + float(seconds)
        self.update_cpu_peak()

    def increment(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(amount)

    def set(self, name: str, value: Any) -> None:
        self.metadata[name] = value

    def count_parameters(self, model: Any) -> None:
        total = 0
        trainable = 0
        for param in model.parameters():
            n = int(param.numel())
            total += n
            if param.requires_grad:
                trainable += n
        self.metadata["trainable_parameters"] = trainable
        self.metadata["total_parameters"] = total

    def finalize(
        self,
        *,
        final_model_path: str | Path | None = None,
        corpus_embedding_cache_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        self.update_cpu_peak()
        peak_alloc = 0
        peak_reserved = 0
        gpu_name = None
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device_idx)))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device_idx)))
                if gpu_name is None:
                    gpu_name = torch.cuda.get_device_name(device_idx)
        return {
            "total_wall_time_sec": time.perf_counter() - self.start_time,
            "generation_time_sec": self.segments.get("generation_time_sec", 0.0),
            "pseudo_labeling_time_sec": self.segments.get("pseudo_labeling_time_sec", 0.0),
            "filtering_time_sec": self.segments.get("filtering_time_sec", 0.0),
            "training_time_sec": self.segments.get("training_time_sec", 0.0),
            "corpus_encoding_time_sec": self.segments.get("corpus_encoding_time_sec", 0.0),
            "query_encoding_time_sec": self.segments.get("query_encoding_time_sec", 0.0),
            "faiss_indexing_time_sec": self.segments.get("faiss_indexing_time_sec", 0.0),
            "faiss_search_time_sec": self.segments.get("faiss_search_time_sec", 0.0),
            "validation_eval_time_sec": self.segments.get("validation_eval_time_sec", 0.0),
            "test_eval_time_sec": self.segments.get("test_eval_time_sec", 0.0),
            "peak_gpu_memory_allocated_mb": peak_alloc / (1024.0 * 1024.0),
            "peak_gpu_memory_reserved_mb": peak_reserved / (1024.0 * 1024.0),
            "peak_cpu_memory_mb": self.peak_cpu_rss_bytes / (1024.0 * 1024.0),
            "gpu_name": gpu_name,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda,
            "precision_mode": self.precision_mode,
            "generated_queries_per_sec": 0.0,
            "pseudo_labels_per_sec": 0.0,
            "training_steps_per_sec": safe_div(
                self.counters.get("training_steps", 0.0),
                self.segments.get("training_time_sec", 0.0),
            ),
            "training_examples_per_sec": safe_div(
                self.counters.get("training_examples", 0.0),
                self.segments.get("training_time_sec", 0.0),
            ),
            "corpus_documents_encoded_per_sec": safe_div(
                self.counters.get("corpus_docs_encoded", 0.0),
                self.segments.get("corpus_encoding_time_sec", 0.0),
            ),
            "queries_encoded_per_sec": safe_div(
                self.counters.get("queries_encoded", 0.0),
                self.segments.get("query_encoding_time_sec", 0.0),
            ),
            "retrieval_queries_per_sec": self.metadata.get("retrieval_queries_per_sec", 0.0),
            "average_query_latency_ms": self.metadata.get("average_query_latency_ms", 0.0),
            "p50_query_latency_ms": self.metadata.get("p50_query_latency_ms", 0.0),
            "p95_query_latency_ms": self.metadata.get("p95_query_latency_ms", 0.0),
            "final_model_size_mb": path_size_mb(final_model_path),
            "generated_data_size_mb": 0.0,
            "pseudo_label_data_size_mb": 0.0,
            "corpus_embedding_cache_size_mb": path_size_mb(corpus_embedding_cache_path),
            "faiss_index_size_mb": self.metadata.get("faiss_index_size_mb", 0.0),
            "trainable_parameters": self.metadata.get("trainable_parameters", 0),
            "total_parameters": self.metadata.get("total_parameters", 0),
        }


def safe_div(x: float, y: float) -> float:
    return float(x / y) if y and y > 0 else 0.0


def path_size_mb(path: str | Path | None) -> float:
    if path is None:
        return 0.0
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024.0 * 1024.0)
    total = 0
    for child in p.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024.0 * 1024.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_dataset_format(cfg: "SupervisedE5Config") -> str:
    if cfg.dataset_format != "auto":
        return cfg.dataset_format
    root = resolve_legalbench_rag_data_root(cfg.legalbench_rag_root or cfg.dataset_dir)
    if legalbench_rag_has_dirs(root) or legalbench_rag_has_zip_pair(root):
        return "legalbench_rag"
    return "standard"


def load_corpus(path: str | Path) -> List[CorpusDoc]:
    docs: List[CorpusDoc] = []
    seen: set[str] = set()
    for obj in load_jsonl(path):
        doc_id = str(obj.get("_id", obj.get("doc_id", obj.get("id", "")))).strip()
        if not doc_id:
            raise ValueError(f"Corpus row missing id in {path}: {obj}")
        if doc_id in seen:
            raise ValueError(f"Duplicate corpus doc id: {doc_id}")
        seen.add(doc_id)
        title = str(obj.get("title", "")).strip()
        text = str(obj.get("text", obj.get("contents", obj.get("abstract", "")))).strip()
        if not text:
            if not title:
                raise ValueError(f"Corpus row {doc_id} has neither text nor title")
            text = title
        docs.append(CorpusDoc(doc_id=doc_id, text=text, title=title))
    return docs


def load_queries(path: str | Path) -> List[QueryRecord]:
    queries: List[QueryRecord] = []
    seen: set[str] = set()
    for obj in load_jsonl(path):
        if "_id" not in obj or "text" not in obj:
            raise ValueError(f"Bad query row in {path}: {obj}")
        qid = str(obj["_id"]).strip()
        if qid in seen:
            raise ValueError(f"Duplicate query id: {qid}")
        seen.add(qid)
        query_type = str(obj.get("type", "")).strip()
        queries.append(
            QueryRecord(
                qid=qid,
                text=str(obj["text"]),
                query_type=query_type,
                base_type=base_query_type(query_type) if query_type else "",
            )
        )
    return queries


def _is_number_like(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _looks_like_qrels_header(parts: Sequence[str]) -> bool:
    normalized = {p.strip().lower() for p in parts}
    header_tokens = {
        "query-id",
        "query_id",
        "qid",
        "corpus-id",
        "corpus_id",
        "doc_id",
        "document_id",
        "score",
        "relevance",
        "label",
        "q0",
    }
    return bool(normalized & header_tokens) and not all(_is_number_like(p) for p in normalized)


def load_qrels(path: str | Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if line_no == 1 and _looks_like_qrels_header(parts):
                continue
            if len(parts) == 3:
                qid, doc_id, score = parts[0], parts[1], parts[2]
            elif len(parts) >= 4:
                qid, doc_id, score = parts[0], parts[-2], parts[-1]
            else:
                raise ValueError(f"Unsupported qrels row in {path}:{line_no}: {parts}")
            score_f = float(score)
            if score_f <= 0:
                continue
            qrels[str(qid).strip()][str(doc_id).strip()] = max(
                score_f, qrels[str(qid).strip()].get(str(doc_id).strip(), float("-inf"))
            )
    return {qid: dict(rels) for qid, rels in qrels.items()}


def validate_qrels(
    queries: Sequence[QueryRecord], corpus: Sequence[CorpusDoc], qrels: Dict[str, Dict[str, float]]
) -> None:
    qids = {q.qid for q in queries}
    doc_ids = {d.doc_id for d in corpus}
    missing_qids = sorted(qid for qid in qrels if qid not in qids)
    missing_doc_ids = sorted(
        {doc_id for rels in qrels.values() for doc_id in rels if doc_id not in doc_ids}
    )
    if missing_qids:
        raise ValueError(f"Qrels contain query ids not in queries.jsonl: {missing_qids[:10]}")
    if missing_doc_ids:
        raise ValueError(f"Qrels contain doc ids not in corpus.jsonl: {missing_doc_ids[:10]}")


def filter_query_ids(
    queries: Sequence[QueryRecord],
    qrels: Dict[str, Dict[str, float]],
    query_type_filter: str,
    query_types: str,
) -> List[str]:
    custom = {item.strip() for item in query_types.split(",") if item.strip()}

    def keep(query: QueryRecord) -> bool:
        if query.qid not in qrels:
            return False
        if query_type_filter == "all":
            return True
        if query_type_filter == "negation":
            return query.base_type in NEGATION_BASE_TYPES
        if query_type_filter == "conjunction":
            return query.base_type in CONJUNCTION_BASE_TYPES
        if query_type_filter == "union":
            return query.base_type in UNION_BASE_TYPES
        if query_type_filter == "projection":
            return query.base_type in PROJECTION_BASE_TYPES
        if query_type_filter == "custom":
            if not custom:
                raise ValueError("query_type_filter=custom requires query_types")
            return query.query_type in custom or query.base_type in custom
        raise ValueError(f"Unknown query_type_filter: {query_type_filter}")

    return [query.qid for query in queries if keep(query)]


def deterministic_fallback_split(
    qids: Sequence[str], seed: int, val_ratio: float, test_ratio: float
) -> Dict[str, List[str]]:
    qids_list = list(qids)
    rng = random.Random(seed)
    rng.shuffle(qids_list)
    n = len(qids_list)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    if n > 0 and test_n == 0:
        test_n = 1
    if n > 1 and val_n == 0:
        val_n = 1
    if test_n + val_n > n:
        overflow = test_n + val_n - n
        val_n = max(0, val_n - overflow)
    return {
        "train": qids_list[test_n + val_n :],
        "val": qids_list[test_n : test_n + val_n],
        "test": qids_list[:test_n],
    }


def save_split(path: Path, split: SplitSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(split), handle, indent=2)


def load_split(path: Path) -> SplitSpec:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitSpec(**payload)


def resolve_qrels_dir(dataset_dir: Path) -> Path:
    qrels_dir = dataset_dir / "qrels"
    if qrels_dir.is_dir():
        return qrels_dir
    if (dataset_dir / "qrels.tsv").exists():
        tmp_dir = dataset_dir / "_single_qrels"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / "test.tsv"
        if not target.exists():
            target.write_text(
                (dataset_dir / "qrels.tsv").read_text(encoding="utf-8"), encoding="utf-8"
            )
        return tmp_dir
    raise FileNotFoundError(f"Could not find qrels dir or qrels.tsv under {dataset_dir}")


def load_standard_dataset(
    dataset_dir: str | Path,
    dataset_name: str,
    split_path: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    query_type_filter: str,
    query_types: str,
    force_rebuild_split: bool,
) -> DatasetBundle:
    dataset_dir = Path(dataset_dir)
    corpus = load_corpus(dataset_dir / "corpus.jsonl")
    queries = load_queries(dataset_dir / "queries.jsonl")
    qrels_dir = resolve_qrels_dir(dataset_dir)

    if split_path.exists() and not force_rebuild_split:
        split = load_split(split_path)
        if (
            split.dataset_name != dataset_name
            or split.dataset_dir != str(dataset_dir)
            or split.seed != seed
        ):
            raise ValueError(
                f"Existing split file {split_path} does not match requested dataset/seed"
            )
        if split.query_type_filter != query_type_filter or split.query_types != query_types:
            raise ValueError(
                f"Existing split file {split_path} does not match requested query filter settings"
            )
        if split.split_source == "official":
            official = find_official_qrel_paths(qrels_dir)
            if official is None:
                raise FileNotFoundError(
                    f"Expected official qrels for split reuse under {qrels_dir}"
                )
            qrels = merge_qrels(*[load_qrels(path) for path in official.values()])
        else:
            qrels = load_qrels(qrels_dir / "test.tsv")
        validate_qrels(queries, corpus, qrels)
        return DatasetBundle(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            qrels_dir=qrels_dir,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            split=split,
        )

    official = find_official_qrel_paths(qrels_dir)
    if official:
        split_qrels = {name: load_qrels(path) for name, path in official.items()}
        split_sets = {name: set(split_qrels[name].keys()) for name in ("train", "val", "test")}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"Official qrel split overlap between {left} and {right}: {sorted(overlap)[:10]}"
                )
        qrels = merge_qrels(*split_qrels.values())
        split_ids = {
            name: filter_query_ids(queries, split_qrels[name], query_type_filter, query_types)
            for name in ("train", "val", "test")
        }
        split = SplitSpec(
            dataset_name=dataset_name,
            dataset_dir=str(dataset_dir),
            qrels_source=str(qrels_dir),
            split_source="official",
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            query_type_filter=query_type_filter,
            query_types=query_types,
            train_qids=split_ids["train"],
            val_qids=split_ids["val"],
            test_qids=split_ids["test"],
        )
    else:
        test_path = qrels_dir / "test.tsv"
        if not test_path.exists():
            raise FileNotFoundError(f"No official qrels split and no qrels/test.tsv in {qrels_dir}")
        qrels = load_qrels(test_path)
        qids = filter_query_ids(queries, qrels, query_type_filter, query_types)
        split_ids = deterministic_fallback_split(qids, seed, val_ratio, test_ratio)
        split = SplitSpec(
            dataset_name=dataset_name,
            dataset_dir=str(dataset_dir),
            qrels_source=str(test_path),
            split_source="random_from_single_qrels_file",
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            query_type_filter=query_type_filter,
            query_types=query_types,
            train_qids=split_ids["train"],
            val_qids=split_ids["val"],
            test_qids=split_ids["test"],
        )
    save_split(split_path, split)
    validate_qrels(queries, corpus, qrels)
    return DatasetBundle(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        qrels_dir=qrels_dir,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        split=split,
    )


def load_dataset_for_cfg(cfg: "SupervisedE5Config", split_path: Path) -> DatasetBundle:
    dataset_format = infer_dataset_format(cfg)
    if dataset_format == "legalbench_rag":
        dataset_name = normalize_legalbench_dataset_name(cfg.dataset_name)
        prepared_root = cfg.legalbench_prepared_dir or os.path.join(
            cfg.output_dir, "legalbench_rag_prepared"
        )
        prepared_dir = os.path.join(prepared_root, dataset_name)
        corpus_path = os.path.join(prepared_dir, "corpus.jsonl")
        queries_path = os.path.join(prepared_dir, "queries.jsonl")
        qrels_test_path = os.path.join(prepared_dir, "qrels", "test.tsv")
        metadata_path = Path(prepared_dir) / "legalbench_metadata.json"
        if cfg.force or not (
            os.path.exists(corpus_path)
            and os.path.exists(queries_path)
            and os.path.exists(qrels_test_path)
            and metadata_path.exists()
        ):
            prepare_legalbench_retrieval_dataset(cfg, dataset_name, prepared_dir)
        dataset = load_standard_dataset(
            dataset_dir=prepared_dir,
            dataset_name=dataset_name,
            split_path=split_path,
            seed=cfg.seed,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            query_type_filter=cfg.query_type_filter,
            query_types=cfg.query_types,
            force_rebuild_split=cfg.force,
        )
        dataset.legalbench_metadata_path = metadata_path
        return dataset

    return load_standard_dataset(
        dataset_dir=cfg.dataset_dir,
        dataset_name=cfg.dataset_name,
        split_path=split_path,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        query_type_filter=cfg.query_type_filter,
        query_types=cfg.query_types,
        force_rebuild_split=cfg.force,
    )


def subset_for_smoke_test(
    dataset: DatasetBundle, max_docs: int = 512, max_queries_per_split: int = 32
) -> DatasetBundle:
    query_map = {q.qid: q for q in dataset.queries}
    keep_qids = (
        dataset.train_qids[:max_queries_per_split]
        + dataset.val_qids[:max_queries_per_split]
        + dataset.test_qids[:max_queries_per_split]
    )
    keep_qids = list(dict.fromkeys(qid for qid in keep_qids if qid in dataset.qrels))
    referenced_doc_ids: List[str] = []
    for qid in keep_qids:
        for doc_id in dataset.qrels.get(qid, {}):
            if doc_id not in referenced_doc_ids:
                referenced_doc_ids.append(doc_id)
    corpus_docs: List[CorpusDoc] = []
    seen: set[str] = set()
    referenced_set = set(referenced_doc_ids)
    for doc in dataset.corpus:
        if doc.doc_id in referenced_set or len(corpus_docs) < max_docs:
            if doc.doc_id not in seen:
                corpus_docs.append(doc)
                seen.add(doc.doc_id)
        if len(corpus_docs) >= max_docs and referenced_set.issubset(seen):
            break
    filtered_qrels = {
        qid: {doc_id: score for doc_id, score in rels.items() if doc_id in seen}
        for qid, rels in dataset.qrels.items()
        if qid in keep_qids
    }
    filtered_qrels = {qid: rels for qid, rels in filtered_qrels.items() if rels}
    keep_qids = [qid for qid in keep_qids if qid in filtered_qrels]
    split = SplitSpec(
        dataset_name=dataset.split.dataset_name,
        dataset_dir=dataset.split.dataset_dir,
        qrels_source=dataset.split.qrels_source,
        split_source=f"{dataset.split.split_source}_smoke",
        seed=dataset.split.seed,
        val_ratio=dataset.split.val_ratio,
        test_ratio=dataset.split.test_ratio,
        query_type_filter=dataset.split.query_type_filter,
        query_types=dataset.split.query_types,
        train_qids=[
            qid for qid in dataset.train_qids[:max_queries_per_split] if qid in filtered_qrels
        ],
        val_qids=[qid for qid in dataset.val_qids[:max_queries_per_split] if qid in filtered_qrels],
        test_qids=[
            qid for qid in dataset.test_qids[:max_queries_per_split] if qid in filtered_qrels
        ],
    )
    return DatasetBundle(
        dataset_name=dataset.dataset_name,
        dataset_dir=dataset.dataset_dir,
        qrels_dir=dataset.qrels_dir,
        corpus=corpus_docs,
        queries=[query_map[qid] for qid in keep_qids],
        qrels=filtered_qrels,
        split=split,
        legalbench_metadata_path=dataset.legalbench_metadata_path,
    )


def load_e5_model(model_name: str, device: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def encode_texts(
    model: SentenceTransformer, texts: Sequence[str], prefix: str, batch_size: int, device: str
) -> torch.Tensor:
    formatted = [f"{prefix}{text}" if prefix else str(text) for text in texts]
    emb = model.encode(
        formatted,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )
    return emb.detach().float().cpu().contiguous()


def _cache_metadata(model_name: str, prefix: str, count: int, embedding_dim: int) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "prefix": prefix,
        "count": count,
        "embedding_dim": embedding_dim,
    }


def _cache_ok(
    saved: Any, ids_key: str, expected_ids: Sequence[str], expected_meta: Dict[str, Any]
) -> bool:
    if not isinstance(saved, dict) or ids_key not in saved or "embeddings" not in saved:
        return False
    if [str(x) for x in saved[ids_key]] != [str(x) for x in expected_ids]:
        raise ValueError(f"Embedding cache id order mismatch for key={ids_key}")
    metadata = saved.get("metadata", {})
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Embedding cache metadata mismatch: {key} saved={metadata.get(key)!r} expected={value!r}"
            )
    return True


def build_or_load_embeddings(
    model: SentenceTransformer,
    ids: Sequence[str],
    texts: Sequence[str],
    prefix: str,
    batch_size: int,
    device: str,
    cache_path: Path,
    ids_key: str,
    model_name: str,
) -> torch.Tensor:
    expected_meta = _cache_metadata(
        model_name, prefix, len(ids), model.get_sentence_embedding_dimension()
    )
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if _cache_ok(saved, ids_key, ids, expected_meta):
            return saved["embeddings"].float().contiguous().cpu()
    embeddings = encode_texts(model, texts, prefix, batch_size, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"embeddings": embeddings, ids_key: list(ids), "metadata": expected_meta}, cache_path
    )
    return embeddings


def build_qrels_by_qid_idx(
    qrels: Dict[str, Dict[str, float]], doc_id_to_index: Dict[str, int]
) -> Dict[str, Dict[int, float]]:
    return {
        qid: {doc_id_to_index[doc_id]: float(score) for doc_id, score in rels.items()}
        for qid, rels in qrels.items()
    }


def build_rankings(
    inds: np.ndarray,
    examples: Sequence[EvalExample],
    corpus: Sequence[CorpusDoc],
    qrels_by_qid_idx: Dict[str, Dict[int, float]],
    top_k: int,
) -> List[dict]:
    rankings = []
    for row, example in enumerate(examples):
        rels = qrels_by_qid_idx[example.qid]
        hits = []
        for rank, doc_idx in enumerate(inds[row][:top_k].tolist(), start=1):
            doc = corpus[int(doc_idx)]
            hits.append(
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "score": None,
                    "relevance": float(rels.get(int(doc_idx), 0.0)),
                    "title": doc.title,
                    "text": doc.text,
                }
            )
        rankings.append(
            {
                "qid": example.qid,
                "query_text": example.query_text,
                "query_type": example.query_type,
                "base_type": example.base_type,
                "ranking": hits,
            }
        )
    return rankings


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_rankings_jsonl(path: Path, rankings: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_model_on_split(
    model: SentenceTransformer,
    model_name: str,
    dataset: DatasetBundle,
    output_dir: Path,
    split_name: str,
    split_qids: Sequence[str],
    ks: Sequence[int],
    query_batch_size: int,
    corpus_batch_size: int,
    device: str,
    tracker: ComputeTracker,
) -> Dict[int, Dict[str, float]]:
    cache_dir = output_dir / "cache"
    doc_ids = [doc.doc_id for doc in dataset.corpus]
    doc_texts = [doc.text for doc in dataset.corpus]
    query_by_id = {q.qid: q for q in dataset.queries}
    selected_queries = [
        query_by_id[qid] for qid in split_qids if qid in query_by_id and qid in dataset.qrels
    ]
    query_ids = [query.qid for query in selected_queries]
    query_texts = [query.text for query in selected_queries]
    with tracker.track("corpus_encoding_time_sec"):
        corpus_embeddings = build_or_load_embeddings(
            model,
            doc_ids,
            doc_texts,
            PASSAGE_PREFIX,
            corpus_batch_size,
            device,
            cache_dir / f"corpus_{model_name.replace('/', '__')}.pt",
            "doc_ids",
            model_name,
        )
    tracker.increment("corpus_docs_encoded", len(doc_ids))
    with tracker.track("query_encoding_time_sec"):
        query_embeddings = build_or_load_embeddings(
            model,
            query_ids,
            query_texts,
            QUERY_PREFIX,
            query_batch_size,
            device,
            cache_dir / f"queries_{split_name}_{model_name.replace('/', '__')}.pt",
            "query_ids",
            model_name,
        )
    tracker.increment("queries_encoded", len(query_ids))

    query_id_to_index = {qid: idx for idx, qid in enumerate(query_ids)}
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    qrels_by_qid_idx = build_qrels_by_qid_idx(dataset.qrels, doc_id_to_index)
    examples = build_eval_examples(
        query_ids, query_by_id, query_id_to_index, dataset.qrels, EvalExample
    )
    segment_name = "validation_eval_time_sec" if split_name == "val" else "test_eval_time_sec"
    with tracker.track(segment_name):
        index = SearchIndex(corpus_embeddings)
        tracker.add_time("faiss_indexing_time_sec", index.build_time_sec)
        query_np = query_embeddings.detach().cpu().numpy().astype(np.float32)
        t0 = time.perf_counter()
        inds = index.search(query_np, max(int(k) for k in ks))
        search_elapsed = time.perf_counter() - t0
        tracker.add_time("faiss_search_time_sec", search_elapsed)
        tracker.set("retrieval_queries_per_sec", safe_div(len(examples), search_elapsed))
        per_query_ms = (search_elapsed / max(1, len(examples))) * 1000.0
        tracker.set("average_query_latency_ms", per_query_ms)
        tracker.set("p50_query_latency_ms", per_query_ms)
        tracker.set("p95_query_latency_ms", per_query_ms)
        tracker.set("faiss_index_size_mb", index.size_mb())
        metrics = eval_metrics_multi_k_from_inds(inds, examples, qrels_by_qid_idx, ks)
        legalbench_char_metrics = legalbench_char_metrics_from_inds(
            inds, examples, dataset.legalbench_metadata_path, ks
        )
        for k, values in legalbench_char_metrics.items():
            if k in metrics:
                metrics[k].update(values)
        rankings = build_rankings(
            inds, examples, dataset.corpus, qrels_by_qid_idx, top_k=max(int(k) for k in ks)
        )
    save_json(output_dir / f"metrics_{split_name}.json", {str(k): v for k, v in metrics.items()})
    save_rankings_jsonl(output_dir / f"rankings_{split_name}.jsonl", rankings)
    return metrics


def save_compute_metrics(path: Path, payload: Dict[str, Any]) -> None:
    save_json(path, payload)


@dataclass
class SupervisedE5Config:
    dataset_dir: str
    dataset_name: str
    dataset_format: str = "auto"
    model_name: str = "intfloat/e5-large-v2"
    output_dir: str = "outputs/supervised_e5"
    seed: int = 42
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    query_type_filter: str = "all"
    query_types: str = ""
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    corpus_batch_size: int = 64
    query_batch_size: int = 128
    max_train_positives_per_query: int = 1
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    device: str = "cpu"
    fp16: bool = False
    base_only: bool = False
    eval_only: bool = False
    force: bool = False
    smoke_test: bool = False
    legalbench_rag_root: str = ""
    legalbench_prepared_dir: str = ""
    legalbench_extract_dir: str = ""
    legalbench_chunk_size: int = 500
    legalbench_chunk_strategy: str = "naive"


def build_supervised_train_examples(
    dataset: DatasetBundle, max_train_positives_per_query: int
) -> List[Any]:
    from sentence_transformers import InputExample

    query_map = {q.qid: q for q in dataset.queries}
    doc_map = {d.doc_id: d for d in dataset.corpus}
    examples: List[Any] = []
    for qid in dataset.train_qids:
        if qid not in dataset.qrels or qid not in query_map:
            continue
        rels = dataset.qrels[qid]
        ranked_doc_ids = sorted(rels.keys(), key=lambda doc_id: (-float(rels[doc_id]), doc_id))
        if max_train_positives_per_query > 0:
            ranked_doc_ids = ranked_doc_ids[:max_train_positives_per_query]
        for doc_id in ranked_doc_ids:
            if doc_id not in doc_map:
                continue
            examples.append(
                InputExample(
                    texts=[
                        QUERY_PREFIX + query_map[qid].text,
                        PASSAGE_PREFIX + doc_map[doc_id].text,
                    ]
                )
            )
    return examples


def train_supervised_e5(
    model: SentenceTransformer,
    train_examples: List[Any],
    cfg: SupervisedE5Config,
    tracker: ComputeTracker,
) -> None:
    from sentence_transformers import losses

    if not train_examples:
        raise ValueError("No supervised training examples were created from the train split qrels")

    loader = DataLoader(train_examples, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = math.ceil(len(loader) * cfg.epochs * cfg.warmup_ratio)
    tracker.increment("training_examples", len(train_examples) * cfg.epochs)
    tracker.increment("training_steps", len(loader) * cfg.epochs)
    with tracker.track("training_time_sec"):
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=cfg.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": cfg.lr},
            show_progress_bar=True,
            use_amp=cfg.fp16,
        )


def run(cfg: SupervisedE5Config) -> dict:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(cfg))

    split_path = output_dir.parent / "splits" / f"seed_{cfg.seed}_split.json"
    dataset = load_dataset_for_cfg(cfg, split_path)
    if cfg.smoke_test:
        dataset = subset_for_smoke_test(dataset)

    precision_mode = "fp16" if cfg.fp16 else "fp32"
    tracker = ComputeTracker(precision_mode=precision_mode)
    model_dir = output_dir / "model"

    if cfg.base_only:
        model = load_e5_model(cfg.model_name, cfg.device, cfg.max_seq_length)
        model_id = cfg.model_name
    elif model_dir.exists() and cfg.eval_only and not cfg.force:
        model = SentenceTransformer(str(model_dir), device=cfg.device)
        model.max_seq_length = cfg.max_seq_length
        model_id = str(model_dir)
    else:
        model = load_e5_model(cfg.model_name, cfg.device, cfg.max_seq_length)
        model_id = cfg.model_name
        tracker.count_parameters(model)
        if cfg.smoke_test:
            cfg = SupervisedE5Config(
                **{
                    **asdict(cfg),
                    "epochs": 1,
                    "max_train_positives_per_query": min(cfg.max_train_positives_per_query, 1),
                }
            )
        if not cfg.eval_only:
            train_examples = build_supervised_train_examples(
                dataset, cfg.max_train_positives_per_query
            )
            if cfg.smoke_test:
                train_examples = train_examples[: min(256, len(train_examples))]
            train_supervised_e5(model, train_examples, cfg, tracker)
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir))
            model_id = str(model_dir)

    tracker.count_parameters(model)
    metrics_val = evaluate_model_on_split(
        model,
        model_id,
        dataset,
        output_dir,
        "val",
        dataset.val_qids,
        cfg.ks,
        cfg.query_batch_size,
        cfg.corpus_batch_size,
        cfg.device,
        tracker,
    )
    metrics_test = evaluate_model_on_split(
        model,
        model_id,
        dataset,
        output_dir,
        "test",
        dataset.test_qids,
        cfg.ks,
        cfg.query_batch_size,
        cfg.corpus_batch_size,
        cfg.device,
        tracker,
    )
    final_model_path = None if cfg.base_only else model_dir
    compute = tracker.finalize(
        final_model_path=final_model_path,
        corpus_embedding_cache_path=output_dir
        / "cache"
        / f"corpus_{model_id.replace('/', '__')}.pt",
    )
    save_compute_metrics(output_dir / "compute_metrics.json", compute)
    return {"metrics_val": metrics_val, "metrics_test": metrics_test, "compute_metrics": compute}


def parse_args(argv: Optional[Sequence[str]] = None) -> SupervisedE5Config:
    parser = argparse.ArgumentParser(
        description="Standalone cluster-ready supervised E5 training + dense retrieval evaluation"
    )
    parser.add_argument(
        "--dataset_format", choices=["auto", "standard", "legalbench_rag"], default="auto"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", default="intfloat/e5-large-v2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument(
        "--query_type_filter",
        choices=["all", "negation", "conjunction", "union", "projection", "custom"],
        default="all",
    )
    parser.add_argument("--query_types", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--corpus_batch_size", type=int, default=64)
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--max_train_positives_per_query", type=int, default=1)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--legalbench_rag_root",
        default="",
        help="Root containing LegalBench-RAG corpus/ and benchmarks/. Defaults to --dataset_dir when dataset_format=legalbench_rag.",
    )
    parser.add_argument(
        "--legalbench_prepared_dir",
        default="",
        help="Optional directory for prepared chunk-level LegalBench-RAG data.",
    )
    parser.add_argument(
        "--legalbench_extract_dir",
        default="",
        help="Optional extraction directory when LegalBench-RAG is provided as corpus.zip and benchmarks.zip.",
    )
    parser.add_argument("--legalbench_chunk_size", type=int, default=500)
    parser.add_argument("--legalbench_chunk_strategy", choices=["naive", "rcts"], default="naive")
    args = parser.parse_args(argv)
    return SupervisedE5Config(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        dataset_format=args.dataset_format,
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        query_type_filter=args.query_type_filter,
        query_types=args.query_types,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        corpus_batch_size=args.corpus_batch_size,
        query_batch_size=args.query_batch_size,
        max_train_positives_per_query=args.max_train_positives_per_query,
        ks=tuple(args.ks),
        device=args.device,
        fp16=args.fp16,
        eval_only=args.eval_only,
        force=args.force,
        smoke_test=args.smoke_test,
        legalbench_rag_root=args.legalbench_rag_root,
        legalbench_prepared_dir=args.legalbench_prepared_dir,
        legalbench_extract_dir=args.legalbench_extract_dir,
        legalbench_chunk_size=args.legalbench_chunk_size,
        legalbench_chunk_strategy=args.legalbench_chunk_strategy,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    main()
