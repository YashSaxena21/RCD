"""Dataset loading, validation, and deterministic split handling."""

from __future__ import annotations

import csv
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .comlq import (
    ALL_COMLQ_TYPES,
    CONJUNCTION_BASE_TYPES,
    NEGATION_BASE_TYPES,
    PROJECTION_BASE_TYPES,
    UNION_BASE_TYPES,
    choose_qids_by_filter,
)
from .config import Cfg
from .data_utils import (
    base_query_type,
    find_official_qrel_paths,
    load_jsonl,
    merge_qrels,
)
from .legalbench import has_archive_pair as legalbench_rag_has_zip_pair
from .legalbench import has_data_directories as legalbench_rag_has_dirs
from .legalbench import resolve_data_root as resolve_legalbench_rag_data_root
from .records import PassageRecord, QueryRecord
from .runtime import log_section, logger


def load_passages(path: str) -> List[PassageRecord]:
    passages = []
    seen = set()
    for obj in load_jsonl(path):
        if "_id" not in obj or "text" not in obj:
            raise ValueError(f"{path} requires fields _id and text. Bad row: {obj}")
        pid = str(obj["_id"])
        if pid in seen:
            raise ValueError(f"Duplicate passage _id: {pid}")
        seen.add(pid)
        text = str(obj["text"]).strip()
        if not text:
            title = str(obj.get("title", "")).strip()
            if not title:
                raise ValueError(f"Corpus row {pid} has neither text nor title")
            text = title
        passages.append(PassageRecord(pid=pid, text=text))
    return passages


def load_queries(path: str) -> List[QueryRecord]:
    queries = []
    seen = set()
    for obj in load_jsonl(path):
        if "_id" not in obj or "text" not in obj or "type" not in obj:
            raise ValueError(f"{path} requires fields _id, text, type. Bad row: {obj}")
        qid = str(obj["_id"])
        if qid in seen:
            raise ValueError(f"Duplicate query _id: {qid}")
        seen.add(qid)
        query_type = str(obj["type"])
        queries.append(
            QueryRecord(
                qid=qid,
                text=str(obj["text"]),
                query_type=query_type,
                base_type=base_query_type(query_type),
            )
        )
    return queries


def infer_dataset_format(cfg: Cfg) -> str:
    if cfg.dataset_format != "auto":
        return cfg.dataset_format
    root = resolve_legalbench_rag_data_root(cfg.legalbench_rag_root or cfg.dataset_dir)
    if legalbench_rag_has_dirs(root) or legalbench_rag_has_zip_pair(root):
        return "legalbench_rag"
    cwd_root = resolve_legalbench_rag_data_root(os.getcwd())
    if (
        not cfg.legalbench_rag_root
        and not os.path.exists(cfg.dataset_dir)
        and legalbench_rag_has_zip_pair(cwd_root)
    ):
        cfg.dataset_dir = os.getcwd()
        return "legalbench_rag"
    return "standard"


def load_qrels_tsv(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} must have tab-separated header with fields {sorted(required)}"
            )
        for row in reader:
            qid = str(row["query-id"])
            pid = str(row["corpus-id"])
            score = float(row["score"])
            if score <= 0:
                continue
            qrels[qid][pid] = max(score, qrels[qid].get(pid, float("-inf")))
    return {qid: dict(rels) for qid, rels in qrels.items()}


def make_split_qids(
    cfg: Cfg,
    qrels_dir: str,
    queries: List[QueryRecord],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]], str]:
    official = find_official_qrel_paths(qrels_dir)
    if official:
        split_qrels = {name: load_qrels_tsv(str(path)) for name, path in official.items()}
        split_sets = {name: set(split_qrels[name]) for name in ("train", "val", "test")}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"Official qrel split query IDs overlap between {left} and {right}; "
                    f"count={len(overlap)}, sample={sorted(overlap)[:10]}"
                )
        merged = merge_qrels(*split_qrels.values())
        split_qids = {
            name: choose_qids_by_filter(queries, split_qrels[name], cfg)
            for name in ("train", "val", "test")
        }
        return split_qids, merged, "official"

    test_path = os.path.join(qrels_dir, "test.tsv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"No official train/dev/test qrels and no fallback qrels/test.tsv in {qrels_dir}"
        )
    qrels = load_qrels_tsv(test_path)
    qids = choose_qids_by_filter(queries, qrels, cfg)
    rng = random.Random(cfg.seed)
    rng.shuffle(qids)
    count = len(qids)
    test_count = int(count * cfg.test_ratio)
    validation_count = int(count * cfg.val_ratio)
    if count > 0 and test_count == 0:
        test_count = 1
    if count > 1 and validation_count == 0:
        validation_count = 1
    return (
        {
            "train": qids[test_count + validation_count :],
            "val": qids[test_count : test_count + validation_count],
            "test": qids[:test_count],
        },
        qrels,
        "random_from_single_qrels_file",
    )


def validate_references(
    queries: List[QueryRecord],
    passages: List[PassageRecord],
    qrels: Dict[str, Dict[str, float]],
) -> None:
    query_ids = {query.qid for query in queries}
    passage_ids = {passage.pid for passage in passages}
    missing_qids = sorted(qid for qid in qrels if qid not in query_ids)
    missing_pids = sorted(
        {pid for relevances in qrels.values() for pid in relevances if pid not in passage_ids}
    )
    if missing_qids:
        raise ValueError(
            f"Qrels contain query IDs not in queries.jsonl, first 10: {missing_qids[:10]}"
        )
    if missing_pids:
        raise ValueError(
            f"Qrels contain passage IDs not in corpus.jsonl, first 10: {missing_pids[:10]}"
        )


def log_schema_summary(
    cfg: Cfg,
    passages: List[PassageRecord],
    queries: List[QueryRecord],
    qrels: Dict[str, Dict[str, float]],
    split_qids: Dict[str, List[str]],
    split_source: str,
) -> None:
    log_section("SCHEMA AND SPLIT SUMMARY")
    qrel_counts = Counter(len(relevances) for relevances in qrels.values())
    labels = Counter(score for relevances in qrels.values() for score in relevances.values())
    query_types = Counter(query.query_type for query in queries)
    base_types = Counter(query.base_type for query in queries)
    logger.info("dataset_dir: %s", cfg.dataset_dir)
    logger.info("top_level_files: %s", sorted(os.listdir(cfg.dataset_dir)))
    logger.info("passages: %s | corpus_file: %s", len(passages), cfg.corpus_file)
    logger.info("queries: %s | queries_file: %s", len(queries), cfg.queries_file)
    logger.info("qrel_queries: %s | qrels_dir: %s", len(qrels), cfg.qrels_dir)
    logger.info("qrel_labels: %s", dict(sorted(labels.items())))
    logger.info("qrels_per_query: %s", dict(sorted(qrel_counts.items())))
    logger.info("actual_query_type_distribution: %s", dict(sorted(query_types.items())))
    logger.info("base_query_type_distribution: %s", dict(sorted(base_types.items())))
    logger.info(
        "query_type_filter: %s | query_types: %s",
        cfg.query_type_filter,
        cfg.query_types or "(none)",
    )
    logger.info(
        "logical_subset_query_counts: negation=%s, conjunction=%s, union=%s, projection=%s",
        sum(1 for query in queries if query.base_type in NEGATION_BASE_TYPES),
        sum(1 for query in queries if query.base_type in CONJUNCTION_BASE_TYPES),
        sum(1 for query in queries if query.base_type in UNION_BASE_TYPES),
        sum(1 for query in queries if query.base_type in PROJECTION_BASE_TYPES),
    )
    logger.info("known_comlq_base_types_present: %s", sorted(set(base_types) & ALL_COMLQ_TYPES))
    logger.info("graded_relevance_present: %s", any(float(label) != 1.0 for label in labels))
    logger.info("multiple_relevant_passages_present: %s", any(value > 1 for value in qrel_counts))
    logger.info("split_source: %s", split_source)
    logger.info(
        "split_sizes: %s",
        {name: len(split_qids[name]) for name in ("train", "val", "test")},
    )
    if cfg.legalbench_metadata_path:
        logger.warning(
            "LegalBench-RAG character spans were converted to overlapping chunk qrels; "
            "no additional relevance judgments were invented."
        )
    if split_source != "official":
        logger.warning(
            "Official train/dev/test qrels were not found. Using a deterministic random split "
            "from qrels/test.tsv for dataset=%s.",
            cfg.dataset_name or cfg.dataset_dir,
        )


def build_qrels_by_qid_idx(
    qrels: Dict[str, Dict[str, float]],
    passage_id_to_index: Dict[str, int],
) -> Dict[str, Dict[int, float]]:
    return {
        qid: {passage_id_to_index[pid]: float(score) for pid, score in relevances.items()}
        for qid, relevances in qrels.items()
    }
