"""Shared LegalBench-RAG archive and chunking utilities."""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DATASETS = ("privacy_qa", "contractnli")
DATASET_ALIASES = {
    "privacyqa": "privacy_qa",
    "privacy_qa": "privacy_qa",
    "privacy-qa": "privacy_qa",
    "contractnli": "contractnli",
    "contract_nli": "contractnli",
    "contract-nli": "contractnli",
}


def has_data_directories(root: str | Path) -> bool:
    root = str(root)
    return os.path.isdir(os.path.join(root, "corpus")) and os.path.isdir(
        os.path.join(root, "benchmarks")
    )


def has_archive_pair(root: str | Path) -> bool:
    root = str(root)
    return os.path.exists(os.path.join(root, "corpus.zip")) and os.path.exists(
        os.path.join(root, "benchmarks.zip")
    )


def resolve_data_root(root: str | Path) -> str:
    root = str(root)
    for candidate in (root, os.path.join(root, "data")):
        if has_data_directories(candidate) or has_archive_pair(candidate):
            return candidate
    return root


def _find_nested_data_root(root: str | Path) -> str | None:
    for dirpath, dirnames, _filenames in os.walk(root):
        if "corpus" in dirnames and "benchmarks" in dirnames:
            return dirpath
    return None


def _safe_member_parts(name: str, archive_path: str | Path) -> tuple[str, ...]:
    normalized = name.replace("\\", "/").lstrip("/")
    parts = tuple(part for part in normalized.split("/") if part and part != ".")
    if any(part == ".." for part in parts):
        raise ValueError(f"Refusing unsafe zip member path {name!r} in {archive_path}")
    if parts and parts[0] == "__MACOSX":
        return tuple()
    return parts


def _archive_has_root_component(archive_path: str | Path, component: str) -> bool:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            parts = _safe_member_parts(name, archive_path)
            if not parts:
                continue
            if parts[0] == component or (
                len(parts) > 1 and parts[0] == "data" and parts[1] == component
            ):
                return True
    return False


def _safe_extract(archive_path: str | Path, extract_dir: str | Path) -> None:
    archive_path = str(archive_path)
    extract_dir = str(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    root = os.path.abspath(extract_dir)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parts = _safe_member_parts(member.filename, archive_path)
            if not parts:
                continue
            target = os.path.abspath(os.path.join(extract_dir, *parts))
            if not target.startswith(root + os.sep) and target != root:
                raise ValueError(
                    f"Refusing unsafe zip member path {member.filename!r} in {archive_path}"
                )
            if member.is_dir() or member.filename.endswith("/"):
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(member, "r") as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)


def _directory_has_file(path: str | Path) -> bool:
    if not os.path.isdir(path):
        return False
    return any(filenames for _dirpath, _dirnames, filenames in os.walk(path))


def _component_is_extracted(extract_root: str | Path, component: str) -> bool:
    return _directory_has_file(Path(extract_root) / component) or _directory_has_file(
        Path(extract_root) / "data" / component
    )


def ensure_data_root(cfg: Any) -> str:
    source_root = resolve_data_root(cfg.legalbench_rag_root or cfg.dataset_dir)
    if has_data_directories(source_root):
        return source_root
    if not has_archive_pair(source_root):
        raise FileNotFoundError(
            "LegalBench-RAG source must contain either corpus/ and benchmarks/ directories "
            f"or corpus.zip and benchmarks.zip. Checked: {source_root}"
        )

    extract_root = cfg.legalbench_extract_dir or os.path.join(
        cfg.output_dir, "legalbench_rag_extracted"
    )
    os.makedirs(extract_root, exist_ok=True)
    for component in ("corpus", "benchmarks"):
        archive_path = os.path.join(source_root, f"{component}.zip")
        extract_dir = (
            extract_root
            if _archive_has_root_component(archive_path, component)
            else os.path.join(extract_root, component)
        )
        if _component_is_extracted(extract_root, component):
            logger.info(
                "LegalBench-RAG %s archive already extracted under %s", component, extract_root
            )
            continue
        logger.info("Extracting LegalBench-RAG %s -> %s", archive_path, extract_dir)
        _safe_extract(archive_path, extract_dir)

    resolved = resolve_data_root(extract_root)
    if not has_data_directories(resolved):
        resolved = _find_nested_data_root(extract_root) or resolved
    if not has_data_directories(resolved):
        raise FileNotFoundError(
            "Could not find corpus/ and benchmarks/ after extracting LegalBench-RAG "
            f"archives into {extract_root}"
        )
    return resolved


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(f"Unsupported LegalBench-RAG dataset {name!r}. Expected one of {DATASETS}")


def load_benchmark_tests(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tests = payload.get("tests", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(tests, list):
        raise ValueError(f"LegalBench-RAG benchmark must contain a tests list: {path}")
    return tests


def _recursive_character_spans(text: str, chunk_size: int) -> list[tuple[int, int]]:
    separators = ["\n\n", "\n", "!", "?", ".", ":", ";", ",", " "]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        if hard_end == len(text):
            end = hard_end
        else:
            window = text[start:hard_end]
            cuts = [(window.rfind(separator), len(separator)) for separator in separators]
            best_cut, separator_length = max(cuts, key=lambda value: value[0])
            end = (
                hard_end
                if best_cut <= max(16, chunk_size // 4)
                else start + best_cut + separator_length
            )
        if end <= start:
            end = min(start + chunk_size, len(text))
        spans.append((start, end))
        start = end
    return spans


def chunk_spans(text: str, chunk_size: int, strategy: str) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("--legalbench_chunk_size must be positive")
    if strategy == "naive":
        return [
            (start, min(start + chunk_size, len(text))) for start in range(0, len(text), chunk_size)
        ]
    if strategy == "rcts":
        return _recursive_character_spans(text, chunk_size)
    raise ValueError("--legalbench_chunk_strategy must be naive or rcts")


def span_overlap(first: Sequence[int], second: Sequence[int]) -> int:
    lower = max(int(first[0]), int(second[0]))
    upper = min(int(first[1]), int(second[1]))
    return max(0, upper - lower)


def prepare_retrieval_dataset(
    cfg: Any,
    dataset_name: str,
    prepared_dir: str | Path,
    *,
    include_paper_note: bool = False,
    progress_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Materialize one LegalBench-RAG benchmark as a chunk-level retrieval dataset."""
    dataset_name = normalize_dataset_name(dataset_name)
    prepared_dir = str(prepared_dir)
    source_root = ensure_data_root(cfg)
    corpus_root = os.path.join(source_root, "corpus")
    benchmark_path = os.path.join(source_root, "benchmarks", f"{dataset_name}.json")
    if not os.path.isdir(corpus_root):
        raise FileNotFoundError(f"LegalBench-RAG corpus directory not found: {corpus_root}")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"LegalBench-RAG benchmark file not found: {benchmark_path}")

    tests = load_benchmark_tests(benchmark_path)
    referenced_files = sorted(
        {
            str(snippet["file_path"])
            for test in tests
            for snippet in test.get("snippets", [])
            if "file_path" in snippet and "span" in snippet
        }
    )
    if not referenced_files:
        raise ValueError(f"No referenced files found in {benchmark_path}")

    os.makedirs(os.path.join(prepared_dir, "qrels"), exist_ok=True)
    passages: list[dict[str, Any]] = []
    passage_meta: dict[str, dict[str, Any]] = {}
    file_to_chunk_ids: defaultdict[str, list[str]] = defaultdict(list)
    for file_path in referenced_files:
        absolute_path = os.path.join(corpus_root, file_path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(
                f"LegalBench-RAG referenced corpus file missing: {absolute_path}"
            )
        with open(absolute_path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        for start, end in chunk_spans(
            text, cfg.legalbench_chunk_size, cfg.legalbench_chunk_strategy
        ):
            chunk_text = text[start:end]
            if not chunk_text:
                continue
            passage_id = f"{dataset_name}::doc::{file_path}::span::{start}-{end}"
            file_to_chunk_ids[file_path].append(passage_id)
            passage_meta[passage_id] = {
                "file_path": file_path,
                "span": [start, end],
                "dataset": dataset_name,
            }
            passages.append({"_id": passage_id, "text": chunk_text})

    queries: list[dict[str, Any]] = []
    qrels_rows: list[tuple[str, str, float]] = []
    query_meta: dict[str, dict[str, Any]] = {}
    missing_relevant = 0
    for index, test in enumerate(tests):
        query = str(test.get("query", "")).strip()
        if not query:
            continue
        query_id = f"{dataset_name}::q::{index}"
        snippets = [
            {
                "file_path": str(snippet["file_path"]),
                "span": [int(snippet["span"][0]), int(snippet["span"][1])],
            }
            for snippet in test.get("snippets", [])
            if "file_path" in snippet and "span" in snippet
        ]
        queries.append({"_id": query_id, "text": query, "type": dataset_name, "answer": ""})
        query_meta[query_id] = {"dataset": dataset_name, "snippets": snippets}
        relevant_passage_ids = {
            passage_id
            for snippet in snippets
            for passage_id in file_to_chunk_ids.get(snippet["file_path"], [])
            if span_overlap(passage_meta[passage_id]["span"], snippet["span"]) > 0
        }
        if not relevant_passage_ids:
            missing_relevant += 1
        qrels_rows.extend(
            (query_id, passage_id, 1.0) for passage_id in sorted(relevant_passage_ids)
        )

    with open(os.path.join(prepared_dir, "corpus.jsonl"), "w", encoding="utf-8") as handle:
        for row in passages:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(prepared_dir, "queries.jsonl"), "w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(prepared_dir, "qrels", "test.tsv"), "w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, passage_id, score in qrels_rows:
            handle.write(f"{query_id}\t{passage_id}\t{score}\n")

    metadata = {
        "dataset_name": dataset_name,
        "source_root": source_root,
        "benchmark_path": benchmark_path,
        "chunk_strategy": cfg.legalbench_chunk_strategy,
        "chunk_size": cfg.legalbench_chunk_size,
        "passage_count": len(passages),
        "query_count": len(queries),
        "qrel_count": len(qrels_rows),
        "referenced_file_count": len(referenced_files),
        "missing_relevant_query_count": missing_relevant,
        "passage_id_order": [row["_id"] for row in passages],
        "passages": passage_meta,
        "queries": query_meta,
    }
    if include_paper_note:
        metadata["paper_note"] = (
            "LegalBench-RAG gold labels are exact character spans. This prepared view splits raw "
            "documents into character chunks and marks overlapping chunks as relevant for the "
            "existing dense-retrieval training/evaluation flow."
        )
    metadata_path = os.path.join(prepared_dir, "legalbench_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if progress_logger is not None:
        progress_logger.info(
            "Prepared LegalBench-RAG dataset=%s | passages=%s | queries=%s | "
            "qrels=%s | missing_relevant=%s | path=%s",
            dataset_name,
            len(passages),
            len(queries),
            len(qrels_rows),
            missing_relevant,
            prepared_dir,
        )
    return metadata
