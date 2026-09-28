#!/usr/bin/env python3
"""Create shared SciFact and NFCorpus split and train-fraction manifests.

The prepared directories are intended to be passed to every training method so
that supervised retrievers, teacher adapters, and distillation objectives use exactly
the same query split. Source datasets are never modified.

Default protocol:

* SciFact: split the official training queries into train/dev and keep the
  official test queries unchanged.
* NFCorpus: preserve the official train/dev/test files.
* Construct nested 10%, 30%, 50%, and 100% prefixes of the resulting training
  query IDs. Fractions never include a dev or test query.

Run after installing this package with:

    rcd-prepare-beir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

DEFAULT_DATASETS = ("scifact", "nfcorpus")
DEFAULT_FRACTIONS = (0.10, 0.30, 0.50, 1.00)
PROTOCOL_VERSION = 2


@dataclass(frozen=True)
class QrelRow:
    qid: str
    doc_id: str
    score: float


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_sha1(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def normalize_dataset(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    aliases = {
        "scifact": "scifact",
        "sci-fact": "scifact",
        "nfcorpus": "nfcorpus",
        "nf-corpus": "nfcorpus",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported BEIR dataset: {value!r}")
    return aliases[key]


def fraction_tag(fraction: float) -> str:
    return f"fraction_{int(round(float(fraction) * 100.0)):03d}"


def deterministic_order(qids: Iterable[str], seed: int) -> List[str]:
    return sorted(
        set(str(qid) for qid in qids),
        key=lambda qid: (
            hashlib.sha1(f"{int(seed)}\0{qid}".encode("utf-8")).hexdigest(),
            qid,
        ),
    )


def nested_fraction_qids(
    train_qids: Sequence[str], fractions: Sequence[float], fraction_seed: int
) -> Tuple[List[str], Dict[float, List[str]]]:
    order = deterministic_order(train_qids, fraction_seed)
    selections: Dict[float, List[str]] = {}
    for fraction in sorted(set(float(value) for value in fractions)):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Training fractions must be in (0, 1], got {fraction}")
        count = min(len(order), max(1, int(math.ceil(len(order) * fraction)))) if order else 0
        selections[fraction] = order[:count]
    return order, selections


def _looks_like_header(parts: Sequence[str]) -> bool:
    tokens = {value.strip().lower() for value in parts}
    return bool(
        tokens
        & {
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
    )


def load_qrel_rows(path: Path) -> List[QrelRow]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[QrelRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if line_number == 1 and _looks_like_header(parts):
                continue
            if len(parts) == 3:
                qid, doc_id, score = parts
            elif len(parts) >= 4:
                qid, doc_id, score = parts[0], parts[-2], parts[-1]
            else:
                raise ValueError(f"Unsupported qrels row in {path}:{line_number}: {parts}")
            rows.append(QrelRow(str(qid), str(doc_id), float(score)))
    if not rows:
        raise ValueError(f"No qrel rows found in {path}")
    return rows


def positive_qids(rows: Sequence[QrelRow]) -> List[str]:
    return sorted({row.qid for row in rows if row.score > 0.0})


def write_qrel_rows(path: Path, rows: Sequence[QrelRow], qids: Iterable[str]) -> int:
    selected = set(str(qid) for qid in qids)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for row in rows:
            if row.qid in selected:
                writer.writerow([row.qid, row.doc_id, format(row.score, ".12g")])
                count += 1
    return count


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if destination.is_file() and file_sha1(destination) == file_sha1(source):
            return
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def normalize_queries(source: Path, destination: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    added_type = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or "_id" not in row or "text" not in row:
                raise ValueError(f"{source}:{line_number} requires _id and text")
            qid = str(row["_id"])
            if qid in seen:
                raise ValueError(f"Duplicate query ID {qid!r} in {source}")
            seen.add(qid)
            normalized = dict(row)
            normalized["_id"] = qid
            normalized["text"] = str(row["text"])
            if not str(normalized.get("type", "")).strip():
                normalized["type"] = "beir"
                added_type += 1
            rows.append(normalized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return {
        "query_count": len(rows),
        "query_ids_sha1": json_sha1(sorted(seen)),
        "added_type_count": added_type,
        "prepared_queries_sha1": file_sha1(destination),
    }


def _source_split_rows(
    dataset: str,
    source_dir: Path,
    split_seed: int,
    val_ratio: float,
) -> Tuple[Dict[str, List[QrelRow]], Dict[str, List[str]], str, Dict[str, str]]:
    qrels_dir = source_dir / "qrels"
    if dataset == "nfcorpus":
        paths = {
            "train": qrels_dir / "train.tsv",
            "val": qrels_dir / "dev.tsv",
            "test": qrels_dir / "test.tsv",
        }
        rows = {name: load_qrel_rows(path) for name, path in paths.items()}
        ids = {name: positive_qids(values) for name, values in rows.items()}
        policy = "official_train_dev_test"
    elif dataset == "scifact":
        train_path = qrels_dir / "train.tsv"
        test_path = qrels_dir / "test.tsv"
        official_train = load_qrel_rows(train_path)
        official_test = load_qrel_rows(test_path)
        train_order = deterministic_order(positive_qids(official_train), split_seed)
        if len(train_order) < 2:
            raise ValueError("SciFact official train qrels need at least two positive queries")
        # Reserve a deterministic validation subset while preserving the official test set.
        dev_count = min(
            len(train_order) - 1,
            max(1, int(math.ceil(len(train_order) * val_ratio))),
        )
        ids = {
            "val": train_order[:dev_count],
            "train": train_order[dev_count:],
            "test": positive_qids(official_test),
        }
        rows = {"train": official_train, "val": official_train, "test": official_test}
        paths = {"train": train_path, "test": test_path}
        policy = "official_train_partitioned_train_dev_official_test_unchanged"
    else:
        raise ValueError(dataset)
    return rows, ids, policy, {name: file_sha1(path) for name, path in paths.items()}


def _assert_split_contract(split_ids: Mapping[str, Sequence[str]]) -> None:
    sets = {name: set(values) for name, values in split_ids.items()}
    for name in ("train", "val", "test"):
        if not sets.get(name):
            raise ValueError(f"Prepared {name} split is empty")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise AssertionError(f"Split overlap between {left} and {right}: {sorted(overlap)[:5]}")


def prepare_dataset(
    dataset: str,
    data_root: Path,
    output_root: Path,
    split_seed: int = 42,
    fraction_seed: int = 42,
    val_ratio: float = 0.10,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    dataset = normalize_dataset(dataset)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"SciFact validation ratio must be in (0, 1), got {val_ratio}")
    source_dir = (data_root / dataset).resolve()
    required = (source_dir / "corpus.jsonl", source_dir / "queries.jsonl")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source dataset files: {missing}")
    prepared_dir = (output_root / dataset / f"split_seed_{int(split_seed)}").resolve()
    rows, split_ids, policy, source_qrel_sha1 = _source_split_rows(
        dataset, source_dir, split_seed, val_ratio
    )
    _assert_split_contract(split_ids)
    full_train_order, fraction_ids = nested_fraction_qids(
        split_ids["train"], fractions, fraction_seed
    )
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "source_dataset_dir": str(source_dir),
        "policy": policy,
        "split_seed": int(split_seed),
        "fraction_seed": int(fraction_seed),
        "scifact_validation_ratio": float(val_ratio) if dataset == "scifact" else None,
        "source_corpus_sha1": file_sha1(source_dir / "corpus.jsonl"),
        "source_queries_sha1": file_sha1(source_dir / "queries.jsonl"),
        "source_qrels_sha1": source_qrel_sha1,
        "train_query_count": len(split_ids["train"]),
        "val_query_count": len(split_ids["val"]),
        "test_query_count": len(split_ids["test"]),
        "train_qids_sha1": json_sha1(split_ids["train"]),
        "val_qids_sha1": json_sha1(split_ids["val"]),
        "test_qids_sha1": json_sha1(split_ids["test"]),
        "all_split_qids_sha1": json_sha1(
            sorted(split_ids["train"] + split_ids["val"] + split_ids["test"])
        ),
        "full_fraction_order_sha1": json_sha1(full_train_order),
        "fractions": [float(value) for value in sorted(fraction_ids)],
        "fraction_qids_sha1": {
            fraction_tag(fraction): json_sha1(qids) for fraction, qids in fraction_ids.items()
        },
        "fraction_query_counts": {
            fraction_tag(fraction): len(qids) for fraction, qids in fraction_ids.items()
        },
    }
    metadata_path = prepared_dir / "split_manifest.json"
    existing: Dict[str, Any] = {}
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches and not force_rebuild:
            raise ValueError(
                f"Existing shared split at {prepared_dir} has a different contract: {mismatches}. "
                "Use a different shared split root or split seed."
            )

    link_or_copy(source_dir / "corpus.jsonl", prepared_dir / "corpus.jsonl")
    query_metadata = normalize_queries(source_dir / "queries.jsonl", prepared_dir / "queries.jsonl")
    qrel_counts = {
        name: write_qrel_rows(
            prepared_dir / "qrels" / ("dev.tsv" if name == "val" else f"{name}.tsv"),
            rows[name],
            split_ids[name],
        )
        for name in ("train", "val", "test")
    }
    for fraction, selected in fraction_ids.items():
        tag = fraction_tag(fraction)
        fraction_dir = prepared_dir / "fractions" / tag
        atomic_write_json(
            fraction_dir / "train_qids.json",
            {
                "dataset": dataset,
                "fraction": fraction,
                "fraction_seed": fraction_seed,
                "selected_count": len(selected),
                "selected_qids_sha1": json_sha1(selected),
                "selected_qids": selected,
            },
        )
        write_qrel_rows(fraction_dir / "train.tsv", rows["train"], selected)
    metadata = {
        **expected,
        "prepared_dataset_dir": str(prepared_dir),
        "query_metadata": query_metadata,
        "qrel_row_counts": qrel_counts,
        "prepared_at": existing.get("prepared_at", now_iso()),
    }
    metadata["split_contract_sha1"] = json_sha1(expected)
    atomic_write_json(metadata_path, metadata)
    return metadata


def prepare_suite(
    datasets: Sequence[str],
    data_root: Path,
    output_root: Path,
    split_seed: int,
    fraction_seed: int,
    val_ratio: float,
    fractions: Sequence[float],
    force_rebuild: bool = False,
) -> Dict[str, Dict[str, Any]]:
    manifest = {
        normalize_dataset(dataset): prepare_dataset(
            dataset,
            data_root,
            output_root,
            split_seed,
            fraction_seed,
            val_ratio,
            fractions,
            force_rebuild,
        )
        for dataset in datasets
    }
    atomic_write_json(output_root / "beir_low_resource_split_manifest.json", manifest)
    return manifest


def run_tests() -> None:
    ordered, fractions = nested_fraction_qids([f"q{i}" for i in range(20)], DEFAULT_FRACTIONS, 42)
    assert fractions[0.10] == fractions[0.30][: len(fractions[0.10])]
    assert fractions[0.30] == fractions[0.50][: len(fractions[0.30])]
    assert fractions[0.50] == fractions[1.00][: len(fractions[0.50])]
    assert fractions[1.00] == ordered
    assert normalize_dataset("sci-fact") == "scifact"
    assert normalize_dataset("nf-corpus") == "nfcorpus"
    print("BEIR low-resource split tests passed", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--output_root", default="runs/beir_low_resource_splits")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--fraction_seed", type=int, default=42)
    parser.add_argument("--scifact_val_ratio", type=float, default=0.10)
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--run_tests", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_tests:
        run_tests()
        return
    manifest = prepare_suite(
        args.datasets,
        Path(args.data_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
        args.split_seed,
        args.fraction_seed,
        args.scifact_val_ratio,
        args.fractions,
        args.force_rebuild,
    )
    for dataset, metadata in manifest.items():
        print(
            f"{dataset}: train={metadata['train_query_count']} "
            f"dev={metadata['val_query_count']} test={metadata['test_query_count']} "
            f"dir={metadata['prepared_dataset_dir']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
