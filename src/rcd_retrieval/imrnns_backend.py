from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import core
from .sampling import deterministic_fraction_qids

IMRNNS_VERSION = "0.2.3"


def _load_api() -> tuple[Any, ...]:
    try:
        version = importlib.metadata.version("imrnns")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Install the teacher backend with: pip install imrnns==0.2.3") from exc
    if version != IMRNNS_VERSION:
        raise RuntimeError(f"Expected imrnns=={IMRNNS_VERSION}, found {version}")

    model = importlib.import_module("imrnns.model")
    training = importlib.import_module("imrnns.training")
    checkpoints = importlib.import_module("imrnns.checkpoints")
    return (
        model.IMRNN,
        model.ModelConfig,
        training.initialize_projector,
        training.train_model,
        training.TrainingConfig,
        checkpoints.save_checkpoint,
        checkpoints.load_model,
    )


def _sha1_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fraction(cfg: Any) -> float:
    return float(getattr(cfg, "_teacher_fraction", 1.0))


def _fraction_seed(cfg: Any) -> int:
    return int(getattr(cfg, "_teacher_fraction_seed", 42))


def _dataset(cfg: Any) -> str:
    return str(cfg.dataset_name or Path(cfg.dataset_dir).name)


class CachedContrastiveDataset:
    def __init__(
        self,
        examples: Sequence[tuple[str, int, int, Sequence[int]]],
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        num_negatives: int,
    ) -> None:
        self.examples = list(examples)
        self.query_embeddings = query_embeddings
        self.document_embeddings = document_embeddings
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        qid, query_index, positive, raw_negatives = self.examples[index]
        negatives = list(map(int, raw_negatives[: self.num_negatives]))
        if not negatives:
            raise RuntimeError(f"Teacher example {qid} has no hard negative")
        negatives.extend([negatives[-1]] * (self.num_negatives - len(negatives)))
        return {
            "qid": qid,
            "query_embedding": self.query_embeddings[query_index].float(),
            "documents": self.document_embeddings[[positive, *negatives]].float(),
        }


def _eligible_qids(
    split_ids: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[int, float]],
    query_indices: Mapping[str, int],
) -> list[str]:
    return [
        str(qid)
        for qid in split_ids["train"]
        if qid in query_indices and any(score > 0 for score in qrels.get(qid, {}).values())
    ]


def build_teacher_examples(
    cfg: Any,
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[int, float]],
    query_indices: Mapping[str, int],
    raw_index: Any,
    query_embeddings: torch.Tensor,
    document_count: int,
) -> tuple[list[tuple[str, int, int, list[int]]], dict[str, Any]]:
    examples: list[tuple[str, int, int, list[int]]] = []
    no_positive = no_negative = 0
    pool_size = min(int(cfg.teacher_train_negative_pool), document_count)
    batch_size = max(1, int(cfg.eval_batch_size))

    for start in range(0, len(qids), batch_size):
        chunk = [str(qid) for qid in qids[start : start + batch_size] if qid in query_indices]
        if not chunk:
            continue
        rows = raw_index.search(
            query_embeddings[[query_indices[qid] for qid in chunk]].numpy().astype(np.float32),
            pool_size,
        )
        for row, qid in enumerate(chunk):
            positives = sorted(
                (
                    (int(index), float(score))
                    for index, score in qrels.get(qid, {}).items()
                    if score > 0
                ),
                key=lambda item: (-item[1], item[0]),
            )
            if not positives:
                no_positive += 1
                continue
            relevant = {index for index, _ in positives}
            negatives: list[int] = []
            seen: set[int] = set()
            for candidate in rows[row]:
                index = int(candidate)
                if index < 0 or index in relevant or index in seen:
                    continue
                seen.add(index)
                negatives.append(index)
                if len(negatives) == int(cfg.teacher_train_num_negatives):
                    break
            if not negatives:
                no_negative += 1
                continue
            examples.append((qid, query_indices[qid], positives[0][0], negatives))

    return examples, {
        "examples": len(examples),
        "skipped_no_positive": no_positive,
        "skipped_no_negative": no_negative,
        "positive_selection": "highest_relevance_then_lowest_document_index",
        "negative_source": "frozen_retriever_top_k_excluding_qrel_positives",
        "negative_pool_k": pool_size,
    }


def _rank(
    model: Any,
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    query_indices: Sequence[int],
    candidate_indices: Any,
    max_k: int,
    device: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for query_index, candidates in zip(query_indices, candidate_indices):
            valid = [int(index) for index in candidates if int(index) >= 0]
            if not valid:
                rows.append(np.full(max_k, -1, dtype=np.int64))
                continue
            _, _, scores = model.score_candidates(
                query_embeddings[query_index].to(device).float(),
                document_embeddings[valid].to(device).float(),
            )
            order = torch.argsort(scores.float(), descending=True).cpu().tolist()
            ranked = [valid[position] for position in order[:max_k]]
            rows.append(np.asarray(ranked + [-1] * (max_k - len(ranked)), dtype=np.int64))
    return np.stack(rows)


def _validation_score(
    model: Any,
    cfg: Any,
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[int, float]],
    query_indices: Mapping[str, int],
    raw_index: Any,
    document_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
) -> float:
    from imrnns.evaluation import compute_mrr, compute_ndcg, compute_recall

    qids = [qid for qid in qids if qid in query_indices and qrels.get(qid)]
    if not qids:
        raise ValueError("The validation split has no qrel-bearing queries")
    values = {"ndcg": [], "recall": [], "mrr": []}
    batch_size = max(1, int(cfg.eval_batch_size))
    for start in range(0, len(qids), batch_size):
        chunk = qids[start : start + batch_size]
        indices = [query_indices[qid] for qid in chunk]
        candidates = raw_index.search(
            query_embeddings[indices].numpy().astype(np.float32), min(100, raw_index.ntotal)
        )
        ranked = _rank(
            model, query_embeddings, document_embeddings, indices, candidates, 10, cfg.device
        )
        for row, qid in enumerate(chunk):
            result = [str(int(index)) for index in ranked[row] if index >= 0]
            relevance = {str(index): score for index, score in qrels[qid].items() if score > 0}
            values["ndcg"].append(float(compute_ndcg(result, relevance, 10)))
            values["recall"].append(float(compute_recall(result, relevance, 10)))
            values["mrr"].append(float(compute_mrr(result, relevance, 10)))
    means = {name: sum(scores) / len(scores) for name, scores in values.items()}
    return sum(means.values()) / 3.0


def train_teacher_checkpoint(
    cfg: Any,
    checkpoint_path: Path,
    split_ids: dict[str, list[str]],
    qrels: dict[str, dict[int, float]],
    query_indices: dict[str, int],
    raw_index: Any,
    document_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    split_source: str,
) -> Path:
    IMRNN, ModelConfig, initialize_projector, train_model, TrainingConfig, save, _ = _load_api()
    eligible = _eligible_qids(split_ids, qrels, query_indices)
    full_order, selected = deterministic_fraction_qids(
        eligible, _fraction(cfg), _fraction_seed(cfg)
    )
    validation_qids = [qid for qid in split_ids["val"] if qid in query_indices and qrels.get(qid)]
    if cfg.dry_run:
        selected = selected[:8]
        validation_qids = validation_qids[:4]
    if not selected or not validation_qids:
        raise ValueError("IMRNNs training needs non-empty labeled train and validation splits")

    train_examples, train_stats = build_teacher_examples(
        cfg, selected, qrels, query_indices, raw_index, query_embeddings, len(document_embeddings)
    )
    val_examples, val_stats = build_teacher_examples(
        cfg,
        validation_qids,
        qrels,
        query_indices,
        raw_index,
        query_embeddings,
        len(document_embeddings),
    )
    if not train_examples or not val_examples:
        raise ValueError("Could not build IMRNNs examples with a positive and hard negative")

    dataset_args = (query_embeddings, document_embeddings, int(cfg.teacher_train_num_negatives))
    train_dataset = CachedContrastiveDataset(train_examples, *dataset_args)
    val_dataset = CachedContrastiveDataset(val_examples, *dataset_args)
    core.set_seed(int(cfg.seed))
    model = IMRNN(ModelConfig(input_dim=int(document_embeddings.shape[1])))
    initialize_projector(model)
    config = TrainingConfig(
        batch_size=int(cfg.teacher_train_batch_size),
        epochs=1 if cfg.dry_run else int(cfg.teacher_train_epochs),
        lr=float(cfg.teacher_train_lr),
        weight_decay=float(cfg.teacher_train_weight_decay),
        num_negatives=int(cfg.teacher_train_num_negatives),
        patience=int(getattr(cfg, "_teacher_patience", 7)),
        seed=int(cfg.seed),
        improvement_margin=float(getattr(cfg, "_teacher_improvement_margin", 0.05)),
    )
    metrics = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=str(cfg.device),
        validation_metric_fn=lambda current: _validation_score(
            current,
            cfg,
            validation_qids,
            qrels,
            query_indices,
            raw_index,
            document_embeddings,
            query_embeddings,
        ),
        validation_metric_name="mean(NDCG@10,Recall@10,MRR@10)",
    )
    metadata = {
        "teacher_implementation": "imrnns_pypi",
        "imrnns_version": IMRNNS_VERSION,
        "dataset": _dataset(cfg),
        "dataset_format": str(cfg.source_dataset_format or cfg.dataset_format),
        "dataset_dir": str(Path(cfg.dataset_dir).resolve()),
        "encoder_model_name": cfg.model_name,
        "embedding_dimension": int(cfg.embedding_dim),
        "seed": int(cfg.seed),
        "split_source": split_source,
        "teacher_training_uses_qrels": True,
        "student_training_uses_qrels": False,
        "teacher_training_fraction": _fraction(cfg),
        "fraction_selection": {
            "seed": _fraction_seed(cfg),
            "eligible_query_count": len(eligible),
            "selected_query_count": len(selected),
            "full_ordered_train_qids_sha1": _sha1_json(full_order),
            "selected_train_qids_sha1": _sha1_json(selected),
            "selected_train_qids": selected,
        },
        "training": {
            "objective": "improvement_margin",
            "metrics": metrics,
            "train_examples": train_stats,
            "validation_examples": val_stats,
        },
        "model_config": model.config.to_dict(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save(checkpoint_path, model, metadata)
    _atomic_json(
        checkpoint_path.parent / "teacher_training_subset.json", metadata["fraction_selection"]
    )
    return checkpoint_path


def _validate_metadata(
    cfg: Any,
    metadata: Mapping[str, Any],
    path: Path,
    expected_fraction_selection: Mapping[str, Any] | None = None,
) -> None:
    checks = {
        "teacher_implementation": (metadata.get("teacher_implementation"), "imrnns_pypi"),
        "imrnns_version": (metadata.get("imrnns_version"), IMRNNS_VERSION),
        "dataset": (metadata.get("dataset"), _dataset(cfg)),
        "encoder_model_name": (metadata.get("encoder_model_name"), cfg.model_name),
        "seed": (metadata.get("seed"), int(cfg.seed)),
    }
    mismatches = [
        f"{name}: checkpoint={actual!r}, expected={expected!r}"
        for name, (actual, expected) in checks.items()
        if str(actual) != str(expected)
    ]
    saved_fraction = float(metadata.get("teacher_training_fraction", -1.0))
    if not math.isclose(saved_fraction, _fraction(cfg)):
        mismatches.append(
            f"teacher fraction: checkpoint={saved_fraction}, expected={_fraction(cfg)}"
        )
    if expected_fraction_selection is not None:
        saved_selection = metadata.get("fraction_selection", {})
        for key in (
            "seed",
            "eligible_query_count",
            "selected_query_count",
            "full_ordered_train_qids_sha1",
            "selected_train_qids_sha1",
        ):
            if saved_selection.get(key) != expected_fraction_selection.get(key):
                mismatches.append(
                    f"fraction_selection.{key}: checkpoint={saved_selection.get(key)!r}, "
                    f"expected={expected_fraction_selection.get(key)!r}"
                )
    if mismatches:
        raise ValueError(f"Incompatible IMRNNs checkpoint {path}:\n- " + "\n- ".join(mismatches))


def ensure_teacher_checkpoint(
    cfg: Any,
    split_ids: dict[str, list[str]],
    qrels: dict[str, dict[int, float]],
    query_indices: dict[str, int],
    raw_index: Any,
    document_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    split_source: str,
) -> Path:
    _, _, _, _, _, _, load = _load_api()
    raw_path = str(cfg.teacher_checkpoint_path).strip()
    auto_path = raw_path.lower() in {"", "auto", "__auto_train_teacher__", "auto_train"}
    path = (
        core.default_trained_teacher_checkpoint_path(cfg)
        if auto_path
        else Path(raw_path).expanduser()
    ).resolve()
    if path.is_file():
        model, metadata, missing, unexpected = load(path, model_config=None, device="cpu")
        del model
        if missing or unexpected:
            raise ValueError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
        eligible = _eligible_qids(split_ids, qrels, query_indices)
        full_order, selected = deterministic_fraction_qids(
            eligible, _fraction(cfg), _fraction_seed(cfg)
        )
        if cfg.dry_run:
            selected = selected[:8]
        expected_selection = {
            "seed": _fraction_seed(cfg),
            "eligible_query_count": len(eligible),
            "selected_query_count": len(selected),
            "full_ordered_train_qids_sha1": _sha1_json(full_order),
            "selected_train_qids_sha1": _sha1_json(selected),
        }
        _validate_metadata(cfg, metadata, path, expected_selection)
        cfg.teacher_checkpoint_path = str(path)
        return path
    if not cfg.auto_train_teacher_if_missing or cfg.mode == core.MODE_EVAL_ONLY:
        raise FileNotFoundError(f"Required IMRNNs teacher checkpoint is missing: {path}")
    trained = train_teacher_checkpoint(
        cfg,
        path,
        split_ids,
        qrels,
        query_indices,
        raw_index,
        document_embeddings,
        query_embeddings,
        split_source,
    )
    cfg.teacher_checkpoint_path = str(trained)
    return trained


class TeacherWrapper:
    def __init__(self, cfg: Any) -> None:
        *_, load = _load_api()
        self.cfg = cfg
        self.device = str(cfg.device)
        path = Path(cfg.teacher_checkpoint_path).expanduser().resolve()
        self.model, self.metadata, missing, unexpected = load(
            path, model_config=None, device=self.device
        )
        if missing or unexpected:
            raise ValueError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
        _validate_metadata(cfg, self.metadata, path)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def _score_one(self, query: torch.Tensor, documents: torch.Tensor) -> tuple[torch.Tensor, ...]:
        q_mod, d_mod, scores, details = self.model.forward_with_details(
            query.unsqueeze(0) if query.ndim == 1 else query,
            documents.unsqueeze(0) if documents.ndim == 2 else documents,
        )
        return (
            q_mod.squeeze(0),
            d_mod.squeeze(0),
            scores.squeeze(0),
            details["projected_queries"].squeeze(0),
            details["projected_documents"].squeeze(0),
        )

    def score_batch(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = query_embeddings.to(self.device).float()
        documents = document_embeddings.to(self.device).float()
        if mask is None:
            with torch.no_grad():
                return self.model(queries, documents)[2].detach()
        valid = mask.to(self.device).bool()
        scores = torch.full(valid.shape, -1e9, device=self.device)
        with torch.no_grad():
            for row in range(len(queries)):
                positions = torch.where(valid[row])[0]
                if len(positions):
                    scores[row, positions] = self._score_one(
                        queries[row], documents[row, positions]
                    )[2].float()
        return scores.detach()

    def extract_modulation_signals(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        queries = query_embeddings.to(self.device).float()
        documents = document_embeddings.to(self.device).float()
        valid = mask.to(self.device).bool()
        batch, candidates, _ = documents.shape
        dimension = int(self.model.config.output_dim)
        q_base = torch.zeros(batch, dimension, device=self.device)
        q_mod = torch.zeros_like(q_base)
        d_base = torch.zeros(batch, candidates, dimension, device=self.device)
        d_mod = torch.zeros_like(d_base)
        scores = torch.full((batch, candidates), -1e9, device=self.device)
        with torch.no_grad():
            for row in range(batch):
                positions = torch.where(valid[row])[0]
                if not len(positions):
                    continue
                values = self._score_one(queries[row], documents[row, positions])
                q_mod[row], d_mod[row, positions], scores[row, positions] = values[:3]
                q_base[row], d_base[row, positions] = values[3:]
        return {
            "q_base_T": q_base.detach(),
            "q_mod_T": q_mod.detach(),
            "delta_q_T": (q_mod - q_base).detach(),
            "d_base_T": d_base.detach(),
            "d_mod_T": d_mod.detach(),
            "delta_d_T": (d_mod - d_base).masked_fill(~valid.unsqueeze(-1), 0).detach(),
            "teacher_scores": scores.detach(),
        }

    def rerank_candidate_indices(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        query_indices: list[int],
        candidate_indices: Any,
        max_k: int,
        batch_q: int = 128,
    ) -> np.ndarray:
        del batch_q
        return _rank(
            self.model,
            query_embeddings,
            document_embeddings,
            query_indices,
            candidate_indices,
            max_k,
            self.device,
        )


def install_hooks(document_micro_batch_size: int = 16) -> None:
    core.train_teacher_checkpoint = train_teacher_checkpoint
    core.ensure_teacher_checkpoint = ensure_teacher_checkpoint
    core.TeacherAdapter = TeacherWrapper

    original = core.TrainableRetriever
    if getattr(original, "_rcd_document_chunking", False):
        return

    class MemorySafeStudent(original):  # type: ignore[misc, valid-type]
        _rcd_document_chunking = True

        def encode_doc_texts_train(self, texts: list[str]) -> torch.Tensor:
            if len(texts) <= document_micro_batch_size:
                return original.encode_doc_texts_train(self, texts)
            return torch.cat(
                [
                    original.encode_doc_texts_train(
                        self, texts[start : start + document_micro_batch_size]
                    )
                    for start in range(0, len(texts), document_micro_batch_size)
                ]
            )

    core.TrainableRetriever = MemorySafeStudent
