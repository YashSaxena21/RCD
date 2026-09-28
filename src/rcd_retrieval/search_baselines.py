#!/usr/bin/env python3
"""Margin-MSE and EmbedDistill baselines using Search-Adaptor teachers.

The default suite runs ComLQ, PrivacyQA, and ContractNLI with
deterministic nested 30%, 50%, and 100% Search-Adaptor teacher-training splits.
For every frozen teacher it trains two independent dense-retriever students:

* ``MARGIN_MSE`` consumes Search-Adaptor candidate scores only.
* ``EMBEDDISTILL`` consumes Search-Adaptor final adapted query/document
  representations and candidate scores. It does not consume adaptation deltas.

Search-Adaptor teacher training is supervised by qrels. Student candidate
construction and both student objectives remain qrel-free. All students use the
same full unlabeled training-query split and identical frozen-E5 candidates.

Example:

    rcd-train margin-mse --teacher search-adaptor \
      --dataset_dir datasets/comlq \
      --legalbench_rag_root /path/to/legalbench-rag \
      --search_adaptor_train_fractions 0.30 0.50 1.00 \
      --search_adaptor_datasets comlq privacy_qa contractnli \
      --seeds 42 --ks 1 2 4 8 16 32 64

Progress is durable. Search-Adaptor trials resume from their latest iteration,
students resume from their latest epoch, and completed objective runs are loaded
from ``final_results.json`` without being recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np
import torch

from . import core as base
from . import embeddistill, margin_mse, provenance, search_adaptor
from .runtime import cuda_total_memory_gb

OBJECTIVE_MARGIN_MSE = "margin_mse"
OBJECTIVE_EMBEDDISTILL = "embeddistill"
SUPPORTED_OBJECTIVES = (OBJECTIVE_MARGIN_MSE, OBJECTIVE_EMBEDDISTILL)
DEFAULT_DATASETS = ("comlq", "privacy_qa", "contractnli")

_ORIGINAL_STUDENT = base.TrainableRetriever
_ORIGINAL_INFER_TEACHER_CONFIG = base.infer_teacher_config_from_state


@dataclass(frozen=True)
class DistillationSuiteOptions:
    objectives: Tuple[str, ...] = SUPPORTED_OBJECTIVES
    student_doc_micro_batch_size: int = 16
    margin_pair_strategy: str = margin_mse.MARGIN_PAIR_TOP_VS_REST
    embeddistill_variant: str = embeddistill.EMBEDDISTILL_VARIANT_RANK_EMBED
    embed_distance: str = embeddistill.EMBED_DISTANCE_L2
    embed_query_weight: float = 1.0
    embed_document_weight: float = 1.0
    embed_loss_weight: float = 1.0
    projection_bias: bool = True
    run_tests: bool = False


ACTIVE_SUITE_OPTIONS = DistillationSuiteOptions()


class MemorySafeSearchAdaptorStudent(_ORIGINAL_STUDENT):
    """Chunk trainable document forwards while preserving one complete loss."""

    def encode_doc_texts_train(self, texts: List[str]) -> torch.Tensor:
        chunk_size = int(ACTIVE_SUITE_OPTIONS.student_doc_micro_batch_size)
        if len(texts) <= chunk_size:
            return _ORIGINAL_STUDENT.encode_doc_texts_train(self, texts)
        if not getattr(self, "_logged_search_distillation_doc_chunks", False):
            base.logger.info(
                "Memory-safe document encoding: "
                f"documents_per_forward={chunk_size}; candidate set and objective are unchanged."
            )
            setattr(self, "_logged_search_distillation_doc_chunks", True)
        chunks = [
            _ORIGINAL_STUDENT.encode_doc_texts_train(self, texts[start : start + chunk_size])
            for start in range(0, len(texts), chunk_size)
        ]
        return torch.cat(chunks, dim=0)


def infer_search_adaptor_teacher_config(
    state_dict: Dict[str, torch.Tensor], cfg: base.Cfg
) -> Tuple[int, int, int, float]:
    """Infer Search-Adaptor's final embedding size for EmbedDistill probing."""
    adapter_weights: List[Tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"adapter\.(\d+)\.weight", key)
        if match and value.ndim == 2:
            adapter_weights.append((int(match.group(1)), value))
    if not adapter_weights:
        return _ORIGINAL_INFER_TEACHER_CONFIG(state_dict, cfg)
    adapter_weights.sort(key=lambda item: item[0])
    input_dim = int(adapter_weights[0][1].shape[1])
    hidden_dim = int(adapter_weights[0][1].shape[0])
    output_dim = int(adapter_weights[-1][1].shape[0])
    if input_dim != int(cfg.embedding_dim) or output_dim != int(cfg.embedding_dim):
        raise ValueError(
            "Search-Adaptor residual adapter must preserve the E5 embedding dimension: "
            f"input={input_dim}, output={output_dim}, configured={cfg.embedding_dim}"
        )
    return input_dim, output_dim, hidden_dim, 0.0


def _search_teacher_metadata(cfg: base.Cfg) -> Dict[str, Any]:
    metadata = dict(getattr(cfg, "_search_adaptor_teacher_metadata", {}))
    if not metadata:
        checkpoint = Path(cfg.teacher_checkpoint_path)
        if checkpoint.exists():
            payload = base.safe_torch_load(checkpoint, map_location="cpu", weights_only=False)
            metadata = dict(payload.get("metadata", {}))
    return metadata


def save_margin_mse_search_adaptor_checkpoint(
    student: base.TrainableRetriever,
    checkpoint_dir: str,
    cfg: margin_mse.MarginMseCfg,
    system_name: str,
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
    teacher_metadata = _search_teacher_metadata(cfg)
    metadata = {
        "checkpoint_format": "dense-retriever-student-margin-mse-search-adaptor-v2",
        "checkpoint_kind": checkpoint_kind,
        "variant": system_name,
        "objective": margin_mse.OBJECTIVE_MARGIN_MSE,
        "pair_strategy": cfg.margin_pair_strategy,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_score": best_score,
        "best_val_metrics": best_metrics,
        "patience_ctr": patience_ctr,
        "completed": completed,
        "teacher_type": search_adaptor.SEARCH_ADAPTOR,
        "teacher_training_fraction": teacher_metadata.get("teacher_training_fraction"),
        "teacher_training_uses_qrels": True,
        "teacher_signal_fields": ["teacher_scores"],
        "qrels_used_in_training": False,
        "qrel_positive_injection": False,
        "qrel_loss_active": False,
        "modulation_supervision": False,
        "student_inference_requires_teacher": False,
        "student_model_name": cfg.model_name,
        "teacher_checkpoint_path": cfg.teacher_checkpoint_path,
        "search_adaptor_teacher_metadata": teacher_metadata,
        "candidate_set_fingerprint_sha1": candidate_metadata.get(
            "candidate_set_fingerprint_sha1", ""
        ),
        "candidate_metadata": candidate_metadata,
        "student_document_encode_micro_batch_size": int(
            ACTIVE_SUITE_OPTIONS.student_doc_micro_batch_size
        ),
        "config": asdict(cfg),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata = provenance.add_run_contract(
        metadata,
        cfg=cfg,
        objective=margin_mse.OBJECTIVE_MARGIN_MSE,
        variant=system_name,
        candidate_metadata=candidate_metadata,
        teacher_checkpoint_path=cfg.teacher_checkpoint_path,
    )
    with open(
        os.path.join(checkpoint_dir, "student_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "metadata": metadata,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    base.logger.info(
        f"Saved {checkpoint_kind} {system_name} from Search-Adaptor scores: {checkpoint_dir}"
    )


def save_embeddistill_search_adaptor_checkpoint(
    student: base.TrainableRetriever,
    projection: embeddistill.SharedLinearProjection,
    checkpoint_dir: str,
    cfg: embeddistill.EmbedDistillCfg,
    system_name: str,
    epoch: int,
    best_score: float,
    best_metrics: Dict[int, Dict[str, float]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    teacher_checkpoint_fingerprint_sha1: str,
    teacher_output_dim: int,
    teacher_metadata: Mapping[str, Any],
    patience_ctr: int = 0,
    completed: bool = False,
    checkpoint_kind: str = "best",
    candidate_metadata: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    student.save(checkpoint_dir)
    candidate_metadata = candidate_metadata or {}
    search_metadata = dict(teacher_metadata) or _search_teacher_metadata(cfg)
    metadata = {
        "checkpoint_format": "dense-retriever-student-embeddistill-search-adaptor-v2",
        "checkpoint_kind": checkpoint_kind,
        "variant": system_name,
        "objective": embeddistill.OBJECTIVE_EMBEDDISTILL,
        "embeddistill_variant": cfg.embeddistill_variant,
        "embedding_distance": cfg.embed_distance,
        "embedding_weights": {
            "query": cfg.embed_query_weight,
            "document": cfg.embed_document_weight,
            "embedding_loss_weight": cfg.embed_loss_weight,
        },
        "teacher_type": search_adaptor.SEARCH_ADAPTOR,
        "teacher_training_fraction": search_metadata.get("teacher_training_fraction"),
        "teacher_training_uses_qrels": True,
        "teacher_target_fields": list(embeddistill.USED_TEACHER_TARGET_FIELDS),
        "score_distillation_enabled": (
            cfg.embeddistill_variant == embeddistill.EMBEDDISTILL_VARIANT_RANK_EMBED
        ),
        "teacher_checkpoint_path": cfg.teacher_checkpoint_path,
        "teacher_checkpoint_fingerprint_sha1": teacher_checkpoint_fingerprint_sha1,
        "search_adaptor_teacher_metadata": search_metadata,
        "teacher_output_dim": int(teacher_output_dim),
        "projection": {
            "architecture": "shared_linear",
            "input_dim": int(cfg.embedding_dim),
            "output_dim": int(teacher_output_dim),
            "bias": bool(projection.bias),
            "training_only": True,
        },
        "delta_supervision": False,
        "query_generation": False,
        "teacher_document_inheritance": False,
        "qrels_used_in_training": False,
        "qrel_positive_injection": False,
        "qrel_loss_active": False,
        "student_inference_requires_teacher": False,
        "candidate_metadata": candidate_metadata,
        "dataset_name": cfg.dataset_name,
        "dataset_format": cfg.source_dataset_format or cfg.dataset_format,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_score": best_score,
        "best_val_metrics": best_metrics,
        "patience_ctr": patience_ctr,
        "completed": completed,
        "student_model_name": cfg.model_name,
        "student_document_encode_micro_batch_size": int(
            ACTIVE_SUITE_OPTIONS.student_doc_micro_batch_size
        ),
        "config": asdict(cfg),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "adaptation_note": (
            "EmbedDistill-style matching of Search-Adaptor final embeddings. The shared "
            "residual Search-Adaptor transforms each query and document independently; "
            "teacher deltas are not consumed, and the projection is training-only."
        ),
    }
    metadata = provenance.add_run_contract(
        metadata,
        cfg=cfg,
        objective=embeddistill.OBJECTIVE_EMBEDDISTILL,
        variant=system_name,
        candidate_metadata=candidate_metadata,
        teacher_checkpoint_path=cfg.teacher_checkpoint_path,
    )
    with open(
        os.path.join(checkpoint_dir, "student_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "projection_head_state": projection.state_dict(),
            "metadata": metadata,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    base.logger.info(
        f"Saved {checkpoint_kind} {system_name} from Search-Adaptor embeddings: {checkpoint_dir}"
    )


def install_integration_hooks() -> None:
    search_adaptor.install_pipeline_hooks()
    base.TrainableRetriever = MemorySafeSearchAdaptorStudent
    base.infer_teacher_config_from_state = infer_search_adaptor_teacher_config
    margin_mse.save_margin_checkpoint = save_margin_mse_search_adaptor_checkpoint
    embeddistill.save_embeddistill_checkpoint = save_embeddistill_search_adaptor_checkpoint


CfgT = TypeVar("CfgT", margin_mse.MarginMseCfg, embeddistill.EmbedDistillCfg)


def _convert_cfg(cls: Type[CfgT], source: base.Cfg, overrides: Mapping[str, Any]) -> CfgT:
    source_values = asdict(source)
    allowed = {field.name for field in fields(cls)}
    values = {key: value for key, value in source_values.items() if key in allowed}
    values.update({key: value for key, value in overrides.items() if key in allowed})
    return cls(**values)


def _fraction_tag(fraction: float) -> str:
    return f"fraction_{int(round(float(fraction) * 100.0)):03d}"


def _teacher_path(fraction_dir: Path, dataset_name: str, fraction: float, seed: int) -> Path:
    filename = (
        f"search-adaptor-{dataset_name}-fraction-{int(round(fraction * 100.0)):03d}-seed-{seed}.pt"
    )
    return fraction_dir / "teacher" / f"seed_{seed}" / filename


def _common_overrides(
    *,
    dataset_cfg: base.Cfg,
    output_dir: Path,
    teacher_path: Path,
    shared_cache: Path,
    seed: int,
    fraction: float,
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "seeds": (int(seed),),
        "output_dir": str(output_dir),
        "log_path": "",
        "teacher_checkpoint_path": str(teacher_path),
        "auto_train_teacher_if_missing": True,
        "candidate_source": "frozen_base_retriever",
        "dynamic_refresh": False,
        "train_queries_file": "",
        "frozen_corpus_emb_path": str(shared_cache / "frozen_base_corpus.pt"),
        "frozen_query_emb_path": str(shared_cache / "frozen_base_queries.pt"),
        "signal_projection_output_dim": int(dataset_cfg.embedding_dim),
    }


def make_margin_cfg(
    dataset_cfg: base.Cfg,
    output_dir: Path,
    teacher_path: Path,
    shared_cache: Path,
    seed: int,
    fraction: float,
) -> margin_mse.MarginMseCfg:
    overrides = _common_overrides(
        dataset_cfg=dataset_cfg,
        output_dir=output_dir,
        teacher_path=teacher_path,
        shared_cache=shared_cache,
        seed=seed,
        fraction=fraction,
    )
    overrides.update(
        {
            "margin_pair_strategy": ACTIVE_SUITE_OPTIONS.margin_pair_strategy,
            "run_loss_tests": False,
        }
    )
    cfg = _convert_cfg(margin_mse.MarginMseCfg, dataset_cfg, overrides)
    setattr(cfg, "_search_adaptor_fraction", float(fraction))
    margin_mse.validate_margin_cfg(cfg)
    return cfg


def make_embeddistill_cfg(
    dataset_cfg: base.Cfg,
    output_dir: Path,
    teacher_path: Path,
    shared_cache: Path,
    seed: int,
    fraction: float,
) -> embeddistill.EmbedDistillCfg:
    overrides = _common_overrides(
        dataset_cfg=dataset_cfg,
        output_dir=output_dir,
        teacher_path=teacher_path,
        shared_cache=shared_cache,
        seed=seed,
        fraction=fraction,
    )
    overrides.update(
        {
            "teacher_checkpoint_template": "",
            "allow_teacher_metadata_mismatch": False,
            "embeddistill_variant": ACTIVE_SUITE_OPTIONS.embeddistill_variant,
            "embed_distance": ACTIVE_SUITE_OPTIONS.embed_distance,
            "embed_query_weight": ACTIVE_SUITE_OPTIONS.embed_query_weight,
            "embed_document_weight": ACTIVE_SUITE_OPTIONS.embed_document_weight,
            "embed_loss_weight": ACTIVE_SUITE_OPTIONS.embed_loss_weight,
            "projection_bias": ACTIVE_SUITE_OPTIONS.projection_bias,
            "run_loss_tests": False,
        }
    )
    cfg = _convert_cfg(embeddistill.EmbedDistillCfg, dataset_cfg, overrides)
    setattr(cfg, "_search_adaptor_fraction", float(fraction))
    embeddistill.validate_embeddistill_cfg(cfg)
    return cfg


def _suite_metadata(
    *,
    objective: str,
    dataset_name: str,
    fraction: float,
    seed: int,
    teacher_path: Path,
    cfg: base.Cfg,
) -> Dict[str, Any]:
    configuration_contract = {
        "training_config": asdict(cfg),
        "search_adaptor_options": asdict(search_adaptor.ACTIVE_OPTIONS),
        "distillation_suite_options": asdict(ACTIVE_SUITE_OPTIONS),
    }
    return {
        "entry_point": Path(__file__).name,
        "objective": objective,
        "dataset": dataset_name,
        "teacher": search_adaptor.SEARCH_ADAPTOR,
        "teacher_training_fraction": float(fraction),
        "teacher_checkpoint_path": str(teacher_path),
        "teacher_training_uses_qrels": True,
        "student_training_uses_qrels": False,
        "student_uses_full_unlabeled_train_split": True,
        "candidate_source": "frozen_base_retriever",
        "dynamic_refresh": False,
        "seed": int(seed),
        "dry_run": bool(cfg.dry_run),
        "configuration_fingerprint_sha1": provenance.sha1_jsonable(configuration_contract),
        "completed": True,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def _load_completed_payload(
    output_dir: Path,
    expected: Mapping[str, Any],
    resume: bool,
) -> Optional[Dict[str, Any]]:
    path = output_dir / "final_results.json"
    if not resume or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("search_adaptor_distillation_suite", {})
        identity_keys = (
            "objective",
            "dataset",
            "teacher_training_fraction",
            "seed",
            "dry_run",
        )
        same_identity = all(metadata.get(key) == expected.get(key) for key in identity_keys)
        if same_identity and metadata.get("configuration_fingerprint_sha1") != expected.get(
            "configuration_fingerprint_sha1"
        ):
            raise ValueError(
                "A completed run exists with different configuration settings. Use a new "
                "--output_dir or pass --no_resume to retrain this objective explicitly."
            )
        teacher_path = Path(str(metadata.get("teacher_checkpoint_path", "")))
        saved_teacher_sha1 = str(metadata.get("teacher_checkpoint_fingerprint_sha1", ""))
        if same_identity and (not teacher_path.is_file()):
            base.logger.warning(
                f"Completed result exists but its teacher checkpoint is missing: {teacher_path}"
            )
            return None
        if same_identity and saved_teacher_sha1:
            current_teacher_sha1 = provenance.file_sha1(teacher_path)
            if current_teacher_sha1 != saved_teacher_sha1:
                raise ValueError(
                    "The Search-Adaptor teacher checkpoint changed after this student result "
                    f"was produced: {teacher_path}"
                )
        keys = (*identity_keys, "configuration_fingerprint_sha1")
        if all(metadata.get(key) == expected.get(key) for key in keys) and metadata.get(
            "completed", False
        ):
            base.logger.info(f"Skipping completed objective run: {output_dir}")
            return payload
    except ValueError:
        raise
    except Exception as exc:
        base.logger.warning(f"Could not reuse completed payload {path}: {exc}")
    return None


def _save_completed_payload(
    output_dir: Path, payload: Dict[str, Any], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["search_adaptor_distillation_suite"] = metadata
    with (output_dir / "final_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def _candidate_fingerprint(payload: Mapping[str, Any], objective: str) -> str:
    metadata = payload.get("candidate_metadata", {})
    if objective == OBJECTIVE_MARGIN_MSE:
        return str(metadata.get("candidate_set_fingerprint_sha1", ""))
    return str(metadata.get("candidate_index_fingerprint_sha1", ""))


def assert_and_write_candidate_fairness(
    fraction_dir: Path,
    seed: int,
    margin_payload: Mapping[str, Any],
    embed_payload: Mapping[str, Any],
) -> None:
    margin_fp = _candidate_fingerprint(margin_payload, OBJECTIVE_MARGIN_MSE)
    embed_fp = _candidate_fingerprint(embed_payload, OBJECTIVE_EMBEDDISTILL)
    if not margin_fp or not embed_fp:
        if not (margin_payload.get("dry_run") or embed_payload.get("dry_run")):
            raise AssertionError("Missing candidate fingerprint for objective fairness check")
    elif margin_fp != embed_fp:
        raise AssertionError(
            "Margin-MSE and EmbedDistill used different ordered student candidate sets: "
            f"margin={margin_fp}, embeddistill={embed_fp}"
        )
    report = {
        "seed": int(seed),
        "margin_mse_candidate_fingerprint": margin_fp,
        "embeddistill_candidate_fingerprint": embed_fp,
        "identical_ordered_candidate_sets": bool(margin_fp and margin_fp == embed_fp),
        "candidate_source": "frozen_base_retriever",
        "qrel_positive_injection": False,
    }
    with (fraction_dir / f"candidate_fairness_seed_{seed}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def _set_state_payload(
    state: Dict[str, Any],
    dataset: str,
    fraction_tag: str,
    objective: str,
    seed: int,
    payload: Dict[str, Any],
) -> None:
    state.setdefault(dataset, {}).setdefault(fraction_tag, {}).setdefault(objective, {})[
        str(seed)
    ] = payload


def write_suite_results(output_dir: Path, state: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results_by_dataset_fraction_objective.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(state, handle, indent=2, sort_keys=True)

    rows: List[Dict[str, Any]] = []
    for dataset, by_fraction in state.items():
        for fraction_tag, by_objective in by_fraction.items():
            fraction = float(fraction_tag.split("_", 1)[1]) / 100.0
            for objective, by_seed in by_objective.items():
                for seed, payload in by_seed.items():
                    for result in base.flatten_results(payload.get("results", {})):
                        rows.append(
                            {
                                "dataset": dataset,
                                "teacher_train_fraction": fraction,
                                "objective": objective,
                                "seed": int(seed),
                                **result,
                            }
                        )
    base.write_csv_rows(str(output_dir / "results_by_dataset_fraction_objective.csv"), rows)

    excluded = {
        "dataset",
        "teacher_train_fraction",
        "objective",
        "seed",
        "system",
        "slice",
        "k",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and isinstance(value, (int, float))
        }
    )
    grouped: Dict[Tuple[str, float, str, str, str, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        group_key = (
            str(row["dataset"]),
            float(row["teacher_train_fraction"]),
            str(row["objective"]),
            str(row["system"]),
            str(row["slice"]),
            int(row["k"]),
        )
        for metric in metric_names:
            if metric in row:
                grouped[group_key][metric].append(float(row[metric]))
    aggregate_rows: List[Dict[str, Any]] = []
    for key, values_by_metric in sorted(grouped.items()):
        dataset, fraction, objective, system, slice_name, k = key
        aggregate: Dict[str, Any] = {
            "dataset": dataset,
            "teacher_train_fraction": fraction,
            "objective": objective,
            "system": system,
            "slice": slice_name,
            "k": k,
        }
        for metric, values in values_by_metric.items():
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
        aggregate_rows.append(aggregate)
    base.write_csv_rows(
        str(output_dir / "aggregate_results_by_dataset_fraction_objective.csv"),
        aggregate_rows,
    )
    with (output_dir / "aggregate_results_by_dataset_fraction_objective.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate_rows, handle, indent=2, sort_keys=True)


def _load_suite_state(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "results_by_dataset_fraction_objective.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except Exception as exc:
        base.logger.warning(f"Could not load prior suite state {path}: {exc}")
        return {}


def _run_objective(
    objective: str,
    cfg: base.Cfg,
    output_dir: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    completed = _load_completed_payload(output_dir, metadata, cfg.resume)
    if completed is not None:
        return completed
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "suite_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({**metadata, "config": asdict(cfg)}, handle, indent=2, sort_keys=True)
    if objective == OBJECTIVE_MARGIN_MSE:
        payload = margin_mse.run_one_seed(cfg)  # type: ignore[arg-type]
    elif objective == OBJECTIVE_EMBEDDISTILL:
        payload = embeddistill.run_one_seed(cfg)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    teacher_path = Path(cfg.teacher_checkpoint_path)
    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Expected completed Search-Adaptor checkpoint after training: {teacher_path}"
        )
    metadata["teacher_checkpoint_fingerprint_sha1"] = provenance.file_sha1(teacher_path)
    return _save_completed_payload(output_dir, payload, metadata)


def run_suite(cfg: base.Cfg) -> None:
    install_integration_hooks()
    cfg.dataset_format = "comlq"
    cfg.auto_train_teacher_if_missing = True
    cfg.signal_projection_output_dim = int(cfg.embedding_dim)
    if cfg.train_queries_file:
        raise ValueError(
            "--train_queries_file is unsupported: every objective and teacher fraction must "
            "use the same full unlabeled student training split."
        )
    total_gb = cuda_total_memory_gb(cfg)
    if total_gb is not None:
        device_index = torch.cuda.current_device()
        print(
            f"GPU: {torch.cuda.get_device_name(device_index)} | total_memory={total_gb:.2f} GiB",
            flush=True,
        )
        if total_gb <= 16.0:
            cfg.eval_batch_size = min(cfg.eval_batch_size, 8)
            cfg.corpus_encode_batch_size = min(cfg.corpus_encode_batch_size, 8)
    suite_output = Path(cfg.output_dir)
    suite_output.mkdir(parents=True, exist_ok=True)
    state = _load_suite_state(suite_output) if cfg.resume else {}

    for raw_dataset in search_adaptor.ACTIVE_OPTIONS.datasets:
        dataset_name = search_adaptor._normalize_dataset_name(raw_dataset)
        dataset_cfg = search_adaptor._dataset_cfg(cfg, dataset_name, suite_output)
        shared_cache = suite_output / dataset_name / "shared_cache"
        for fraction in search_adaptor.ACTIVE_OPTIONS.train_fractions:
            fraction_tag = _fraction_tag(fraction)
            fraction_dir = suite_output / dataset_name / fraction_tag
            fraction_dir.mkdir(parents=True, exist_ok=True)
            objective_payloads: Dict[str, List[Dict[str, Any]]] = {
                objective: [] for objective in ACTIVE_SUITE_OPTIONS.objectives
            }
            for seed in dataset_cfg.seeds:
                seed = int(seed)
                teacher_path = _teacher_path(fraction_dir, dataset_name, float(fraction), seed)
                current_payloads: Dict[str, Dict[str, Any]] = {}
                for objective in ACTIVE_SUITE_OPTIONS.objectives:
                    objective_dir = fraction_dir / objective / f"seed_{seed}"
                    if objective == OBJECTIVE_MARGIN_MSE:
                        objective_cfg: base.Cfg = make_margin_cfg(
                            dataset_cfg,
                            objective_dir,
                            teacher_path,
                            shared_cache,
                            seed,
                            float(fraction),
                        )
                    else:
                        objective_cfg = make_embeddistill_cfg(
                            dataset_cfg,
                            objective_dir,
                            teacher_path,
                            shared_cache,
                            seed,
                            float(fraction),
                        )
                    metadata = _suite_metadata(
                        objective=objective,
                        dataset_name=dataset_name,
                        fraction=float(fraction),
                        seed=seed,
                        teacher_path=teacher_path,
                        cfg=objective_cfg,
                    )
                    payload = _run_objective(objective, objective_cfg, objective_dir, metadata)
                    current_payloads[objective] = payload
                    objective_payloads[objective].append(payload)
                    _set_state_payload(state, dataset_name, fraction_tag, objective, seed, payload)
                    write_suite_results(suite_output, state)
                    base.clear_cuda_cache_if_needed(objective_cfg)

                if all(
                    objective in current_payloads
                    for objective in (OBJECTIVE_MARGIN_MSE, OBJECTIVE_EMBEDDISTILL)
                ):
                    assert_and_write_candidate_fairness(
                        fraction_dir,
                        seed,
                        current_payloads[OBJECTIVE_MARGIN_MSE],
                        current_payloads[OBJECTIVE_EMBEDDISTILL],
                    )

            for objective, per_seed in objective_payloads.items():
                base.aggregate_seed_results(str(fraction_dir / objective), per_seed)
            write_suite_results(suite_output, state)
    print(f"Saved Search-Adaptor distillation suite under {suite_output}", flush=True)


def run_suite_tests() -> None:
    global ACTIVE_SUITE_OPTIONS
    search_adaptor.run_search_adaptor_tests()
    print("Search-Adaptor objective tests passed", flush=True)
    margin_mse.run_margin_mse_loss_tests()
    print("Margin-MSE objective tests passed", flush=True)
    embeddistill.run_embeddistill_loss_tests()
    print("EmbedDistill objective tests passed", flush=True)

    synthetic_state = {
        "adapter.0.weight": torch.randn(5, 4),
        "adapter.0.bias": torch.randn(5),
        "adapter.2.weight": torch.randn(4, 5),
        "adapter.2.bias": torch.randn(4),
    }
    synthetic_cfg = base.Cfg(embedding_dim=4)
    assert infer_search_adaptor_teacher_config(synthetic_state, synthetic_cfg)[:3] == (
        4,
        4,
        5,
    )

    synthetic_teacher = search_adaptor.SearchAdaptorTeacherWrapper.__new__(
        search_adaptor.SearchAdaptorTeacherWrapper
    )
    synthetic_teacher.device = "cpu"
    synthetic_teacher.model = search_adaptor.SearchAdaptorModel(4, 5, 2, "relu")
    synthetic_teacher.model.eval()
    for parameter in synthetic_teacher.model.parameters():
        parameter.requires_grad = False
    query_embeddings = torch.randn(2, 4)
    document_embeddings = torch.randn(2, 3, 4)
    document_mask = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)
    teacher_scores = synthetic_teacher.score_batch(
        query_embeddings, document_embeddings, document_mask
    )
    assert teacher_scores.shape == (2, 3)
    assert float(teacher_scores[0, 2]) == -1e9
    final_targets = embeddistill.extract_final_teacher_embeddings_and_scores(
        synthetic_teacher, query_embeddings, document_embeddings, document_mask
    )
    assert final_targets.teacher_query_embedding.shape == (2, 4)
    assert final_targets.teacher_document_embeddings.shape == (2, 3, 4)
    assert final_targets.teacher_scores.shape == (2, 3)
    assert final_targets.teacher_output_dim == 4
    assert all(parameter.grad is None for parameter in synthetic_teacher.model.parameters())

    original_encode = _ORIGINAL_STUDENT.encode_doc_texts_train
    original_options = ACTIVE_SUITE_OPTIONS
    calls: List[List[str]] = []

    def fake_encode(student: Any, texts: List[str]) -> torch.Tensor:
        calls.append(list(texts))
        values = torch.tensor(
            [float(text.removeprefix("d")) for text in texts], dtype=torch.float32
        ).unsqueeze(-1)
        return values * student.synthetic_scale

    try:
        _ORIGINAL_STUDENT.encode_doc_texts_train = fake_encode
        ACTIVE_SUITE_OPTIONS = replace(ACTIVE_SUITE_OPTIONS, student_doc_micro_batch_size=2)
        student = MemorySafeSearchAdaptorStudent.__new__(MemorySafeSearchAdaptorStudent)
        torch.nn.Module.__init__(student)
        student.synthetic_scale = torch.nn.Parameter(torch.tensor(1.0))
        output = student.encode_doc_texts_train(["d0", "d1", "d2", "d3", "d4"])
        assert calls == [["d0", "d1"], ["d2", "d3"], ["d4"]]
        assert torch.equal(output.detach().squeeze(-1), torch.arange(5).float())
        output.sum().backward()
        assert student.synthetic_scale.grad is not None
        assert float(student.synthetic_scale.grad) == 10.0
    finally:
        _ORIGINAL_STUDENT.encode_doc_texts_train = original_encode
        ACTIVE_SUITE_OPTIONS = original_options

    margin_payload = {
        "candidate_metadata": {"candidate_set_fingerprint_sha1": "same"},
        "dry_run": False,
    }
    embed_payload = {
        "candidate_metadata": {"candidate_index_fingerprint_sha1": "same"},
        "dry_run": False,
    }
    assert _candidate_fingerprint(margin_payload, OBJECTIVE_MARGIN_MSE) == (
        _candidate_fingerprint(embed_payload, OBJECTIVE_EMBEDDISTILL)
    )
    print("Search-Adaptor distillation suite tests passed", flush=True)


def _suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--distillation_objectives",
        nargs="+",
        choices=list(SUPPORTED_OBJECTIVES),
        default=list(SUPPORTED_OBJECTIVES),
    )
    parser.add_argument(
        "--student_doc_micro_batch_size",
        type=int,
        default=16,
        help=(
            "Candidate documents encoded per trainable-E5 forward. This changes only "
            "memory usage, not candidates or losses. Use 1-2 for 11 GB GPUs."
        ),
    )
    parser.add_argument(
        "--margin_pair_strategy",
        choices=[
            margin_mse.MARGIN_PAIR_TOP_VS_REST,
            margin_mse.MARGIN_PAIR_ALL_PAIRS,
        ],
        default=margin_mse.MARGIN_PAIR_TOP_VS_REST,
    )
    parser.add_argument(
        "--embeddistill_variant",
        choices=[
            embeddistill.EMBEDDISTILL_VARIANT_RANK_EMBED,
            embeddistill.EMBEDDISTILL_VARIANT_EMBED_ONLY,
        ],
        default=embeddistill.EMBEDDISTILL_VARIANT_RANK_EMBED,
    )
    parser.add_argument(
        "--embed_distance",
        choices=[
            embeddistill.EMBED_DISTANCE_L2,
            embeddistill.EMBED_DISTANCE_COSINE,
            embeddistill.EMBED_DISTANCE_MSE,
        ],
        default=embeddistill.EMBED_DISTANCE_L2,
    )
    parser.add_argument("--embed_query_weight", type=float, default=1.0)
    parser.add_argument("--embed_document_weight", type=float, default=1.0)
    parser.add_argument("--embed_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--projection_bias", dest="projection_bias", action="store_true", default=True
    )
    parser.add_argument("--no_projection_bias", dest="projection_bias", action="store_false")
    parser.add_argument("--run_suite_tests", action="store_true")
    return parser


@contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterable[None]:
    previous = sys.argv
    sys.argv = [previous[0], *argv]
    try:
        yield
    finally:
        sys.argv = previous


def parse_args() -> Tuple[base.Cfg, search_adaptor.SearchAdaptorOptions, DistillationSuiteOptions]:
    parser = _suite_parser()
    custom, remaining = parser.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        print("\nSearch-Adaptor distillation-suite arguments:\n")
        print(parser.format_help())
    with _temporary_argv(remaining):
        cfg, search_options = search_adaptor.parse_args()
    if "--output_dir" not in remaining:
        cfg.output_dir = "runs/search_adaptor_distillation_baselines"
    options = DistillationSuiteOptions(
        objectives=tuple(custom.distillation_objectives),
        student_doc_micro_batch_size=int(custom.student_doc_micro_batch_size),
        margin_pair_strategy=str(custom.margin_pair_strategy),
        embeddistill_variant=str(custom.embeddistill_variant),
        embed_distance=str(custom.embed_distance),
        embed_query_weight=float(custom.embed_query_weight),
        embed_document_weight=float(custom.embed_document_weight),
        embed_loss_weight=float(custom.embed_loss_weight),
        projection_bias=bool(custom.projection_bias),
        run_tests=bool(custom.run_suite_tests),
    )
    if len(set(search_options.train_fractions)) != len(search_options.train_fractions):
        raise ValueError("Search-Adaptor training fractions must be unique")
    if options.student_doc_micro_batch_size < 1:
        raise ValueError("--student_doc_micro_batch_size must be positive")
    if not options.objectives:
        raise ValueError("--distillation_objectives cannot be empty")
    if options.embed_query_weight < 0.0 or options.embed_document_weight < 0.0:
        raise ValueError("EmbedDistill query/document weights must be non-negative")
    if options.embed_loss_weight <= 0.0:
        raise ValueError("--embed_loss_weight must be positive")
    return cfg, search_options, options


def main() -> None:
    global ACTIVE_SUITE_OPTIONS
    cfg, search_options, ACTIVE_SUITE_OPTIONS = parse_args()
    search_adaptor.ACTIVE_OPTIONS = search_options
    install_integration_hooks()
    if ACTIVE_SUITE_OPTIONS.run_tests:
        run_suite_tests()
        return
    run_suite(cfg)


if __name__ == "__main__":
    main()
