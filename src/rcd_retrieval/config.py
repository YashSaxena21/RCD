"""Shared experiment configuration and public method identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

MODE_UNSUPERVISED = "UNSUPERVISED"
MODE_EVAL_ONLY = "EVAL_ONLY"

VARIANT_E5_BASE = "E5_BASE"
VARIANT_IMRNN_TEACHER = "IMRNN_TEACHER"
VARIANT_RANKING_ONLY = "RANKING_ONLY"
VARIANT_CORRECTION_ONLY = "CORRECTION_ONLY"
VARIANT_FULL_RCD = "FULL_RCD"


@dataclass
class Cfg:
    dataset_format: str = "auto"
    dataset_dir: str = "datasets/comlq/dataset"
    corpus_file: str = "corpus.jsonl"
    queries_file: str = "queries.jsonl"
    qrels_dir: str = "qrels"
    legalbench_rag_root: str = ""
    legalbench_rag_datasets: Tuple[str, ...] = ("privacy_qa", "contractnli")
    legalbench_prepared_dir: str = ""
    legalbench_extract_dir: str = ""
    legalbench_chunk_size: int = 500
    legalbench_chunk_strategy: str = "naive"
    legalbench_metadata_path: str = ""
    dataset_name: str = ""
    source_dataset_format: str = ""

    model_name: str = "intfloat/e5-large-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    embedding_dim: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    teacher_checkpoint_path: str = "auto_train"
    teacher_checkpoint_template: str = ""
    teacher_output_dim: Optional[int] = None
    teacher_hidden_dim: Optional[int] = None
    teacher_dropout: float = 0.1
    auto_train_teacher_if_missing: bool = True
    teacher_train_epochs: int = 10
    teacher_train_batch_size: int = 32
    teacher_train_lr: float = 1e-4
    teacher_train_weight_decay: float = 1e-5
    teacher_train_num_negatives: int = 20
    teacher_train_negative_pool: int = 200
    teacher_loss: str = "listnet"
    teacher_label_temp: float = 1.0
    teacher_max_positives: int = 0
    teacher_max_candidates: int = 0
    teacher_metric_select: bool = True
    teacher_select_recall_k: int = 50
    teacher_select_ndcg_k: int = 10
    teacher_select_mrr_k: int = 10
    teacher_recall_w: float = 1.0
    teacher_ndcg_w: float = 0.25
    teacher_mrr_w: float = 0.10
    allow_weak_teacher: bool = False

    mode: str = MODE_UNSUPERVISED
    resume: bool = True
    variants: Tuple[str, ...] = (VARIANT_FULL_RCD,)
    seeds: Tuple[int, ...] = (42,)
    train_queries_file: str = ""
    dry_run: bool = False

    seed: int = 42
    batch_size: int = 1
    micro_batch_size: int = 1
    num_workers: int = 0
    eval_batch_size: int = 32
    corpus_encode_batch_size: int = 64
    gradient_checkpointing: bool = True
    student_max_seq_length: int = 0
    max_train_candidates_per_query: int = 32
    l40s_safe_mode: bool = True
    epochs: int = 5
    optimizer: str = "auto"
    auto_adafactor_gpu_gb: float = 16.0
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 3
    min_delta: float = 1e-4

    val_ratio: float = 0.10
    test_ratio: float = 0.10
    query_type_filter: str = "all"
    query_types: str = ""
    report_comlq_slices: bool = True

    use_graded_relevance: bool = True
    tau: float = 1.0
    alpha: float = 1.0
    beta_q: float = 0.1
    beta_d: float = 0.3
    delta_eps: float = 1e-6
    signal_projection_output_dim: int = 256
    projection_head_weight_decay: float = 0.0

    candidate_pool_k: int = 64
    feedback_k: int = 100
    ks: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    eval_k: int = 64
    selection_recall_w: float = 1.0
    selection_ndcg_w: float = 0.25
    selection_mrr_w: float = 0.10

    frozen_corpus_emb_path: str = "base_corpus_embeddings.pt"
    frozen_query_emb_path: str = "base_query_embeddings.pt"
    force_rebuild_embedding_cache: bool = False

    log_path: str = ""
    output_dir: str = "runs/rcd"
    debug_teacher_samples: int = 5

