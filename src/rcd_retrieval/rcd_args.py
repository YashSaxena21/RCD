"""Command-line arguments for the RCD training pipeline."""

from __future__ import annotations

import argparse

import torch

from .core import (
    MODE_EVAL_ONLY,
    MODE_UNSUPERVISED,
    VARIANT_FULL_RCD,
    Cfg,
)
from .legalbench import DATASETS as LEGALBENCH_RAG_DATASETS


def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description="Representation Correction Distillation for dense retrievers"
    )
    parser.add_argument(
        "--dataset_format",
        choices=["auto", "standard", "comlq", "legalbench_rag"],
        default="auto",
    )
    parser.add_argument("--dataset_dir", default="datasets/comlq/dataset")
    parser.add_argument("--corpus_file", default="corpus.jsonl")
    parser.add_argument("--queries_file", default="queries.jsonl")
    parser.add_argument("--qrels_dir", default="qrels")
    parser.add_argument("--train_queries_file", default="")
    parser.add_argument(
        "--legalbench_rag_root",
        default="",
        help="Root containing LegalBench-RAG corpus/ and benchmarks/. Defaults to --dataset_dir when dataset_format=legalbench_rag.",
    )
    parser.add_argument(
        "--legalbench_rag_datasets",
        nargs="+",
        default=list(LEGALBENCH_RAG_DATASETS),
        help="LegalBench-RAG datasets to run: privacy_qa contractnli.",
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
    parser.add_argument(
        "--query_type_filter",
        choices=["all", "negation", "conjunction", "union", "projection", "custom"],
        default="all",
    )
    parser.add_argument("--query_types", default="")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--use_binary_relevance", action="store_true")
    parser.add_argument("--no_report_comlq_slices", action="store_true")

    parser.add_argument(
        "--teacher_checkpoint_path",
        default="auto_train",
        help="Frozen IMRNNS checkpoint path. The default trains a dataset- and seed-specific teacher.",
    )
    parser.add_argument(
        "--teacher_checkpoint_template",
        default="",
        help="Dataset/seed-specific checkpoint template with optional {dataset} and {seed} fields.",
    )
    parser.add_argument("--teacher_output_dim", type=int)
    parser.add_argument("--teacher_hidden_dim", type=int)
    parser.add_argument("--teacher_dropout", type=float, default=0.1)
    parser.add_argument(
        "--no_auto_train_teacher_if_missing",
        action="store_true",
        help="Fail instead of training an original adapter-only IMRNN teacher when --teacher_checkpoint_path is missing.",
    )
    parser.add_argument("--teacher_train_epochs", type=int, default=10)
    parser.add_argument("--teacher_train_batch_size", type=int, default=32)
    parser.add_argument("--teacher_train_lr", type=float, default=1e-4)
    parser.add_argument("--teacher_train_weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--teacher_train_num_negatives",
        "--teacher_num_negatives",
        dest="teacher_train_num_negatives",
        type=int,
        default=20,
    )
    parser.add_argument("--teacher_train_negative_pool", type=int, default=200)
    parser.add_argument("--teacher_label_temp", type=float, default=1.0)
    parser.add_argument(
        "--teacher_max_positives",
        type=int,
        default=0,
        help="0 keeps all qrel positives for teacher listwise training.",
    )
    parser.add_argument(
        "--teacher_max_candidates",
        type=int,
        default=0,
        help="0 means no cap except all positives plus teacher negatives; positive docs are preserved and this cap reduces negatives first.",
    )
    parser.add_argument(
        "--teacher_metric_select", dest="teacher_metric_select", action="store_true", default=True
    )
    parser.add_argument(
        "--no_teacher_metric_select", dest="teacher_metric_select", action="store_false"
    )
    parser.add_argument("--teacher_select_recall_k", type=int, default=50)
    parser.add_argument("--teacher_select_ndcg_k", type=int, default=10)
    parser.add_argument("--teacher_select_mrr_k", type=int, default=10)
    parser.add_argument("--teacher_recall_w", type=float, default=1.0)
    parser.add_argument("--teacher_ndcg_w", type=float, default=0.25)
    parser.add_argument("--teacher_mrr_w", type=float, default=0.10)
    parser.add_argument("--allow_weak_teacher", action="store_true")

    parser.add_argument("--model_name", default="intfloat/e5-large-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--student_max_seq_length", type=int, default=0)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    parser.add_argument(
        "--mode", choices=[MODE_UNSUPERVISED, MODE_EVAL_ONLY], default=MODE_UNSUPERVISED
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume variants from latest checkpoints; enabled by default.",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and start trainable variants from scratch.",
    )
    parser.add_argument("--variants", nargs="+", default=[VARIANT_FULL_RCD])
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build data and run one batch through each selected variant loss without training.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--corpus_encode_batch_size", type=int, default=64)
    parser.add_argument(
        "--optimizer",
        choices=["auto", "adamw", "adafactor"],
        default="auto",
        help="Student optimizer. auto uses Adafactor on small CUDA devices to reduce memory.",
    )
    parser.add_argument(
        "--auto_adafactor_gpu_gb",
        type=float,
        default=16.0,
        help="When --optimizer auto, use Adafactor if CUDA total memory is at or below this many GB.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta_q", type=float, default=0.1)
    parser.add_argument("--beta_d", type=float, default=0.3)
    parser.add_argument("--delta_eps", type=float, default=1e-6)
    parser.add_argument("--signal_projection_output_dim", type=int, default=256)
    parser.add_argument("--projection_head_weight_decay", type=float, default=0.0)

    parser.add_argument("--candidate_pool_k", type=int, default=64)
    parser.add_argument(
        "--max_train_candidates_per_query",
        type=int,
        default=32,
        help=(
            "Maximum candidates encoded through the trainable E5 per query during training. "
            "Retrieval still uses --candidate_pool_k first, then this cap is applied after deterministic de-duplication."
        ),
    )
    parser.add_argument("--feedback_k", type=int, default=100)
    parser.add_argument(
        "--disable_l40s_safe_mode",
        action="store_true",
        help="Disable default conservative memory guards for 44GB L40S runs.",
    )

    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--eval_k", type=int, default=64)
    parser.add_argument("--selection_recall_w", type=float, default=1.0)
    parser.add_argument("--selection_ndcg_w", type=float, default=0.25)
    parser.add_argument("--selection_mrr_w", type=float, default=0.10)

    parser.add_argument("--frozen_corpus_emb_path", default="base_corpus_embeddings.pt")
    parser.add_argument("--frozen_query_emb_path", default="base_query_embeddings.pt")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--output_dir", default="runs/rcd")
    parser.add_argument("--log_path", default="")
    parser.add_argument("--debug_teacher_samples", type=int, default=5)
    args = parser.parse_args()

    return Cfg(
        dataset_format=args.dataset_format,
        dataset_dir=args.dataset_dir,
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        qrels_dir=args.qrels_dir,
        train_queries_file=args.train_queries_file,
        legalbench_rag_root=args.legalbench_rag_root,
        legalbench_rag_datasets=tuple(args.legalbench_rag_datasets),
        legalbench_prepared_dir=args.legalbench_prepared_dir,
        legalbench_extract_dir=args.legalbench_extract_dir,
        legalbench_chunk_size=args.legalbench_chunk_size,
        legalbench_chunk_strategy=args.legalbench_chunk_strategy,
        query_type_filter=args.query_type_filter,
        query_types=args.query_types,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_graded_relevance=not args.use_binary_relevance,
        report_comlq_slices=not args.no_report_comlq_slices,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        teacher_checkpoint_template=args.teacher_checkpoint_template,
        teacher_output_dim=args.teacher_output_dim,
        teacher_hidden_dim=args.teacher_hidden_dim,
        teacher_dropout=args.teacher_dropout,
        auto_train_teacher_if_missing=not args.no_auto_train_teacher_if_missing,
        teacher_train_epochs=args.teacher_train_epochs,
        teacher_train_batch_size=args.teacher_train_batch_size,
        teacher_train_lr=args.teacher_train_lr,
        teacher_train_weight_decay=args.teacher_train_weight_decay,
        teacher_train_num_negatives=args.teacher_train_num_negatives,
        teacher_train_negative_pool=args.teacher_train_negative_pool,
        teacher_label_temp=args.teacher_label_temp,
        teacher_max_positives=args.teacher_max_positives,
        teacher_max_candidates=args.teacher_max_candidates,
        teacher_metric_select=args.teacher_metric_select,
        teacher_select_recall_k=args.teacher_select_recall_k,
        teacher_select_ndcg_k=args.teacher_select_ndcg_k,
        teacher_select_mrr_k=args.teacher_select_mrr_k,
        teacher_recall_w=args.teacher_recall_w,
        teacher_ndcg_w=args.teacher_ndcg_w,
        teacher_mrr_w=args.teacher_mrr_w,
        allow_weak_teacher=args.allow_weak_teacher,
        model_name=args.model_name,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        embedding_dim=args.embedding_dim,
        device=args.device,
        amp=not args.no_amp,
        student_max_seq_length=args.student_max_seq_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        mode=MODE_EVAL_ONLY if args.eval_only else args.mode,
        resume=args.resume,
        variants=tuple(args.variants),
        dry_run=args.dry_run,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
        corpus_encode_batch_size=args.corpus_encode_batch_size,
        optimizer=args.optimizer,
        auto_adafactor_gpu_gb=args.auto_adafactor_gpu_gb,
        max_train_candidates_per_query=args.max_train_candidates_per_query,
        l40s_safe_mode=not args.disable_l40s_safe_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        tau=args.tau,
        alpha=args.alpha,
        beta_q=args.beta_q,
        beta_d=args.beta_d,
        delta_eps=args.delta_eps,
        signal_projection_output_dim=args.signal_projection_output_dim,
        projection_head_weight_decay=args.projection_head_weight_decay,
        candidate_pool_k=args.candidate_pool_k,
        feedback_k=args.feedback_k,
        ks=tuple(args.ks),
        eval_k=args.eval_k,
        selection_recall_w=args.selection_recall_w,
        selection_ndcg_w=args.selection_ndcg_w,
        selection_mrr_w=args.selection_mrr_w,
        frozen_corpus_emb_path=args.frozen_corpus_emb_path,
        frozen_query_emb_path=args.frozen_query_emb_path,
        force_rebuild_embedding_cache=args.force_rebuild_cache,
        output_dir=args.output_dir,
        log_path=args.log_path,
        debug_teacher_samples=args.debug_teacher_samples,
    )
