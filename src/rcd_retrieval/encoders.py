from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EncoderPreset:
    name: str
    model_name: str
    query_prefix: str
    passage_prefix: str
    dimension: int
    max_sequence_length: int


E5 = EncoderPreset(
    name="e5",
    model_name="intfloat/e5-large-v2",
    query_prefix="query: ",
    passage_prefix="passage: ",
    dimension=1024,
    max_sequence_length=512,
)
QWEN = EncoderPreset(
    name="qwen",
    model_name="Qwen/Qwen3-Embedding-0.6B",
    query_prefix=(
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery:"
    ),
    passage_prefix="",
    dimension=1024,
    max_sequence_length=512,
)
PRESETS = {preset.name: preset for preset in (E5, QWEN)}


def get_preset(name: str) -> EncoderPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder {name!r}; choose from {sorted(PRESETS)}") from exc


def configure_qwen_model(model: Any, expected_dimension: int = 1024) -> Any:
    import torch

    model.float()
    bad_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    }
    if bad_dtypes:
        raise RuntimeError(
            f"Qwen parameters must be FP32 before AMP: {sorted(map(str, bad_dtypes))}"
        )

    model.max_seq_length = QWEN.max_sequence_length
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.padding_side = "left"
    first = model._first_module() if hasattr(model, "_first_module") else None
    if first is not None:
        if hasattr(first, "max_seq_length"):
            first.max_seq_length = QWEN.max_sequence_length
        if getattr(first, "tokenizer", None) is not None:
            first.tokenizer.padding_side = "left"
        auto_model = getattr(first, "auto_model", None)
        if auto_model is not None and hasattr(auto_model, "gradient_checkpointing_enable"):
            auto_model.gradient_checkpointing_enable()
            if hasattr(auto_model, "config") and hasattr(auto_model.config, "use_cache"):
                auto_model.config.use_cache = False

    dimension = int(model.get_sentence_embedding_dimension())
    if dimension != expected_dimension:
        raise ValueError(f"Qwen embedding dimension is {dimension}, expected {expected_dimension}")
    return model


def install_qwen_factory(module: Any) -> None:
    factory = module.SentenceTransformer
    if getattr(factory, "_rcd_qwen_factory", False):
        return

    def qwen_factory(*args: Any, **kwargs: Any) -> Any:
        return configure_qwen_model(factory(*args, **kwargs))

    qwen_factory._rcd_qwen_factory = True  # type: ignore[attr-defined]
    module.SentenceTransformer = qwen_factory


def configure_modules(encoder: str, modules: Iterable[Any]) -> EncoderPreset:
    preset = get_preset(encoder)
    for module in modules:
        if hasattr(module, "VARIANT_E5_BASE"):
            old_name = module.VARIANT_E5_BASE
            module.VARIANT_E5_BASE = "QWEN_BASE" if encoder == "qwen" else "E5_BASE"
            if hasattr(module, "BASELINE_VARIANTS"):
                module.BASELINE_VARIANTS = {
                    module.VARIANT_E5_BASE if value == old_name else value
                    for value in module.BASELINE_VARIANTS
                }
        if encoder == "qwen":
            if hasattr(module, "SentenceTransformer"):
                install_qwen_factory(module)
    return preset


def apply_preset(cfg: Any, preset: EncoderPreset) -> Any:
    cfg.model_name = preset.model_name
    if hasattr(cfg, "query_prefix"):
        cfg.query_prefix = preset.query_prefix
    if hasattr(cfg, "passage_prefix"):
        cfg.passage_prefix = preset.passage_prefix
    if hasattr(cfg, "embedding_dim"):
        cfg.embedding_dim = preset.dimension
    if hasattr(cfg, "student_max_seq_length") and not cfg.student_max_seq_length:
        cfg.student_max_seq_length = preset.max_sequence_length
    if hasattr(cfg, "signal_projection_output_dim"):
        cfg.signal_projection_output_dim = preset.dimension
    return cfg
