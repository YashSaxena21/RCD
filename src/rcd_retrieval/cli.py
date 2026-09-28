from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from . import core, embeddistill, margin_mse, rcd_args, search_adaptor, search_baselines, supervised
from .encoders import apply_preset, configure_modules
from .imrnns_backend import install_hooks

RCD_VARIANTS = (
    "RANKING_ONLY",
    "CORRECTION_ONLY",
    "FULL_RCD",
)
COMMANDS = ("base", "supervised", "rcd", "margin-mse", "embeddistill")


@contextmanager
def _argv(arguments: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [previous[0], *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def _front_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--encoder", choices=["e5", "qwen"], default="e5")
    if command in {"rcd", "margin-mse", "embeddistill"}:
        parser.add_argument("--teacher", choices=["imrnns", "search-adaptor"], default="imrnns")
        parser.add_argument("--teacher-fraction", type=float, default=1.0)
        parser.add_argument("--teacher-fraction-seed", type=int, default=42)
        parser.add_argument("--student-doc-micro-batch-size", type=int, default=16)
    return parser


def _print_help() -> None:
    print(
        "usage: rcd-train COMMAND [repository options] [experiment options]\n\n"
        "commands:\n"
        "  base          evaluate the pretrained E5 or Qwen retriever\n"
        "  supervised    supervised retriever fine-tuning\n"
        "  rcd           Representation Correction Distillation\n"
        "  margin-mse    pairwise score-margin distillation\n"
        "  embeddistill  final-representation and score distillation\n\n"
        "repository options:\n"
        "  --encoder {e5,qwen}\n"
        "  --teacher {imrnns,search-adaptor}\n"
        "  --teacher-fraction FLOAT\n"
        "  --teacher-fraction-seed INT\n"
        "  --student-doc-micro-batch-size INT\n\n"
        "Run `rcd-train COMMAND --help` for the experiment-specific options."
    )


def _parse_front(command: str, arguments: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = _front_parser(command)
    options, remaining = parser.parse_known_args(arguments)
    if hasattr(options, "teacher_fraction") and not 0 < options.teacher_fraction <= 1:
        parser.error("--teacher-fraction must be in (0, 1]")
    return options, remaining


def _prepare_cfg(cfg: object, front: argparse.Namespace) -> object:
    preset = configure_modules(front.encoder, (core, supervised))
    apply_preset(cfg, preset)
    if hasattr(front, "teacher_fraction"):
        cfg._teacher_fraction = front.teacher_fraction
        cfg._teacher_fraction_seed = front.teacher_fraction_seed
    return cfg


def _search_arguments(front: argparse.Namespace, remaining: list[str]) -> list[str]:
    if "--search_adaptor_train_fractions" not in remaining:
        remaining += ["--search_adaptor_train_fractions", str(front.teacher_fraction)]
    if "--search_adaptor_fraction_seed" not in remaining:
        remaining += ["--search_adaptor_fraction_seed", str(front.teacher_fraction_seed)]
    return remaining


def _run_supervised(command: str, front: argparse.Namespace, remaining: list[str]) -> None:
    configure_modules(front.encoder, (supervised,))
    cfg = supervised.parse_args(remaining)
    preset = configure_modules(front.encoder, (supervised,))
    cfg.model_name = preset.model_name
    cfg.max_seq_length = preset.max_sequence_length
    supervised.QUERY_PREFIX = preset.query_prefix
    supervised.PASSAGE_PREFIX = preset.passage_prefix
    if command == "base":
        cfg.base_only = True
        cfg.eval_only = True
    supervised.run(cfg)


def _run_rcd(front: argparse.Namespace, remaining: list[str]) -> None:
    if not any(value == "--variants" or value.startswith("--variants=") for value in remaining):
        remaining += ["--variants", "FULL_RCD"]
    if front.teacher == "search-adaptor":
        remaining = _search_arguments(front, remaining)
        with _argv(remaining):
            cfg, options = search_adaptor.parse_args()
        invalid = sorted(set(cfg.variants) - set(RCD_VARIANTS))
        if invalid:
            raise ValueError(f"RCD exposes only {RCD_VARIANTS}; got {invalid}")
        _prepare_cfg(cfg, front)
        search_adaptor.ACTIVE_OPTIONS = options
        if options.run_tests:
            search_adaptor.run_search_adaptor_tests()
            return
        search_adaptor.run_search_adaptor_suite(cfg)
        return

    with _argv(remaining):
        cfg = rcd_args.parse_args()
    invalid = sorted(set(cfg.variants) - set(RCD_VARIANTS))
    if invalid:
        raise ValueError(f"RCD exposes only {RCD_VARIANTS}; got {invalid}")
    _prepare_cfg(cfg, front)
    install_hooks(front.student_doc_micro_batch_size)
    core.main_unsup(cfg)


def _run_imrnns_baseline(command: str, front: argparse.Namespace, remaining: list[str]) -> None:
    install_hooks(front.student_doc_micro_batch_size)
    if command == "margin-mse":
        with _argv(remaining):
            cfg = margin_mse.parse_margin_args()
        _prepare_cfg(cfg, front)
        margin_mse.validate_margin_cfg(cfg)
        if cfg.run_loss_tests:
            margin_mse.run_margin_mse_loss_tests()
        else:
            margin_mse.main_margin(cfg)
        return

    with _argv(remaining):
        cfg = embeddistill.parse_embeddistill_args()
    _prepare_cfg(cfg, front)
    embeddistill.validate_embeddistill_cfg(cfg)
    if cfg.run_loss_tests:
        embeddistill.run_embeddistill_loss_tests()
    else:
        embeddistill.main_embeddistill(cfg)


def _run_search_baseline(command: str, front: argparse.Namespace, remaining: list[str]) -> None:
    objective = "margin_mse" if command == "margin-mse" else "embeddistill"
    remaining = _search_arguments(front, remaining)
    remaining += ["--distillation_objectives", objective]
    with _argv(remaining):
        cfg, search_options, suite_options = search_baselines.parse_args()
    _prepare_cfg(cfg, front)
    search_adaptor.ACTIVE_OPTIONS = search_options
    search_baselines.ACTIVE_SUITE_OPTIONS = suite_options
    search_baselines.install_integration_hooks()
    if suite_options.run_tests:
        search_baselines.run_suite_tests()
    else:
        search_baselines.run_suite(cfg)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return
    command = arguments.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}. Choose from: {', '.join(COMMANDS)}")
    front, remaining = _parse_front(command, arguments)

    if "--help" in remaining or "-h" in remaining:
        print(_front_parser(command).format_help())
    if command in {"base", "supervised"}:
        _run_supervised(command, front, remaining)
    elif command == "rcd":
        _run_rcd(front, remaining)
    elif front.teacher == "imrnns":
        _run_imrnns_baseline(command, front, remaining)
    else:
        _run_search_baseline(command, front, remaining)
