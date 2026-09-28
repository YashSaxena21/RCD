import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rcd_retrieval import beir_splits, cli, core, legalbench, supervised
from rcd_retrieval.encoders import E5, QWEN, apply_preset, configure_modules
from rcd_retrieval.imrnns_backend import (
    IMRNNS_VERSION,
    _validate_metadata,
    deterministic_fraction_qids,
)


class DummyConfig:
    model_name = "old"
    query_prefix = "old"
    passage_prefix = "old"
    embedding_dim = 1
    student_max_seq_length = 0
    signal_projection_output_dim = 1


class ConfigurationTests(unittest.TestCase):
    def test_dataset_scope_is_current_benchmark_set(self):
        self.assertEqual(beir_splits.DEFAULT_DATASETS, ("scifact", "nfcorpus"))
        self.assertEqual(legalbench.DATASETS, ("privacy_qa", "contractnli"))
        with self.assertRaisesRegex(ValueError, "Unsupported BEIR dataset"):
            beir_splits.normalize_dataset("unsupported")
        with self.assertRaisesRegex(ValueError, "Unsupported LegalBench-RAG dataset"):
            legalbench.normalize_dataset_name("unsupported")

    def test_encoder_presets(self):
        for preset in (E5, QWEN):
            cfg = apply_preset(DummyConfig(), preset)
            self.assertEqual(cfg.model_name, preset.model_name)
            self.assertEqual(cfg.query_prefix, preset.query_prefix)
            self.assertEqual(cfg.passage_prefix, preset.passage_prefix)
            self.assertEqual(cfg.embedding_dim, preset.dimension)
            self.assertEqual(cfg.signal_projection_output_dim, preset.dimension)

    def test_teacher_fraction_is_deterministic_and_nested(self):
        qids = [f"q{index}" for index in range(20)]
        order_30, selected_30 = deterministic_fraction_qids(qids, 0.3, 42)
        order_50, selected_50 = deterministic_fraction_qids(qids, 0.5, 42)
        self.assertEqual(order_30, order_50)
        self.assertEqual(selected_30, selected_50[: len(selected_30)])
        self.assertEqual(len(selected_30), 6)
        self.assertEqual(len(selected_50), 10)

    def test_invalid_teacher_fraction_fails(self):
        with self.assertRaises(ValueError):
            deterministic_fraction_qids(["q1"], 0.0, 42)

    def test_qwen_result_label_is_not_e5(self):
        module = SimpleNamespace(
            VARIANT_E5_BASE="E5_BASE",
            BASELINE_VARIANTS={"E5_BASE", "IMRNN_TEACHER"},
        )
        configure_modules("qwen", (module,))
        self.assertEqual(module.VARIANT_E5_BASE, "QWEN_BASE")
        self.assertEqual(module.BASELINE_VARIANTS, {"QWEN_BASE", "IMRNN_TEACHER"})

    def test_teacher_template_resolves_per_seed_and_dataset(self):
        cfg = core.Cfg(
            dataset_name="contractnli",
            output_dir="runs/example",
            seeds=(42, 43),
            teacher_checkpoint_template="teachers/{dataset}/seed_{seed}.pt",
        )
        seed_cfg = core.make_seed_cfg(cfg, 43)
        self.assertEqual(
            seed_cfg.teacher_checkpoint_path,
            "teachers/contractnli/seed_43.pt",
        )

    def test_one_explicit_teacher_cannot_cover_multiple_seeds(self):
        cfg = core.Cfg(
            seeds=(42, 43),
            teacher_checkpoint_path="teachers/seed_42.pt",
            auto_train_teacher_if_missing=False,
        )
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            core.make_seed_cfg(cfg, 42)

    def test_one_explicit_teacher_cannot_cover_dataset_suite(self):
        cfg = core.Cfg(
            teacher_checkpoint_path="teachers/comlq.pt",
            auto_train_teacher_if_missing=False,
        )
        with self.assertRaisesRegex(ValueError, "multi-dataset suite"):
            core._suite_teacher_path(cfg, 2)

    def test_removed_labeled_mode_fails(self):
        cfg = core.Cfg(mode="LABEL_AWARE")
        with self.assertRaisesRegex(ValueError, "UNSUPERVISED or EVAL_ONLY"):
            core.validate_mode_and_variants(cfg)

    def test_teacher_metadata_checks_exact_selected_subset(self):
        cfg = core.Cfg(dataset_name="comlq", model_name="intfloat/e5-large-v2", seed=42)
        cfg._teacher_fraction = 0.5
        selection = {
            "seed": 42,
            "eligible_query_count": 10,
            "selected_query_count": 5,
            "full_ordered_train_qids_sha1": "full",
            "selected_train_qids_sha1": "selected",
        }
        metadata = {
            "teacher_implementation": "imrnns_pypi",
            "imrnns_version": IMRNNS_VERSION,
            "dataset": "comlq",
            "encoder_model_name": "intfloat/e5-large-v2",
            "seed": 42,
            "teacher_training_fraction": 0.5,
            "fraction_selection": dict(selection),
        }
        _validate_metadata(cfg, metadata, Path("teacher.pt"), selection)
        metadata["fraction_selection"]["selected_train_qids_sha1"] = "wrong"
        with self.assertRaisesRegex(ValueError, "selected_train_qids_sha1"):
            _validate_metadata(cfg, metadata, Path("teacher.pt"), selection)

    def test_base_command_is_pretrained_eval_only(self):
        with patch.object(supervised, "run") as run:
            cli.main(
                [
                    "base",
                    "--dataset_dir",
                    "datasets/example",
                    "--dataset_name",
                    "example",
                    "--output_dir",
                    "runs/example",
                ]
            )
        cfg = run.call_args.args[0]
        self.assertTrue(cfg.base_only)
        self.assertTrue(cfg.eval_only)
        self.assertEqual(cfg.model_name, E5.model_name)


if __name__ == "__main__":
    unittest.main()
