import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from rcd_retrieval import data_utils
from rcd_retrieval.legalbench import prepare_retrieval_dataset


class DataUtilityTests(unittest.TestCase):
    def test_jsonl_loader_reports_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id": 1}\n["not", "an", "object"]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected JSON object"):
                data_utils.load_jsonl(path)

    def test_merge_qrels_keeps_highest_score(self):
        merged = data_utils.merge_qrels(
            {"q1": {"d1": 1.0, "d2": 2.0}},
            {"q1": {"d1": 3.0}, "q2": {"d3": 1.0}},
        )
        self.assertEqual(
            merged,
            {"q1": {"d1": 3.0, "d2": 2.0}, "q2": {"d3": 1.0}},
        )

    def test_eval_example_builder_filters_queries_without_qrels(self):
        query = SimpleNamespace(text="query", query_type="type", base_type="base")
        examples = data_utils.build_eval_examples(
            ["q1", "q2"],
            {"q1": query, "q2": query},
            {"q1": 3, "q2": 4},
            {"q1": {"d1": 1.0}},
            SimpleNamespace,
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].qid, "q1")
        self.assertEqual(examples[0].q_idx, 3)

    def test_official_qrels_accept_dev_as_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("train.tsv", "dev.tsv", "test.tsv"):
                (root / filename).touch()
            paths = data_utils.find_official_qrel_paths(root)
            self.assertIsNotNone(paths)
            assert paths is not None
            self.assertEqual(paths["val"], root / "dev.tsv")

    def test_search_index_preserves_exact_cosine_ranking(self):
        corpus = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        with patch.object(data_utils, "faiss", None):
            index = data_utils.SearchIndex(corpus)
            ranking = index.search(np.array([[1.0, 0.0]], dtype=np.float32), 2)
            empty = index.search(np.empty((0, 2), dtype=np.float32), 2)
        np.testing.assert_array_equal(ranking, np.array([[0, 1]]))
        self.assertEqual(empty.shape, (0, 0))


class LegalBenchPreparationTests(unittest.TestCase):
    def test_shared_preparation_preserves_caller_specific_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "corpus").mkdir()
            (root / "benchmarks").mkdir()
            (root / "corpus" / "document.txt").write_text("abcdefgh", encoding="utf-8")
            benchmark = {
                "tests": [
                    {
                        "query": "Which span is relevant?",
                        "snippets": [{"file_path": "document.txt", "span": [2, 6]}],
                    }
                ]
            }
            (root / "benchmarks" / "privacy_qa.json").write_text(
                json.dumps(benchmark), encoding="utf-8"
            )
            cfg = SimpleNamespace(
                legalbench_rag_root=str(root),
                dataset_dir=str(root),
                legalbench_extract_dir="",
                output_dir=str(root / "runs"),
                legalbench_chunk_size=4,
                legalbench_chunk_strategy="naive",
            )
            plain_dir = root / "plain"
            noted_dir = root / "noted"
            logger = Mock()

            plain = prepare_retrieval_dataset(cfg, "privacyqa", plain_dir)
            noted = prepare_retrieval_dataset(
                cfg,
                "privacy_qa",
                noted_dir,
                include_paper_note=True,
                progress_logger=logger,
            )

            self.assertNotIn("paper_note", plain)
            self.assertIn("paper_note", noted)
            self.assertEqual(plain["passage_count"], 2)
            self.assertEqual(plain["qrel_count"], 2)
            for relative_path in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
                self.assertEqual(
                    (plain_dir / relative_path).read_text(encoding="utf-8"),
                    (noted_dir / relative_path).read_text(encoding="utf-8"),
                )
            logger.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
