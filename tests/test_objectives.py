import unittest

import torch

from rcd_retrieval import core, embeddistill, margin_mse, search_adaptor


class RCDContractTests(unittest.TestCase):
    def test_public_variants_have_expected_components(self):
        expected = {
            "RANKING_ONLY": (True, False),
            "CORRECTION_ONLY": (False, True),
            "FULL_RCD": (True, True),
        }
        for variant, (rank, modulation) in expected.items():
            self.assertEqual(core.variant_uses_rank(variant), rank)
            self.assertEqual(core.variant_uses_modulation(variant), modulation)

    def test_unsupervised_loss_rejects_nonzero_labels(self):
        cfg = core.Cfg(variants=("RANKING_ONLY",))
        student_scores = torch.tensor([[1.0, 0.0]], requires_grad=True)
        mask = torch.ones_like(student_scores, dtype=torch.bool)
        embeddings = {
            "query": torch.tensor([[1.0, 0.0]], requires_grad=True),
            "docs": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True),
        }
        teacher = {"teacher_scores": torch.tensor([[1.0, 0.0]])}
        with self.assertRaises(AssertionError):
            core.compute_variant_loss(
                "RANKING_ONLY",
                cfg,
                student_scores,
                teacher,
                embeddings,
                {key: value.detach().clone() for key, value in embeddings.items()},
                None,
                torch.tensor([[1.0, 0.0]]),
                mask,
                core.MODE_UNSUPERVISED,
            )


class MarginMSETests(unittest.TestCase):
    def test_matching_margins_and_constant_shift(self):
        teacher = torch.tensor([[0.2, 0.9, -0.1]], requires_grad=True)
        mask = torch.tensor([[True, True, True]])
        for student in (teacher.detach().clone(), teacher.detach().clone() + 17.0):
            student.requires_grad_(True)
            result = margin_mse.margin_mse_per_example(student, teacher, mask)
            self.assertEqual(result.top_indices.tolist(), [1])
            self.assertAlmostEqual(result.total.item(), 0.0, places=7)
            result.total.backward()
            self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_padding_and_single_candidate(self):
        student = torch.tensor([[2.0, 1.0, 1e20], [3.0, -1e20, 5e20]], requires_grad=True)
        teacher = torch.tensor([[2.5, 1.5, -1e20], [7.0, 1e20, -1e20]], requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        result = margin_mse.margin_mse_per_example(student, teacher, mask)
        self.assertTrue(torch.isfinite(result.total))
        self.assertEqual(result.valid_query_mask.tolist(), [True, False])
        self.assertEqual(result.total_pair_count.item(), 1.0)
        result.total.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_all_pairs_counts_each_pair_once(self):
        scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        result = margin_mse.margin_mse_per_example(
            scores, scores.detach(), torch.ones_like(scores, dtype=torch.bool), "all_pairs"
        )
        self.assertEqual(result.total_pair_count.item(), 3.0)


class EmbedDistillTests(unittest.TestCase):
    def test_l2_is_euclidean_and_padding_is_ignored(self):
        student_q = torch.tensor([[3.0, 4.0]], requires_grad=True)
        teacher_q = torch.zeros_like(student_q, requires_grad=True)
        student_d = torch.tensor([[[0.0, 0.0], [1000.0, 1000.0]]], requires_grad=True)
        teacher_d = torch.tensor([[[0.0, 0.0], [-1000.0, -1000.0]]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        result = embeddistill.embeddistill_alignment_per_example(
            student_q, student_d, teacher_q, teacher_d, mask, query_weight=1.0, document_weight=1.0
        )
        self.assertAlmostEqual(result.total.item(), 5.0, places=6)
        result.total.backward()
        self.assertIsNotNone(student_q.grad)
        self.assertIsNotNone(student_d.grad)
        self.assertIsNone(teacher_q.grad)
        self.assertIsNone(teacher_d.grad)

    def test_document_loss_is_averaged_within_query(self):
        query = torch.zeros((2, 2), requires_grad=True)
        docs = torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0]], [[2.0, 0.0], [99.0, 0.0]]],
            requires_grad=True,
        )
        teacher_query = torch.zeros((2, 2))
        teacher_docs = torch.zeros((2, 2, 2))
        mask = torch.tensor([[True, True], [True, False]])
        result = embeddistill.embeddistill_alignment_per_example(
            query, docs, teacher_query, teacher_docs, mask, query_weight=0.0
        )
        self.assertEqual(result.document_loss_per_query.tolist(), [2.0, 2.0])
        self.assertEqual(result.total.item(), 2.0)

    def test_nonzero_labels_are_rejected(self):
        with self.assertRaises(AssertionError):
            embeddistill.assert_zero_labels(torch.tensor([[0.0, 1.0]]))


class SearchAdaptorTests(unittest.TestCase):
    def test_objective_has_gradients(self):
        model = search_adaptor.SearchAdaptorModel(4, 6, num_layers=2)
        queries = torch.randn(2, 4)
        documents = torch.randn(2, 3, 4)
        relevance = torch.tensor([[2.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        mask = torch.ones((2, 3), dtype=torch.bool)
        result = search_adaptor.search_adaptor_objective(
            model, queries, documents, relevance, mask, alpha=0.1, beta=0.01
        )
        self.assertTrue(torch.isfinite(result["total"]))
        self.assertGreater(result["ranking_pair_count"], 0)
        result["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.adapter.parameters()))
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.query_predictor.parameters())
        )


if __name__ == "__main__":
    unittest.main()
