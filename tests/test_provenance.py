import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rcd_retrieval.provenance import (
    assert_resume_contract,
    augment_candidate_metadata,
    build_run_contract,
)


@dataclass(frozen=True)
class Passage:
    pid: str


@dataclass(frozen=True)
class Example:
    qid: str
    candidate_doc_idxs: tuple[int, ...]


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.passages = [Passage("d0"), Passage("d1"), Passage("d2")]
        self.examples = [Example("q0", (2, 0)), Example("q1", (1,))]

    def test_candidate_fingerprints_cover_order_ids_and_corpus(self):
        first = augment_candidate_metadata({}, self.examples, self.passages)
        reordered = augment_candidate_metadata(
            {}, [Example("q0", (0, 2)), Example("q1", (1,))], self.passages
        )
        reordered_corpus = augment_candidate_metadata(
            {}, self.examples, [Passage("d2"), Passage("d1"), Passage("d0")]
        )
        self.assertNotEqual(
            first["candidate_index_fingerprint_sha1"],
            reordered["candidate_index_fingerprint_sha1"],
        )
        self.assertNotEqual(
            first["candidate_id_fingerprint_sha1"],
            reordered["candidate_id_fingerprint_sha1"],
        )
        self.assertNotEqual(
            first["corpus_order_fingerprint_sha1"],
            reordered_corpus["corpus_order_fingerprint_sha1"],
        )

    def test_resume_requires_identical_contract(self):
        metadata = augment_candidate_metadata({}, self.examples, self.passages)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher.pt"
            teacher.write_bytes(b"teacher-a")
            cfg = SimpleNamespace(seed=42, lr=2e-5, output_dir="ignored")
            contract = build_run_contract(
                cfg=cfg,
                objective="rcd",
                variant="FULL_RCD",
                candidate_metadata=metadata,
                teacher_checkpoint_path=teacher,
            )
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "student_metadata.json").write_text(
                json.dumps({"run_contract": contract}), encoding="utf-8"
            )
            assert_resume_contract(checkpoint, contract)

            changed = build_run_contract(
                cfg=SimpleNamespace(seed=42, lr=3e-5, output_dir="ignored"),
                objective="rcd",
                variant="FULL_RCD",
                candidate_metadata=metadata,
                teacher_checkpoint_path=teacher,
            )
            with self.assertRaisesRegex(ValueError, "changed config"):
                assert_resume_contract(checkpoint, changed)

    def test_resume_rejects_legacy_checkpoint_without_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "student_metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predates strict run contracts"):
                assert_resume_contract(checkpoint, {"sha1": "expected"})


if __name__ == "__main__":
    unittest.main()
