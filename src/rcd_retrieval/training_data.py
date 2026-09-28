"""Batch construction for candidate-list student training."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .records import PassageRecord, TrainExample


class StudentCandidateDataset(Dataset):
    def __init__(self, examples: list[TrainExample], passages: list[PassageRecord]):
        self.examples = examples
        self.passages = passages

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainExample:
        return self.examples[index]


def collate_student_candidates(
    batch: list[TrainExample], passages: list[PassageRecord]
) -> dict[str, Any]:
    batch_size = len(batch)
    max_docs = max(len(example.candidate_doc_idxs) for example in batch)
    doc_idxs = torch.full((batch_size, max_docs), -1, dtype=torch.long)
    labels = torch.zeros((batch_size, max_docs), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_docs), dtype=torch.bool)
    candidate_texts: list[list[str]] = []
    for row, example in enumerate(batch):
        row_texts = []
        for col, doc_idx in enumerate(example.candidate_doc_idxs):
            doc_idxs[row, col] = int(doc_idx)
            labels[row, col] = float(example.labels[col])
            mask[row, col] = True
            row_texts.append(passages[int(doc_idx)].text)
        row_texts.extend([""] * (max_docs - len(row_texts)))
        candidate_texts.append(row_texts)
    return {
        "qids": [example.qid for example in batch],
        "q_idx": torch.tensor([example.q_idx for example in batch], dtype=torch.long),
        "query_texts": [example.query_text for example in batch],
        "doc_idxs": doc_idxs,
        "candidate_texts": candidate_texts,
        "labels": labels,
        "mask": mask,
    }


def slice_training_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    row_mask = batch["mask"][start:end]
    if row_mask.numel() == 0:
        raise ValueError("Empty microbatch slice")
    max_docs = max(1, int(row_mask.sum(dim=1).max().item()))
    return {
        "qids": batch["qids"][start:end],
        "q_idx": batch["q_idx"][start:end],
        "query_texts": batch["query_texts"][start:end],
        "doc_idxs": batch["doc_idxs"][start:end, :max_docs],
        "candidate_texts": [row[:max_docs] for row in batch["candidate_texts"][start:end]],
        "labels": batch["labels"][start:end, :max_docs],
        "mask": batch["mask"][start:end, :max_docs],
    }
