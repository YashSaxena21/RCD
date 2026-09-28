"""Typed records passed between dataset, training, and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PassageRecord:
    pid: str
    text: str


@dataclass
class QueryRecord:
    qid: str
    text: str
    query_type: str
    base_type: str


@dataclass
class EvalExample:
    qid: str
    q_idx: int
    query_text: str
    query_type: str
    base_type: str


@dataclass
class TrainExample:
    qid: str
    q_idx: int
    query_text: str
    query_type: str
    base_type: str
    candidate_doc_idxs: List[int]
    labels: List[float]


@dataclass
class CandidateBuildResult:
    examples: List[TrainExample]
    metadata: Dict[str, Any]
