"""ComLQ query-family filtering and evaluation slices."""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from .config import Cfg
from .records import EvalExample, QueryRecord

logger = logging.getLogger("rcd_retrieval")

NEGATION_BASE_TYPES = {"2in", "3in", "inp", "pin", "pni"}
CONJUNCTION_BASE_TYPES = {"2i", "3i", "pi", "ip"}
UNION_BASE_TYPES = {"2u", "up"}
PURE_PROJECTION_BASE_TYPES = {"1p", "2p", "3p"}
COMBINATION_BASE_TYPES = CONJUNCTION_BASE_TYPES | UNION_BASE_TYPES
LOGICAL_COMBINATION_BASE_TYPES = COMBINATION_BASE_TYPES | NEGATION_BASE_TYPES
ALL_COMLQ_TYPES = {
    "1p",
    "2p",
    "3p",
    "2i",
    "3i",
    "pi",
    "ip",
    "2u",
    "up",
    "2in",
    "3in",
    "inp",
    "pin",
    "pni",
}
PROJECTION_BASE_TYPES = {"1p", "2p", "3p", "pi", "ip", "up", "inp", "pin", "pni"}
COMLQ_HOP_BY_BASE_TYPE = {
    "1p": 1,
    "2p": 2,
    "3p": 3,
    "2i": 2,
    "3i": 3,
    "pi": 2,
    "ip": 2,
    "2u": 2,
    "up": 2,
    "2in": 2,
    "3in": 3,
    "inp": 2,
    "pin": 2,
    "pni": 2,
}


def choose_qids_by_filter(
    queries: List[QueryRecord], qrels: Dict[str, Dict[str, float]], cfg: Cfg
) -> List[str]:
    custom = {item.strip() for item in cfg.query_types.split(",") if item.strip()}

    def keep(query: QueryRecord) -> bool:
        if query.qid not in qrels:
            return False
        if cfg.query_type_filter == "all":
            return True
        if cfg.query_type_filter == "negation":
            return query.base_type in NEGATION_BASE_TYPES
        if cfg.query_type_filter == "conjunction":
            return query.base_type in CONJUNCTION_BASE_TYPES
        if cfg.query_type_filter == "union":
            return query.base_type in UNION_BASE_TYPES
        if cfg.query_type_filter == "projection":
            return query.base_type in PROJECTION_BASE_TYPES
        if cfg.query_type_filter == "custom":
            if not custom:
                raise ValueError("--query_type_filter custom requires --query_types")
            return query.query_type in custom or query.base_type in custom
        raise ValueError(f"Unknown query_type_filter: {cfg.query_type_filter}")

    return [query.qid for query in queries if keep(query)]


def subset_by_base_type(examples: List[EvalExample], base_types: Set[str]) -> List[EvalExample]:
    return [example for example in examples if example.base_type in base_types]


def subset_by_hop(examples: List[EvalExample], hop: int) -> List[EvalExample]:
    return [
        example for example in examples if COMLQ_HOP_BY_BASE_TYPE.get(example.base_type) == int(hop)
    ]


def build_slices(
    val_examples: List[EvalExample], test_examples: List[EvalExample]
) -> Dict[str, List[EvalExample]]:
    slices: Dict[str, List[EvalExample]] = {
        "VAL": val_examples,
        "VAL NEGATION": subset_by_base_type(val_examples, NEGATION_BASE_TYPES),
        "VAL CONJUNCTION": subset_by_base_type(val_examples, CONJUNCTION_BASE_TYPES),
        "VAL COMBINATION": subset_by_base_type(val_examples, COMBINATION_BASE_TYPES),
        "VAL UNION": subset_by_base_type(val_examples, UNION_BASE_TYPES),
        "VAL PROJECTION": subset_by_base_type(val_examples, PROJECTION_BASE_TYPES),
        "VAL PURE_PROJECTION": subset_by_base_type(val_examples, PURE_PROJECTION_BASE_TYPES),
        "TEST": test_examples,
        "TEST NEGATION": subset_by_base_type(test_examples, NEGATION_BASE_TYPES),
        "TEST CONJUNCTION": subset_by_base_type(test_examples, CONJUNCTION_BASE_TYPES),
        "TEST COMBINATION": subset_by_base_type(test_examples, COMBINATION_BASE_TYPES),
        "TEST LOGICAL_COMBINATION": subset_by_base_type(
            test_examples, LOGICAL_COMBINATION_BASE_TYPES
        ),
        "TEST UNION": subset_by_base_type(test_examples, UNION_BASE_TYPES),
        "TEST PROJECTION": subset_by_base_type(test_examples, PROJECTION_BASE_TYPES),
        "TEST PURE_PROJECTION": subset_by_base_type(test_examples, PURE_PROJECTION_BASE_TYPES),
    }
    for hop in sorted(set(COMLQ_HOP_BY_BASE_TYPE.values())):
        slices[f"TEST HOP {hop}"] = subset_by_hop(test_examples, hop)
        slices[f"VAL HOP {hop}"] = subset_by_hop(val_examples, hop)
    for base_type in sorted(ALL_COMLQ_TYPES):
        slices[f"TEST TYPE {base_type}"] = subset_by_base_type(test_examples, {base_type})
    logger.info(
        "ComLQ evaluation slices: %s",
        ", ".join(
            f"{name}={len(examples)}"
            for name, examples in slices.items()
            if name == "TEST" or name.startswith("TEST ")
        ),
    )
    return slices
