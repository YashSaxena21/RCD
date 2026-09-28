"""Deterministic sampling shared by teacher backends."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


def deterministic_fraction_qids(
    qids: Sequence[str], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Return one stable hash order and the requested nested prefix."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"Teacher fraction must be in (0, 1], got {fraction}")
    ordered = sorted(
        set(map(str, qids)),
        key=lambda qid: (
            hashlib.sha1(f"{int(seed)}\0{qid}".encode("utf-8")).hexdigest(),
            qid,
        ),
    )
    count = min(len(ordered), max(1, math.ceil(len(ordered) * float(fraction)))) if ordered else 0
    return ordered, ordered[:count]
