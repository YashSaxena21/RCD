"""Reproducibility contracts shared by all student-training objectives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# These values control execution rather than the scientific identity of a run.
_OPERATIONAL_CONFIG_FIELDS = {
    "auto_train_teacher_if_missing",
    "device",
    "force_rebuild_cache",
    "frozen_corpus_emb_path",
    "frozen_query_emb_path",
    "legalbench_extract_dir",
    "log_path",
    "output_dir",
    "resume",
    "seeds",
}


def sha1_jsonable(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def file_sha1(path: str | Path) -> str:
    source = Path(path).expanduser()
    if not source.is_file():
        return ""
    digest = hashlib.sha1()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_fingerprints(
    examples: Sequence[Any], passages: Sequence[Any] | None
) -> dict[str, str]:
    ordered_indices = [
        [str(example.qid), [int(doc_idx) for doc_idx in example.candidate_doc_idxs]]
        for example in examples
    ]
    metadata = {
        "candidate_set_fingerprint_sha1": sha1_jsonable(ordered_indices),
        "candidate_index_fingerprint_sha1": sha1_jsonable(ordered_indices),
    }
    if passages is None:
        metadata.update(
            {
                "candidate_id_fingerprint_sha1": "",
                "corpus_order_fingerprint_sha1": "",
            }
        )
        return metadata

    ordered_ids = [
        [
            str(example.qid),
            [str(passages[int(doc_idx)].pid) for doc_idx in example.candidate_doc_idxs],
        ]
        for example in examples
    ]
    metadata.update(
        {
            "candidate_id_fingerprint_sha1": sha1_jsonable(ordered_ids),
            "corpus_order_fingerprint_sha1": sha1_jsonable(
                [str(passage.pid) for passage in passages]
            ),
        }
    )
    return metadata


def augment_candidate_metadata(
    metadata: Mapping[str, Any],
    examples: Sequence[Any],
    passages: Sequence[Any] | None,
) -> dict[str, Any]:
    output = dict(metadata)
    output.update(candidate_fingerprints(examples, passages))
    output["candidate_query_count"] = len(examples)
    output["candidate_doc_total"] = int(
        sum(len(example.candidate_doc_idxs) for example in examples)
    )
    return output


def _config_contract(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        values = asdict(cfg)
    else:
        values = {key: value for key, value in vars(cfg).items() if not key.startswith("_")}
    return {key: value for key, value in values.items() if key not in _OPERATIONAL_CONFIG_FIELDS}


def build_run_contract(
    *,
    cfg: Any,
    objective: str,
    variant: str,
    candidate_metadata: Mapping[str, Any],
    teacher_checkpoint_path: str | Path,
) -> dict[str, Any]:
    candidate_identity = {
        key: candidate_metadata.get(key, "")
        for key in (
            "candidate_set_fingerprint_sha1",
            "candidate_index_fingerprint_sha1",
            "candidate_id_fingerprint_sha1",
            "corpus_order_fingerprint_sha1",
        )
    }
    contract = {
        "schema_version": 1,
        "objective": objective,
        "variant": variant,
        "config": _config_contract(cfg),
        "teacher_checkpoint_path": str(Path(teacher_checkpoint_path).expanduser().resolve()),
        "teacher_checkpoint_sha1": file_sha1(teacher_checkpoint_path),
        "candidate_identity": candidate_identity,
    }
    return {**contract, "sha1": sha1_jsonable(contract)}


def add_run_contract(
    metadata: Mapping[str, Any],
    *,
    cfg: Any,
    objective: str,
    variant: str,
    candidate_metadata: Mapping[str, Any],
    teacher_checkpoint_path: str | Path,
) -> dict[str, Any]:
    output = dict(metadata)
    contract = build_run_contract(
        cfg=cfg,
        objective=objective,
        variant=variant,
        candidate_metadata=candidate_metadata,
        teacher_checkpoint_path=teacher_checkpoint_path,
    )
    output["run_contract"] = contract
    output["run_contract_sha1"] = contract["sha1"]
    return output


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "student_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")
    return payload


def assert_resume_contract(
    checkpoint_dir: str | Path,
    expected_contract: Mapping[str, Any],
) -> None:
    metadata = load_checkpoint_metadata(checkpoint_dir)
    saved = metadata.get("run_contract")
    if not isinstance(saved, dict) or not saved.get("sha1"):
        raise ValueError(
            f"Checkpoint {checkpoint_dir} predates strict run contracts. "
            "Use --no_resume with a clean output directory."
        )
    if saved.get("sha1") == expected_contract.get("sha1"):
        return

    changed = []
    for key in ("objective", "variant", "teacher_checkpoint_sha1", "candidate_identity", "config"):
        if saved.get(key) != expected_contract.get(key):
            changed.append(key)
    details = ", ".join(changed) or "unknown fields"
    raise ValueError(
        f"Checkpoint run contract mismatch at {checkpoint_dir}: changed {details}. "
        "Use the original configuration or --no_resume with a clean output directory."
    )
