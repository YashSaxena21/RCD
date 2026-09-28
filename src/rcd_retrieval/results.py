"""Result serialization and cross-seed aggregation."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from .config import Cfg
from .runtime import logger

NestedResults = dict[str, dict[str, dict[int, dict[str, float]]]]


def flatten_results(results: NestedResults) -> list[dict[str, Any]]:
    rows = []
    for system, by_slice in results.items():
        for slice_name, by_k in by_slice.items():
            for k, metrics in by_k.items():
                rows.append({"system": system, "slice": slice_name, "k": int(k), **metrics})
    return rows


def write_csv_rows(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comlq_test_breakdown_results(cfg: Cfg, results: NestedResults) -> None:
    if not cfg.report_comlq_slices:
        return
    test_only = {
        system: {
            slice_name: by_k
            for slice_name, by_k in by_slice.items()
            if slice_name == "TEST" or slice_name.startswith("TEST ")
        }
        for system, by_slice in results.items()
    }
    test_only = {system: slices for system, slices in test_only.items() if slices}
    if not test_only:
        return
    json_path = os.path.join(cfg.output_dir, "final_results_comlq_test_breakdown.json")
    csv_path = os.path.join(cfg.output_dir, "final_results_comlq_test_breakdown.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "ks": list(cfg.ks),
                "note": (
                    "ComLQ TEST-only breakdown by overall, logical family, exact base query "
                    "type, and derived hop count."
                ),
                "results": test_only,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    write_csv_rows(csv_path, flatten_results(test_only))
    logger.info("Saved ComLQ TEST breakdown JSON: %s", json_path)
    logger.info("Saved ComLQ TEST breakdown CSV: %s", csv_path)


def _metric_names(rows: list[dict[str, Any]], dimensions: set[str]) -> list[str]:
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in dimensions and isinstance(value, (int, float))
        }
    )


def aggregate_seed_results(parent_output_dir: str, per_seed: list[dict[str, Any]]) -> None:
    rows = []
    for payload in per_seed:
        for result in flatten_results(payload["results"]):
            rows.append({**result, "seed": payload["seed"]})
    os.makedirs(parent_output_dir, exist_ok=True)
    with open(
        os.path.join(parent_output_dir, "results_per_seed.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(per_seed, handle, indent=2, sort_keys=True)
    write_csv_rows(os.path.join(parent_output_dir, "results_per_seed.csv"), rows)

    metric_names = _metric_names(rows, {"system", "slice", "k", "seed", "dataset"})
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (str(row["system"]), str(row["slice"]), int(row["k"]))
        for metric in metric_names:
            if metric in row:
                grouped[key][metric].append(float(row[metric]))

    aggregate_rows = []
    aggregate_json: dict[str, Any] = {}
    for (system, slice_name, k), metric_values in sorted(grouped.items()):
        output = {"system": system, "slice": slice_name, "k": k}
        json_key = f"{system}|{slice_name}|{k}"
        aggregate_json[json_key] = {}
        for metric, values in metric_values.items():
            mean = float(np.mean(values))
            std = float(np.std(values))
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            aggregate_json[json_key][f"{metric}_mean"] = mean
            aggregate_json[json_key][f"{metric}_std"] = std
        aggregate_rows.append(output)
    with open(
        os.path.join(parent_output_dir, "aggregate_results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate_json, handle, indent=2, sort_keys=True)
    write_csv_rows(os.path.join(parent_output_dir, "aggregate_results.csv"), aggregate_rows)
    write_csv_rows(
        os.path.join(parent_output_dir, "ablation_summary.csv"),
        [
            row
            for row in aggregate_rows
            if row["slice"] == "TEST" and int(row["k"]) in {8, 16, 32, 64}
        ],
    )


def write_legalbench_suite_results(
    parent_output_dir: str,
    dataset_payloads: dict[str, list[dict[str, Any]]],
) -> None:
    os.makedirs(parent_output_dir, exist_ok=True)
    with open(
        os.path.join(parent_output_dir, "legalbench_rag_results_by_dataset.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(dataset_payloads, handle, indent=2, sort_keys=True)

    rows = []
    for dataset_name, per_seed in dataset_payloads.items():
        for payload in per_seed:
            for result in flatten_results(payload.get("results", {})):
                rows.append({**result, "dataset": dataset_name, "seed": payload.get("seed")})
    write_csv_rows(os.path.join(parent_output_dir, "legalbench_rag_results_by_dataset.csv"), rows)

    metric_names = _metric_names(rows, {"dataset", "system", "slice", "k", "seed"})
    grouped: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["system"]),
            str(row["slice"]),
            int(row["k"]),
        )
        for metric in metric_names:
            if metric in row:
                grouped[key][metric].append(float(row[metric]))

    aggregate_rows = []
    for (dataset, system, slice_name, k), metric_values in sorted(grouped.items()):
        output = {"dataset": dataset, "system": system, "slice": slice_name, "k": k}
        for metric, values in metric_values.items():
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_std"] = float(np.std(values))
        aggregate_rows.append(output)
    write_csv_rows(
        os.path.join(parent_output_dir, "legalbench_rag_results_by_dataset_aggregate.csv"),
        aggregate_rows,
    )
