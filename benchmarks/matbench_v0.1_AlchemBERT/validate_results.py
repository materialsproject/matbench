"""Validate and summarize the archived submission without changing its bytes.

Run from the Matbench repository root after installing that checkout::

    python benchmarks/matbench_v0.1_AlchemBERT/validate_results.py

The JSON report summarizes *stored* scores; it does not rerun training or
recompute metrics from ground-truth targets. No datasets or models are loaded.
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ARCHIVED_SHA256 = "a0bb054a3a32b09041940aa90a7adc4e63a538811af430194057736ea5bd3daf"
DEFAULT_RESULTS = Path(__file__).resolve().with_name("results.json.gz")


def summarize_stored_results(raw):
    """Summarize serialized predictions and scores, without rescoring them."""
    tasks = {}
    for name, task in sorted(raw["tasks"].items()):
        folds = {}
        metric_values = {}
        for fold_name, fold in sorted(task["results"].items()):
            predictions = list(fold["data"].values())
            if not predictions or not all(
                isinstance(value, (bool, int, float)) and math.isfinite(value)
                for value in predictions
            ):
                raise ValueError(
                    f"{name}/{fold_name}: missing or non-finite predictions"
                )
            scores = fold["scores"]
            for metric, value in scores.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{name}/{fold_name}: invalid {metric} score")
                metric_values.setdefault(metric, []).append(value)
            fold_summary = {
                "prediction_count": len(predictions),
                "prediction_types": sorted(
                    {type(value).__name__ for value in predictions}
                ),
                "prediction_min": min(predictions),
                "prediction_max": max(predictions),
                "unique_prediction_count": len(set(predictions)),
                "record_parameters": fold["parameters"],
                "stored_scores": scores,
            }
            if "rocauc" in scores:
                outside = sum(not 0 <= value <= 1 for value in predictions)
                if outside:
                    raise ValueError(
                        f"{name}/{fold_name}: predictions outside [0, 1]"
                    )
                fold_summary["predictions_outside_unit_interval"] = outside
                fold_summary["non_binary_prediction_count"] = sum(
                    value not in (0, 1) for value in predictions
                )
                fold_summary["stored_rocauc_equals_balanced_accuracy"] = (
                    math.isclose(
                        scores["rocauc"],
                        scores["balanced_accuracy"],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            folds[fold_name] = fold_summary
        if any(len(values) != len(folds) for values in metric_values.values()):
            raise ValueError(f"{name}: inconsistent metric names across folds")
        tasks[name] = {
            "fold_count": len(folds),
            "stored_score_summary": {
                metric: {
                    "mean": statistics.mean(values),
                    "population_variance": statistics.pvariance(values),
                    "population_std": statistics.pstdev(values),
                }
                for metric, values in sorted(metric_values.items())
            },
            "folds": folds,
        }
    return tasks


def validate_results(path, expected_sha256=ARCHIVED_SHA256):
    """Check archive identity and Matbench validity, completeness and recording."""
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Archive SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    raw = json.loads(gzip.decompress(blob))
    tasks = summarize_stored_results(raw)

    # The caller must install/use the intended Matbench checkout. Importing here
    # keeps the archive-inspection helper usable without training dependencies.
    from matbench.bench import MatbenchBenchmark
    from matbench.constants import VERSION

    benchmark = MatbenchBenchmark.from_file(str(path))
    checks = {
        "is_valid": benchmark.is_valid,
        "is_complete": benchmark.is_complete,
        "is_recorded": benchmark.is_recorded,
    }
    if not all(checks.values()):
        raise ValueError(f"Matbench validation failed: {checks}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Archive changed during validation")
    return {
        "artifact": {
            "filename": path.name,
            "sha256": digest,
            "serialized_matbench_version": raw.get("version"),
            "serialized_datestamp": raw.get("datestamp"),
            "serialized_content_hash": raw.get("hash"),
            "user_metadata": raw.get("user_metadata"),
        },
        "validation": {"matbench_version": VERSION, **checks},
        "score_provenance": "stored fold scores; not recomputed from targets",
        "task_count": len(tasks),
        "total_fold_count": sum(task["fold_count"] for task in tasks.values()),
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected-sha256", default=ARCHIVED_SHA256)
    args = parser.parse_args()
    # Matbench initializes a stdout logger on import. Keep stdout as one JSON
    # document so that callers can parse or redirect the report reliably.
    with contextlib.redirect_stdout(sys.stderr):
        report = validate_results(args.results, args.expected_sha256)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
