"""Refresh cached metrics from verified labels, preserving every saved prediction.

The historical artifact is hash-locked. All 13 tasks and 65 folds are rescored
with the current, identified Matbench scorer before a new artifact is written.
No model is trained, no prediction is changed, and existing files are never
overwritten. Review the new output before promoting it to results.json.gz.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from rescore_results import (
    DATASET_SHA256,
    build_report,
    read_verified_json,
    write_report,
)
from validate_results import ARCHIVED_SHA256, HISTORICAL_RESULTS


def canonical(value):
    """A type-sensitive comparison: 1, 1.0 and true remain different."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def without_scores(tasks):
    result = copy.deepcopy(tasks)
    for task in result.values():
        for fold in task["results"].values():
            del fold["scores"]
    return result


def check_complete_report(source, report):
    """Reject partial rescoring or a report for a different archived artifact."""
    task_names = set(DATASET_SHA256)
    folds = {f"fold_{number}" for number in range(5)}
    if (
        report["artifact_sha256"] != ARCHIVED_SHA256
        or report["task_count"] != 13
        or report["fold_count"] != 65
        or report["not_rescored"]
        or set(source["tasks"]) != task_names
        or set(report["tasks"]) != task_names
    ):
        raise ValueError(
            "Migration requires all 13 tasks and 65 folds of the archive"
        )
    for name in task_names:
        original = source["tasks"][name]["results"]
        recomputed = report["tasks"][name]["folds"]
        if set(original) != folds or set(recomputed) != folds:
            raise ValueError(f"{name}: migration requires all five folds")
        for fold in folds:
            old = original[fold]
            new = recomputed[fold]
            scores = new["recomputed_scores"]
            if (
                new["n_samples"] != len(old["data"])
                or canonical(new["stored_scores"]) != canonical(old["scores"])
                or set(scores) != set(old["scores"])
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in scores.values()
                )
            ):
                raise ValueError(
                    f"{name}/{fold}: rescoring report does not match archive"
                )


def check_new_output(path, suffix):
    path = Path(path)
    if not path.name.endswith(suffix):
        raise ValueError(f"Output must end in {suffix}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not path.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {path.parent}")
    return path


def migrate_results(archive, dataset_dir, output, report_output=None):
    """Rescore immutable source data and publish a validated, distinct artifact."""
    from matbench.bench import MatbenchBenchmark

    archive = Path(archive)
    output = check_new_output(output, ".json.gz")
    if report_output is not None:
        report_output = check_new_output(report_output, ".json")
    source = read_verified_json(archive, ARCHIVED_SHA256, compressed=True)
    report = build_report(results=archive, dataset_dir=dataset_dir)
    check_complete_report(source, report)
    benchmark = MatbenchBenchmark.from_file(str(archive))
    if not all((benchmark.is_valid, benchmark.is_complete, benchmark.is_recorded)):
        raise ValueError("Historical benchmark is not valid, complete and recorded")
    preserved = canonical(without_scores(source["tasks"]))
    for task in benchmark.tasks:
        for fold, values in report["tasks"][task.dataset_name]["folds"].items():
            task.results[fold]["scores"] = dict(values["recomputed_scores"])
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    benchmark.add_metadata(
        {
            "score_migration": {
                "purpose": (
                    "Recompute cached scores from unchanged historical predictions"
                ),
                "original_artifact_sha256": ARCHIVED_SHA256,
                "original_content_hash": source["hash"],
                "original_matbench_version": source["version"],
                "original_datestamp": source["datestamp"],
                "scorer_upstream_commit": report["upstream_commit"],
                "software": report["software"],
                "official_spec_sha256": report["spec_sha256"],
                "label_sources": {
                    name: task["label_source"]
                    for name, task in report["tasks"].items()
                },
                "rescoring_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "migration_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "predictions_parameters_and_uncertainties_unchanged": True,
                "retrained": False,
                "historical_training_reproduced": False,
            }
        }
    )
    if canonical(without_scores(benchmark.as_dict()["tasks"])) != preserved:
        raise ValueError(
            "Migration changed predictions, parameters or uncertainties"
        )
    if not all((benchmark.is_valid, benchmark.is_complete, benchmark.is_recorded)):
        raise ValueError("Migrated benchmark is not valid, complete and recorded")
    # Matbench's serialization recomputes its content hash; deserialize the staged
    # bytes and check them before publishing an exclusive hard link to the output.
    with tempfile.TemporaryDirectory(
        prefix=".alchembert-migrate-", dir=output.parent
    ) as tmp:
        staged = Path(tmp) / "results.json.gz"
        benchmark.to_file(str(staged))
        restored = MatbenchBenchmark.from_file(str(staged))
        if not all((restored.is_valid, restored.is_complete, restored.is_recorded)):
            raise ValueError("Serialized migration failed Matbench validation")
        if canonical(without_scores(restored.as_dict()["tasks"])) != preserved:
            raise ValueError(
                "Serialization changed predictions, parameters or uncertainties"
            )
        for task in restored.tasks:
            for fold, values in report["tasks"][task.dataset_name]["folds"].items():
                if canonical(task.results[fold]["scores"]) != canonical(
                    values["recomputed_scores"]
                ):
                    raise ValueError(
                        "Serialized metrics differ from the verified report"
                    )
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVED_SHA256:
            raise RuntimeError("Historical artifact changed during migration")
        os.link(staged, output)
    if report_output is not None:
        write_report(report, report_output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=HISTORICAL_RESULTS)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        report = migrate_results(
            args.archive, args.dataset_dir, args.output, args.report_output
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "task_count": report["task_count"],
                "fold_count": report["fold_count"],
                "predictions_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
