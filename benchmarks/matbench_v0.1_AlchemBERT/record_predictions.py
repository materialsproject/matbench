"""Record a new, explicitly identified prediction export without replacing results.

CSV columns: task,fold,sample_id,prediction. ``sample_id`` is the Matbench
index (mbid), not a row number. All five folds of every selected task are
required. Legacy CSVs without IDs are deliberately rejected: matching their
length does not establish which prediction belongs to which material.

Run from the repository root, after installing Matbench, for example::

    python benchmarks/matbench_v0.1_AlchemBERT/record_predictions.py predictions.csv

This utility does not establish that a model or its training inputs reproduced
the historical AlchemBERT experiment. It only checks and records a new export.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

CSV_COLUMNS = {"task", "fold", "sample_id", "prediction"}


def load_predictions(path):
    """Read the bytes once so the recorded digest covers the actual input."""
    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames is None or set(reader.fieldnames) != CSV_COLUMNS:
        raise ValueError(
            "CSV must have exactly task,fold,sample_id,prediction columns"
        )
    if len(reader.fieldnames) != len(CSV_COLUMNS):
        raise ValueError("CSV has duplicate column names")
    groups = {}
    for line, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {line}")
        task, sample_id = row["task"], row["sample_id"]
        if (
            not task
            or task != task.strip()
            or not sample_id
            or sample_id != sample_id.strip()
        ):
            raise ValueError(f"Blank or padded task/sample_id at row {line}")
        try:
            fold = int(row["fold"])
            prediction = float(row["prediction"])
        except ValueError as error:
            raise ValueError(f"Invalid fold or prediction at row {line}") from error
        if fold < 0 or str(fold) != row["fold"]:
            raise ValueError(f"Fold must be a nonnegative integer at row {line}")
        if not math.isfinite(prediction):
            raise ValueError(f"Nonfinite prediction at row {line}")
        values = groups.setdefault((task, fold), {})
        if sample_id in values:
            raise ValueError(
                f"Duplicate sample_id {sample_id!r} in {task} fold {fold}"
            )
        values[sample_id] = prediction
    if not groups:
        raise ValueError("CSV contains no predictions")
    return groups, hashlib.sha256(raw).hexdigest()


def align_predictions(values, expected_ids, classification=False):
    """Reject mismatched IDs, then align scores to the official test fold."""
    expected_ids = [str(sample_id) for sample_id in expected_ids]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Official test fold unexpectedly contains duplicate IDs")
    missing = set(expected_ids) - set(values)
    extra = set(values) - set(expected_ids)
    if missing or extra:
        raise ValueError(
            f"Prediction IDs do not match the official test fold: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    predictions = [float(values[sample_id]) for sample_id in expected_ids]
    if any(not math.isfinite(value) for value in predictions):
        raise ValueError("Predictions must be finite")
    if classification and any(not 0.0 <= value <= 1.0 for value in predictions):
        raise ValueError(
            "Classification predictions must be probabilities in [0, 1]"
        )
    return predictions


def check_output_path(path):
    path = Path(path)
    if path.name == "results.json.gz" or path.resolve().name == "results.json.gz":
        raise ValueError(
            "The historical results.json.gz is protected; choose a new filename"
        )
    if not path.name.endswith(".json.gz"):
        raise ValueError("Output filename must end in .json.gz")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not path.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {path.parent}")
    return path


def write_new_benchmark(benchmark, output):
    """Publish complete bytes atomically, with an exclusive no-overwrite link."""
    output = check_output_path(output)
    with tempfile.TemporaryDirectory(
        prefix=".alchembert-", dir=output.parent
    ) as directory:
        temporary = Path(directory) / "new.json.gz"
        benchmark.to_file(str(temporary))
        # Unlike replace/rename, link fails if another process created output.
        os.link(temporary, output)


def record_predictions(
    benchmark, groups, source_sha256, source_name, parameters=None
):
    """Record every selected fold using official IDs and untouched probabilities."""
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict):
        raise ValueError(
            "Parameters JSON must be an object keyed by task, then fold"
        )
    tasks = list(benchmark.tasks)
    # MatbenchTask.record logs and skips an already-recorded fold instead of
    # raising. Reject it up front so old predictions cannot acquire new CSV
    # provenance metadata while the requested recording silently does nothing.
    if any(any(task.is_recorded.values()) for task in tasks):
        raise ValueError(
            "Recording requires a fresh benchmark with no recorded folds"
        )
    expected_groups = {
        (task.dataset_name, fold) for task in tasks for fold in task.folds
    }
    if set(groups) != expected_groups:
        raise ValueError(
            "CSV must contain every fold of exactly the selected tasks: "
            f"missing={sorted(expected_groups - set(groups))}, "
            f"extra={sorted(set(groups) - expected_groups)}"
        )
    task_names = {task.dataset_name for task in tasks}
    if set(parameters) - task_names:
        raise ValueError("Parameters JSON contains unselected tasks")
    # Validate parameter structure before loading any datasets.
    for task in tasks:
        task_parameters = parameters.get(task.dataset_name, {})
        if not isinstance(task_parameters, dict):
            raise ValueError(
                "Each task's parameters must be keyed by fold number strings"
            )
        if set(task_parameters) - {str(fold) for fold in task.folds}:
            raise ValueError(f"Unknown parameter fold for {task.dataset_name}")
        if any(not isinstance(value, dict) for value in task_parameters.values()):
            raise ValueError("Each fold's parameters must be a JSON object")
    for task in tasks:
        task.load()
        for fold in task.folds:
            test_inputs = task.get_test_data(fold, include_target=False)
            predictions = align_predictions(
                groups[(task.dataset_name, fold)],
                test_inputs.index,
                classification=task.metadata.task_type == "classification",
            )
            task.record(
                fold,
                predictions,
                params={
                    "prediction_csv_sha256": source_sha256,
                    "supplied_training_parameters": parameters.get(
                        task.dataset_name, {}
                    ).get(str(fold), {}),
                },
            )
    benchmark.add_metadata(
        {
            "prediction_csv": source_name,
            "prediction_csv_sha256": source_sha256,
            "recording_protocol": (
                "explicit Matbench sample IDs; official fold order; "
                "unthresholded probabilities"
            ),
            "historical_reproduction_verified": False,
        }
    )
    if not benchmark.is_valid or not benchmark.is_recorded:
        raise ValueError(
            "Recorded benchmark failed validity or fold-completeness checks"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "predictions", type=Path, help="CSV with task,fold,sample_id,prediction"
    )
    parser.add_argument("--output", type=Path, default=Path("fresh_results.json.gz"))
    parser.add_argument(
        "--tasks", nargs="+", help="Explicit task subset; default: all 13 tasks"
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        help="Optional known parameters: {task: {fold: {...}}}",
    )
    args = parser.parse_args(argv)
    try:
        check_output_path(args.output)
        groups, digest = load_predictions(args.predictions)
        parameters = (
            json.loads(args.params_json.read_text()) if args.params_json else {}
        )
        # Help and pure CSV validation do not require any training dependencies.
        from matbench.bench import MatbenchBenchmark

        benchmark = MatbenchBenchmark(autoload=False, subset=args.tasks)
        record_predictions(
            benchmark, groups, digest, args.predictions.name, parameters
        )
        if args.tasks is None and not benchmark.is_complete:
            raise ValueError("Full submission does not include all 13 tasks")
        write_new_benchmark(benchmark, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(
        f"Wrote new predictions to {args.output}; "
        "historical results were not changed."
    )


if __name__ == "__main__":
    main()
