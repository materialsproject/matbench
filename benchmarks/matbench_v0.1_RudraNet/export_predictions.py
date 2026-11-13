#!/usr/bin/env python3
"""Convert V13A per-fold predictions into a MatbenchBenchmark using official APIs.

This is an export path for *new* V13A training runs, not evidence about how
historic PR predictions were originally generated. The training script writes
trm_models_v13/v13a_fold_predictions_by_mbid.json with the inputs for this tool.

Usage:
  python export_predictions.py --input trm_models_v13/v13a_fold_predictions_by_mbid.json
"""

import argparse
import json
import math
from pathlib import Path


def export_predictions(prediction_file: Path, output_file: Path) -> None:
    from matbench.bench import MatbenchBenchmark

    with prediction_file.open(encoding="utf-8") as stream:
        data = json.load(stream)
    expected_folds = {f"fold_{i}" for i in range(5)}
    if set(data) != expected_folds:
        raise ValueError(f"Expected five folds: {sorted(expected_folds)}")
    if prediction_file.resolve() == output_file.resolve():
        raise ValueError("Do not overwrite the raw predictions input")

    mb = MatbenchBenchmark(
        benchmark="matbench_v0.1", subset=["matbench_steels"], autoload=True
    )
    task = mb.tasks_map["matbench_steels"]
    for fold in task.folds:
        train = task.get_train_and_val_data(fold, as_type="df")
        test = task.get_test_data(fold, as_type="df")
        expected_ids = list(test.index)
        if set(train.index).intersection(expected_ids):
            raise ValueError(f"Fold {fold} train/test IDs overlap")
        predictions_by_id = data[f"fold_{fold}"]
        if set(predictions_by_id) != set(expected_ids):
            raise ValueError(f"Fold {fold} predictions do not match official test IDs")
        predictions = [float(predictions_by_id[key]) for key in expected_ids]
        if not all(math.isfinite(x) for x in predictions):
            raise ValueError(f"Fold {fold} has nonfinite predictions")
        task.record(fold, predictions, params={
            "model": "RudraNet / TRIADS V13A",
            "ensemble_seeds": [42, 123, 7, 0, 99],
            "prediction_file": prediction_file.name,
        })

    mb.add_metadata({
        "algorithm": "RudraNet / TRIADS V13A",
        "code": "https://github.com/Rtx09x/matbench",
        "research_archive": "https://github.com/Rtx09x/TRIADS",
        "prediction_input": prediction_file.name,
    })
    mb.to_file(str(output_file))
    reloaded = MatbenchBenchmark.from_file(str(output_file))
    if not reloaded.is_valid or not reloaded.is_recorded:
        raise ValueError("Generated results failed Matbench validation")
    print(f"PASS: recorded all 5 folds using the official Matbench API: {output_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.rebuilt.json.gz"))
    args = parser.parse_args()
    export_predictions(args.input, args.output)


if __name__ == "__main__":
    main()
