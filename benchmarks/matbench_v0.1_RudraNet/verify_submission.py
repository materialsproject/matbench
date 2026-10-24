#!/usr/bin/env python3
"""Read-only checks for the RudraNet matbench_steels submission.

These checks establish fold membership and consistency with the archived V13A
summary. They do NOT establish checkpoint-level prediction provenance, exclude
historical test-based model selection, or retrain the model.

Usage (from benchmark submission directory):
  python verify_submission.py
  python verify_submission.py --official
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from sklearn.model_selection import KFold

TASK = "matbench_steels"
SPLIT_SEED = 18012019
N_SAMPLES = 312
EXPECTED_FOLD_SIZES = (63, 63, 62, 62, 62)
EXPECTED_V13A_MAES = (114.3248, 81.4639, 80.5513, 90.4915, 89.1827)
EXPECTED_MEAN_MAE = 91.2028
TOLERANCE = 0.0002  # archived summary rounds MAE to four decimals


def verify(results_path: Path, official: bool = False) -> None:
    with gzip.open(results_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload.get("benchmark_name") == "matbench_v0.1", "Wrong benchmark"
    task = payload["tasks"][TASK]
    folds = task["results"]
    assert set(folds) == {f"fold_{i}" for i in range(5)}, "Unexpected folds"

    official_task = None
    if official:
        from matbench.bench import MatbenchBenchmark
        benchmark = MatbenchBenchmark.from_file(str(results_path))
        assert benchmark.is_valid, "MatbenchBenchmark.is_valid is false"
        official_task = benchmark.tasks_map[TASK]
        official_task.load()
        assert len(official_task.df) == N_SAMPLES, "Unexpected dataset length"

    all_ids = set()
    maes = []
    for fold_index, (_, test_indices) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(range(N_SAMPLES))
    ):
        fold_data = folds[f"fold_{fold_index}"]
        preds = fold_data["data"]
        submitted = set(preds)
        expected = {f"mb-steels-{row_index + 1:03d}" for row_index in test_indices}
        assert len(submitted) == EXPECTED_FOLD_SIZES[fold_index], f"Wrong fold {fold_index} size"
        assert submitted == expected, f"Fold {fold_index} indices differ from seeded Matbench protocol"
        assert all(math.isfinite(float(p)) for p in preds.values()), "Nonfinite prediction"
        assert not (submitted & all_ids), "Repeated sample across folds"
        all_ids.update(submitted)
        mae = float(fold_data["scores"]["mae"])
        assert abs(mae - EXPECTED_V13A_MAES[fold_index]) < TOLERANCE, (
            f"Fold {fold_index} MAE differs from archived V13A summary"
        )
        maes.append(mae)
        if official_task is not None:
            train_df = official_task.get_train_and_val_data(fold_index, as_type="df")
            test_df = official_task.get_test_data(fold_index, as_type="df")
            assert not set(train_df.index).intersection(test_df.index), "Train/test overlap"
            assert set(test_df.index) == expected, f"Official fold {fold_index} differs"

        print(f"Fold {fold_index}: {len(submitted)} IDs; MAE {mae:.4f} MPa; OK")

    assert len(all_ids) == N_SAMPLES, "Not all 312 samples included"
    mean_mae = sum(maes) / len(maes)
    assert abs(mean_mae - EXPECTED_MEAN_MAE) < TOLERANCE, "Mean MAE mismatch"
    print(f"PASS: {len(all_ids)} distinct IDs; mean MAE {mean_mae:.4f} MPa")
    if official:
        print("PASS: MatbenchBenchmark validator and official fold IDs match")
    else:
        print("Use --official to additionally check Matbench's packaged split definitions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path(__file__).resolve().with_name("results.json.gz"),
        help="Submitted MatbenchBenchmark results.json.gz",
    )
    parser.add_argument(
        "--official", action="store_true",
        help="Also load the official Matbench data and check its split membership",
    )
    args = parser.parse_args()
    verify(args.results, args.official)


if __name__ == "__main__":
    main()
