"""Regression tests for independent verification of archived prediction scores."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from matbench.bench import MatbenchBenchmark

from verify_submission import verify


RESULTS = Path(__file__).with_name("results.json.gz")


class TestVerifySubmission(unittest.TestCase):
    def test_historical_predictions_pass_official_verification(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify(RESULTS, official=True)

    def check_recomputed_score_rejects(self, benchmark, metric):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json.gz"
            # Serialization updates the benchmark hash without recalculating scores.
            # Thus the test reaches numerical verification, not a hash rejection.
            benchmark.to_file(str(path))
            self.assertTrue(MatbenchBenchmark.from_file(str(path)).is_valid)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    AssertionError, f"Fold 0 {metric} differs from recomputed official score"
                ):
                    verify(path, official=True)

    def test_changed_prediction_with_stale_scores_is_rejected(self):
        benchmark = MatbenchBenchmark.from_file(str(RESULTS))
        predictions = benchmark.tasks_map["matbench_steels"].results["fold_0"]["data"]
        sample_id = next(iter(predictions))
        predictions[sample_id] += 10000.0
        self.check_recomputed_score_rejects(benchmark, "mae")

    def test_non_mae_scores_are_recomputed(self):
        for metric in ("rmse", "mape", "max_error"):
            with self.subTest(metric=metric):
                benchmark = MatbenchBenchmark.from_file(str(RESULTS))
                scores = benchmark.tasks_map["matbench_steels"].results["fold_0"]["scores"]
                scores[metric] += 1.0
                self.check_recomputed_score_rejects(benchmark, metric)


if __name__ == "__main__":
    unittest.main()
