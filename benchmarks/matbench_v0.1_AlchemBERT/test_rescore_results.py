"""Tests for trusted labels, ID alignment and probability-based rescoring."""

import hashlib
import gzip
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from rescore_results import (
    DATASET_SHA256,
    build_report,
    dataset_labels,
    read_verified_json,
    rescore_task,
    write_report,
)


class TestHistoricalRescoring(unittest.TestCase):
    def setUp(self):
        self.split = {"fold_0": {"test": ["a", "b", "c", "d"]}}
        self.truth = {"a": False, "b": True, "c": False, "d": True}
        self.task = {
            "results": {
                "fold_0": {
                    # Deliberately different insertion order from official IDs.
                    "data": {"d": 0.9, "b": 0.4, "a": 0.1, "c": 0.6},
                    "scores": {
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "f1": 0.5,
                        "rocauc": 0.5,
                    },
                }
            }
        }

    def test_probability_auc_and_id_alignment(self):
        before = json.dumps(self.task, sort_keys=True)
        report = rescore_task(self.task, self.truth, self.split, "classification")
        scores = report["folds"]["fold_0"]["recomputed_scores"]
        self.assertEqual(scores["rocauc"], 0.75)
        self.assertEqual(scores["balanced_accuracy"], 0.5)
        self.assertEqual(json.dumps(self.task, sort_keys=True), before)

    def test_unexpected_scorer_is_rejected(self):
        with patch("rescore_results.SCORER_SHA256", "incorrect"):
            with self.assertRaisesRegex(ValueError, "scorer SHA256 mismatch"):
                build_report()

    def test_mismatched_truth_or_prediction_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Truth IDs"):
            rescore_task(
                self.task,
                {**self.truth, "extra": False},
                self.split,
                "classification",
            )
        self.task["results"]["fold_0"]["data"]["extra"] = 0.1
        with self.assertRaisesRegex(ValueError, "Prediction IDs"):
            rescore_task(self.task, self.truth, self.split, "classification")

    def test_untrusted_data_and_error_pages_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.json.gz"
            good = b'{"index": [0], "data": [[true]]}'
            path.write_bytes(good)
            expected = hashlib.sha256(good).hexdigest()
            self.assertEqual(read_verified_json(path, expected)["index"], [0])
            path.write_bytes(b"<html>403 Forbidden</html>")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                read_verified_json(path, expected, compressed=True)

    def test_invalid_probability_and_truth_fail(self):
        self.task["results"]["fold_0"]["data"]["a"] = 1.2
        with self.assertRaisesRegex(ValueError, "probabilities"):
            rescore_task(self.task, self.truth, self.split, "classification")
        self.truth["a"] = float("nan")
        with self.assertRaisesRegex(ValueError, "truth label"):
            rescore_task(self.task, self.truth, self.split, "classification")

    def test_existing_reports_and_results_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            write_report({"value": 1}, path)
            with self.assertRaises(FileExistsError):
                write_report({"value": 2}, path)
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            with self.assertRaises(ValueError):
                write_report({}, Path(temp) / "results.json.gz")

    def test_streamed_dataset_targets_keep_original_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "matbench_glass"
            raw = {
                "index": [0, 1],
                "columns": ["structure", "target"],
                "data": [[{"nested": [1, 2]}, False], [{"nested": [3]}, True]],
            }
            data = gzip.compress(json.dumps(raw).encode())
            (root / f"{name}.json.gz").write_bytes(data)
            with patch.dict(
                DATASET_SHA256, {name: hashlib.sha256(data).hexdigest()}
            ):
                labels, _ = dataset_labels(
                    name, {"n_samples": 2, "target": "target"}, root
                )
            self.assertEqual(labels, {"mb-glass-1": False, "mb-glass-2": True})


if __name__ == "__main__":
    unittest.main()
