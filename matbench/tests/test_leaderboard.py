"""Classification scoring tests use five synthetic folds; no downloads."""

import copy
import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from matbench.constants import CLF_KEY, CLF_METRICS, MBV01_KEY, REG_KEY
from matbench.leaderboard import (
    prepare_classification_scores,
    probability_eligibility,
)
from matbench.util import MSONable2File, RecursiveDotDict


class SyntheticTask:
    """Five recorded folds with deliberately reordered prediction dictionaries."""

    def __init__(self, predictions=None, name="synthetic_classification"):
        self.dataset_name = name
        self.benchmark_name = MBV01_KEY
        self.metadata = RecursiveDotDict(
            {"task_type": CLF_KEY, "input_type": "composition", "target": "target"}
        )
        self.folds_map = {number: f"fold_{number}" for number in range(5)}
        self.folds_nums = self.folds = list(self.folds_map)
        self.folds_keys = list(self.folds_map.values())
        self.validation = RecursiveDotDict(
            {
                key: {"test": [f"positive_{number}", f"negative_{number}"]}
                for number, key in self.folds_map.items()
            }
        )
        self.truth = pd.Series(
            [True, False] * 5,
            index=[
                sample for fold in self.validation.values() for sample in fold.test
            ],
        )
        self.df = None
        self.load_calls = 0
        self.unload_calls = 0
        self.results = RecursiveDotDict({})
        if predictions is None:
            predictions = [
                [0.9, 0.8],
                [0.2, 0.1],
                [0.9, 0.8],
                [0.2, 0.1],
                [0.9, 0.8],
            ]
        for fold, values in enumerate(predictions):
            self.results[self.folds_map[fold]] = {
                "data": {
                    f"negative_{fold}": values[1],
                    f"positive_{fold}": values[0],
                },
                "scores": dict.fromkeys(CLF_METRICS, 0.5),
                "parameters": {},
                "uncertainty": None,
            }

    @property
    def scores(self):
        return RecursiveDotDict(
            {
                metric: {
                    op: float(
                        getattr(np, op)(
                            [
                                self.results[key]["scores"][metric]
                                for key in self.folds_keys
                            ]
                        )
                    )
                    for op in ("mean", "std", "min", "max")
                }
                for metric in CLF_METRICS
            }
        )

    def load(self):
        self.load_calls += 1
        self.df = pd.DataFrame({"composition": "H", "target": self.truth})

    def unload(self):
        self.unload_calls += 1
        self.df = None

    def get_test_data(self, fold_number, as_type="tuple", include_target=False):
        if not include_target:
            raise AssertionError("Scoring must explicitly request test targets")
        ids = self.validation[self.folds_map[fold_number]].test
        return pd.Series("H", index=ids), self.truth.loc[ids]


def snapshot(value):
    """Copy RecursiveDotDict without invoking its dynamic dunder attributes."""
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return copy.deepcopy(value)


def packet_for(*tasks):
    return {
        "synthetic": {
            "results": SimpleNamespace(benchmark_name=MBV01_KEY, tasks=list(tasks)),
            "info": {"algorithm": "Synthetic"},
            "dir_name_short": "synthetic",
        }
    }


class TestProbabilityLeaderboard(unittest.TestCase):
    def evaluate(self, task):
        packets = packet_for(task)
        prepare_classification_scores(packets)
        return packets["synthetic"]["classification_scores"][task.dataset_name]

    def test_recomputes_probabilities_by_id_and_averages_folds(self):
        task = SyntheticTask()
        original = snapshot(task.results)
        entry = self.evaluate(task)
        self.assertTrue(entry["eligible"])
        self.assertTrue(entry["ba_available"])
        self.assertTrue(entry["auc_available"])
        self.assertIsNone(entry["reason"])
        self.assertEqual(entry["scores"].rocauc.mean, 1.0)
        self.assertEqual(entry["scores"].rocauc.std, 0.0)
        self.assertEqual(len(entry["fold_scores"]), 5)
        for fold_scores in entry["fold_scores"].values():
            self.assertEqual(fold_scores["rocauc"], 1.0)
            self.assertEqual(fold_scores["balanced_accuracy"], 0.5)
        self.assertEqual(entry["stored_scores"].rocauc.mean, 0.5)
        self.assertEqual(task.results, original)
        pooled_auc = roc_auc_score(
            task.truth, [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1, 0.9, 0.8]
        )
        self.assertNotEqual(entry["scores"].rocauc.mean, pooled_auc)

    def test_fold_distributions_are_independent_and_use_population_std(self):
        task = SyntheticTask(
            [[0.9, 0.8], [0.1, 0.2], [0.9, 0.2], [0.1, 0.9], [0.9, 0.8]]
        )
        entry = self.evaluate(task)
        auc = [1.0, 0.0, 1.0, 0.0, 1.0]
        ba = [0.5, 0.5, 1.0, 0.0, 0.5]
        self.assertEqual(entry["scores"].rocauc.mean, np.mean(auc))
        self.assertEqual(entry["scores"].rocauc.std, np.std(auc, ddof=0))
        self.assertEqual(entry["scores"].balanced_accuracy.std, np.std(ba, ddof=0))
        self.assertNotEqual(
            entry["scores"].rocauc.std, entry["scores"].balanced_accuracy.std
        )
        self.assertEqual(entry["scores"].rocauc.min, 0.0)
        self.assertEqual(entry["scores"].rocauc.max, 1.0)

    def test_hard_labels_recompute_ba_without_restoring_probability_auc(self):
        for cast in (bool, int, np.bool_, np.int64):
            with self.subTest(cast=cast):
                task = SyntheticTask([[cast(1), cast(0)]] * 5)
                original = snapshot(task.results)
                entry = self.evaluate(task)
                self.assertFalse(entry["eligible"])
                self.assertFalse(entry["auc_available"])
                self.assertTrue(entry["ba_available"])
                self.assertIsNone(entry["ba_reason"])
                self.assertIn("cannot be recovered", entry["auc_reason"])
                self.assertEqual(entry["prediction_type"], "hard_labels")
                self.assertEqual(entry["scores"].balanced_accuracy.mean, 1.0)
                self.assertEqual(entry["scores"].balanced_accuracy.std, 0.0)
                self.assertEqual(entry["stored_scores"].balanced_accuracy.mean, 0.5)
                self.assertIsNone(entry["scores"].rocauc)
                for fold in entry["fold_scores"].values():
                    self.assertEqual(fold["balanced_accuracy"], 1.0)
                    self.assertIsNone(fold["rocauc"])
                self.assertEqual(task.results, original)
                self.assertEqual(task.load_calls, 1)

    def test_invalid_predictions_cannot_fall_back_to_ba(self):
        invalid_pairs = [
            [True, False],
            [1, 0],
            [0.8, False],
            [0.8, 0],
            [float("nan"), 0.2],
            [float("inf"), 0.2],
            [1.01, 0.2],
            [-0.01, 0.2],
            ["0.8", "0.2"],
            [None, 0.2],
        ]
        for values in invalid_pairs:
            with self.subTest(values=values):
                task = SyntheticTask([[0.9, 0.1]] * 4 + [values])
                entry = self.evaluate(task)
                self.assertFalse(entry["eligible"])
                self.assertFalse(entry["ba_available"])
                self.assertTrue(entry["ba_reason"])
                self.assertIsNone(entry["scores"])
                self.assertIsNone(entry["fold_scores"])
                self.assertEqual(task.load_calls, 0)
        for values in [[2, 0], [-1, 1], [True, 0]]:
            with self.subTest(hard_labels=values):
                entry = self.evaluate(SyntheticTask([values] * 5))
                self.assertFalse(entry["ba_available"])
                self.assertFalse(entry["auc_available"])

    def test_accepts_numpy_floats_and_probability_endpoints(self):
        task = SyntheticTask(
            [
                [np.float32(1.0), np.float32(0.0)],
                [np.float64(0.8), np.float64(0.2)],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        entry = self.evaluate(task)
        self.assertTrue(entry["eligible"])
        self.assertEqual(entry["scores"].rocauc.mean, 1.0)
        self.assertEqual(entry["scores"].balanced_accuracy.mean, 1.0)
        # Endpoint-only probability outputs are also numerically valid.
        for values, expected in [([1.0, 0.0], 1.0), ([0.0, 1.0], 0.0)]:
            entry = self.evaluate(SyntheticTask([values] * 5))
            self.assertTrue(entry["auc_available"])
            self.assertEqual(entry["scores"].rocauc.mean, expected)

    def test_probability_decisions_use_strict_greater_than_half(self):
        entry = self.evaluate(SyntheticTask([[0.5, 0.5]] * 5))
        self.assertEqual(entry["scores"].balanced_accuracy.mean, 0.5)
        self.assertEqual(entry["scores"].f1.mean, 0.0)

    def test_regression_task_does_not_enter_classification_scores(self):
        task = SimpleNamespace(
            dataset_name="regression", metadata=SimpleNamespace(task_type=REG_KEY)
        )
        packets = packet_for(task)
        prepare_classification_scores(packets)
        self.assertEqual(packets["synthetic"]["classification_scores"], {})

    def test_loads_targets_once_for_both_prediction_formats_and_releases_data(self):
        first = SyntheticTask([[True, False]] * 5)
        second = SyntheticTask()
        packets = packet_for(first)
        packets["another"] = packet_for(second)["synthetic"]
        prepare_classification_scores(packets)
        self.assertEqual(first.load_calls + second.load_calls, 1)
        self.assertEqual(first.unload_calls + second.unload_calls, 1)
        self.assertIsNone(first.df)
        self.assertIsNone(second.df)
        self.assertTrue(
            packets["another"]["classification_scores"][second.dataset_name][
                "eligible"
            ]
        )

    def test_keeps_preloaded_task_data_loaded(self):
        task = SyntheticTask()
        task.load()
        original_df = task.df
        self.evaluate(task)
        self.assertEqual(task.load_calls, 1)
        self.assertEqual(task.unload_calls, 0)
        self.assertIs(task.df, original_df)

    def test_sample_id_mismatch_aborts_for_probabilities_and_hard_labels(self):
        for values in [[0.9, 0.1], [True, False]]:
            for mode in ("missing", "extra", "wrong"):
                with self.subTest(values=values, mode=mode):
                    task = SyntheticTask([values] * 5)
                    data = task.results["fold_1"]["data"]
                    if mode != "extra":
                        data.pop("positive_1")
                    if mode != "missing":
                        data["unknown_id"] = values[0]
                    with self.assertRaisesRegex(ValueError, "IDs"):
                        self.evaluate(task)

    def test_duplicate_official_ids_abort_for_both_formats(self):
        for values in [[0.9, 0.1], [True, False]]:
            task = SyntheticTask([values] * 5)
            task.validation.fold_1.test = ["positive_1", "positive_1", "negative_1"]
            with self.assertRaisesRegex(ValueError, "IDs"):
                self.evaluate(task)
            task = SyntheticTask([values] * 5)
            task.validation.fold_1.test = ["positive_0", "negative_0"]
            task.results.fold_1.data = snapshot(task.results.fold_0.data)
            with self.assertRaisesRegex(ValueError, "duplicate test IDs"):
                self.evaluate(task)

    def test_undefined_metrics_abort_for_both_formats(self):
        for values in [[0.9, 0.1], [True, False]]:
            task = SyntheticTask([values] * 5)
            task.truth.loc[["positive_1", "negative_1"]] = True
            with self.assertRaisesRegex(ValueError, "binary target classes"):
                self.evaluate(task)

    def test_missing_empty_extra_or_incomplete_fold_mapping_is_excluded(self):
        for values in [[0.9, 0.1], [True, False]]:
            for mode in ("missing", "empty", "extra", "mapping"):
                with self.subTest(values=values, mode=mode):
                    task = SyntheticTask([values] * 5)
                    if mode == "missing":
                        task.results.pop("fold_1")
                    elif mode == "empty":
                        task.results["fold_1"]["data"] = {}
                    elif mode == "extra":
                        task.results["fold_5"] = snapshot(task.results["fold_0"])
                    else:
                        task.folds_map.pop(4)
                    original = snapshot(task.results)
                    eligible, reason = probability_eligibility(task)
                    self.assertFalse(eligible)
                    self.assertTrue(reason)
                    entry = self.evaluate(task)
                    self.assertFalse(entry["ba_available"])
                    self.assertFalse(entry["auc_available"])
                    self.assertEqual(task.load_calls, 0)
                    self.assertEqual(task.results, original)

    def test_repeated_preparation_is_stable_and_preserves_stored_scores(self):
        for values in [[0.9, 0.1], [True, False]]:
            task = SyntheticTask([values] * 5)
            self.assertEqual(self.evaluate(task), self.evaluate(task))
            self.assertEqual(task.scores.rocauc.mean, 0.5)

    def test_duplicate_json_sample_ids_are_rejected_before_deserialization(self):
        class Passthrough(MSONable2File):
            @classmethod
            def from_dict(cls, data):
                return data

        invalid = '{"fold_0": {"data": {"sample": true, "sample": false}}}'
        valid = '{"fold_0": {"data": {"sample": true, "other": false}}}'
        with tempfile.TemporaryDirectory() as directory:
            for suffix, opener in [(".json", open), (".json.gz", gzip.open)]:
                path = Path(directory) / ("result" + suffix)
                with opener(path, "wt") as handle:
                    handle.write(invalid)
                with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                    Passthrough.from_file(str(path))
                with opener(path, "wt") as handle:
                    handle.write(valid)
                original = path.read_bytes()
                self.assertEqual(
                    Passthrough.from_file(str(path))["fold_0"]["data"],
                    {"sample": True, "other": False},
                )
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
