"""Probability leaderboard tests use synthetic, ID-aligned folds; no downloads."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from matbench.constants import CLF_KEY, CLF_METRICS, MBV01_KEY, REG_KEY
from matbench.leaderboard import prepare_classification_scores, probability_eligibility
from matbench.util import RecursiveDotDict


class SyntheticTask:
    """Small recorded task with deliberately reordered prediction dictionaries."""

    def __init__(self, predictions=None, name="synthetic_classification"):
        self.dataset_name = name
        self.benchmark_name = MBV01_KEY
        self.metadata = RecursiveDotDict({
            "task_type": CLF_KEY, "input_type": "composition", "target": "target"
        })
        self.folds_map = {0: "fold_0", 1: "fold_1"}
        self.folds_nums = self.folds = [0, 1]
        self.folds_keys = list(self.folds_map.values())
        self.validation = RecursiveDotDict({
            "fold_0": {"test": ["positive_0", "negative_0"]},
            "fold_1": {"test": ["positive_1", "negative_1"]},
        })
        self.truth = pd.Series(
            [True, False, True, False],
            index=["positive_0", "negative_0", "positive_1", "negative_1"],
        )
        self.df = None
        self.load_calls = 0
        self.unload_calls = 0
        self.results = RecursiveDotDict({})
        predictions = predictions or [[0.9, 0.8], [0.2, 0.1]]
        for fold, values in enumerate(predictions):
            # Saved JSON key order must never stand in for sample alignment.
            self.results[self.folds_map[fold]] = {
                "data": {f"negative_{fold}": values[1], f"positive_{fold}": values[0]},
                "scores": dict.fromkeys(CLF_METRICS, 0.5),
                "parameters": {},
                "uncertainty": None,
            }

    @property
    def scores(self):
        return RecursiveDotDict({
            metric: {
                op: float(getattr(np, op)([
                    self.results[key]["scores"][metric] for key in self.folds_keys
                ]))
                for op in ("mean", "std", "min", "max")
            }
            for metric in CLF_METRICS
        })

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
        original = {key: {field: dict(value) if isinstance(value, dict) else value
                          for field, value in fold.items()}
                    for key, fold in task.results.items()}
        entry = self.evaluate(task)
        self.assertTrue(entry["eligible"])
        self.assertIsNone(entry["reason"])
        self.assertEqual(entry["scores"].rocauc.mean, 1.0)
        self.assertEqual(entry["scores"].rocauc.std, 0.0)
        for fold_scores in entry["fold_scores"].values():
            self.assertEqual(fold_scores["rocauc"], 1.0)
            self.assertEqual(fold_scores["balanced_accuracy"], 0.5)
        self.assertEqual(entry["stored_scores"].rocauc.mean, 0.5)
        self.assertEqual(task.results, original)
        pooled_auc = roc_auc_score(task.truth, [0.9, 0.8, 0.2, 0.1])
        self.assertEqual(pooled_auc, 0.75)
        self.assertNotEqual(entry["scores"].rocauc.mean, pooled_auc)

    def test_fold_distribution_uses_recomputed_values(self):
        task = SyntheticTask([[0.9, 0.8], [0.1, 0.2]])
        entry = self.evaluate(task)
        self.assertEqual(entry["scores"].rocauc.mean, 0.5)
        self.assertEqual(entry["scores"].rocauc.std, 0.5)
        self.assertEqual(entry["scores"].rocauc.min, 0.0)
        self.assertEqual(entry["scores"].rocauc.max, 1.0)

    def test_rejects_non_probability_predictions_in_any_fold(self):
        bad_pairs = [
            [True, False], [np.bool_(True), np.bool_(False)], [1, 0],
            [0.8, False], [0.8, 0], [float("nan"), 0.2],
            [float("inf"), 0.2], [1.01, 0.2], [-0.01, 0.2],
        ]
        for values in bad_pairs:
            with self.subTest(values=values):
                task = SyntheticTask([[0.9, 0.1], values])
                original = {key: {field: dict(value) if isinstance(value, dict) else value
                          for field, value in fold.items()}
                    for key, fold in task.results.items()}
                entry = self.evaluate(task)
                self.assertFalse(entry["eligible"])
                self.assertTrue(entry["reason"])
                self.assertIsNone(entry["scores"])
                self.assertIsNone(entry["fold_scores"])
                # NaN is not equal to itself, so compare keys/types/other values.
                for key in task.results:
                    self.assertEqual(task.results[key]["scores"], original[key]["scores"])
                self.assertEqual(task.load_calls, 0)

    def test_accepts_numpy_floats_and_probability_endpoints(self):
        task = SyntheticTask([
            [np.float32(1.0), np.float32(0.0)],
            [np.float64(0.8), np.float64(0.2)],
        ])
        entry = self.evaluate(task)
        self.assertTrue(entry["eligible"])
        self.assertEqual(entry["scores"].rocauc.mean, 1.0)
        self.assertEqual(entry["scores"].balanced_accuracy.mean, 1.0)

    def test_regression_task_does_not_enter_classification_scores(self):
        task = SimpleNamespace(
            dataset_name="synthetic_regression", metadata=SimpleNamespace(task_type=REG_KEY)
        )
        packets = packet_for(task)
        prepare_classification_scores(packets)
        self.assertEqual(packets["synthetic"]["classification_scores"], {})

    def test_loads_targets_once_per_dataset_and_releases_loaded_data(self):
        first = SyntheticTask()
        second = SyntheticTask()
        packets = packet_for(first)
        packets["another"] = packet_for(second)["synthetic"]
        prepare_classification_scores(packets)
        self.assertEqual(first.load_calls + second.load_calls, 1)
        self.assertEqual(first.unload_calls + second.unload_calls, 1)
        self.assertIsNone(first.df)
        self.assertIsNone(second.df)
        self.assertTrue(packets["another"]["classification_scores"][second.dataset_name]["eligible"])

    def test_keeps_preloaded_task_data_loaded(self):
        task = SyntheticTask()
        task.load()
        original_df = task.df
        self.evaluate(task)
        self.assertEqual(task.load_calls, 1)
        self.assertEqual(task.unload_calls, 0)
        self.assertIs(task.df, original_df)

    def test_sample_id_mismatch_aborts_instead_of_misaligning_scores(self):
        task = SyntheticTask()
        task.results["fold_1"]["data"]["unknown_id"] = task.results["fold_1"]["data"].pop("positive_1")
        with self.assertRaises(ValueError):
            self.evaluate(task)

    def test_undefined_auc_aborts_instead_of_publishing_nan(self):
        task = SyntheticTask()
        task.truth.loc[["positive_1", "negative_1"]] = True
        with self.assertRaises(ValueError):
            self.evaluate(task)

    def test_missing_or_empty_fold_is_ineligible(self):
        for mode in ("missing", "empty"):
            with self.subTest(mode=mode):
                task = SyntheticTask()
                if mode == "missing":
                    task.results.pop("fold_1")
                else:
                    task.results["fold_1"]["data"] = {}
                eligible, reason = probability_eligibility(task)
                self.assertFalse(eligible)
                self.assertTrue(reason)
                self.assertEqual(task.load_calls, 0)

    def test_repeated_preparation_is_stable_and_does_not_rewrite_stored_scores(self):
        task = SyntheticTask()
        first = self.evaluate(task)
        second = self.evaluate(task)
        self.assertEqual(first, second)
        self.assertEqual(task.scores.rocauc.mean, 0.5)


if __name__ == "__main__":
    unittest.main()
