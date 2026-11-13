"""Safety checks for ID alignment and preservation of the submitted artifact."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).with_name("record_predictions.py")
SPEC = importlib.util.spec_from_file_location("record_predictions", SCRIPT)
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


class RecordingSafetyTests(unittest.TestCase):
    def test_reorders_explicit_ids_without_thresholding(self):
        actual = recorder.align_predictions(
            {"mb-2": 0.65, "mb-1": 0.7}, ["mb-1", "mb-2"], True
        )
        self.assertEqual(actual, [0.7, 0.65])

    def test_rejects_missing_extra_nonfinite_and_bad_probabilities(self):
        for values in (
            {"mb-1": 0.3},
            {"mb-1": 0.3, "unknown": 0.4},
            {"mb-1": float("nan"), "mb-2": 0.2},
            {"mb-1": float("inf"), "mb-2": 0.2},
            {"mb-1": 1.01, "mb-2": 0.2},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                recorder.align_predictions(values, ["mb-1", "mb-2"], True)

    def test_rejects_legacy_and_duplicate_csv_rows(self):
        csvs = (
            "predictions,ratings\n0.7,1\n",
            "task,fold,sample_id,prediction\nt,0,mb-1,0.7\nt,0,mb-1,0.8\n",
            "task,fold,sample_id,prediction\nt,0,mb-1,nan\n",
            "task,fold,sample_id,prediction\nt,0,mb-1,0.7,extra\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            for content in csvs:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    recorder.load_predictions(path)

    def test_preserves_old_results_and_existing_outputs(self):
        class Benchmark:
            def to_file(self, filename):
                Path(filename).write_bytes(b"new result")

        with tempfile.TemporaryDirectory() as directory:
            historical = Path(directory) / "results.json.gz"
            historical.write_bytes(b"original result")
            with self.assertRaises(ValueError):
                recorder.write_new_benchmark(Benchmark(), historical)
            self.assertEqual(historical.read_bytes(), b"original result")
            output = Path(directory) / "fresh_results.json.gz"
            recorder.write_new_benchmark(Benchmark(), output)
            with self.assertRaises(FileExistsError):
                recorder.write_new_benchmark(Benchmark(), output)
            self.assertEqual(output.read_bytes(), b"new result")

    def test_record_uses_target_free_inputs_and_preserves_probability_order(self):
        calls = []

        class Task:
            dataset_name = "test_task"
            folds = [0]
            is_recorded = {0: False}
            metadata = SimpleNamespace(task_type="classification")

            def load(self):
                pass

            def get_test_data(self, fold, include_target):
                self_test.assertFalse(include_target)
                return SimpleNamespace(index=["mb-1", "mb-2"])

            def record(self, fold, predictions, params):
                calls.append((fold, predictions, params))

        class Benchmark:
            tasks = [Task()]
            is_valid = True
            is_recorded = True

            def add_metadata(self, metadata):
                self.metadata = metadata

        self_test = self
        benchmark = Benchmark()
        recorder.record_predictions(
            benchmark,
            {("test_task", 0): {"mb-2": 0.65, "mb-1": 0.7}},
            "verified-input-digest",
            "predictions.csv",
            {"test_task": {"0": {"seed": 42}}},
        )
        self.assertEqual(calls[0][1], [0.7, 0.65])
        self.assertEqual(calls[0][2]["supplied_training_parameters"], {"seed": 42})
        self.assertFalse(benchmark.metadata["historical_reproduction_verified"])
        with self.assertRaises(ValueError):
            recorder.record_predictions(benchmark, {}, "hash", "empty.csv")

    def test_rejects_previously_recorded_benchmark_before_changing_provenance(self):
        benchmark = SimpleNamespace(
            tasks=[SimpleNamespace(is_recorded={0: True})],
            metadata={"prediction_csv": "original.csv"},
        )
        with self.assertRaisesRegex(ValueError, "fresh benchmark"):
            recorder.record_predictions(benchmark, {}, "new-hash", "new.csv")
        self.assertEqual(benchmark.metadata, {"prediction_csv": "original.csv"})


if __name__ == "__main__":
    unittest.main()
