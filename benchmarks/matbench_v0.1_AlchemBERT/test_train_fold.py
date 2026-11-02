"""Protocol tests require no model downloads, GPU or historical checkpoints."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import train_fold as runner


class FakeTask:
    metadata = {"task_type": "regression"}

    def get_train_and_val_data(self, fold):
        ids = [f"train-{index}" for index in range(10)]
        return pd.Series(["material"] * 10, index=ids), pd.Series(
            np.arange(1.0, 11.0), index=ids
        )

    def get_test_data(self, fold, include_target=False):
        if include_target is not False:
            raise AssertionError("Outer test targets must never be requested")
        return pd.Series(["material", "material"], index=["test-1", "test-2"])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_rows(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def rows(self, ids):
        return [
            {"sample_id": value, "instruction": "Predict property", "input": value}
            for value in ids
        ]

    def test_inner_validation_labels_do_not_fit_transform(self):
        labels = np.array(
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        )
        train, validation, transformed, state = runner.split_and_transform(
            labels, seed=42, validation_fraction=0.2, transform=True
        )
        changed = labels.copy()
        changed[validation] *= 1000
        train2, validation2, transformed2, state2 = runner.split_and_transform(
            changed, seed=42, validation_fraction=0.2, transform=True
        )
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(validation, validation2)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(len(set(train) | set(validation)), len(labels))
        self.assertEqual(state, state2)
        np.testing.assert_array_equal(transformed[train], transformed2[train])
        # The persisted JSON, not a live Python transformer, must support replay.
        restored = json.loads(json.dumps(state))
        np.testing.assert_allclose(
            runner.inverse_targets(transformed, restored), labels
        )

    def test_inverse_transform_handles_both_signs_and_rejects_invalid_domain(self):
        from sklearn.preprocessing import PowerTransformer

        original = np.array([-100.0, -4.0, -0.25, 0.0, 0.1, 3.0, 200.0])
        for power in (0.0, 2.0, -0.4, 2.5):
            fitted = PowerTransformer(standardize=False)
            fitted.lambdas_ = np.array([power])
            state = {
                "method": "yeo-johnson",
                "lambda": power,
                "mean": 0.0,
                "scale": 1.0,
            }
            values = fitted.transform(original[:, None]).ravel()
            np.testing.assert_allclose(
                runner.inverse_targets(values, state), original, atol=1e-10
            )
        invalid = {
            "method": "yeo-johnson",
            "lambda": -1.0,
            "mean": 0.0,
            "scale": 1.0,
        }
        with self.assertRaises(FloatingPointError):
            runner.inverse_targets([2.0], invalid)

    def test_input_ids_align_and_targets_are_rejected(self):
        path = self.write_rows("inputs.jsonl", self.rows(["b", "a"]))
        texts = runner.load_texts(path, runner.digest(path), ["a", "b"])
        self.assertEqual(texts, ["Predict property a", "Predict property b"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.load_texts(path, "0" * 64, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "official fold"):
            runner.load_texts(path, runner.digest(path), ["a", "c"])
        path = self.write_rows(
            "labels.jsonl", [{**self.rows(["a"])[0], "output": 3}]
        )
        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            runner.load_texts(path, runner.digest(path), ["a"])
        path = self.write_rows("duplicate.jsonl", self.rows(["a", "a"]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.load_texts(path, runner.digest(path), ["a"])

    def test_official_training_labels_and_target_free_test_inputs(self):
        train_path = self.write_rows(
            "train.jsonl",
            self.rows([f"train-{index}" for index in reversed(range(10))]),
        )
        test_path = self.write_rows("test.jsonl", self.rows(["test-2", "test-1"]))
        config = {
            **runner.DEFAULTS,
            "task": "matbench_steels",
            "fold": 0,
            "train_file": str(train_path),
            "train_sha256": runner.digest(train_path),
            "test_file": str(test_path),
            "test_sha256": runner.digest(test_path),
        }
        prepared = runner.prepare_fold(config, FakeTask())
        self.assertEqual(prepared["test_ids"], ["test-1", "test-2"])
        self.assertEqual(
            prepared["test_texts"],
            ["Predict property test-1", "Predict property test-2"],
        )
        np.testing.assert_allclose(
            runner.inverse_targets(prepared["labels"], prepared["target_transform"]),
            np.arange(1.0, 11.0),
        )

    def test_config_requires_immutable_inputs_and_model(self):
        config = {
            "task": "matbench_steels",
            "fold": 0,
            "model_name": "bert-base-cased",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "train_file": "train.jsonl",
            "train_sha256": "c" * 64,
            "test_file": "test.jsonl",
            "test_sha256": "d" * 64,
            "input_provenance": {"template_version": "new-inputs-v1"},
            "output_dir": "new-run",
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        parsed = runner.read_config(path)
        self.assertEqual(parsed["train_file"], str(self.root / "train.jsonl"))
        self.assertEqual(parsed["seed"], 42)
        self.assertEqual(parsed["learning_rate"], 1e-5)
        config["model_revision"] = "main"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "immutable"):
            runner.read_config(path)
        config["model_revision"] = "a" * 40
        del config["input_provenance"]
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "missing"):
            runner.read_config(path)

    def test_probability_export_is_unrounded_and_exclusive(self):
        path = self.root / "predictions.csv"
        values = [0.123456789, 0.987654321]
        config = {"task": "matbench_glass", "fold": 2}
        runner.write_predictions(path, config, ["a", "b"], values, True)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["prediction"]) for row in rows], values)
        with self.assertRaises(FileExistsError):
            runner.write_predictions(path, config, ["a", "b"], values, True)
        for invalid in ([float("nan")], [1.1], [-0.1]):
            with self.assertRaises(ValueError):
                runner.write_predictions(
                    self.root / "bad.csv", config, ["a"], invalid, True
                )
        self.assertFalse((self.root / "bad.csv").exists())


if __name__ == "__main__":
    unittest.main()
