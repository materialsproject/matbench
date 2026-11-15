"""Protect identities, target isolation and newly corrected template behavior."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import prepare_inputs as prep


class FakeTask:
    dataset_name = "matbench_glass"
    folds = [0, 1, 2, 3, 4]

    def __init__(self):
        self.train = pd.Series(["Fe2O3", "NaCl"], index=["mb-glass-2", "mb-glass-1"])
        self.test = pd.Series(["Al2O3"], index=["mb-glass-3"])
        self.called = []

    def get_train_and_val_data(self, fold):
        self.called.append(("train", fold))
        return self.train, pd.Series(["SECRET_TARGET_A", "SECRET_TARGET_B"])

    def get_test_data(self, fold, include_target):
        if include_target is not False:
            raise AssertionError("Test targets must not be requested")
        self.called.append(("test", fold, include_target))
        return self.test


class PrepareTests(unittest.TestCase):
    def test_official_ids_are_preserved_and_targets_are_never_serialized(self):
        task = FakeTask()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "inputs"
            manifest = prep.prepare(task, 0, output)
            train = (output / "train.jsonl").read_text()
            test = (output / "test.jsonl").read_text()
            rows = [json.loads(line) for line in train.splitlines()]
            self.assertEqual(
                [row["sample_id"] for row in rows], list(task.train.index)
            )
            self.assertTrue(
                all(
                    set(row) == {"sample_id", "instruction", "input"} for row in rows
                )
            )
            self.assertNotIn("SECRET_TARGET", train + test)
            self.assertEqual(
                manifest["train_sha256"], prep.sha256(output / "train.jsonl")
            )
            self.assertEqual(
                manifest["test_sha256"], prep.sha256(output / "test.jsonl")
            )
            self.assertFalse(
                manifest["input_provenance"]["historical_v1_reproduction"]
            )
            self.assertEqual(task.called, [("train", 0), ("test", 0, False)])

    def test_duplicate_and_overlapping_ids_fail_without_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            for duplicate in (True, False):
                task = FakeTask()
                if duplicate:
                    task.train.index = ["mb-glass-1", "mb-glass-1"]
                else:
                    task.test.index = ["mb-glass-1"]
                output = Path(folder) / "inputs"
                with self.assertRaises(ValueError):
                    prep.prepare(task, 0, output)
                self.assertFalse(output.exists())

    def test_existing_directory_and_contents_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            marker = output / "keep.txt"
            marker.write_text("original")
            task = FakeTask()
            with self.assertRaises(FileExistsError):
                prep.prepare(task, 0, output)
            self.assertEqual(marker.read_text(), "original")
            self.assertEqual(task.called, [])

    def test_generation_failure_leaves_no_completed_or_partial_directory(self):
        def failed_renderer(task_name, value):
            raise RuntimeError("Cannot describe this structure")

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "inputs"
            with self.assertRaisesRegex(RuntimeError, "Cannot describe"):
                prep.prepare(FakeTask(), 0, output, renderer=failed_renderer)
            self.assertFalse(output.exists())

    def test_periodic_images_keep_their_actual_neighbor_distance(self):
        from pymatgen.core import Lattice, Structure

        structure = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = prep.get_structure_nl(structure)
        self.assertIn("Fe (three point zero Angstrom away)", text)
        self.assertNotIn("None", text)
        self.assertEqual(prep.n2e(0.0), "zero")

    def test_unordered_groups_are_stable_and_neighbor_errors_are_not_hidden(self):
        from pymatgen.core import Lattice, Structure

        self.assertEqual(prep.join_unique(["z", "a", "z"]), "a, z")
        structure = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        with patch(
            "pymatgen.analysis.local_env.CrystalNN.get_nn_info",
            side_effect=ValueError("bad neighbor"),
        ):
            with self.assertRaisesRegex(ValueError, "bad neighbor"):
                prep.get_structure_nl(structure)


if __name__ == "__main__":
    unittest.main()
