import unittest

import numpy as np
from pymatgen import Structure

from matbench.data import load
from matbench.constants import datasets, validation, REG_KEY, CLF_KEY


class TestMetadata(unittest.TestCase):
    def test_datasets_metadata_file(self):
        for metadata in datasets.values():
            for key in ["task_type", "n_samples", "input_type", "target"]:
                self.assertIn(key, metadata.keys())

                if key == "input_type":
                    self.assertIn(metadata[key], ["composition", "structure"])

                elif key == "task_type":
                    self.assertIn(metadata[key], [REG_KEY, CLF_KEY])
        self.assertEqual(len(list(datasets.values())), 13)

    def test_validation_metadata_file(self):
        self.assertIn("common", validation.keys())
        self.assertIn("seed", validation["common"].keys())
        self.assertIn("n_spits", validation["common"].keys())


class TestLoad(unittest.TestCase):

    def test_downloads(self):
        for dataset, metadata in datasets.items():
            print(f"Testing {dataset} with metadata {metadata}")
            df = load(dataset)

            self.assertEqual(df.shape[0], metadata["n_samples"])
            self.assertIn(metadata["target"], df.columns)

            input_type = metadata["input_type"]
            test_type = Structure if input_type == "structure" else str
            self.assertTrue(all([isinstance(input, test_type) for input in
                                 df[metadata["input_type"]]]))

            problem_type = metadata["task_type"]
            test_types = [np.bool_, bool] if problem_type == CLF_KEY else [
                np.float_, np.float32, np.float64, float]
            self.assertIn(df[metadata["target"]].dtypes, test_types)
