import unittest

import numpy as np
from pymatgen import Structure

from matbench.data import load
from matbench.constants import DATASETS, VALIDATION, REG_KEY, CLF_KEY


class TestMetadata(unittest.TestCase):
    def test_datasets_metadata_file(self):
        for metadata in DATASETS.values():
            for key in ["problem_type", "n_samples", "input_type", "target"]:
                self.assertIn(key, metadata.keys())

                if key == "input_type":
                    self.assertIn(metadata[key], ["composition", "structure"])

                elif key == "problem_type":
                    self.assertIn(metadata[key], [REG_KEY, CLF_KEY])
        self.assertEqual(len(list(DATASETS.values())), 13)

    def test_validation_metadata_file(self):
        self.assertIn("common", VALIDATION.keys())
        self.assertIn("seed", VALIDATION["common"].keys())
        self.assertIn("n_spits", VALIDATION["common"].keys())


class TestLoad(unittest.TestCase):

    def test_downloads(self):
        for dataset, metadata in DATASETS.items():
            print(f"Testing {dataset} with metadata {metadata}")
            df = load(dataset)

            self.assertEqual(df.shape[0], metadata["n_samples"])
            self.assertIn(metadata["target"], df.columns)

            input_type = metadata["input_type"]
            test_type = Structure if input_type == "structure" else str
            self.assertTrue(all([isinstance(input, test_type) for input in
                                 df[metadata["input_type"]]]))

            problem_type = metadata["problem_type"]
            test_types = [np.bool_, bool] if problem_type == CLF_KEY else [
                np.float_, np.float32, np.float64, float]
            self.assertIn(df[metadata["target"]].dtypes, test_types)
