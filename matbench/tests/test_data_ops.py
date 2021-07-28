import unittest

import numpy as np
from pymatgen.core import Structure

from matbench.constants import CLF_KEY, REG_KEY
from matbench.data_ops import load, mean_absolute_percentage_error, score_array
from matbench.metadata import mbv01_metadata
from matbench.tests.util import FULL_TEST


class TestDataOps(unittest.TestCase):
    def test_downloads_mbv01(self):

        if FULL_TEST:
            metadata = mbv01_metadata
        else:
            restricted_set = [
                "matbench_dielectric",
                "matbench_steels",
                "matbench_glass",
            ]
            metadata = {
                k: v for k, v in mbv01_metadata.items() if k in restricted_set
            }

        for dataset, metadata in metadata.items():
            df = load(dataset)

            self.assertEqual(df.shape[0], metadata["n_samples"])
            self.assertIn(metadata["target"], df.columns)

            input_type = metadata["input_type"]
            test_type = Structure if input_type == "structure" else str
            self.assertTrue(
                all(
                    [
                        isinstance(input, test_type)
                        for input in df[metadata["input_type"]]
                    ]
                )
            )

            problem_type = metadata["task_type"]
            test_types = (
                [np.bool_, bool]
                if problem_type == CLF_KEY
                else [np.float_, np.float32, np.float64, float]
            )
            self.assertIn(df[metadata["target"]].dtypes, test_types)

    def test_score_array(self):
        # test for regression
        true = [1, 2, 3, 4]
        test = [1, 3, 3, 4]
        ans = score_array(true, test, task_type=REG_KEY)
        true_ans = {"mae": 0.25, "rmse": 0.5, "mape": 0.125, "max_error": 1}
        self.assertDictEqual(ans, true_ans)

        # test for classification
        true = [True, False]
        test = [True, True]
        ans = score_array(true, test, task_type=CLF_KEY)
        true_ans = {
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "f1": 0.6666666666666666,
            "rocauc": 0.5,
        }
        self.assertDictEqual(ans, true_ans)

    def test_mean_absolute_percentage_error(self):

        true = [1, 100, 1000, 0, 0.00000001]
        test = [0.9, 81, 1010, 1, 0.0028933]

        # make sure the small or zero values are masked
        mape = mean_absolute_percentage_error(true, test, threshold=1e-6)
        mape_masked = mean_absolute_percentage_error(true[:4], test[:4])
        self.assertAlmostEqual(mape, 0.09999999999999999)
        self.assertAlmostEqual(mape, mape_masked)
