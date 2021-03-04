import unittest

import pandas as pd
import numpy as np
from pymatgen import Structure

from matbench.task import MatbenchTask
from matbench.metadata import metadata


import warnings
import traceback
import sys



class TestMatbenchTask(unittest.TestCase):

    def setUp(self) -> None:
        self.test_datasets = ["matbench_dielectric", "matbench_steels", "matbench_glass"]

        # def warn_with_traceback(message, category, filename, lineno, file=None,
        #                         line=None):
        #     log = file if hasattr(file, 'write') else sys.stderr
        #     traceback.print_stack(file=log)
        #     log.write(
        #         warnings.formatwarning(message, category, filename, lineno,
        #                                line))
        #
        # warnings.showwarning = warn_with_traceback

    def test_instantiation(self):
        for ds in self.test_datasets:
            mbt = MatbenchTask(ds)

    def test_get_train_and_val_data(self):


        entries_manual_check = {
            "matbench_dielectric": ("K3V5O14", 2.040549),
            "matbench_steels": ("Fe0.696C0.00878Mn0.000101Si0.00989Cr0.135Ni0.00927Mo0.0113V0.000109Nb0.000120Co0.129Al0.000617", 1376.3),
            "matbench_glass": ("Al21(CoNi)2", False)
        }


        # Assuming 5-fold nested cross validation
        for ds in self.test_datasets:
            mbt = MatbenchTask(ds)
            inputs, outputs = mbt.get_train_and_val_data(fold_number=0, as_type="tuple", shuffle_seed=1001)


            self.assertListEqual(inputs.index.tolist(), outputs.index.tolist())
            self.assertEqual(inputs.shape[0], int(np.floor(mbt.df.shape[0] * 4/5)))
            self.assertEqual(outputs.shape[0], int(np.floor(mbt.df.shape[0] * 4/5)))

            input_type = Structure if mbt.metadata.input_type == "structure" else str
            output_type = float if mbt.metadata.task_type == "regression" else bool
            self.assertTrue(all([isinstance(d, input_type) for d in inputs]))
            self.assertTrue(all([isinstance(d, output_type) for d in outputs]))


            if ds == "matbench_dielectric":
                sio2 = inputs.iloc[101].composition.reduced_formula
                self.assertEqual(sio2, "SiO2")
                self.assertEqual(sio2, inputs.loc[2658].composition.reduced_formula) # loc index corresponds to the iloc in the original df
                self.assertEqual(sio2, mbt.df["structure"].iloc[2658].composition.reduced_formula)                # make sure the index matches the original df
                n = 1.5191342929445046
                self.assertAlmostEqual(outputs.iloc[101], n, places=10)
                self.assertAlmostEqual(outputs.loc[2658], n, places=10)
                self.assertAlmostEqual(mbt.df["n"].iloc[2658], n, places=10)


            # self.assertEqual()
            print(inputs.iloc[101])
            print(inputs.index[101])
            print(outputs.iloc[101])
            print(outputs.index[101])


