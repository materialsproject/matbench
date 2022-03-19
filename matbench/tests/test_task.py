import copy
import glob
import json
import os
import unittest
from warnings import warn

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from matbench.constants import (
    CLF_METRICS,
    COMPOSITION_KEY,
    FOLD_DIST_METRICS,
    MBV01_KEY,
    REG_KEY,
    REG_METRICS,
    STRUCTURE_KEY,
    TEST_KEY,
)
from matbench.task import MatbenchTask
from matbench.tests.util import TEST_DIR, model_random


class TestMatbenchTask(unittest.TestCase):
    test_datasets = ("matbench_dielectric", "matbench_steels", "matbench_glass")

    def tearDown(self) -> None:
        # remove all temporary output files
        for f in glob.glob(os.path.join(TEST_DIR, "*_output.json")):
            os.remove(f)

    def test_instantiation(self):
        for ds in self.test_datasets:
            MatbenchTask(ds, autoload=True)

    def test_get_train_and_val_data(self):
        # Assuming 5-fold nested cross validation
        for ds in self.test_datasets:
            mbt = MatbenchTask(ds, autoload=False)
            mbt.load()
            # shuffle seed must be set because it shuffles
            # the (same) training data in a deterministic manner

            inputs, outputs = mbt.get_train_and_val_data(
                fold_number=0, as_type="tuple"
            )

            self.assertListEqual(inputs.index.tolist(), outputs.index.tolist())
            self.assertEqual(inputs.shape[0], int(np.floor(mbt.df.shape[0] * 4 / 5)))
            self.assertEqual(
                outputs.shape[0], int(np.floor(mbt.df.shape[0] * 4 / 5))
            )

            input_type = (
                Structure if mbt.metadata.input_type == STRUCTURE_KEY else str
            )
            output_type = float if mbt.metadata.task_type == REG_KEY else bool
            self.assertTrue(all([isinstance(d, input_type) for d in inputs]))
            self.assertTrue(all([isinstance(d, output_type) for d in outputs]))

            if ds == "matbench_dielectric":
                mat = inputs.loc["mb-dielectric-1985"]
                f = mat.composition.reduced_formula
                val = outputs.loc["mb-dielectric-1985"]
                self.assertEqual(f, "Re3(TeBr)7")
                self.assertEqual(
                    inputs.iloc[1621], mat
                )  # ensure the ordering is correct via iloc
                n = 2.5230272821931656
                self.assertAlmostEqual(val, n, places=10)
                self.assertAlmostEqual(outputs.iloc[1621], n, places=10)
            elif ds == "matbench_steels":
                alloy = (
                    "Fe0.692C0.00968Mn0.000101Si0.0144Cr0.133Ni0.00887"
                    "Mo0.0114V0.000109Nb0.000477Co0.130Al0.000616"
                )
                mat = inputs.loc["mb-steels-095"]
                val = outputs.loc["mb-steels-095"]
                self.assertEqual(alloy, mat)
                self.assertEqual(alloy, inputs.iloc[75])
                yield_strength = 1369.5
                self.assertAlmostEqual(val, yield_strength, places=5)
                self.assertAlmostEqual(outputs.iloc[75], yield_strength, places=5)
            elif ds == "matbench_glass":
                alloy = "Ce2Al5Cu43"
                mat = inputs.loc["mb-glass-0600"]
                val = outputs.loc["mb-glass-0600"]
                self.assertEqual(alloy, mat)
                self.assertEqual(alloy, inputs.iloc[480])
                gfa = False
                self.assertEqual(val, gfa)
                self.assertEqual(outputs.iloc[480], gfa)

    def test_get_test_data(self):
        for ds in self.test_datasets:
            mbt = MatbenchTask(ds, autoload=False)
            mbt.load()
            folds = []
            for fold in mbt.folds:
                inputs, outputs = mbt.get_test_data(
                    fold_number=fold, as_type="tuple", include_target=True
                )

                self.assertListEqual(inputs.index.tolist(), outputs.index.tolist())

                upper_bound = int(np.ceil(mbt.df.shape[0] / 5))
                allowed_fold_sizes = (upper_bound - 1, upper_bound)
                self.assertTrue(inputs.shape[0] in allowed_fold_sizes)
                self.assertTrue(outputs.shape[0] in allowed_fold_sizes)
                input_type = (
                    Structure if mbt.metadata.input_type == STRUCTURE_KEY else str
                )
                output_type = float if mbt.metadata.task_type == REG_KEY else bool
                self.assertTrue(all([isinstance(d, input_type) for d in inputs]))
                self.assertTrue(all([isinstance(d, output_type) for d in outputs]))
                folds.append((inputs, outputs))

                df_test = mbt.get_test_data(
                    fold_number=fold, as_type="df", include_target=True
                )
                self.assertListEqual(
                    df_test.columns.tolist(),
                    [mbt.metadata.input_type, mbt.metadata.target],
                )

                df_test = mbt.get_test_data(
                    fold_number=fold, as_type="df", include_target=False
                )
                self.assertTrue(isinstance(df_test, pd.DataFrame))
                self.assertListEqual(
                    df_test.columns.tolist(), [mbt.metadata.input_type]
                )

            # check if all entries from original df are in exactly
            # one test fold exactly once
            original_input_df = mbt.df[mbt.metadata.input_type]
            inputs_from_folds = pd.concat([f[0] for f in folds])
            self.assertEqual(inputs_from_folds.shape[0], original_input_df.shape[0])
            self.assertTrue(
                original_input_df.apply(
                    lambda i: i in inputs_from_folds.tolist()
                ).all()
            )

            # Test individual samples from an individual test set
            inputs, outputs = folds[0]
            if ds == "matbench_dielectric":
                ki = inputs.iloc[12].composition.reduced_formula
                self.assertEqual(ki, "KI")
                self.assertEqual(
                    ki, inputs.loc["mb-dielectric-0076"].composition.reduced_formula
                )
                self.assertEqual(
                    ki,
                    mbt.df[STRUCTURE_KEY]
                    .loc["mb-dielectric-0076"]
                    .composition.reduced_formula,
                )
                n = 1.7655027612552967
                self.assertAlmostEqual(outputs.iloc[12], n, places=10)
                self.assertAlmostEqual(
                    outputs.loc["mb-dielectric-0076"], n, places=10
                )
                self.assertAlmostEqual(
                    mbt.df[mbt.metadata.target].loc["mb-dielectric-0076"],
                    n,
                    places=10,
                )
            elif ds == "matbench_steels":
                alloy = (
                    "Fe0.682C0.00877Mn0.000202Si0.00967Cr0.134"
                    "Ni0.00907Mo0.00861V0.00501Nb0.0000597Co0.142Al0.000616"
                )
                self.assertEqual(alloy, inputs.loc["mb-steels-068"])
                self.assertEqual(alloy, inputs.iloc[12])
                self.assertEqual(alloy, mbt.df[COMPOSITION_KEY].loc["mb-steels-068"])
                yield_strength = 1241.0
                self.assertAlmostEqual(outputs.iloc[12], yield_strength, places=5)
                self.assertAlmostEqual(
                    outputs.loc["mb-steels-068"], yield_strength, places=5
                )
                self.assertAlmostEqual(
                    mbt.df[mbt.metadata.target].loc["mb-steels-068"],
                    yield_strength,
                    places=5,
                )
            elif ds == "matbench_glass":
                alloy = "Al13VCu6"
                self.assertEqual(alloy, inputs.iloc[12])
                self.assertEqual(alloy, inputs.loc["mb-glass-0056"])
                self.assertEqual(alloy, mbt.df[COMPOSITION_KEY].loc["mb-glass-0056"])
                gfa = True
                self.assertEqual(outputs.iloc[12], gfa)
                self.assertEqual(outputs.loc["mb-glass-0056"], gfa)
                self.assertEqual(
                    mbt.df[mbt.metadata.target].loc["mb-glass-0056"], gfa
                )

    def test_get_info(self):
        mbt = MatbenchTask("matbench_steels", autoload=False)
        mbt.get_info()
        self.assertTrue("citations" in mbt.info.lower())
        self.assertTrue("SHA256 Hash Digest" in mbt.info)

    def test_record(self):
        self._test_record()

    def test_record_std(self):
        # ensure that conversion from std to ci is correct
        mbt = MatbenchTask(self.test_datasets[0], autoload=True)
        _, test_outputs = mbt.get_test_data(0, as_type="tuple", include_target=True)
        n_test = len(test_outputs)
        mbt.record(0, [0.0] * n_test, std=[1.0] * n_test)
        ci_0_0 = list(mbt.results["fold_0"]["uncertainty"].items())[0][1]
        ci_lower = ci_0_0["ci_lower"]
        ci_upper = ci_0_0["ci_upper"]
        # mean:0 std:1, 95% CI-->c.a. 1.96
        # https://stackoverflow.com/a/29562808/13697228
        ci_upp_check = 1.959963984540054
        ci_low_check = -ci_upp_check
        msg = f"""Conversion from normal distribution standard deviation to
        lower 95% confidence bound ({ci_upper}) is not within tolerance of
        expected confidence bound ({ci_upp_check})"""
        self.assertAlmostEqual(ci_upper, ci_upp_check, msg=msg)
        msg = f"""Conversion from normal distribution standard deviation
        to upper 95% confidence bound ({ci_lower}) is not within tolerance
        of expected confidence bound ({ci_low_check})"""
        self.assertAlmostEqual(ci_lower, ci_low_check, msg=msg)

        # carry out test as normal
        self._test_record(uq_type="std")

    def test_record_ci(self):
        # ensure that conversion from ci to std is correct
        mbt = MatbenchTask(self.test_datasets[0], autoload=True)
        _, test_outputs = mbt.get_test_data(0, as_type="tuple", include_target=True)
        n_test = len(test_outputs)
        ci_upper = 1.959963984540054
        ci_lower = -ci_upper
        mbt.record(0, [0.0] * n_test, ci=[[ci_lower, ci_upper]] * n_test)
        ci_0_0 = list(mbt.results["fold_0"]["uncertainty"].items())[0][1]
        std = ci_0_0["std"]
        std_check = 1.0
        # mean:0 std:1, 95% CI-->c.a. 1.96
        # https://stackoverflow.com/a/29562808/13697228
        msg = f"""Conversion from symmetric 95% confidence bounds to
        standard deviation ({std}) is not within tolerance of expected
        confidence bound ({std_check})"""
        self.assertAlmostEqual(std, std_check, msg=msg)

        self._test_record(uq_type="ci")

    def _test_record(self, uq_type=None):
        for ds in self.test_datasets:
            if ds == "matbench_glass":
                uq_type = None
                warn(
                    """Overriding uq_type as None due to incompatibility
                     with classification tasks (expected behavior)."""
                )
            # Testing two scenarios: model is perfect, and model is random
            for model_is_perfect in (True, False):
                mbt = MatbenchTask(ds, autoload=False)
                mbt.load()

                # test to make sure raw data output is correct,
                # using a random model
                for fold, fold_key in mbt.folds_map.items():
                    _, training_outputs = mbt.get_train_and_val_data(
                        fold, as_type="tuple"
                    )
                    if model_is_perfect:
                        test_inputs, test_outputs = mbt.get_test_data(
                            fold, as_type="tuple", include_target=True
                        )
                        model_response = test_outputs
                    else:
                        test_inputs = mbt.get_test_data(
                            fold, as_type="tuple", include_target=False
                        )
                        model_response = model_random(
                            training_outputs,
                            test_inputs,
                            response_type=mbt.metadata.task_type,
                        )

                    n_test = len(test_inputs)

                    # uncertainty quantification parameter
                    uq_param = {}
                    if uq_type == "ci":
                        uq_shape = (n_test, 2)
                    elif uq_type == "std":
                        uq_shape = (n_test,)
                    if uq_type is not None:
                        uq_param[uq_type] = 2 * np.random.rand(*uq_shape)

                    dummy_params = {
                        "test_param": 1,
                        "other_param": "string",
                        "hyperparam": True,
                    }

                    mbt.record(
                        fold,
                        predictions=model_response,
                        params=dummy_params,
                        **uq_param,
                    )
                    self.assertEqual(
                        len(mbt.results[fold_key].data.values()), n_test
                    )
                    self.assertEqual(mbt.results[fold_key].parameters.test_param, 1)
                    self.assertEqual(
                        mbt.results[fold_key].parameters.other_param, "string"
                    )
                    self.assertEqual(
                        mbt.results[fold_key].parameters.hyperparam, True
                    )

                if ds == "matbench_dielectric":
                    mae = mbt.results.fold_0.scores.mae
                    val = mbt.results.fold_0.data["mb-dielectric-0008"]
                    if model_is_perfect:
                        self.assertAlmostEqual(mae, 0.0, places=10)
                        self.assertAlmostEqual(val, 2.0323401126123875, places=10)
                    else:
                        self.assertAlmostEqual(mae, 29.790913986352297, places=10)
                        self.assertAlmostEqual(val, 43.36354273040313, places=10)
                elif ds == "matbench_steels":
                    mae = mbt.results.fold_0.scores.mae
                    if model_is_perfect:
                        self.assertAlmostEqual(mae, 0.0, places=10)
                    else:
                        self.assertAlmostEqual(mae, 488.97286237333986, places=10)
                elif ds == "matbench_glass":
                    rocauc = mbt.results.fold_0.scores.rocauc
                    if model_is_perfect:
                        self.assertAlmostEqual(rocauc, 1.0, places=10)
                    else:
                        self.assertAlmostEqual(rocauc, 0.5141975796883651, places=10)

                self.assertTrue(mbt.all_folds_recorded)

            mbt = MatbenchTask(self.test_datasets[0], autoload=True)
            # Test to make sure bad predictions won't be recorded
            with self.assertRaises(ValueError):
                mbt.record(0, [0.0, 1.0])

            with self.assertRaises(ValueError):
                mbt.record(0, ["not", "a number"])

    def test_MSONability(self):
        for ds in self.test_datasets:
            mbt = MatbenchTask(ds, autoload=False)
            mbt.load()

            for fold in mbt.folds:
                _, training_outputs = mbt.get_train_and_val_data(
                    fold, as_type="tuple"
                )
                test_inputs, test_outputs = mbt.get_test_data(
                    fold, as_type="tuple", include_target=True
                )
                mbt.record(
                    fold,
                    predictions=test_outputs,
                    params={"some_param": 1, "another param": 30349.4584},
                )

            d = mbt.as_dict()

            self.assertEqual(d["@module"], "matbench.task")
            self.assertEqual(d["@class"], "MatbenchTask")
            self.assertEqual(d[mbt._BENCHMARK_KEY], MBV01_KEY)
            self.assertEqual(d[mbt._DATASET_KEY], ds)
            self.assertEqual(len(d["results"]), len(mbt.validation.keys()))

            for fold, fold_key in mbt.folds_map.items():
                res = d["results"][fold_key]
                self.assertIn(MatbenchTask._PARAMS_KEY, res)
                self.assertIn(MatbenchTask._SCORES_KEY, res)
                self.assertIn(MatbenchTask._DATA_KEY, res)

                # make sure test set as per MbT and the recorded predictions
                # are the same shape inside dict
                self.assertEqual(
                    len(res["data"]), len(mbt.validation[fold_key][TEST_KEY])
                )

            mbt.to_file(os.path.join(TEST_DIR, f"msonability_{ds}_output.json"))

            # todo: uncomment to regenerate test files
            # todo: these can be used as the score_matbench_*_perfect.json
            # todo: files as well if renamed.
            # mbt.to_file(os.path.join(TEST_DIR, f"msonability_{ds}.json"))
            # mbt.to_file(os.path.join(TEST_DIR, f"scores_{ds}_perfect.json"))

            # Test ingestion from ground truth json files
            truth_fname = os.path.join(TEST_DIR, f"msonability_{ds}.json")

            with open(truth_fname, "r") as f:
                truth = json.loads(f.read())
            MatbenchTask.from_file(truth_fname)
            MatbenchTask.from_dict(truth)

            # Ensure errors are thrown for bad json

            missing_results = copy.deepcopy(truth)
            missing_results.pop("results")

            with self.assertRaises(KeyError):
                MatbenchTask.from_dict(missing_results)

            for key in [
                MatbenchTask._PARAMS_KEY,
                MatbenchTask._DATA_KEY,
                MatbenchTask._SCORES_KEY,
            ]:
                missing_key = copy.deepcopy(truth)
                missing_key["results"]["fold_3"].pop(key)

                with self.assertRaises((ValueError, KeyError)):
                    MatbenchTask.from_dict(missing_key)

            # If an otherwise perfect json is missing a required index
            missing_ix_fold0 = copy.deepcopy(truth)

            missing_ix = mbt.validation.fold_0.test[0]
            missing_ix_fold0["results"]["fold_0"]["data"].pop(missing_ix)

            with self.assertRaises(ValueError):
                MatbenchTask.from_dict(missing_ix_fold0)

            # If an otherwise perfect json has an extra index
            extra_ix_fold0 = copy.deepcopy(truth)

            extra_viable_ix = mbt.validation.fold_0.train[0]
            extra_ix_fold0["results"]["fold_0"]["data"][extra_viable_ix] = 1.92

            with self.assertRaises(ValueError):
                MatbenchTask.from_dict(extra_ix_fold0)

            # If an otherwise perfect json has a wrong datatype
            wrong_dtype = copy.deepcopy(truth)
            wrong_dtype["results"]["fold_2"]["data"][
                mbt.validation.fold_2.test[4]
            ] = "any string"

            with self.assertRaises(TypeError):
                MatbenchTask.from_dict(wrong_dtype)

    def test_autoload(self):
        mbt = MatbenchTask("matbench_steels", autoload=False)
        with self.assertRaises(ValueError):
            mbt._check_is_loaded()

        with self.assertRaises(ValueError):
            mbt.get_test_data(0)

        with self.assertRaises(ValueError):
            mbt.get_train_and_val_data(0)

        mbt.load()
        mbt._check_is_loaded()
        mbt.get_test_data(0)
        mbt.get_train_and_val_data(0)

        MatbenchTask("matbench_steels", autoload=True)

    def test_scores(self):
        mbt = MatbenchTask.from_file(
            os.path.join(TEST_DIR, "scores_matbench_dielectric_perfect.json")
        )

        for metric in REG_METRICS:
            for fdm in FOLD_DIST_METRICS:
                self.assertAlmostEqual(0.0, mbt.scores[metric][fdm], places=10)

        mbt = MatbenchTask.from_file(
            os.path.join(TEST_DIR, "scores_matbench_glass_perfect.json")
        )

        for metric in CLF_METRICS:
            for fdm in FOLD_DIST_METRICS:
                test_val = 0.0 if fdm == "std" else 1.0
                self.assertAlmostEqual(test_val, mbt.scores[metric][fdm], places=10)

    def test_usage(self):
        # access some common attrs
        mbt_clf = MatbenchTask.from_file(
            os.path.join(TEST_DIR, "scores_matbench_dielectric_perfect.json")
        )
        mbt_reg = MatbenchTask.from_file(
            os.path.join(TEST_DIR, "scores_matbench_glass_perfect.json")
        )

        for mbt in (mbt_clf, mbt_reg):
            for index, datum in mbt.results.fold_2.data.items():
                self.assertTrue(isinstance(datum, (bool, float)))
                self.assertTrue("mb-" in index)

        self.assertTrue(
            isinstance(mbt.results.fold_3.parameters, (dict, type(None)))
        )

    def test_clf_conversion(self):
        mbt = MatbenchTask("matbench_glass", autoload=True)

        for fold in mbt.folds:
            trdf = mbt.get_test_data(fold, as_type="df")
            pred = np.random.random(len(trdf))
            mbt.record(fold, pred)

        # With random sampling it should almost never exceed
        # ROCAUC of 0.6
        self.assertLess(mbt.scores["rocauc"]["mean"], 0.6)

        # Try the same model as perfect
        mbt = MatbenchTask("matbench_glass", autoload=True)

        for fold in mbt.folds:
            inputs, targets = mbt.get_test_data(fold, include_target=True)
            pred = [1.0 if i else 0.0 for i in targets]
            mbt.record(fold, pred)

        self.assertAlmostEqual(mbt.scores["rocauc"]["mean"], 1.0, places=5)
        self.assertAlmostEqual(mbt.scores["rocauc"]["std"], 0.0, places=5)

    def test_has_polymorphs(self):
        # Test a composition input type
        mbt = MatbenchTask("matbench_steels", autoload=True)
        self.assertFalse(mbt.has_polymorphs)

        # Test similarly for a composition dataframe with polymorphs
        df_comp_poly = pd.DataFrame(
            {"composition": ["Al2O3", "Cr2O3", "Al4O6"], "property": [1.0, 1.1, 1.2]}
        )
        mbt.df = df_comp_poly
        self.assertTrue(mbt.has_polymorphs)

        # Test a structure input type
        mbt = MatbenchTask("matbench_dielectric", autoload=True)
        self.assertTrue(mbt.has_polymorphs)

        # Test similarly for a structure dataframe without polymorphs
        mbt.df = mbt.df.iloc[:10]
        self.assertFalse(mbt.has_polymorphs)
