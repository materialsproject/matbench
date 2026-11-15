import copy
import json
import os
import unittest

from matbench.bench import MatbenchBenchmark
from matbench.constants import (
    CLF_KEY,
    COMPOSITION_KEY,
    MBV01_KEY,
    REG_KEY,
    STRUCTURE_KEY,
)
from matbench.task import MatbenchTask
from matbench.tests.util import FULL_TEST, TEST_DIR, model_random


class TestMatbenchBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.msonability_tmp_path = os.path.join(
            TEST_DIR, "msonability_test_augmented.json"
        )
        self.static_full_bench_json = os.path.join(
            TEST_DIR, "mb_all_tasks_random.json"
        )
        self.static_3_bench_json = os.path.join(TEST_DIR, "mb_3_tasks_random.json")

    def tearDown(self) -> None:
        for f in [self.msonability_tmp_path]:
            if os.path.exists(f):
                os.remove(f)

    def test_from_preset(self):
        for preset, n_tasks in {
            REG_KEY: 10,
            CLF_KEY: 3,
            STRUCTURE_KEY: 9,
            COMPOSITION_KEY: 4,
            "all": 13,
        }.items():
            mb = MatbenchBenchmark.from_preset(
                benchmark=MBV01_KEY, preset_name=preset, autoload=False
            )
            self.assertEqual(len(mb.tasks_map.keys()), n_tasks)

    def test_scores(self):
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        scores = mb.scores
        self.assertAlmostEqual(
            14.169387576348942, scores.matbench_dielectric.mape.mean, places=10
        )
        self.assertAlmostEqual(
            2.376419875235826, scores.matbench_mp_e_form.rmse.mean, places=10
        )
        self.assertAlmostEqual(
            0.49560975609756097, scores.matbench_expt_is_metal.f1.min, places=10
        )

    def test_info(self):
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        mb.info
        mb.get_info()

    def test_add_metadata(self):
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        example_metadata = {"some": "metadata"}
        mb.add_metadata(example_metadata)
        self.assertDictEqual(example_metadata, mb.user_metadata)

        bad_metadata = [1, 2, 3]
        mb.add_metadata(bad_metadata)
        self.assertDictEqual(example_metadata, mb.user_metadata)

    def test_complete_valid_recorded(self):
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        self.assertTrue(mb.is_valid)
        self.assertTrue(mb.is_recorded)
        self.assertTrue(mb.is_complete)

        # if one sample is missing from one task fold, throw error
        mb.matbench_dielectric.results.fold_0.data.pop("mb-dielectric-0010")
        self.assertFalse(mb.is_valid)
        self.assertTrue(len(mb.validate().keys()), 1)

        # if one task is missing one fold
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        mb.matbench_mp_e_form.results.pop("fold_1")
        self.assertFalse(mb.is_valid)
        self.assertTrue(len(mb.validate().keys()), 1)

        # if one fold is not recorded
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        mb.matbench_phonons.results.fold_1 = {}
        self.assertFalse(mb.is_recorded)

        # if one or more tasks is missing
        mb = MatbenchBenchmark.from_file(self.static_3_bench_json)
        self.assertFalse(mb.is_complete)

    def test_MSONability(self):
        # Test serialization
        self.generate_benchmark_json_files(write=False, full_set=FULL_TEST)

        # Test deserialization
        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        self.assertTrue(mb.is_complete)
        self.assertTrue(mb.is_valid)
        self.assertTrue(mb.is_recorded)

        # Ensure object can be added to and re/de-serialized without issue
        example_metadata = {"some": "metadata"}
        mb.add_metadata(example_metadata)
        mb.to_file(self.msonability_tmp_path)
        mb = MatbenchBenchmark.from_file(self.msonability_tmp_path)
        self.assertTrue(mb.is_complete)
        self.assertTrue(mb.is_valid)
        self.assertTrue(mb.is_recorded)
        self.assertDictEqual(example_metadata, mb.user_metadata)

        with open(self.static_full_bench_json) as f:
            d_truth = json.load(f)

        # Test if hash is wrong (hash error should occur)
        d_test = copy.deepcopy(d_truth)
        d_test[MatbenchBenchmark._HASH_KEY] = "not a real hash"
        with self.assertRaises(ValueError):
            MatbenchBenchmark.from_dict(d_test)

        # Test if datetime is wrong (hash error should occur)
        d_test = copy.deepcopy(d_truth)
        d_test[MatbenchBenchmark._DATESTAMP_KEY] = "2021.02.02 06:33.21"
        with self.assertRaises(ValueError):
            MatbenchBenchmark.from_dict(d_test)

        # Test if one or more task names is wrong
        d_test = copy.deepcopy(d_truth)
        d_test[MatbenchBenchmark._TASKS_KEY]["matbench_dielectric"][
            MatbenchTask._DATASET_KEY
        ] = "matbench_expt_gap"
        d_test[MatbenchBenchmark._TASKS_KEY]["matbench_expt_gap"][
            MatbenchTask._DATASET_KEY
        ] = "matbench_dielectric"
        with self.assertRaises(ValueError):
            MatbenchBenchmark.from_dict(d_test)

        # Test if one or more benchmark names is wrong
        d_test = copy.deepcopy(d_truth)
        d_test[MatbenchBenchmark._TASKS_KEY]["matbench_dielectric"][
            MatbenchTask._BENCHMARK_KEY
        ] = "matbench_v0.2"
        with self.assertRaises(ValueError):
            MatbenchBenchmark.from_dict(d_test)

    def test_partial_MSONability(self):
        ds = "matbench_steels"
        mb = MatbenchBenchmark(subset=[ds], autoload=False)

        # An untouched benchmark can be saved and resumed, but is not valid.
        mb.to_file(self.msonability_tmp_path)
        mb = MatbenchBenchmark.from_file(self.msonability_tmp_path)
        self.assertEqual(mb.tasks_map[ds].results, {})
        self.assertFalse(mb.is_recorded)
        self.assertFalse(mb.is_valid)
        self.assertEqual(set(mb.validate()), {ds})

        task = mb.tasks_map[ds]
        task.load()
        recorded_folds = [1, 3]
        for fold in recorded_folds:
            _, outputs = task.get_test_data(fold, include_target=True)
            task.record(fold, outputs, params={"seed": 42})
        recorded_results = json.loads(json.dumps(task.as_dict()))["results"]

        mb.to_file(self.msonability_tmp_path)
        resumed = MatbenchBenchmark.from_file(self.msonability_tmp_path)
        resumed_task = resumed.tasks_map[ds]
        self.assertEqual(resumed_task.results, recorded_results)
        self.assertFalse(resumed.is_recorded)
        self.assertFalse(resumed.is_valid)
        self.assertEqual(set(resumed.validate()), {ds})
        self.assertEqual(resumed_task.results, recorded_results)

        # Serialize corruption with a fresh valid hash so the task check runs.
        corrupt = MatbenchBenchmark.from_file(self.msonability_tmp_path)
        fold_data = corrupt.tasks_map[ds].results["fold_1"]["data"]
        fold_data[next(iter(fold_data))] = "invalid prediction"
        corrupt.to_file(self.msonability_tmp_path)
        with self.assertRaises(TypeError):
            MatbenchBenchmark.from_file(self.msonability_tmp_path)

        resumed.load()
        for fold in resumed_task.folds:
            if fold not in recorded_folds:
                _, outputs = resumed_task.get_test_data(fold, include_target=True)
                resumed_task.record(fold, outputs)
        for fold in recorded_folds:
            key = resumed_task.folds_map[fold]
            self.assertEqual(resumed_task.results[key], recorded_results[key])
        self.assertEqual(resumed.validate(), {})
        self.assertTrue(resumed.is_recorded)
        self.assertTrue(resumed.is_valid)
        resumed.to_file(self.msonability_tmp_path)
        self.assertTrue(
            MatbenchBenchmark.from_file(self.msonability_tmp_path).is_valid
        )

    def test_datestamp(self):
        mb = MatbenchBenchmark(benchmark=MBV01_KEY, subset=["matbench_steels"])
        self.assertIsNone(mb.datestamp)

        with open(self.static_full_bench_json) as f:
            datestamp_truth = json.load(f)[MatbenchBenchmark._DATESTAMP_KEY]

        mb = MatbenchBenchmark.from_file(self.static_full_bench_json)
        self.assertEqual(mb.datestamp, datestamp_truth)

    def test_completeness(self):
        mb = MatbenchBenchmark(benchmark=MBV01_KEY, subset=["matbench_steels"])

        self.assertFalse(mb.is_complete)
        self.assertFalse(mb.is_composition_complete)
        self.assertFalse(mb.is_structure_complete)

        mb = MatbenchBenchmark.from_preset(MBV01_KEY, preset_name=COMPOSITION_KEY)
        self.assertFalse(mb.is_complete)
        self.assertTrue(mb.is_composition_complete)
        self.assertFalse(mb.is_structure_complete)

        mb = MatbenchBenchmark.from_preset(MBV01_KEY, preset_name=STRUCTURE_KEY)
        self.assertFalse(mb.is_complete)
        self.assertFalse(mb.is_composition_complete)
        self.assertTrue(mb.is_structure_complete)

        mb = MatbenchBenchmark.from_preset(MBV01_KEY, preset_name="all")
        self.assertTrue(mb.is_complete)
        self.assertTrue(mb.is_composition_complete)
        self.assertTrue(mb.is_structure_complete)

        mb = MatbenchBenchmark.from_preset(MBV01_KEY, preset_name=REG_KEY)
        self.assertFalse(mb.is_complete)
        self.assertTrue(mb.is_regression_complete)
        self.assertFalse(mb.is_classification_complete)

        mb = MatbenchBenchmark.from_preset(MBV01_KEY, preset_name=CLF_KEY)
        self.assertFalse(mb.is_complete)
        self.assertFalse(mb.is_regression_complete)
        self.assertTrue(mb.is_classification_complete)

    def generate_benchmark_json_files(self, write=True, full_set=False):
        """
        Generate new benchmark files from a random model.

        Args:
            write:
            full_set:

        Returns:

        """

        if full_set:
            subset = None  # use all tasks
        else:
            subset = [
                "matbench_dielectric",
                "matbench_steels",
                "matbench_glass",
            ]  # to generate 3 task file

        mb = MatbenchBenchmark(benchmark=MBV01_KEY, autoload=False, subset=subset)

        for task in mb.tasks:
            task.load()
            for fold in task.folds:
                train_inputs, train_outputs = task.get_train_and_val_data(fold)
                test_inputs = task.get_test_data(fold, include_target=False)
                model_response = model_random(
                    train_outputs, test_inputs, response_type=task.metadata.task_type
                )
                task.record(
                    fold,
                    model_response,
                    params={"some_param": 1, "other_param": 12.39348},
                )

        if subset:
            if write:
                mb.to_file(self.static_3_bench_json)
            self.assertTrue(mb.is_recorded)
            self.assertTrue(mb.is_valid)
            self.assertFalse(mb.is_complete)
        else:
            if write:
                mb.to_file(self.static_full_bench_json)
            self.assertTrue(mb.is_recorded)
            self.assertTrue(mb.is_valid)
            self.assertTrue(mb.is_complete)
