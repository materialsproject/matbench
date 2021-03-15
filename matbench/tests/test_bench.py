import unittest

import numpy as np

from matbench.constants import REG_KEY, CLF_KEY, STRUCTURE_KEY, COMPOSITION_KEY, MBV01_KEY
from matbench.bench import MatbenchBenchmark, hash_dictionary, immutify_dictionary



class TestMatbenchBenchmark(unittest.TestCase):

    def setUp(self) -> None:
        pass


    def test_from_preset(self):
        for preset, n_tasks in {REG_KEY: 10, CLF_KEY: 3, STRUCTURE_KEY: 9, COMPOSITION_KEY: 4}.items():
            mb = MatbenchBenchmark.from_preset(benchmark=MBV01_KEY, preset_name=preset, autoload=False)
            self.assertEqual(len(mb.tasks.keys()), n_tasks)

    def test_scores(self):
        pass

    def test_info(self):
        pass

    def test_add_metadata(self):
        pass

    def test_complete_valid_recorded(self):
        pass

    def test_MSONability(self):
        pass




class TestHashingDictionaryFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.d = {"q": [1, 2, 3],
             "b": {"c": "12", "d": 15, "e": np.asarray([4, 5, 6]),
                   "f": {"z": 12, "a": [7, 8]}}, "c": np.int64(400)}
        self.d_same = {"q": [1, 2, 3],
                  "b": {"c": "12", "d": 15, "e": np.asarray([4, 5, 6]),
                        "f": {"a": [7, 8], "z": 12}}, "c": 400}
        self.d_different = {"q": [1, 2, 3],
                       "b": {"c": "12", "d": 15, "e": np.asarray([4, 5, 6]),
                             "f": {"z": 12, "a": [7, 9]}}, "c": 400}

    def test_immutify_dictionary(self):
        d_immutable = immutify_dictionary(self.d)
        d_truth = {"q": (1, 2, 3),
             "b": {"c": "12", "d": 15, "e": (4, 5, 6),
                   "f": {"z": 12, "a": (7, 8)}}, "c": 400}
        self.assertDictEqual(d_immutable, d_truth)

    def test_hash_dictionary(self):
        self.assertEqual(hash_dictionary(self.d), hash_dictionary(self.d_same))
        self.assertNotEqual(hash_dictionary(self.d), hash_dictionary(self.d_different))