import logging
import os
import unittest

import numpy as np

from matbench.tests.util import TEST_DIR
from matbench.util import (
    MSONable2File,
    RecursiveDotDict,
    hash_dictionary,
    immutify_dictionary,
    initialize_logger,
)


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.json_file = os.path.join(TEST_DIR, "example.json")
        self.log_file = os.path.join(TEST_DIR, "matbench_test.log")

    def tearDown(self) -> None:
        if os.path.exists(self.json_file):
            os.remove(self.json_file)

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_RecursiveDotDict(self):

        d = {"a": {"b": [1, 2, 3], "c": "string"}, "b": 12}

        rd = RecursiveDotDict(d)

        self.assertDictEqual(rd.a, {"b": [1, 2, 3], "c": "string"})
        self.assertListEqual(rd.a.b, [1, 2, 3])
        self.assertEqual(rd.b, 12)

    def test_MSONable2File(self):
        class MSONable2FileChild(MSONable2File):
            def __init__(self, prop1):
                self.prop1 = prop1

            def as_dict(self):
                return {"prop1": self.prop1}

            @classmethod
            def from_dict(cls, d):
                return cls(prop1=d["prop1"])

        ms = MSONable2FileChild(prop1="blue")
        ms.to_file(self.json_file)

        ms2 = MSONable2FileChild.from_file(self.json_file)
        self.assertEqual(ms2.prop1, "blue")

    def test_logging(self):
        logger = initialize_logger(
            "matbench_test", log_dir=TEST_DIR, level=logging.DEBUG
        )

        logger.info("example msg 1")
        logger.debug("Example msg 2")
        logger.critical("Example msg 3")

        self.assertTrue(os.path.exists(self.log_file))

        with open(self.log_file, "r") as f:
            logtxt = f.read()
            self.assertIn("INFO", logtxt)
            self.assertIn("DEBUG", logtxt)
            self.assertIn("CRITICAL", logtxt)
            self.assertEqual(len(logtxt.split("\n")), 4)


class TestHashingDictionaryFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.d = {
            "q": [1, 2, 3],
            "b": {
                "c": "12",
                "d": 15,
                "e": np.asarray([4, 5, 6]),
                "f": {"z": 12, "a": [7, 8]},
            },
            "c": np.int64(400),
        }
        self.d_same = {
            "q": [1, 2, 3],
            "b": {
                "c": "12",
                "d": 15,
                "e": np.asarray([4, 5, 6]),
                "f": {"a": [7, 8], "z": 12},
            },
            "c": 400,
        }
        self.d_different = {
            "q": [1, 2, 3],
            "b": {
                "c": "12",
                "d": 15,
                "e": np.asarray([4, 5, 6]),
                "f": {"z": 12, "a": [7, 9]},
            },
            "c": 400,
        }

    def test_immutify_dictionary(self):
        d_immutable = immutify_dictionary(self.d)
        d_truth = {
            "q": (1, 2, 3),
            "b": {"c": "12", "d": 15, "e": (4, 5, 6), "f": {"z": 12, "a": (7, 8)}},
            "c": 400,
        }
        self.assertDictEqual(d_immutable, d_truth)

    def test_hash_dictionary(self):
        self.assertEqual(hash_dictionary(self.d), hash_dictionary(self.d_same))
        self.assertNotEqual(
            hash_dictionary(self.d), hash_dictionary(self.d_different)
        )
