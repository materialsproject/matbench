import unittest
import os

from matbench.util import RecursiveDotDict, MSONable2File
from matbench.tests.util import TEST_DIR


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.json_file = os.path.join(TEST_DIR, "example.json")

    def tearDown(self) -> None:
        if os.path.exists(self.json_file):
            os.remove(self.json_file)

    def test_RecursiveDotDict(self):

        d = {"a": {"b": [1,2,3], "c": "string"}, "b": 12}

        rd = RecursiveDotDict(d)

        self.assertDictEqual(rd.a, {"b": [1,2,3], "c": "string"})
        self.assertListEqual(rd.a.b, [1,2,3])
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