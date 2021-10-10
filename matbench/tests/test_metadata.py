import unittest

from matbench.constants import CLF_KEY, REG_KEY
from matbench.metadata import mbv01_metadata, mbv01_validation


class TestMetadata(unittest.TestCase):
    def test_mbv01_metadata(self):

        # for matbench v0.1
        for ds, metadata in mbv01_metadata.items():
            for key in ["task_type", "n_samples", "input_type", "target"]:
                self.assertIn(key, metadata.keys())

                if key == "input_type":
                    self.assertIn(metadata[key], ["composition", "structure"])

                elif key == "task_type":
                    self.assertIn(metadata[key], [REG_KEY, CLF_KEY])
        self.assertEqual(len(list(mbv01_metadata.values())), 13)

        self.assertIn("metadata", mbv01_validation)
        self.assertIn("splits", mbv01_validation)
        self.assertEqual(len(mbv01_validation.splits), 13)

        for k in mbv01_metadata:
            self.assertEqual(len(mbv01_validation.splits[k]), 5)
