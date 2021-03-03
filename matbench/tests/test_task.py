import unittest

from matbench.task import MatbenchTask
from matbench.metadata import metadata



class TestMatbenchTask(unittest.TestCase):

    def setUp(self) -> None:
        self.test_datasets = ["matbench_dielectric", "matbench_steels", "matbench_glass"]


    def test_instantiation(self):

        for ds in self.test_datasets:
            mbt = MatbenchTask(ds)
