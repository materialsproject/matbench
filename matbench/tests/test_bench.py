import unittest

from matbench.constants import REG_KEY, CLF_KEY, STRUCTURE_KEY, COMPOSITION_KEY

from matbench.bench import MatbenchBenchmark



class TestMatbenchBenchmark(unittest.TestCase):

    def setUp(self) -> None:
        pass


    def test_from_preset(self):
        mb = MatbenchBenchmark.from_preset(REG_KEY)

        print(mb.tasks.matbench_steels.metadata)