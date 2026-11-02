import importlib.util
from pathlib import Path
import tempfile
import unittest

from matbench.bench import MatbenchBenchmark
from matbench.constants import REG_METRICS

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ph_reproduce_test", HERE / "reproduce.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
TASKS = ("matbench_log_gvrh", "matbench_log_kvrh")


def write_single_task(path, name):
    """Synthetic merge fixture only; never used by the reproduction driver."""
    benchmark = MatbenchBenchmark(subset=[name], autoload=False)
    task = benchmark.tasks_map[name]
    for fold, split in task.validation.items():
        task.results[fold] = {
            "data": dict.fromkeys(split.test, 1.0),
            "parameters": {"feature_track": "B_comp+ph"},
            "scores": dict.fromkeys(REG_METRICS, 1.0),
            "uncertainty": None,
        }
    benchmark.to_file(str(path))


class PHProtocolTest(unittest.TestCase):
    def test_merge_preserves_both_tasks_and_rejects_duplicate_or_missing_task(self):
        self.assertTrue(
            callable(getattr(runner, "merge_results", None)),
            "reproduction must actually merge the two validated task runs",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / (name + ".json.gz") for name in TASKS]
            for path, name in zip(paths, TASKS):
                write_single_task(path, name)
            out = root / "merged.json.gz"
            runner.merge_results(paths, out)
            merged = MatbenchBenchmark.from_file(str(out))
            self.assertEqual(set(merged.tasks_map), set(TASKS))
            self.assertTrue(merged.is_valid)
            for name in TASKS:
                self.assertEqual(
                    next(iter(merged.tasks_map[name].results.fold_0.data.values())),
                    1.0,
                )
            with self.assertRaises(ValueError):
                runner.merge_results([paths[0]], out)
            with self.assertRaises(ValueError):
                runner.merge_results([paths[0], paths[0]], out)

    def test_structure_cache_can_be_reused_with_the_other_tasks_official_ids(self):
        import numpy as np
        import pandas as pd
        from pymatgen.core import Lattice, Structure
        import sys

        sys.path.insert(0, str(HERE / "src"))
        import protocol

        structures = [
            Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            for a in (3.0, 4.0)
        ]
        shear = pd.Series(
            structures, index=["mb-log-gvrh-00001", "mb-log-gvrh-00002"]
        )
        bulk = pd.Series(
            structures, index=["mb-log-kvrh-00001", "mb-log-kvrh-00002"]
        )
        with tempfile.TemporaryDirectory() as cache:
            first, _ = protocol.feature_matrix(shear, False, cache, workers=1)
            second, metadata = protocol.feature_matrix(bulk, False, cache, workers=1)
            self.assertEqual(second.index.tolist(), bulk.index.tolist())
            self.assertEqual(second.shape, (2, 207))
            np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
            self.assertTrue(metadata["cache_hit"])


if __name__ == "__main__":
    unittest.main()
