import os
import glob
import unittest

from monty.serialization import loadfn

from matbench.bench import MatbenchBenchmark


BENCHMARKS_DIR = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir), "benchmarks")

LOCAL = False
# LOCAL = True

INFO_FILE = "info.json"
RESULTS_FILE = "results.json.gz"
SRC_NB_GLOB = "*.ipynb"
SRC_PY_GLOB = "*.py"


class BenchmarkSubmissionTest(unittest.TestCase):

    def setUp(self) -> None:
        print(f"Running test in benchmark dir: {BENCHMARKS_DIR}")
        print(f"Running test locally? {LOCAL}")

    def test_dir_structure_correct(self):
        self.assertTrue(os.path.exists(BENCHMARKS_DIR))

        # make sure the dir only has submissions as directories
        for file_or_dir in os.listdir(BENCHMARKS_DIR):
            full_path = os.path.join(BENCHMARKS_DIR, file_or_dir)
            if not full_path.endswith(".DS_Store"):
                self.assertTrue(os.path.isdir(full_path))

            if not LOCAL:
                for banned_file in (".ipynb_checkpoints",):
                    self.assertFalse(os.path.exists(os.path.join(full_path, banned_file)))

    def test_submissions(self):

        algo_names = []
        for bmark_d in os.listdir(BENCHMARKS_DIR):
            bmark_path = os.path.join(BENCHMARKS_DIR, bmark_d)
            if os.path.isdir(bmark_path):
                print(f"checking benchmark {bmark_path}")
                for required_file in (INFO_FILE, RESULTS_FILE):

                    full_path = os.path.join(bmark_path, required_file)
                    self.assertTrue(os.path.exists(full_path))

                    if required_file == INFO_FILE:
                        info = loadfn(full_path)
                        self.assertIn("notes", info)

                        for field_must_not_be_empty in ("authors", "algorithm", "algorithm_long", "bibtex_refs", "requirements"):
                            self.assertIsNotNone(info.get(field_must_not_be_empty, None))

                        algo_names.append(info["algorithm"])

                    if required_file == RESULTS_FILE:
                        mb = MatbenchBenchmark.from_file(full_path)
                        is_valid = mb.is_valid
                        self.assertTrue(is_valid)

                        print(f"{RESULTS_FILE} file in {bmark_path}: valid ({is_valid}), complete ({mb.is_complete}), recorded ({mb.is_recorded})")


                # benchmark must either contain a notebook or at least one .py file
                nb_paths = glob.glob(os.path.join(bmark_path, SRC_NB_GLOB))
                py_paths = glob.glob(os.path.join(bmark_path, SRC_PY_GLOB))
                self.assertGreater(len(nb_paths + py_paths), 0)

                # ensure there are no large (10MB+) files in the submission
                # apart from the results
                for file in os.listdir(bmark_path):
                    if file != RESULTS_FILE:
                        fpath = os.path.join(bmark_path, file)
                        fsize_mb = os.stat(fpath).st_size/1e6
                        self.assertLessEqual(fsize_mb, 10)

            else:
                print(f"path {bmark_path} skipped!")

        # Each algo name must be unique
        self.assertEqual(len(algo_names), len(set(algo_names)))


if __name__ == "__main__":
    unittest.main()