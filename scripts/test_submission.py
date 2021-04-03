import os
from unittest import TestCase

from monty.serialization import loadfn

from matbench.bench import MatbenchBenchmark


BENCHMARKS_DIR = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir), "benchmarks")

LOCAL = False
# LOCAL = True

INFO_FILE = "info.json"
NOTEBOOK_FILE = "notebook.ipynb"
RESULTS_FILE = "results.json.gz"

class BenchmarkSubmissionTest(TestCase):

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
                for required_file in (INFO_FILE, RESULTS_FILE, NOTEBOOK_FILE):

                    full_path = os.path.join(bmark_path, required_file)
                    self.assertTrue(os.path.exists(full_path))

                    if required_file == INFO_FILE:
                        info = loadfn(full_path)
                        self.assertIn("notes", info)

                        for field_must_not_be_empty in ("authors", "algorithm", "algorithm_long", "bibtex_refs"):
                            self.assertTrue(info.get(field_must_not_be_empty))

                        algo_names.append(info["algorithm"])

                    if required_file == RESULTS_FILE:
                        mb = MatbenchBenchmark.from_file(full_path)
                        is_valid = mb.is_valid
                        self.assertTrue(is_valid)

                        print(f"{RESULTS_FILE} file in {bmark_path}: valid ({is_valid}), complete ({mb.is_complete}), recorded ({mb.is_recorded})")

            else:
                print(f"path {bmark_path} skipped!")

        # Each algo name must be unique
        self.assertEqual(len(algo_names), len(set(algo_names)))
