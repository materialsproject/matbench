import argparse
import glob
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from monty.serialization import loadfn

from matbench.bench import MatbenchBenchmark
from matbench.leaderboard import prepare_classification_scores

BENCHMARKS_DIR = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir), "benchmarks")

LOCAL = False
# LOCAL = True

INFO_FILE = "info.json"
RESULTS_FILE = "results.json.gz"
SRC_NB_GLOB = "*.ipynb"
SRC_PY_GLOB = "*.py"


REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def validate_classification_submission(benchmark):
    """Require probability predictions and honest cached metrics in a new result.

    Historical deserialization stays permissive; this stricter gate is applied
    only when results are added or replaced. Canonical test IDs and targets are
    checked by the same scorer that generates the current leaderboard.
    """
    packet = {"results": benchmark}
    prepare_classification_scores({"submission": packet})
    for task in benchmark.tasks:
        entry = packet["classification_scores"].get(task.dataset_name)
        if entry is None:
            continue
        if not entry["eligible"]:
            raise ValueError(f"{task.dataset_name}: {entry['reason']}")
        for fold_key, recomputed in entry["fold_scores"].items():
            stored = task.results[fold_key]["scores"]
            for metric, actual in recomputed.items():
                submitted = stored.get(metric)
                if (isinstance(submitted, bool) or
                        not isinstance(submitted, (int, float)) or
                        not math.isfinite(submitted) or
                        not math.isclose(submitted, actual, rel_tol=1e-8, abs_tol=1e-10)):
                    raise ValueError(
                        f"{task.dataset_name}/{fold_key}/{metric}: stored score "
                        f"{submitted!r} disagrees with recomputed {actual:.12g}. "
                        "Record probabilities using the current scoring code."
                    )
    return packet["classification_scores"]


def changed_result_files(base_ref, repo_dir=REPOSITORY_DIR):
    """Return only added/replaced result files; never re-gate untouched history."""
    repo_dir = Path(repo_dir)
    base = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}"],
        cwd=repo_dir, text=True,
    ).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=AMRT", "-z",
         base, "HEAD", "--", "benchmarks"], cwd=repo_dir,
    ).decode().split("\0")
    return [repo_dir / name for name in changed
            if len(Path(name).parts) == 3 and
            Path(name).parts[0] == "benchmarks" and
            Path(name).name == RESULTS_FILE and
            ((repo_dir / name).is_file() or (repo_dir / name).is_symlink())]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate new classification submissions against canonical test targets."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--base-ref", help="Git base commit; check changed results only")
    selector.add_argument("--submission", action="append", metavar="DIRECTORY",
                          help="Submission directory under benchmarks (repeatable)")
    args = parser.parse_args(argv)
    if args.base_ref:
        results = changed_result_files(args.base_ref)
    else:
        results = [Path(BENCHMARKS_DIR) / name / RESULTS_FILE
                   for name in args.submission]
    for result_path in results:
        if result_path.is_symlink():
            raise ValueError(f"{result_path}: symlink result artifacts are not accepted")
        benchmark = MatbenchBenchmark.from_file(str(result_path))
        validate_classification_submission(benchmark)
        print(f"Probability submission check passed: {result_path}")
    print(f"Validated {len(results)} new or updated result file(s).")


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
    if len(sys.argv) == 1:
        unittest.main()
    else:
        main()
