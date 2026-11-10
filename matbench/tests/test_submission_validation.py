"""Submission-policy tests; archived results use separate structural checks."""

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from matbench.constants import CLF_KEY, REG_KEY
from matbench.data_ops import score_array
from scripts.test_submission import (
    changed_result_files,
    main,
    validate_classification_submission,
)


class TestClassificationSubmission(unittest.TestCase):
    def make_task(self, predictions=None):
        target = pd.Series([False, True], index=["negative", "positive"])
        if predictions is None:
            # Saved dictionary order deliberately differs from official target order.
            predictions = {"positive": 0.4, "negative": 0.2}
        expected = score_array(target, [0.2, 0.4], CLF_KEY)
        task = SimpleNamespace(
            dataset_name="synthetic", benchmark_name="matbench_v0.1",
            metadata=SimpleNamespace(task_type=CLF_KEY),
            folds_map={0: "fold_0"}, scores={},
            results={"fold_0": {"data": predictions, "scores": expected}},
        )
        return task, {"fold_0": target}

    def validate(self, task, targets):
        with patch("matbench.leaderboard._load_fold_targets", return_value=targets):
            return validate_classification_submission(SimpleNamespace(tasks=[task]))

    def test_correct_probability_metrics_pass_without_mutation(self):
        task, targets = self.make_task()
        saved = copy.deepcopy(task.results)
        result = self.validate(task, targets)
        self.assertEqual(result["synthetic"]["scores"]["rocauc"]["mean"], 1.0)
        self.assertEqual(task.results, saved)

    def test_old_thresholded_auc_is_rejected(self):
        task, targets = self.make_task()
        task.results["fold_0"]["scores"]["rocauc"] = 0.5
        with self.assertRaisesRegex(ValueError, "rocauc.*disagrees"):
            self.validate(task, targets)

    def test_other_stale_classification_metrics_are_rejected(self):
        task, targets = self.make_task()
        task.results["fold_0"]["scores"]["accuracy"] = 1.0
        with self.assertRaisesRegex(ValueError, "accuracy.*disagrees"):
            self.validate(task, targets)

    def test_hard_labels_and_invalid_probabilities_rejected(self):
        for values in [(False, True), (0, 1), (0.2, True),
                       (float("nan"), 0.4), (-0.1, 0.4), (0.2, 1.1)]:
            with self.subTest(values=values):
                task, targets = self.make_task(dict(zip(["negative", "positive"], values)))
                with self.assertRaises(ValueError):
                    self.validate(task, targets)

    def test_cached_nonfinite_or_non_numeric_metrics_rejected(self):
        for invalid in [float("nan"), float("inf"), True, "1.0", None]:
            with self.subTest(invalid=invalid):
                task, targets = self.make_task()
                task.results["fold_0"]["scores"]["rocauc"] = invalid
                with self.assertRaisesRegex(ValueError, "rocauc.*disagrees"):
                    self.validate(task, targets)

    def test_incorrect_sample_ids_abort(self):
        task, targets = self.make_task({"wrong": 0.2, "positive": 0.4})
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.validate(task, targets)

    def test_regression_only_submission_does_not_load_classification_data(self):
        task = SimpleNamespace(dataset_name="regression", metadata=SimpleNamespace(task_type=REG_KEY))
        with patch("matbench.leaderboard._load_fold_targets") as load:
            self.assertEqual(validate_classification_submission(SimpleNamespace(tasks=[task])), {})
            load.assert_not_called()


class TestChangedSubmissionSelection(unittest.TestCase):
    def test_cli_rejects_symlink_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission = root / "symlink_submission"
            submission.mkdir()
            (submission / "results.json.gz").symlink_to("missing.json.gz")
            with patch("scripts.test_submission.BENCHMARKS_DIR", root), patch(
                "scripts.test_submission.MatbenchBenchmark.from_file"
            ) as load:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    main(["--submission", "symlink_submission"])
                load.assert_not_called()

    def test_type_changed_dangling_symlink_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "benchmarks" / "changed" / "results.json.gz"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("original artifact")

            def git(*args):
                return subprocess.check_output(
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
                    cwd=root, stderr=subprocess.DEVNULL, text=True,
                ).strip()

            git("init")
            git("add", ".")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD")
            artifact.unlink()
            artifact.symlink_to("missing.json.gz")
            git("add", ".")
            git("commit", "-m", "replace artifact with symlink")
            self.assertEqual(changed_result_files(base, root), [artifact])

    def test_only_new_and_replaced_result_files_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
                    cwd=root, stderr=subprocess.DEVNULL, text=True,
                ).strip()

            def write(relative, text):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)

            git("init")
            for name in ["untouched", "changed", "metadata_only", "deleted"]:
                write(f"benchmarks/{name}/results.json.gz", "old artifact")
                write(f"benchmarks/{name}/info.json", "old metadata")
            git("add", ".")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD")
            write("benchmarks/changed/results.json.gz", "new predictions")
            write("benchmarks/new model/results.json.gz", "new submission")
            write("benchmarks/metadata_only/info.json", "updated citation")
            (root / "benchmarks/deleted/results.json.gz").unlink()
            git("add", ".")
            git("commit", "-m", "update")
            selected = {path.relative_to(root).as_posix() for path in changed_result_files(base, root)}
            self.assertEqual(selected, {
                "benchmarks/changed/results.json.gz", "benchmarks/new model/results.json.gz"
            })


if __name__ == "__main__":
    unittest.main()
