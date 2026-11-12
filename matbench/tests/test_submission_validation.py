"""Submission-policy tests; archived results use separate structural checks."""

import copy
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
            dataset_name="synthetic",
            benchmark_name="matbench_v0.1",
            metadata=SimpleNamespace(task_type=CLF_KEY),
            folds_map={i: f"fold_{i}" for i in range(5)},
            scores={},
            results={
                f"fold_{i}": {
                    "data": {
                        f"fold_{i}_{key}": value
                        for key, value in predictions.items()
                    },
                    "scores": expected.copy(),
                }
                for i in range(5)
            },
        )
        return task, {
            f"fold_{i}": pd.Series(
                target.to_numpy(), index=[f"fold_{i}_negative", f"fold_{i}_positive"]
            )
            for i in range(5)
        }

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
        for values in [
            (False, True),
            (0, 1),
            (0.2, True),
            (float("nan"), 0.4),
            (-0.1, 0.4),
            (0.2, 1.1),
        ]:
            with self.subTest(values=values):
                task, targets = self.make_task(
                    dict(zip(["negative", "positive"], values))
                )
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
        task = SimpleNamespace(
            dataset_name="regression", metadata=SimpleNamespace(task_type=REG_KEY)
        )
        with patch("matbench.leaderboard._load_fold_targets") as load:
            self.assertEqual(
                validate_classification_submission(SimpleNamespace(tasks=[task])), {}
            )
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
            artifact.symlink_to("missing.json.gz")
            base_sha = "a" * 40
            with patch(
                "scripts.test_submission.subprocess.check_output",
                side_effect=[
                    base_sha + "\n",
                    b"benchmarks/changed/results.json.gz\0",
                ],
            ) as git:
                self.assertEqual(
                    changed_result_files("origin/main", root), [artifact]
                )
            self.assertEqual(
                git.call_args_list[0].args[0],
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    "origin/main^{commit}",
                ],
            )
            self.assertEqual(
                git.call_args_list[1].args[0],
                [
                    "git",
                    "diff",
                    "--name-only",
                    "--diff-filter=AMRT",
                    "-z",
                    base_sha,
                    "HEAD",
                    "--",
                    "benchmarks",
                ],
            )

    def test_only_new_and_replaced_result_files_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in [
                "benchmarks/untouched/results.json.gz",
                "benchmarks/changed/results.json.gz",
                "benchmarks/new model/results.json.gz",
                "benchmarks/metadata_only/info.json",
                "benchmarks/nested/extra/results.json.gz",
                "unrelated/results.json.gz",
            ]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture artifact")
            # This is the NUL-separated output of diff --diff-filter=AMRT:
            # untouched and deleted files must not be selected; spaces are literal.
            changed = [
                "benchmarks/changed/results.json.gz",
                "benchmarks/new model/results.json.gz",
                "benchmarks/metadata_only/info.json",
                "benchmarks/nested/extra/results.json.gz",
                "benchmarks/missing/results.json.gz",
                "unrelated/results.json.gz",
            ]
            with patch(
                "scripts.test_submission.subprocess.check_output",
                side_effect=["a" * 40 + "\n", ("\0".join(changed) + "\0").encode()],
            ) as git:
                selected = {
                    path.relative_to(root).as_posix()
                    for path in changed_result_files("origin/main", root)
                }
            self.assertEqual(
                selected,
                {
                    "benchmarks/changed/results.json.gz",
                    "benchmarks/new model/results.json.gz",
                },
            )
            self.assertEqual(git.call_count, 2)
            self.assertTrue(
                all(call.kwargs["cwd"] == root for call in git.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
