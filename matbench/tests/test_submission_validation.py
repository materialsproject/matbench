"""Submission-policy tests; archived results use separate structural checks."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
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

    def validate(self, task, targets, **kwargs):
        with patch("matbench.leaderboard._load_fold_targets", return_value=targets):
            return validate_classification_submission(SimpleNamespace(tasks=[task]), **kwargs)

    def make_hard_label_task(self):
        task, targets = self.make_task({"positive": True, "negative": False})
        for key, fold in task.results.items():
            fold["scores"] = score_array(
                targets[key].to_numpy(), [False, True], CLF_KEY
            )
        return task, targets

    @contextmanager
    def reviewed_artifact(self, **overrides):
        """Use a reviewed identity fixture without weakening scorer validation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "benchmarks" / "reviewed" / "results.json.gz"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"unchanged original artifact fixture")
            entry = {
                "path": artifact.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "submitted_at": "2026-09-21T14:29:03Z",
                "original_commit_date": "2026-09-21T14:17:29Z",
                "hard_label_tasks": ["synthetic"],
                **overrides,
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"artifacts": [entry]}))
            with patch("scripts.test_submission.REPOSITORY_DIR", root), patch(
                "scripts.test_submission.HISTORICAL_CLASSIFICATION_MANIFEST", manifest
            ):
                yield artifact

    def test_reviewed_original_hard_labels_pass_as_ba_only_without_mutation(self):
        task, targets = self.make_hard_label_task()
        saved = copy.deepcopy(task.results)
        with self.reviewed_artifact() as artifact:
            result = self.validate(task, targets, result_path=artifact)["synthetic"]
        self.assertTrue(result["ba_available"])
        self.assertFalse(result["eligible"])
        self.assertFalse(result["auc_available"])
        self.assertIsNone(result["scores"]["rocauc"])
        self.assertEqual(task.results, saved)

    def test_unlisted_path_or_mutated_artifact_remains_probability_only(self):
        for change in ("unknown_path", "changed_bytes"):
            with self.subTest(change=change), self.reviewed_artifact() as artifact:
                if change == "unknown_path":
                    other = artifact.parent.parent / "unreviewed" / artifact.name
                    other.parent.mkdir()
                    other.write_bytes(artifact.read_bytes())
                    artifact = other
                else:
                    artifact.write_bytes(artifact.read_bytes() + b"changed")
                task, targets = self.make_hard_label_task()
                with self.assertRaisesRegex(ValueError, "probabilities"):
                    self.validate(task, targets, result_path=artifact)

    def test_review_cannot_admit_a_post_gate_artifact_or_unlisted_task(self):
        for key in ("submitted_at", "original_commit_date"):
            with self.subTest(key=key), self.reviewed_artifact(
                **{key: "2026-11-10T00:00:00Z"}
            ) as artifact:
                task, targets = self.make_hard_label_task()
                with self.assertRaisesRegex(ValueError, "predate"):
                    self.validate(task, targets, result_path=artifact)
        with self.reviewed_artifact(hard_label_tasks=["different_task"]) as artifact:
            task, targets = self.make_hard_label_task()
            with self.assertRaisesRegex(ValueError, "probabilities"):
                self.validate(task, targets, result_path=artifact)

    def test_reviewed_hard_labels_still_require_every_cached_metric(self):
        with self.reviewed_artifact() as artifact:
            for metric in ("accuracy", "balanced_accuracy", "f1", "rocauc"):
                for invalid in (0.0, float("nan"), float("inf"), True, "1.0", None):
                    with self.subTest(metric=metric, invalid=invalid):
                        task, targets = self.make_hard_label_task()
                        task.results["fold_0"]["scores"][metric] = invalid
                        with self.assertRaisesRegex(ValueError, metric + ".*disagrees"):
                            self.validate(task, targets, result_path=artifact)

    def test_reviewed_identity_never_hides_invalid_predictions_or_folds(self):
        with self.reviewed_artifact() as artifact:
            for damage in ("missing_fold", "extra_fold", "wrong_id", "mixed", "nan", "nonbinary"):
                with self.subTest(damage=damage):
                    task, targets = self.make_hard_label_task()
                    values = task.results["fold_0"]["data"]
                    if damage == "missing_fold":
                        del task.results["fold_4"]
                    elif damage == "extra_fold":
                        task.results["fold_5"] = copy.deepcopy(task.results["fold_0"])
                    elif damage == "wrong_id":
                        values["wrong_id"] = values.pop("fold_0_negative")
                    else:
                        values["fold_0_negative"] = {
                            "mixed": 0.0, "nan": float("nan"), "nonbinary": 2
                        }[damage]
                    with self.assertRaises(ValueError):
                        self.validate(task, targets, result_path=artifact)

    def test_reviewed_identity_does_not_relax_probability_metric_checks(self):
        with self.reviewed_artifact() as artifact:
            task, targets = self.make_task()
            result = self.validate(task, targets, result_path=artifact)["synthetic"]
            self.assertTrue(result["auc_available"])
            task.results["fold_0"]["scores"]["rocauc"] = 0.5
            with self.assertRaisesRegex(ValueError, "rocauc.*disagrees"):
                self.validate(task, targets, result_path=artifact)

    def test_reviewed_cli_checks_structure_and_reports_ba_only(self):
        with self.reviewed_artifact() as artifact:
            for is_valid in (False, True):
                with self.subTest(is_valid=is_valid):
                    task, targets = self.make_hard_label_task()
                    benchmark = SimpleNamespace(tasks=[task], is_valid=is_valid)
                    output = io.StringIO()
                    with patch("scripts.test_submission.BENCHMARKS_DIR", artifact.parent.parent), patch(
                        "scripts.test_submission.MatbenchBenchmark.from_file", return_value=benchmark
                    ) as load, patch(
                        "matbench.leaderboard._load_fold_targets", return_value=targets
                    ) as load_targets, redirect_stdout(output):
                        if is_valid:
                            main(["--submission", "reviewed"])
                            self.assertIn("Historical hard-label/BA check passed", output.getvalue())
                            self.assertNotIn("Probability submission check passed", output.getvalue())
                        else:
                            with self.assertRaisesRegex(ValueError, "structural"):
                                main(["--submission", "reviewed"])
                            load_targets.assert_not_called()
                    load.assert_called_once_with(str(artifact))

    def test_symlink_cannot_claim_reviewed_artifact_identity(self):
        with self.reviewed_artifact() as artifact:
            alias = artifact.parent / "alias.json.gz"
            alias.symlink_to(artifact)
            task, targets = self.make_hard_label_task()
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.validate(task, targets, result_path=alias)

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
