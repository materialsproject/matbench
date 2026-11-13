"""Test migration invariants using the real archive and synthetic score reports.

Scoring itself is covered separately; these tests require no dataset downloads.
"""

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from migrate_results import canonical, migrate_results, without_scores
from rescore_results import read_verified_json
from validate_results import ARCHIVED_SHA256, HISTORICAL_RESULTS


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_verified_json(HISTORICAL_RESULTS, ARCHIVED_SHA256, True)
        tasks = {}
        for name, task in cls.source["tasks"].items():
            folds = {}
            for fold, data in task["results"].items():
                scores = dict(data["scores"])
                metric = "rocauc" if "rocauc" in scores else "mae"
                scores[metric] = scores[metric] / 2
                folds[fold] = {
                    "n_samples": len(data["data"]),
                    "stored_scores": dict(data["scores"]),
                    "recomputed_scores": scores,
                }
            tasks[name] = {"folds": folds, "label_source": {"kind": "TEST-ONLY"}}
        cls.report = {
            "artifact_sha256": ARCHIVED_SHA256,
            "upstream_commit": "TEST-ONLY",
            "task_count": 13,
            "fold_count": 65,
            "not_rescored": [],
            "tasks": tasks,
            "software": {},
            "spec_sha256": {},
        }

    def test_roundtrip_preserves_predictions_parameters_and_original_bytes(self):
        from matbench.bench import MatbenchBenchmark

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "migrated.json.gz"
            report_output = Path(folder) / "rescoring.json"
            with patch(
                "migrate_results.build_report",
                return_value=copy.deepcopy(self.report),
            ) as scorer:
                migrate_results(
                    HISTORICAL_RESULTS, Path(folder), output, report_output
                )
            scorer.assert_called_once_with(
                results=HISTORICAL_RESULTS, dataset_dir=Path(folder)
            )
            raw = json.loads(gzip.decompress(output.read_bytes()))
            self.assertEqual(
                canonical(without_scores(raw["tasks"])),
                canonical(without_scores(self.source["tasks"])),
            )
            for name, task in raw["tasks"].items():
                for fold, values in task["results"].items():
                    self.assertEqual(
                        values["scores"],
                        self.report["tasks"][name]["folds"][fold][
                            "recomputed_scores"
                        ],
                    )
            self.assertNotEqual(raw["hash"], self.source["hash"])
            loaded = MatbenchBenchmark.from_file(str(output))
            self.assertTrue(
                loaded.is_valid and loaded.is_complete and loaded.is_recorded
            )
            metadata = raw["user_metadata"]["score_migration"]
            self.assertTrue(
                metadata["predictions_parameters_and_uncertainties_unchanged"]
            )
            self.assertFalse(metadata["retrained"])
            self.assertEqual(
                metadata["rescoring_report_sha256"],
                hashlib.sha256(report_output.read_bytes()).hexdigest(),
            )
        self.assertEqual(
            hashlib.sha256(HISTORICAL_RESULTS.read_bytes()).hexdigest(),
            ARCHIVED_SHA256,
        )

    def test_missing_fold_report_is_rejected_before_writing(self):
        report = copy.deepcopy(self.report)
        del report["tasks"]["matbench_glass"]["folds"]["fold_4"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "migrated.json.gz"
            with patch("migrate_results.build_report", return_value=report):
                with self.assertRaisesRegex(ValueError, "all five folds"):
                    migrate_results(HISTORICAL_RESULTS, Path(folder), output)
            self.assertFalse(output.exists())

    def test_existing_output_or_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "migrated.json.gz"
            report_output = Path(folder) / "report.json"
            output.write_bytes(b"keep-result")
            with patch("migrate_results.build_report") as scorer:
                with self.assertRaises(FileExistsError):
                    migrate_results(HISTORICAL_RESULTS, Path(folder), output)
                scorer.assert_not_called()
            self.assertEqual(output.read_bytes(), b"keep-result")
            output.unlink()
            report_output.write_bytes(b"keep-report")
            with patch("migrate_results.build_report") as scorer:
                with self.assertRaises(FileExistsError):
                    migrate_results(
                        HISTORICAL_RESULTS, Path(folder), output, report_output
                    )
                scorer.assert_not_called()
            self.assertEqual(report_output.read_bytes(), b"keep-report")
            self.assertFalse(output.exists())

    def test_changed_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "changed.json.gz"
            archive.write_bytes(HISTORICAL_RESULTS.read_bytes() + b"tamper")
            output = Path(folder) / "migrated.json.gz"
            with patch("migrate_results.build_report") as scorer:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    migrate_results(archive, Path(folder), output)
                scorer.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
