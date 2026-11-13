"""Protect archived source identity while preserving external original paths."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_source import verify_sources


class SourceArchiveTests(unittest.TestCase):
    def make_fixture(self, root):
        local = root / "submission"
        upstream = local / "upstream"
        external = root / "external"
        upstream.mkdir(parents=True)
        external.mkdir()
        original = b"torch==2.2.1+cu118\n"
        manifest = {
            "upstream_sha256": {
                "requirements.txt": hashlib.sha256(original).hexdigest()
            },
            "local_archive_paths": {"requirements.txt": "requirements.txt.archive"},
            "data_nl_v1_sha256": {},
        }
        (local / "provenance.json").write_text(json.dumps(manifest))
        (upstream / "requirements.txt.archive").write_bytes(original)
        (external / "requirements.txt").write_bytes(original)
        return local, external, original

    def test_real_archive_matches_original_source_hashes(self):
        root = Path(__file__).resolve().parent
        counts = verify_sources(root)
        self.assertEqual(counts["archived_source_files"], 10)
        self.assertIsNone(counts["external_source_files"])
        self.assertIsNone(counts["external_input_archives"])

    def test_changed_local_archive_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            local, _, _ = self.make_fixture(Path(folder))
            archived = local / "upstream" / "requirements.txt.archive"
            archived.write_bytes(b"torch==2.13.0\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_sources(local)

    def test_external_checkout_uses_original_name_without_local_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            local, external, _ = self.make_fixture(Path(folder))
            (external / "requirements.txt.archive").write_bytes(b"not source")
            counts = verify_sources(local, external)
            self.assertEqual(counts["archived_source_files"], 1)
            self.assertEqual(counts["external_source_files"], 1)
            self.assertEqual(counts["external_input_archives"], 0)
            (external / "requirements.txt").write_bytes(b"modified source")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_sources(local, external)

    def test_external_checkout_cannot_substitute_a_renamed_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            local, external, original = self.make_fixture(Path(folder))
            (external / "requirements.txt").unlink()
            (external / "requirements.txt.archive").write_bytes(original)
            with self.assertRaises(FileNotFoundError):
                verify_sources(local, external)


if __name__ == "__main__":
    unittest.main()
