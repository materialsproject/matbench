"""CPU-only regression: an ordinary HF cache filename is a symlink to a blob."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from jdft2d_review_reproduce import checkpoint_directory


class CheckpointPathTests(unittest.TestCase):
    def test_hf_snapshot_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = root / 'blobs' / 'hash-only-filename'
            blob.parent.mkdir()
            blob.write_bytes(b'fixture-checkpoint')
            snapshot = root / 'snapshots' / 'revision' / 'tabpfn-v2-regressor-v2_default.ckpt'
            snapshot.parent.mkdir(parents=True)
            snapshot.symlink_to(blob)
            self.assertFalse((snapshot.resolve().parent / snapshot.name).exists())
            model = checkpoint_directory(snapshot, root / 'stage-model', hashlib.sha256(blob.read_bytes()).hexdigest())
            self.assertEqual((model / snapshot.name).read_bytes(), blob.read_bytes())
            self.assertTrue(snapshot.is_symlink())

    def test_explicit_nonstandard_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'download.bin'
            source.write_bytes(b'fixture-checkpoint')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            model = checkpoint_directory(source, root / 'stage-model', digest)
            self.assertEqual((model / 'tabpfn-v2-regressor-v2_default.ckpt').read_bytes(), source.read_bytes())
            with self.assertRaises(AssertionError):
                checkpoint_directory(source, root / 'bad-model', '0' * 64)
            self.assertFalse((root / 'bad-model').exists())


if __name__ == '__main__':
    unittest.main()
