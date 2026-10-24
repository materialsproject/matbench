import importlib.util
import os
import tempfile
import unittest
from unittest import mock

# rebuild_docs.py needs the packages in requirements-docs.txt
DOCS_DEPS = ("plotly", "pymatviz", "tqdm")
HAS_DOCS_DEPS = all(importlib.util.find_spec(dep) is not None for dep in DOCS_DEPS)


@unittest.skipUnless(HAS_DOCS_DEPS, "requires the packages in requirements-docs.txt")
class RebuildDocsTest(unittest.TestCase):

    def test_index_points_to_matbench_discovery(self):
        # Matbench Discovery has its own website; the index page should only
        # link to it, not host its leaderboard (issue #388)
        from scripts import rebuild_docs

        with tempfile.TemporaryDirectory() as docs_dir:
            static_dir = os.path.join(docs_dir, "static")
            os.makedirs(static_dir)
            with mock.patch.object(rebuild_docs, "DOCS_DIR", docs_dir), \
                    mock.patch.object(rebuild_docs, "STATIC_DOCS_DIR", static_dir):
                rebuild_docs.generate_general_purpose_leaderboard_and_plot(
                    {}, {"algos": 1, "tasks": 2, "benchmarks": 1}
                )
            with open(os.path.join(docs_dir, "index.md"), encoding="utf-8") as f:
                index = f.read()

        self.assertIn(rebuild_docs.MATBENCH_DISCOVERY_NOTE, index)
        self.assertIn("https://matbench-discovery.materialsproject.org/", index)
        self.assertNotIn("Leaderboard-Discovery", index)
        self.assertNotIn("matbench_discovery", index)


if __name__ == "__main__":
    unittest.main()
