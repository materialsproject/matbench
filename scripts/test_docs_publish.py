"""Check the final static site, including a real MkDocs clean rebuild."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from scripts import docs_publish


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.hrefs = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        self.hrefs.extend(value for key, value in attributes if key == "href")


class TestDocsPublish(unittest.TestCase):
    REPORT = Path("static/classification_scores.json")
    CSS = Path("assets/_mkdocstrings.css")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_resources(self, root):
        for relative, body in (
            (self.REPORT, b'{"records": [{"algorithm": "AlchemBERT"}]}\n'),
            (self.CSS, b".doc-heading { color: inherit; }\n"),
        ):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)

    def digest(self, root, relative):
        return hashlib.sha256((root / relative).read_bytes()).hexdigest()

    def assert_version(self, href, expected):
        self.assertEqual(parse_qs(urlsplit(href).query)["v"], [expected])

    def test_local_urls_preserve_context_and_preparation_is_idempotent(self):
        self.write_resources(self.root)
        nested = self.root / "How To Use" / "probability-migration" / "index.html"
        nested.parent.mkdir(parents=True)
        nested.write_text(
            '<a href="../../static/classification_scores.json?download=1'
            '&amp;v=old&amp;tag=a&amp;tag=b&amp;v=older#record-1">Audit</a>\n'
            '<link href="../../assets/_mkdocstrings.css?v=old#styles" '
            'rel="stylesheet">\n',
            encoding="utf-8",
        )
        external = (
            "https://example.org/static/classification_scores.json?v=external",
            "//example.org/assets/_mkdocstrings.css?v=external",
        )
        homepage = self.root / "index.html"
        homepage.write_text(
            '<a href="./static/classification_scores.json">Audit</a>\n'
            '<a href="/static/classification_scores.json#top">Root audit</a>\n'
            '<a href="other.json?keep=1&amp;v=untouched">Other</a>\n'
            '<div data-href="static/classification_scores.json">Data</div>\n'
            + "".join(f'<a href="{href}">External</a>\n' for href in external),
            encoding="utf-8",
        )

        docs_publish.prepare_site(self.root)

        report_hash = self.digest(self.root, self.REPORT)
        css_hash = self.digest(self.root, self.CSS)
        nested_text = nested.read_text(encoding="utf-8")
        audit, stylesheet = Links(nested_text).hrefs
        self.assert_version(audit, report_hash)
        self.assertEqual(
            urlsplit(audit).path, "../../static/classification_scores.json"
        )
        self.assertEqual(urlsplit(audit).fragment, "record-1")
        self.assertEqual(
            parse_qs(urlsplit(audit).query),
            {"download": ["1"], "tag": ["a", "b"], "v": [report_hash]},
        )
        self.assertIn("&amp;", nested_text)
        self.assert_version(stylesheet, css_hash)
        self.assertEqual(urlsplit(stylesheet).fragment, "styles")
        root_links = Links(homepage.read_text(encoding="utf-8")).hrefs
        for href in root_links[:2]:
            self.assert_version(href, report_hash)
        self.assertEqual(
            urlsplit(root_links[1]).path, "/static/classification_scores.json"
        )
        self.assertEqual(urlsplit(root_links[1]).fragment, "top")
        self.assertEqual(root_links[2], "other.json?keep=1&v=untouched")
        self.assertEqual(root_links[3:], list(external))
        self.assertIn(
            'data-href="static/classification_scores.json"',
            homepage.read_text(encoding="utf-8"),
        )
        self.assertTrue((self.root / ".nojekyll").is_file())

        before = {path: path.read_bytes() for path in (homepage, nested)}
        docs_publish.prepare_site(self.root)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_missing_required_publication_resource_fails(self):
        for missing in (self.REPORT, self.CSS):
            with self.subTest(missing=missing):
                site = self.root / missing.stem
                self.write_resources(site)
                (site / "index.html").write_text("<html></html>", encoding="utf-8")
                (site / missing).unlink()
                with self.assertRaises(FileNotFoundError):
                    docs_publish.prepare_site(site)

    @unittest.skipUnless(
        importlib.util.find_spec("mkdocs"),
        "Install requirements-docs.txt for the real MkDocs build test",
    )
    def test_mkdocs_build_versions_every_page_and_clean_rebuild_restores_marker(
        self,
    ):
        source = self.root / "source"
        site = self.root / "site"
        self.write_resources(source)
        pages = (
            Path("index.md"),
            Path("Leaderboards Per-Task/task.md"),
            Path("Full Benchmark Data/AlchemBERT.md"),
            Path("How To Use/probability-migration.md"),
        )
        for page in pages:
            destination = source / page
            destination.parent.mkdir(parents=True, exist_ok=True)
            prefix = "./" if page.parent == Path(".") else "../"
            destination.write_text(
                "# Publication check\n\n"
                f"[Classification score audit]({prefix}{self.REPORT.as_posix()})\n",
                encoding="utf-8",
            )
        config = self.root / "mkdocs.yml"
        config.write_text(
            "site_name: Publication check\n"
            f"docs_dir: {json.dumps(str(source))}\n"
            f"site_dir: {json.dumps(str(site))}\n"
            "extra_css:\n  - assets/_mkdocstrings.css\n"
            f"hooks:\n  - {json.dumps(str(Path(docs_publish.__file__).resolve()))}\n",
            encoding="utf-8",
        )

        def build():
            result = subprocess.run(
                [sys.executable, "-m", "mkdocs", "build", "-f", str(config)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        def check_pages():
            report_hash = self.digest(site, self.REPORT)
            css_hash = self.digest(site, self.CSS)
            self.assertTrue((site / ".nojekyll").is_file())
            for page in pages:
                built = (
                    site / "index.html"
                    if page == Path("index.md")
                    else site / page.with_suffix("") / "index.html"
                )
                hrefs = Links(built.read_text(encoding="utf-8")).hrefs
                for target, expected in (
                    (self.REPORT, report_hash),
                    (self.CSS, css_hash),
                ):
                    matches = [
                        href
                        for href in hrefs
                        if urlsplit(href).path.endswith(target.as_posix())
                    ]
                    self.assertTrue(matches, f"{built}: missing link to {target}")
                    for href in matches:
                        self.assert_version(href, expected)
                        self.assertEqual(
                            (built.parent / urlsplit(href).path).resolve(),
                            (site / target).resolve(),
                        )
            return report_hash, css_hash

        build()
        original_report_hash, original_css_hash = check_pages()
        (site / ".nojekyll").unlink()
        obsolete = site / "obsolete-output.txt"
        obsolete.write_text("must be removed by the clean build", encoding="utf-8")
        (source / self.REPORT).write_text(
            '{"records": [{"algorithm": "AlchemBERT", "updated": true}]}\n',
            encoding="utf-8",
        )
        build()
        updated_report_hash, updated_css_hash = check_pages()
        self.assertNotEqual(original_report_hash, updated_report_hash)
        self.assertEqual(original_css_hash, updated_css_hash)
        self.assertFalse(obsolete.exists())


if __name__ == "__main__":
    unittest.main()
