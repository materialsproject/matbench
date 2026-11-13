"""Finalize MkDocs output for GitHub Pages and cached static downloads."""

import hashlib
import html
import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

VERSIONED_FILES = (
    "static/classification_scores.json",
    "assets/_mkdocstrings.css",
)
HREF = re.compile(r"((?<![\w:-])href\s*=\s*)([\"'])(.*?)\2", re.IGNORECASE)


def prepare_site(site_dir):
    """Version local audit/style links and preserve underscore files on Pages.

    Work on final HTML so generated pages, handwritten Markdown and theme CSS
    references all use the same version derived from the published file bytes.
    The canonical JSON remains available for existing consumers.
    """
    site_dir = Path(site_dir).resolve()
    versions = {
        (site_dir / name)
        .resolve(): hashlib.sha256((site_dir / name).read_bytes())
        .hexdigest()
        for name in VERSIONED_FILES
    }
    references = dict.fromkeys(versions, 0)

    for page in site_dir.rglob("*.html"):

        def version_href(match):
            url = urlsplit(html.unescape(match[3]))
            if url.scheme or url.netloc or not url.path:
                return match[0]
            path = unquote(url.path)
            target = (
                site_dir / path.lstrip("/")
                if path.startswith("/")
                else page.parent / path
            ).resolve()
            if target not in versions:
                return match[0]
            references[target] += 1
            query = [
                (key, value)
                for key, value in parse_qsl(url.query, keep_blank_values=True)
                if key != "v"
            ]
            query.append(("v", versions[target]))
            versioned = urlunsplit(url._replace(query=urlencode(query)))
            return (
                match[1] + match[2] + html.escape(versioned, quote=True) + match[2]
            )

        original = page.read_text(encoding="utf-8")
        versioned = HREF.sub(version_href, original)
        if versioned != original:
            page.write_text(versioned, encoding="utf-8")

    missing = [
        str(path.relative_to(site_dir))
        for path, count in references.items()
        if not count
    ]
    if missing:
        raise ValueError(f"Published assets have no HTML references: {missing}")
    # MkDocs excludes dotfiles from docs_src by default. Create the marker
    # after every clean build, at the actual GitHub Pages publishing root.
    (site_dir / ".nojekyll").touch()


def on_post_build(config):
    """MkDocs hook; applies equally to local builds and CI previews/deploys."""
    prepare_site(config["site_dir"])
