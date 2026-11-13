"""Check archived source bytes and optional externally cloned v1 input files.

Local archive filenames may differ from the original repository names. The
manifest records that mapping separately from the original names and hashes;
it never changes how an external source checkout is verified.
"""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root, entries, path_map=None):
    root = Path(root)
    path_map = {} if path_map is None else path_map
    if set(path_map) - set(entries):
        raise ValueError("Archive path mapping contains unknown source files")
    for relative, expected in entries.items():
        path = root / path_map.get(relative, relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
    return len(entries)


def verify_sources(root, source=None):
    """Verify local aliases and external original paths against the same hashes."""
    root = Path(root)
    manifest = json.loads((root / "provenance.json").read_text())
    counts = {
        "archived_source_files": verify(
            root / "upstream",
            manifest["upstream_sha256"],
            manifest.get("local_archive_paths", {}),
        ),
        "external_source_files": None,
        "external_input_archives": None,
    }
    if source is not None:
        # An external checkout retains upstream's names. Never apply the local
        # archive mapping here, even when a similarly named file happens to exist.
        counts["external_source_files"] = verify(source, manifest["upstream_sha256"])
        counts["external_input_archives"] = verify(
            source, manifest["data_nl_v1_sha256"]
        )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Pinned external AlchemBERT checkout"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    counts = verify_sources(root, args.source)
    print(f"Verified {counts['archived_source_files']} archived source files.")
    if args.source is not None:
        count = counts["external_input_archives"]
        print(f"Verified {count} external v1 input archives.")
    else:
        print(
            "External v1 input archives were not checked; "
            "supply --source to verify them."
        )


if __name__ == "__main__":
    main()
