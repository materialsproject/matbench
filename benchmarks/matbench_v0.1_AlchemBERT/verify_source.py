"""Check the archived source and optional externally cloned v1 input files."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root, entries):
    for relative, expected in entries.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Pinned external AlchemBERT checkout"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "provenance.json").read_text())
    count = verify(root / "upstream", manifest["upstream_sha256"])
    print(f"Verified {count} archived source files.")
    if args.source is not None:
        verify(args.source, manifest["upstream_sha256"])
        count = verify(args.source, manifest["data_nl_v1_sha256"])
        print(f"Verified {count} external v1 input archives.")
    else:
        print(
            "External v1 input archives were not checked; "
            "supply --source to verify them."
        )


if __name__ == "__main__":
    main()
