#!/usr/bin/env python3
"""Capture or verify SHA-256 hashes of the upstream Mat2Vec assets.

Does not execute/unpickle downloaded models. Do not treat a new manifest as
historical proof: it describes bytes fetched from the upstream URLs *now*.

  python mat2vec_integrity.py capture --cache ./mat2vec_cache
  python mat2vec_integrity.py verify --cache ./mat2vec_cache

The capture command writes mat2vec_sha256.json, suitable for a subsequent
review and commit. Verification fails closed when a hash mismatches.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen

SOURCE = "https://storage.googleapis.com/mat2vec/"
FILES = (
    "pretrained_embeddings",
    "pretrained_embeddings.wv.vectors.npy",
    "pretrained_embeddings.trainables.syn1neg.npy",
)


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def download(name, cache):
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=".mat2vec-", dir=cache)
    os.close(fd)
    tmp = Path(tmpname)
    try:
        req = Request(SOURCE + name, headers={"User-Agent": "RudraNet-Mat2Vec-integrity/1"})
        with urlopen(req, timeout=120) as remote, tmp.open("wb") as dst:
            metadata = {
                "last_modified": remote.headers.get("Last-Modified"),
                "gcs_generation": remote.headers.get("x-goog-generation"),
            }
            while chunk := remote.read(1024 * 1024):
                dst.write(chunk)
        digest, size = sha256_file(tmp)
        if not size:
            raise ValueError(f"Empty downloaded file: {name}")
        tmp.replace(cache / name)
        return {"sha256": digest, "size_bytes": size, **metadata}
    finally:
        tmp.unlink(missing_ok=True)


def capture(cache, manifest):
    entries = {}
    for name in FILES:
        entries[name] = download(name, cache)
        print(f"{name}: {entries[name]['sha256']} ({entries[name]['size_bytes']} bytes)", flush=True)
    result = {
        "source": SOURCE,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "captured-from-current-upstream-not-historically-verified",
        "files": entries,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=".mat2vec-manifest-", dir=manifest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True)
            out.write("\n")
        Path(tmpname).replace(manifest)
    finally:
        Path(tmpname).unlink(missing_ok=True)
    print(f"Wrote manifest: {manifest}")


def verify(cache, manifest):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("source") != SOURCE or set(data.get("files", {})) != set(FILES):
        raise ValueError("Manifest has an unexpected source or file list")
    for name in FILES:
        row = data["files"][name]
        expected = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Missing or invalid pinned SHA-256 for {name}")
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(f"Not cached: {path}; run capture in an online environment")
        actual, size = sha256_file(path)
        if actual != expected or size != row["size_bytes"]:
            raise ValueError(f"SHA-256/size mismatch for {name}; refusing to use unverified bytes")
        print(f"VERIFIED {name}: {actual} ({size} bytes)")
    print("All three Mat2Vec files matched the manifest. This script did not deserialize them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("--cache", type=Path, default=Path("mat2vec_cache"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("mat2vec_sha256.json"))
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.cache, args.manifest)
    else:
        verify(args.cache, args.manifest)


if __name__ == "__main__":
    main()
