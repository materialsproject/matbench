"""Reject incompatible runtimes, changed inputs and reused fit caches."""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def verify(output='fresh_seed417'):
    if platform.python_implementation() != 'CPython' or sys.version_info[:3] != (3,13,16):
        raise RuntimeError('Exact replay requires CPython3.13.16.')
    for name, version in json.loads((ROOT/'environment_lock.json').read_text()).items():
        if md.version(name) != version:
            raise RuntimeError(f'Package version mismatch: {name} expected {version}')
    for name, digest in json.loads((ROOT/'manifest.json').read_text()).items():
        p = ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Entry integrity mismatch: {name}')
    stage = (ROOT/output).resolve()
    if stage.parent != ROOT or stage.exists():
        raise RuntimeError('Output must be a new direct child directory.')
    for parent in (ROOT, ROOT/'inputs'):
        if any(p.is_dir() for p in parent.glob('seed_*')):
            raise RuntimeError('Preexisting input/reference fit cache is forbidden.')
    package = Path(importlib.util.find_spec('matbench').origin).parent
    for row in json.loads((ROOT/'upstream_source_hashes.json').read_text()):
        p=package/row['name']
        if hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f'Matbench source mismatch: {row["name"]}')
    return stage
