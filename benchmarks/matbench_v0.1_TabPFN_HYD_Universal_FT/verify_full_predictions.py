"""Dispatch to the exact task-specific, SHA-pinned scientific snapshot."""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', choices=['phonons', 'dielectric', 'expt_gap'], required=True)
    args, rest = parser.parse_known_args()
    script = Path(__file__).resolve().parent / 'tasks' / args.task / 'verify_full_predictions.py'
    raise SystemExit(subprocess.call([sys.executable, str(script), *rest]))
