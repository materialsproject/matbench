"""Refit the fixed composition HistGB classifier using official folds and scoring."""

from pathlib import Path
import argparse
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from protocol import run_task, validate_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=HERE / "run_outputs")
    parser.add_argument("--output", type=Path, default=HERE / "results.json.gz")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.verify_only:
        print(validate_scores(args.output))
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = run_task(
        "matbench_glass", args.work_dir, workers=1, threads=args.threads
    )
    shutil.copyfile(result, args.output)
    print(validate_scores(args.output))


if __name__ == "__main__":
    main()
