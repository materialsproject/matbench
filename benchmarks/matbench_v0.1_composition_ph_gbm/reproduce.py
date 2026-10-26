"""Refit both fixed-feature tasks, merge them, and validate with official Matbench."""

from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from matbench.bench import MatbenchBenchmark
from protocol import run_task, validate_scores

TASKS = ("matbench_log_gvrh", "matbench_log_kvrh")


def merge_results(paths, output):
    tasks, metadata = {}, {}
    for path in paths:
        benchmark = MatbenchBenchmark.from_file(str(path))
        if not benchmark.is_valid or not benchmark.is_recorded:
            raise ValueError("Cannot merge an invalid/incomplete task run")
        for name, task in benchmark.tasks_map.items():
            if name not in TASKS or name in tasks:
                raise ValueError("Unexpected or duplicate task in merge")
            tasks[name] = task
            metadata[name] = benchmark.user_metadata
    if set(tasks) != set(TASKS):
        raise ValueError("Both GVRH and KVRH task runs are required")
    merged = MatbenchBenchmark(subset=list(TASKS), autoload=False)
    for name, task in tasks.items():
        merged.tasks_map[name] = task
    merged.add_metadata({"task_runs": metadata, "feature_track": "B_comp+ph"})
    merged.to_file(str(output))
    checked = MatbenchBenchmark.from_file(str(output))
    if not checked.is_valid or not checked.is_recorded:
        raise ValueError("Merged result failed official validation")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=HERE / "run_outputs")
    parser.add_argument("--output", type=Path, default=HERE / "results.json.gz")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    if args.verify_only:
        print(validate_scores(args.output))
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in TASKS:
        path = args.work_dir / (name + ".json.gz")
        paths.append(
            path
            if args.merge_only
            else run_task(name, args.work_dir, args.workers, args.threads)
        )
    merge_results(paths, args.output)
    print(validate_scores(args.output))


if __name__ == "__main__":
    main()
