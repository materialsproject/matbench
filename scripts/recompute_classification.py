"""Audit probability ROC AUC without rewriting historical submission artifacts.

Run from the repository root:
    python -m scripts.recompute_classification --output classification_scores.json
"""

import argparse
import json
from pathlib import Path

from matbench.bench import MatbenchBenchmark
from matbench.leaderboard import classification_score_report, prepare_classification_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks-dir", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packets = {}
    for results in sorted(args.benchmarks_dir.glob("*/results.json.gz")):
        packets[results.parent.name] = {
            "results": MatbenchBenchmark.from_file(str(results)),
            "info": json.loads((results.parent / "info.json").read_text()),
            "dir_name_short": results.parent.name,
        }
    prepare_classification_scores(packets)
    report = classification_score_report(packets, args.benchmarks_dir)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    eligible = sum(row["eligible"] for row in report["records"])
    print(f"Wrote {eligible} eligible and {len(report['records']) - eligible} excluded classification records to {args.output}")


if __name__ == "__main__":
    main()
