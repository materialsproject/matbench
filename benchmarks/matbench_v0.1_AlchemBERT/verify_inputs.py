"""Audit composition v1 archive rows against identified official materials.

This read-only check verifies archive/dataset hashes before comparing every row
in all five train/test folds. It does not infer historical use of these files.
For the two expt tasks, the public Figshare originals predate a known formula
typo correction. Accepting those exact historical bytes requires the explicit
--allow-historical-expt option, and discrepancies remain in the report.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from rescore_results import DATASET_SHA256, REPO_ROOT, SPEC_SHA256

HERE = Path(__file__).resolve().parent
TASKS = (
    "matbench_expt_gap",
    "matbench_expt_is_metal",
    "matbench_glass",
    "matbench_steels",
)
HISTORICAL_EXPT = {
    "matbench_expt_gap": {
        "sha256": "1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a",
        "article": 9765779,
    },
    "matbench_expt_is_metal": {
        "sha256": "80fb0854cff5d4657d812793f4748d5c23a9407f6c59bcf5b86c818ebaa6910f",
        "article": 9765803,
    },
}
KINGSBURY_SHA256 = "b36bf5bfe3782c2a2372422c64b8a86fd1d30709559a6dc0c30e4c8ac5ab3e96"
INSTRUCTIONS = {
    "matbench_expt_gap": (
        "Please tell me experiment band gap (eV) of the following composition:"
    ),
    "matbench_expt_is_metal": (
        "Please tell me if it is a metal of the following composition:"
    ),
    "matbench_glass": (
        "Please tell me if it is a glass of the following composition:"
    ),
    "matbench_steels": (
        "Please tell me yield strength (MPa) of the following composition:"
    ),
}
PROPERTY_WORDS = re.compile(
    r"\b(?:band[- ]gap|yield strength|metallic|refractive|glass|target|"
    r"True|False|MPa|eV)\b",
    re.IGNORECASE,
)


def read_hashed(path, expected, compressed=False):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}")
    return json.loads(gzip.decompress(raw) if compressed else raw)


def formula_from_text(text):
    match = re.match(r"^The chemical formula (\S+) consists of:", text)
    if not match:
        raise ValueError("Unrecognized composition text prefix")
    return match.group(1)


def alloy_mismatches(text, formula):
    """Check every element percentage to its displayed rounding precision."""
    parts = re.findall(r"([A-Z][a-z]?)([0-9.]+)", formula)
    if "".join(element + amount for element, amount in parts) != formula:
        raise ValueError("Unrecognized official steels formula")
    expected = {element: float(amount) * 100 for element, amount in parts}
    if len(expected) != len(parts):
        raise ValueError("Duplicate element in official steels formula")
    main = re.findall(r"(\d+(?:\.\d+)?)% [a-z]+ \(([A-Z][a-z]?)\)", text)
    minor = re.findall(r"\(([A-Z][a-z]?)\) \((\d+(?:\.\d+)?)%\)", text)
    values = [(element, value) for value, element in main] + minor
    observed = dict(values)
    if len(values) != len(observed):
        raise ValueError("Duplicate element in steels description")
    differences = []
    if set(observed) != set(expected):
        differences.append(
            {
                "kind": "element_set",
                "missing": sorted(set(expected) - set(observed)),
                "extra": sorted(set(observed) - set(expected)),
            }
        )
    for element in sorted(set(observed) & set(expected)):
        value = observed[element]
        decimals = len(value.split(".")[1]) if "." in value else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-10
        if abs(float(value) - expected[element]) > tolerance:
            differences.append(
                {
                    "kind": "percentage",
                    "element": element,
                    "displayed": value,
                    "official_percent": expected[element],
                }
            )
    return differences


def output_matches(output, target):
    if not isinstance(output, str):
        return False
    if isinstance(target, bool):
        return output == str(target)
    try:
        value = float(output)
    except ValueError:
        return False
    return math.isfinite(value) and value == target


def kingsbury_comparison(dataset_dir, original):
    """Optionally compare the separately published official corrected table."""
    path = Path(dataset_dir) / "expt_gap_kingsbury.json.gz"
    if not path.exists():
        return None
    corrected = read_hashed(path, KINGSBURY_SHA256, compressed=True)
    if corrected["index"] != original["index"]:
        raise ValueError("Kingsbury derivative index differs from the reference")
    formula_column = corrected["columns"].index("formula")
    label_column = corrected["columns"].index("expt_gap")
    old_formula = original["columns"].index("composition")
    old_label = original["columns"].index("gap expt")
    differences = []
    for index, (old, new) in enumerate(zip(original["data"], corrected["data"])):
        if old[old_label] != new[label_column]:
            raise ValueError("Kingsbury derivative target labels differ")
        if old[old_formula] != new[formula_column]:
            differences.append(
                {
                    "row_index": index,
                    "historical_formula": old[old_formula],
                    "corrected_formula": new[formula_column],
                }
            )
    return {
        "filename": path.name,
        "sha256": KINGSBURY_SHA256,
        "url": "https://ndownloader.figshare.com/files/28232598",
        "scope": (
            "Independent official expt_gap_kingsbury derivative, described by "
            "Matminer as Matbench expt_gap with added MP IDs and renamed columns"
        ),
        "all_indices_and_labels_identical": True,
        "formula_changes": differences,
        "note": "Does not identify the current compressed Matbench archive bytes",
    }


def audit(source, dataset_dir, tasks=TASKS, allow_historical_expt=False):
    provenance = json.loads((HERE / "provenance.json").read_text())
    specs = {
        name: read_hashed(REPO_ROOT / "matbench" / name, expected)
        for name, expected in SPEC_SHA256.items()
    }
    splits = specs["matbench_v0.1_validation.json"]["splits"]
    metadata = specs["matbench_v0.1_dataset_metadata.json"]
    report = {
        "source_commit": provenance["source_commit"],
        "scope": (
            "Four composition tasks, all five train/test folds; no structure audit"
        ),
        "spec_sha256": SPEC_SHA256,
        "checks": {
            "identity": "Official ordered fold IDs and source dataset row index",
            "formula": "Exact formula prefix for expt_gap/expt_is_metal/glass",
            "alloy": (
                "Exact element set and percentages within "
                "displayed rounding precision"
            ),
            "labels": (
                "Exact archived output versus reference target; "
                "not passed to any model"
            ),
            "instruction": "Exact task-fixed instruction",
            "input_words": PROPERTY_WORDS.pattern,
        },
        "limitations": [
            "Archive row positions are tested against official fold order, "
            "not assumed correct.",
            "Formula-prefix/alloy checks do not validate every descriptive clause.",
            "Keyword absence is not proof of absence of semantic leakage "
            "or corpus overlap.",
            "Agreement does not establish these archives were used "
            "in the historical run.",
            "Historical expt dataset identity is separate from "
            "the current corrected dataset.",
        ],
        "tasks": {},
    }
    for name in tasks:
        if name not in TASKS:
            raise ValueError(f"Only composition tasks are supported: {name}")
        data_path = Path(dataset_dir) / f"{name}.json.gz"
        dataset_digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        reference_kind = "current_matminer_dataset"
        dataset_url = f"https://ml.materialsproject.org/projects/{name}.json.gz"
        if dataset_digest != DATASET_SHA256[name]:
            historical = HISTORICAL_EXPT.get(name, {})
            if not (
                allow_historical_expt and dataset_digest == historical.get("sha256")
            ):
                raise ValueError(f"Unexpected official dataset bytes: {name}")
            reference_kind = "historical_official_figshare_dataset_before_typo_fix"
            dataset_url = (
                f"https://api.figshare.com/v2/articles/{historical['article']}"
            )
        data = read_hashed(data_path, dataset_digest, compressed=True)
        expected_count = metadata[name]["n_samples"]
        if data["index"] != list(range(expected_count)):
            raise ValueError(f"Unexpected official row index: {name}")
        if len(data["data"]) != expected_count:
            raise ValueError(f"Unexpected official sample count: {name}")
        input_column = data["columns"].index("composition")
        target_column = data["columns"].index(metadata[name]["target"])
        archives = []
        failures = []
        totals = {
            "rows": 0,
            "input_mismatches": 0,
            "output_mismatches": 0,
            "instruction_mismatches": 0,
            "keyword_hits": 0,
        }
        for fold, split in splits[name].items():
            for kind, ids in split.items():
                filename = (
                    f"data_nl_v1/{kind.title()}_{fold.removeprefix('fold_')}_"
                    f"{name}_nl_{len(ids)}.json.gz"
                )
                archive_sha = provenance["data_nl_v1_sha256"][filename]
                rows = read_hashed(Path(source) / filename, archive_sha, True)
                if len(rows) != len(ids):
                    raise ValueError(f"Unexpected archive row count: {filename}")
                archives.append({"path": filename, "sha256": archive_sha})
                for sample_id, row in zip(ids, rows):
                    if set(row) != {"input", "instruction", "output"}:
                        raise ValueError(f"Unexpected archive fields: {filename}")
                    index = int(sample_id.rsplit("-", 1)[1]) - 1
                    material, target = (
                        data["data"][index][input_column],
                        data["data"][index][target_column],
                    )
                    totals["rows"] += 1
                    differences = {}
                    if name == "matbench_steels":
                        comparison = alloy_mismatches(row["input"], material)
                        if comparison:
                            differences["input"] = comparison
                    else:
                        observed = formula_from_text(row["input"])
                        if observed != material:
                            differences["input"] = {
                                "reference_formula": material,
                                "archive_formula": observed,
                            }
                    if "input" in differences:
                        totals["input_mismatches"] += 1
                    if not output_matches(row["output"], target):
                        totals["output_mismatches"] += 1
                        differences["output"] = {
                            "reference": target,
                            "archive": row["output"],
                        }
                    if row["instruction"] != INSTRUCTIONS[name]:
                        totals["instruction_mismatches"] += 1
                        differences["instruction"] = row["instruction"]
                    hits = PROPERTY_WORDS.findall(row["input"])
                    if hits:
                        totals["keyword_hits"] += 1
                        differences["keyword_hits"] = hits
                    if differences:
                        failures.append(
                            {
                                "sample_id": sample_id,
                                "fold": fold,
                                "partition": kind,
                                "differences": differences,
                            }
                        )
        report["tasks"][name] = {
            "reference": {
                "kind": reference_kind,
                "sha256": dataset_digest,
                "url": dataset_url,
                "current_expected_sha256": DATASET_SHA256[name],
            },
            "n_samples": expected_count,
            "archives": archives,
            "totals": totals,
            "mismatches": failures,
        }
        if name == "matbench_expt_gap":
            report["tasks"][name]["corrected_derivative_comparison"] = (
                kingsbury_comparison(dataset_dir, data)
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--allow-historical-expt", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.source, args.dataset_dir, args.tasks, args.allow_historical_expt
    )
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
