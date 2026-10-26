"""Independently rescore archived predictions without changing their artifact.

By default, use the three ground-truth fixtures committed in upstream Matbench.
The upstream test_task.py records test_outputs as predictions when constructing
these fixtures. Their exact hashes and official fold IDs are checked here.
This is an offline label source, not a new model experiment.

For the other tasks, pass --dataset-dir containing the original Matminer
TASK.json.gz downloads. Each file must match its official SHA256 before labels
are read. This script neither downloads data nor accepts unchecked labels.
"""

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import statistics
import sys
from pathlib import Path

from record_predictions import align_predictions
from validate_results import ARCHIVED_SHA256, DEFAULT_RESULTS

UPSTREAM_COMMIT = "4b7d0714849a917814115be380a64e7ace94be55"
SCORER_SHA256 = "f791263c675c45b6938c78d856676a6821ac6879526bc9aca7d82523303c3414"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = {
    "matbench_dielectric": (
        "1d1dbc4c8fcbaed9d6b646cbacae2996" "762a814b39e065b3b6546e8991588760"
    ),
    "matbench_glass": (
        "4d00ea9e9bdb1e0a81178ade08e1d689" "46925b2c99ba858959f2003d1f30e2ca"
    ),
    "matbench_steels": (
        "f0bb429c99e267eb1b1925a6866381e5" "1d6c802475b90f5bac347ba1e0afbed4"
    ),
}
# Original compressed dataset identities published by matminer, independent of
# the AlchemBERT input archives. Never substitute the latter as official labels.
DATASET_SHA256 = {
    "matbench_dielectric": (
        "83befa09bc2ec2f4b6143afc41315782" "7a90e5e2e42c1eb507ccfa01bf26a1d6"
    ),
    "matbench_expt_gap": (
        "783e7d1461eb83b00b2f2942da4b95fd" "a5e58a0d1ae26b581c24cf8a82ca75b2"
    ),
    "matbench_expt_is_metal": (
        "8f2a4f9bacdcbc5c2c73615629ee7986" "f09d39bed40ba7db52b61b2889730887"
    ),
    "matbench_glass": (
        "36beb654e2a463ee2a6572105bea0ca2" "961eee7c7b26a25377bff2c3b338e53a"
    ),
    "matbench_jdft2d": (
        "26057dc4524e193e32abffb296ce819b" "58b6e11d1278cae329a2f97817a4eddf"
    ),
    "matbench_log_gvrh": (
        "098af941f4c663270f1fe21abf20ffad" "6fb85ecbfcba5786ceac03983ac29da7"
    ),
    "matbench_log_kvrh": (
        "44b113ddb7e23aa18731a62c74afa7e5" "aa654199e0db5f951c8248a00955c9cd"
    ),
    "matbench_mp_e_form": (
        "dedcb1d4ba2e3e50dbdd45ba5bc647a0" "0e9c2bcf8f8bf556dc8e92caa39eb21f"
    ),
    "matbench_mp_gap": (
        "58b65746bd88329986ed66031a2ac136" "9c7c522f7bc9f9081528e07097c2c057"
    ),
    "matbench_mp_is_metal": (
        "9a028ed5750a4c76ca36e9f3c8d48fe0" "bf3fb21b76ec2289e58ae7048d527919"
    ),
    "matbench_perovskites": (
        "4641e2417f8ec8b50096d2230864468d" "fa08278dc9d257c327f65d0305278483"
    ),
    "matbench_phonons": (
        "4db551f21ec5f577e6202725f10e34df" "c509aa7df3a6bdaac497da7f6dbbb9b3"
    ),
    "matbench_steels": (
        "473bc4957b2ea5e6465aef84bc29bb48" "ac34db27d69ea4ec5f508745c6fae252"
    ),
}
SPEC_SHA256 = {
    "matbench_v0.1_validation.json": (
        "94c17bc3dac32c67d33d69289927a5fe" "c174497e74bbc47293e23d3d147e9a6d"
    ),
    "matbench_v0.1_dataset_metadata.json": (
        "bd8d2962406a1829bda984c38c36b4f3" "ce6acc20e46d34a42d3ba5294a1df787"
    ),
}
EXPT_GAP_DERIVATIVE_SHA256 = (
    "b36bf5bfe3782c2a2372422c64b8a86fd1d30709559a6dc0c30e4c8ac5ab3e96"
)
FIGSHARE_FILE_IDS = {
    "matbench_dielectric": 17476067,
    "matbench_glass": 17494637,
    "matbench_jdft2d": 17476046,
    "matbench_log_gvrh": 17476064,
    "matbench_log_kvrh": 17476061,
    "matbench_mp_e_form": 17476028,
    "matbench_mp_gap": 17084741,
    "matbench_mp_is_metal": 17476034,
    "matbench_perovskites": 17494805,
    "matbench_phonons": 17476037,
    "matbench_steels": 17476040,
}


def read_verified_json(path, expected_sha256, compressed=False):
    """Check identity before parsing; reject HTML/error pages and wrong files."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {digest}")
    return json.loads(gzip.decompress(data) if compressed else data)


def fixture_labels(task_name, validation, root=REPO_ROOT):
    """Recover ID-keyed truth from the pinned upstream ground-truth fixture."""
    relative = f"matbench/tests/msonability_{task_name}.json"
    digest = FIXTURES[task_name]
    fixture = read_verified_json(root / relative, digest)
    if fixture["dataset_name"] != task_name or set(fixture["results"]) != set(
        validation
    ):
        raise ValueError("Ground-truth fixture task/folds mismatch")
    labels = {}
    for fold_name, split in validation.items():
        values = fixture["results"][fold_name]["data"]
        if set(values) != set(split["test"]) or set(labels).intersection(values):
            raise ValueError("Ground-truth fixture has incorrect/overlapping IDs")
        labels.update(values)
    return labels, {
        "kind": "upstream_ground_truth_fixture",
        "path": relative,
        "sha256": digest,
        "url": (
            "https://github.com/materialsproject/matbench/blob/"
            f"{UPSTREAM_COMMIT}/{relative}"
        ),
        "construction_evidence": (
            "matbench/tests/test_task.py::TestMatbenchTask.test_msonability "
            "records predictions=test_outputs and identifies these fixtures"
        ),
    }


def dataset_labels(task_name, metadata, dataset_dir):
    """Read labels from a verified raw Matminer split-orient JSON dataframe."""
    import ijson

    filename = f"{task_name}.json.gz"
    digest = DATASET_SHA256[task_name]
    source_path = dataset_dir / filename
    derivative = dataset_dir / "expt_gap_kingsbury.json.gz"
    if (
        task_name == "matbench_expt_gap"
        and not source_path.is_file()
        and derivative.is_file()
    ):
        return expt_gap_derivative_labels(derivative, metadata)
    blob = source_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != digest:
        if task_name == "matbench_expt_gap" and derivative.is_file():
            return expt_gap_derivative_labels(derivative, metadata)
        raise ValueError(f"SHA256 mismatch for {filename}")
    count = metadata["n_samples"]
    indices, columns, targets = [], [], []

    # Read just the short header before data, then use the C-backed row builder.
    # Passing every structure parser event through Python is unnecessarily slow.
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == "index.item":
                indices.append(value)
            elif prefix == "columns.item":
                columns.append(value)
            elif prefix == "data" and event == "start_array":
                break
    if indices != list(range(count)) or len(columns) != len(set(columns)):
        raise ValueError(f"{task_name}: unexpected dataset index or columns")
    column = columns.index(metadata["target"])

    # Large structure datasets inflate to gigabytes. Build one row at a time
    # and retain only scalar targets, never all deserialized structures.
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        for row in ijson.items(stream, "data.item", use_float=True):
            if len(row) != len(columns):
                raise ValueError("Dataset has missing columns or malformed rows")
            targets.append(row[column])
    if len(targets) != count:
        raise ValueError(f"{task_name}: unexpected official dataset index")
    width = len(str(count))
    prefix = task_name.replace("matbench", "mb").replace("_", "-")
    labels = {
        f"{prefix}-{i + 1:0{width}d}": value for i, value in enumerate(targets)
    }
    return labels, {
        "kind": "official_matminer_dataset",
        "filename": filename,
        "sha256": digest,
        "url": f"https://ml.materialsproject.org/projects/{filename}",
        "verified_public_mirror": (
            f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_IDS[task_name]}"
            if task_name in FIGSHARE_FILE_IDS
            else None
        ),
    }


def expt_gap_derivative_labels(path, metadata):
    """Use Matminer's explicitly identical target table with added MP IDs.

    The authoritative dataset description states that the materials are
    identical to matbench_expt_gap, with MP IDs added and columns renamed.
    Its compressed hash differs, so report it as a derivative, never as a
    byte-identical copy of the original benchmark dataset.
    """
    raw = read_verified_json(path, EXPT_GAP_DERIVATIVE_SHA256, compressed=True)
    if raw["index"] != list(range(metadata["n_samples"])):
        raise ValueError("Unexpected experimental-gap derivative indices")
    columns = raw["columns"]
    if columns != ["likely_mpid", "expt_gap", "formula"]:
        raise ValueError("Unexpected experimental-gap derivative columns")
    if len(raw["data"]) != metadata["n_samples"] or any(
        len(row) != len(columns) for row in raw["data"]
    ):
        raise ValueError("Unexpected experimental-gap derivative rows")
    labels = {
        f"mb-expt-gap-{i + 1:04d}": row[1] for i, row in enumerate(raw["data"])
    }
    return labels, {
        "kind": "official_matminer_derivative",
        "filename": path.name,
        "sha256": EXPT_GAP_DERIVATIVE_SHA256,
        "url": "https://ndownloader.figshare.com/files/28232598",
        "identity_evidence_url": (
            "https://github.com/hackingmaterials/matminer/blob/"
            "be0f096b6b4f59374a9ae00b8fd84321fc3dccbc/"
            "src/matminer/datasets/dataset_metadata.json"
        ),
        "identity_evidence": (
            "Matminer describes expt_gap_kingsbury as identical to "
            "matbench_expt_gap, with likely MP IDs added and columns renamed. "
            "Use expt_gap as the target in the unchanged integer index order."
        ),
        "original_benchmark_archive_sha256": DATASET_SHA256["matbench_expt_gap"],
    }


def rescore_task(task, labels, validation, task_type):
    """Align by official ID and recompute raw-probability classification scores."""
    from matbench.data_ops import score_array

    if set(task["results"]) != set(validation):
        raise ValueError("Archived results must contain all official folds")
    official_ids = [
        sample for split in validation.values() for sample in split["test"]
    ]
    if len(official_ids) != len(set(official_ids)) or set(labels) != set(
        official_ids
    ):
        raise ValueError("Truth IDs must match the official dataset exactly")
    classification = task_type == "classification"
    for label in labels.values():
        if not isinstance(label, (bool, int, float)) or not math.isfinite(label):
            raise ValueError("Non-finite or non-numeric truth label")
        if classification and label not in (False, True):
            raise ValueError("Classification truth must be binary")
    folds = {}
    for fold_name, split in validation.items():
        archived = task["results"][fold_name]
        ids = split["test"]
        predictions = align_predictions(archived["data"], ids, classification)
        truth = [labels[sample] for sample in ids]
        scores = score_array(truth, predictions, task_type)
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("Non-finite recomputed metric")
        stored = archived["scores"]
        if set(stored) != set(scores):
            raise ValueError("Archived and current metric sets differ")
        folds[fold_name] = {
            "n_samples": len(ids),
            "stored_scores": stored,
            "recomputed_scores": scores,
            "recomputed_minus_stored": {
                metric: scores[metric] - stored[metric] for metric in scores
            },
        }
    return {
        "task_type": task_type,
        "n_samples": len(labels),
        "folds": folds,
        "means": {
            metric: {
                "stored": statistics.mean(
                    f["stored_scores"][metric] for f in folds.values()
                ),
                "recomputed": statistics.mean(
                    f["recomputed_scores"][metric] for f in folds.values()
                ),
            }
            for metric in next(iter(folds.values()))["stored_scores"]
        },
    }


def build_report(results=DEFAULT_RESULTS, tasks=None, dataset_dir=None):
    """Return a separate report; never invoke record() or modify predictions."""
    from matbench import data_ops
    from matbench.constants import VERSION

    scorer_sha256 = hashlib.sha256(Path(data_ops.__file__).read_bytes()).hexdigest()
    if scorer_sha256 != SCORER_SHA256:
        raise ValueError(
            "Install the documented Matbench checkout: scorer SHA256 mismatch"
        )
    source = read_verified_json(results, ARCHIVED_SHA256, compressed=True)
    specs = {
        name: read_verified_json(REPO_ROOT / "matbench" / name, digest)
        for name, digest in SPEC_SHA256.items()
    }
    validation = specs["matbench_v0.1_validation.json"]["splits"]
    metadata = specs["matbench_v0.1_dataset_metadata.json"]
    selected = sorted(
        tasks if tasks is not None else (DATASET_SHA256 if dataset_dir else FIXTURES)
    )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one task without duplicates")
    report_tasks = {}
    for name in selected:
        if name not in source["tasks"] or name not in DATASET_SHA256:
            raise ValueError(f"Unknown task: {name}")
        if dataset_dir is None:
            if name not in FIXTURES:
                raise ValueError(
                    f"No official fixture for {name}; supply --dataset-dir"
                )
            labels, label_source = fixture_labels(name, validation[name])
        else:
            labels, label_source = dataset_labels(
                name, metadata[name], Path(dataset_dir)
            )
        task_report = rescore_task(
            source["tasks"][name],
            labels,
            validation[name],
            metadata[name]["task_type"],
        )
        task_report["label_source"] = label_source
        report_tasks[name] = task_report
    return {
        "artifact_sha256": ARCHIVED_SHA256,
        "upstream_commit": UPSTREAM_COMMIT,
        "task_count": len(report_tasks),
        "fold_count": sum(len(task["folds"]) for task in report_tasks.values()),
        "scope": (
            "Re-score stored ID-keyed predictions; no retraining or proof "
            "of prediction provenance"
        ),
        "prediction_alignment": (
            "Exact official fold-ID sets; values looked up by sample ID"
        ),
        "tasks": report_tasks,
        "not_rescored": sorted(set(source["tasks"]) - set(selected)),
        "not_rescored_data_requirements": {
            name: {
                "filename": f"{name}.json.gz",
                "sha256": DATASET_SHA256[name],
                "url": (
                    "https://ml.materialsproject.org/projects/" f"{name}.json.gz"
                ),
            }
            for name in sorted(set(source["tasks"]) - set(selected))
        },
        "software": {
            "matbench": VERSION,
            "numpy": importlib.metadata.version("numpy"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "ijson": importlib.metadata.version("ijson") if dataset_dir else None,
            "data_ops_sha256": scorer_sha256,
            "rescore_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "spec_sha256": SPEC_SHA256,
    }


def write_report(report, output):
    """Refuse to replace any existing file, including the historical artifact."""
    output = Path(output)
    if output.name == "results.json.gz" or output.suffix != ".json":
        raise ValueError(
            "Choose a new .json report, not a benchmark results archive"
        )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        report = build_report(args.results, args.tasks, args.dataset_dir)
    if args.output:
        write_report(report, args.output)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
