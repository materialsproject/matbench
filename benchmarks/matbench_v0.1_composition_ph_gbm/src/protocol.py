"""Fixed-feature, training-only selection using the pinned official Matbench API."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from matbench import constants, data_ops
from matbench.data_ops import score_array
from monty.json import MontyEncoder
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from composition_features_v2 import FEATURE_NAMES, feature_vector


SDK_REVISION = "4b7d0714849a917814115be380a64e7ace94be55"
EXPECTED_VALIDATION_SHA256 = (
    "94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d"
)
EXPECTED_SCORER_SHA256 = (
    "f791263c675c45b6938c78d856676a6821ac6879526bc9aca7d82523303c3414"
)
SEED = 42
GRID = tuple(
    {"max_iter": iterations, "max_leaf_nodes": leaves, "learning_rate": 0.1}
    for iterations in (300, 600)
    for leaves in (31, 63)
)
PH_SETTINGS = {"r_cut": 6.0, "max_points": 1000, "hist_bins": 8}
HERE = Path(__file__).resolve().parent
EXPOSURE = (
    "The earlier submission inspected outer-fold scores for alternative feature/model "
    "tracks. This revision freezes the declared feature track before refitting; it "
    "does not turn previously observed benchmark folds into an unseen holdout. "
    "No historical preregistration evidence is available."
)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    return digest_bytes(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, cls=MontyEncoder) + "\n"
    )


def source_hashes():
    return {path.name: file_digest(path) for path in sorted(HERE.glob("*.py"))}


def sdk_record():
    if (
        file_digest(constants.MBV01_VALIDATION_DATA_PATH)
        != EXPECTED_VALIDATION_SHA256
        or file_digest(data_ops.__file__) != EXPECTED_SCORER_SHA256
    ):
        raise ValueError(
            "Install the pinned Matbench Git revision from requirements.txt"
        )
    return {
        "git_revision": SDK_REVISION,
        "version": constants.VERSION,
        "validation_sha256": file_digest(constants.MBV01_VALIDATION_DATA_PATH),
        "scorer_sha256": file_digest(data_ops.__file__),
    }


def hardware_record(threads, workers):
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "model_threads": threads,
        "feature_workers": workers,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "matbench",
                "matminer",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "pymatgen",
                "ripser",
                "persim",
                "pyarrow",
                "threadpoolctl",
            )
        },
    }


def _ph_row(structure_dict):
    from pymatgen.core import Structure
    from ph_features import featurize_structure

    structure = Structure.from_dict(structure_dict)
    ph, details = featurize_structure(structure, **PH_SETTINGS)
    comp = feature_vector(structure.composition.reduced_formula)
    return comp + list(ph.values()), details


def feature_matrix(inputs, classification, cache_dir, workers=1):
    """Cache only deterministic, label-free features, keyed by inputs and code."""
    started = time.perf_counter()
    raw = (
        inputs.astype(str).tolist()
        if classification
        else [s.as_dict() for s in inputs]
    )
    identity = {
        "inputs": raw,
        "classification": classification,
        "settings": None if classification else PH_SETTINGS,
        "source": source_hashes(),
        "versions": {
            name: version(name) for name in ("numpy", "pymatgen", "ripser")
        },
    }
    key = digest_bytes(
        json.dumps(identity, sort_keys=True, cls=MontyEncoder).encode()
    )
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / (key + ".npz")
    metadata = root / (key + ".json")
    if cache.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        # The cache key covers the complete ordered inputs, code and versions.
        # GVRH/KVRH have the same structures but different official ID prefixes;
        # bind those validated rows to the requesting task's IDs below.
        if info["cache_key"] != key:
            raise ValueError("Feature cache input/code key mismatch")
        if file_digest(cache) != info["cache_sha256"]:
            raise ValueError("Feature cache checksum mismatch")
        with np.load(cache, allow_pickle=False) as stored:
            matrix = stored["features"].copy()
        info = {**info, "cache_hit": True}
    else:
        if classification:
            matrix = np.asarray(
                [feature_vector(formula) for formula in raw], dtype=float
            )
            details = []
        else:
            if workers == 1:
                rows = map(_ph_row, raw)
                executor = None
            else:
                executor = ProcessPoolExecutor(max_workers=workers)
                rows = executor.map(_ph_row, raw, chunksize=16)
            values, details = [], []
            try:
                for number, (features, detail) in enumerate(rows, 1):
                    values.append(features)
                    details.append(detail)
                    if number % 500 == 0:
                        print(f"PH features {number}/{len(raw)}", flush=True)
            finally:
                if executor is not None:
                    executor.shutdown()
            matrix = np.asarray(values, dtype=float)
        np.savez_compressed(
            cache, features=matrix, ids=inputs.index.to_numpy(dtype=str)
        )
        info = {
            "input_sha256": digest_bytes(
                json.dumps(raw, sort_keys=True, cls=MontyEncoder).encode()
            ),
            "cache_key": key,
            "cache_sha256": file_digest(cache),
            "rows": len(inputs),
            "features": int(matrix.shape[1]),
            "feature_track": (
                "B2_composition_HistGB" if classification else "B_comp+ph"
            ),
            "featurization_seconds": time.perf_counter() - started,
            "expansion_fallback_count": sum(
                int(d["expansion_fallback_used"]) for d in details
            ),
            "hard_cap_count": sum(int(d["hard_cap_used"]) for d in details),
            "cache_hit": False,
        }
        write_json(metadata, info)
    expected_features = len(FEATURE_NAMES) + (0 if classification else 51)
    if (
        matrix.shape != (len(inputs), expected_features)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid feature matrix")
    return pd.DataFrame(matrix, index=inputs.index), info


def select_and_predict(
    x_train, y_train, x_test, classification, threads=2, grid=GRID
):
    """No outer-test targets or scores enter model selection."""
    inner_x, val_x, inner_y, val_y = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=SEED,
        stratify=y_train if classification else None,
    )
    model_type = (
        HistGradientBoostingClassifier
        if classification
        else HistGradientBoostingRegressor
    )
    task_type, primary = (
        ("classification", "rocauc") if classification else ("regression", "mae")
    )
    evaluations = []
    with threadpool_limits(limits=threads):
        for config in grid:
            model = model_type(**config, random_state=SEED)
            model.fit(inner_x, inner_y)
            predictions = (
                model.predict_proba(val_x)[:, 1]
                if classification
                else model.predict(val_x)
            )
            score = float(
                score_array(val_y.tolist(), predictions.tolist(), task_type)[primary]
            )
            evaluations.append({"config": config, "inner_score": score})
        chosen = (max if classification else min)(
            evaluations, key=lambda item: item["inner_score"]
        )
        model = model_type(**chosen["config"], random_state=SEED)
        model.fit(x_train, y_train)
        predictions = (
            model.predict_proba(x_test)[:, 1]
            if classification
            else model.predict(x_test)
        )
    return predictions.tolist(), {
        "feature_track": "B2_composition_HistGB" if classification else "B_comp+ph",
        "feature_selection": "fixed; no outer-score-based track selection",
        "model": model_type.__name__,
        "random_state": SEED,
        "hyperparameter_selection": "90/10 split of official training fold only",
        "inner_metric": primary,
        "best_config": chosen["config"],
        "inner_evaluations": evaluations,
    }


def record_fold(task, matrix, fold, classification, threads=2, grid=GRID):
    train_inputs, train_targets = task.get_train_and_val_data(fold)
    test_inputs = task.get_test_data(fold, include_target=False)
    predictions, parameters = select_and_predict(
        matrix.loc[train_inputs.index],
        train_targets,
        matrix.loc[test_inputs.index],
        classification,
        threads,
        grid,
    )
    # Only recording/scoring has access to outer-test targets.
    task.record(fold, predictions, params=parameters)


def run_task(name, work_dir, workers=1, threads=2):
    started = time.perf_counter()
    sdk_record()
    benchmark = MatbenchBenchmark(subset=[name], autoload=True)
    task = benchmark.tasks_map[name]
    classification = task.metadata.task_type == "classification"
    inputs = task.df[task.metadata.input_type]
    matrix, features = feature_matrix(
        inputs, classification, Path(work_dir) / "features", workers
    )
    fold_times = {}
    for fold in task.folds:
        fold_started = time.perf_counter()
        record_fold(task, matrix, fold, classification, threads)
        fold_times[str(fold)] = time.perf_counter() - fold_started
        print(
            f"{name} fold {fold}: recorded ({fold_times[str(fold)]:.1f}s)",
            flush=True,
        )
    benchmark.add_metadata(
        {
            "sdk": sdk_record(),
            "source_sha256": source_hashes(),
            "protocol": "fixed features; training-only hyperparameter selection",
            "historical_test_exposure": EXPOSURE,
            "dataset": {
                "name": name,
                "url": task.metadata.url,
                "sha256": task.metadata.hash,
            },
            "features": features,
        }
    )
    output = Path(work_dir) / (name + ".json.gz")
    benchmark.to_file(str(output))
    checked = MatbenchBenchmark.from_file(str(output))
    if not checked.is_valid or not checked.is_recorded:
        raise ValueError("Official result round-trip validation failed")
    write_json(
        Path(work_dir) / (name + "_run.json"),
        {
            "task": name,
            "sdk": sdk_record(),
            "features": features,
            "hardware": hardware_record(threads, workers),
            "source_sha256": source_hashes(),
            "fold_seconds": fold_times,
            "elapsed_seconds": time.perf_counter() - started,
            "results_sha256": file_digest(output),
            "scores": dict(task.scores),
            "official_from_file_is_valid": checked.is_valid,
            "historical_test_exposure": EXPOSURE,
        },
    )
    return output


def validate_scores(path):
    """Recompute committed scores without fitting/selecting anything."""
    benchmark = MatbenchBenchmark.from_file(str(path))
    for task in benchmark.tasks:
        task.load()
        for fold in task.folds:
            _, truth = task.get_test_data(fold, include_target=True)
            result = task.results[task.folds_map[fold]]
            predictions = [result.data[identifier] for identifier in truth.index]
            actual = score_array(
                truth.tolist(), predictions, task.metadata.task_type
            )
            if set(actual) != set(result.scores):
                raise ValueError("Result is missing official score keys")
            for metric, score in actual.items():
                if not np.isclose(
                    score, result.scores[metric], rtol=1e-12, atol=1e-12
                ):
                    raise ValueError(
                        f"Stored {metric} does not match official scoring"
                    )
    return {"is_valid": benchmark.is_valid, "tasks": sorted(benchmark.tasks_map)}
