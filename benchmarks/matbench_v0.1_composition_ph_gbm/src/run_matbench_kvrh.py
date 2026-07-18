# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "persim",
#   "pyarrow",
#   "pymatgen",
#   "ripser",
#   "scikit-learn",
# ]
# ///

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import platform
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from pymatgen.core import Structure
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import max_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
OUT = HERE / "run_outputs" / "matbench_kvrh"
DATASET_PATH = HERE / "data" / "matbench_log_kvrh.json.gz"
PH_CACHE_PATH = OUT / "ph_features.parquet"
PH_CONFIG_PATH = OUT / "ph_featurizer_config.json"

DATASET_URL = "https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz"
VALIDATION_URL = (
    "https://raw.githubusercontent.com/materialsproject/matbench/main/"
    "matbench/matbench_v0.1_validation.json"
)
LEADERBOARD_URL = (
    "https://matbench.materialsproject.org/Leaderboards%20Per-Task/"
    "matbench_v0.1_matbench_log_kvrh/"
)

DATASET_NAME = "matbench_log_kvrh"
BENCHMARK_NAME = "matbench_v0.1"
MATBENCH_VERSION = "0.6"
INPUT_COLUMN = "structure"
TARGET_COLUMN = "log10(K_VRH)"
EXPECTED_ROWS = 10_987
REG_METRICS = ("mae", "rmse", "max_error")
PRIMARY_METRIC = "mae"
FOLD_DIST_METRICS = ("mean", "max", "min", "std")
SEED = 42

FOLDS_PATH = OUT / "official_validation_matbench_log_kvrh.json"
LEADERBOARD_HTML_PATH = OUT / "leaderboard_matbench_log_kvrh.html"
LEADERBOARD_JSON_PATH = OUT / "leaderboard_context.json"
RESULTS_PATH = OUT / "results.json.gz"
SCORES_CSV_PATH = OUT / "fold_scores.csv"
SCORES_JSON_PATH = OUT / "fold_scores.json"
SUMMARY_JSON_PATH = OUT / "model_summaries.json"
RUN_METADATA_PATH = OUT / "run_metadata.json"


sys.path.insert(0, str(HERE))
from composition_features_v2 import FEATURE_NAMES as COMP_FEATURE_NAMES  # noqa: E402
from composition_features_v2 import feature_vector as composition_feature_vector  # noqa: E402
from ph_features import compute_cache  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mbids(n_rows: int) -> list[str]:
    width = math.floor(math.log(n_rows, 10)) + 1
    return [f"mb-log-kvrh-{i + 1:0{width}d}" for i in range(n_rows)]


def load_dataset() -> pd.DataFrame:
    with gzip.open(DATASET_PATH, "rt") as handle:
        raw = json.load(handle)
    df = pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])
    if list(df.columns) != [INPUT_COLUMN, TARGET_COLUMN]:
        raise RuntimeError(f"unexpected dataset columns: {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected dataset row count: {len(df)}")
    df.index = mbids(len(df))
    df.index.name = "mbid"
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)
    if not np.all(np.isfinite(df[TARGET_COLUMN].to_numpy())):
        raise RuntimeError("target contains non-finite values")
    return df


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "codex-matbench-kvrh-local/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())


def ensure_dataset() -> None:
    if not DATASET_PATH.exists():
        download(DATASET_URL, DATASET_PATH)


def ensure_ph_cache(df: pd.DataFrame) -> None:
    if PH_CACHE_PATH.exists() and PH_CONFIG_PATH.exists():
        return
    config = compute_cache(
        df.index.tolist(),
        df[INPUT_COLUMN].tolist(),
        dataset_path=DATASET_PATH,
        cache_path=PH_CACHE_PATH,
        config_path=PH_CONFIG_PATH,
        r_cut=6.0,
        max_points=1_000,
        hist_bins=8,
    )
    config["dataset"] = DATASET_NAME
    config["source_module"] = "ph_features.py"
    config["wrapper_module"] = "run_matbench_kvrh.py"
    PH_CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def ensure_official_folds(df: pd.DataFrame) -> dict[str, Any]:
    if FOLDS_PATH.exists():
        fold_doc = json.loads(FOLDS_PATH.read_text())
    else:
        with urllib.request.urlopen(VALIDATION_URL, timeout=180) as response:
            validation = json.loads(response.read().decode("utf-8"))
        fold_doc = {
            "source_url": VALIDATION_URL,
            "metadata": validation["metadata"],
            "splits": validation["splits"][DATASET_NAME],
        }
        FOLDS_PATH.write_text(json.dumps(fold_doc, indent=2, sort_keys=True) + "\n")

    folds = fold_doc["splits"]
    expected_keys = [f"fold_{i}" for i in range(5)]
    if sorted(folds) != expected_keys:
        raise RuntimeError(f"unexpected fold keys: {sorted(folds)}")
    all_ids = set(df.index)
    for fold_key in expected_keys:
        train_ids = folds[fold_key]["train"]
        test_ids = folds[fold_key]["test"]
        if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
            raise RuntimeError(f"{fold_key} contains duplicate ids")
        if set(train_ids).intersection(test_ids):
            raise RuntimeError(f"{fold_key} train/test overlap")
        if set(train_ids).union(test_ids) != all_ids:
            raise RuntimeError(f"{fold_key} does not cover the dataset exactly")
    return fold_doc


def composition_matrix(structures: pd.Series) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    formulas: list[str] = []
    for structure_dict in structures:
        structure = Structure.from_dict(structure_dict)
        formula = structure.composition.reduced_formula
        formulas.append(formula)
        rows.append(composition_feature_vector(formula))
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(structures), len(COMP_FEATURE_NAMES)):
        raise RuntimeError(f"unexpected composition feature shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("non-finite composition features")
    return matrix, formulas


def load_ph_matrix(ids: pd.Index) -> tuple[np.ndarray, dict[str, Any]]:
    if not PH_CACHE_PATH.exists() or not PH_CONFIG_PATH.exists():
        raise RuntimeError("persistent-homology cache/config is missing")
    config = json.loads(PH_CONFIG_PATH.read_text())
    if config["completed_rows"] != len(ids):
        raise RuntimeError(f"incomplete persistent-homology cache: {config['completed_rows']}")
    if config["dataset_sha256"] != sha256_file(DATASET_PATH):
        raise RuntimeError("persistent-homology cache dataset hash mismatch")
    if config["cache_sha256"] != sha256_file(PH_CACHE_PATH):
        raise RuntimeError("persistent-homology cache file hash mismatch")
    frame = pd.read_parquet(PH_CACHE_PATH)
    if frame["mbid"].tolist() != ids.tolist():
        raise RuntimeError("persistent-homology cache ids do not match dataset ids")
    if frame.columns[1:].tolist() != config["feature_names"]:
        raise RuntimeError("persistent-homology cache columns do not match config")
    matrix = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("non-finite persistent-homology features")
    return matrix, config


def regression_scores(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "max_error": float(max_error(y_true, predictions)),
    }


def inner_select_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_inner, x_val, y_inner, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=SEED,
    )
    grid = [
        {"max_iter": max_iter, "max_leaf_nodes": max_leaf_nodes, "learning_rate": 0.1}
        for max_iter in (300, 600)
        for max_leaf_nodes in (31, 63)
    ]
    evaluations: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in grid:
        model = HistGradientBoostingRegressor(**config, random_state=SEED)
        model.fit(x_inner, y_inner)
        predictions = model.predict(x_val).astype(float)
        scores = regression_scores(y_val, predictions)
        evaluations.append({**config, **{f"inner_{key}": value for key, value in scores.items()}})
        if scores[PRIMARY_METRIC] < best_mae:
            best_mae = scores[PRIMARY_METRIC]
            best_config = config
    if best_config is None:
        raise RuntimeError("no HistGradientBoostingRegressor config selected")
    return best_config, evaluations


def fit_predict(
    track: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    best_config, evaluations = inner_select_hgb(x_train, y_train)
    model = HistGradientBoostingRegressor(**best_config, random_state=SEED)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).astype(float)
    return predictions, {
        "track": track,
        "regressor": "HistGradientBoostingRegressor",
        "selected_by": "inner_90_10_train_only_mae",
        "random_state": SEED,
        "best_config": best_config,
        "inner_evaluations": evaluations,
    }


def summarize_scores(fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for track in ("B_comp", "B_ph", "B_comp+ph"):
        records = [record for record in fold_records if record["track"] == track]
        summary[track] = {}
        for metric in REG_METRICS:
            values = np.asarray([record[metric] for record in records], dtype=float)
            summary[track][metric] = {
                operation: float(getattr(np, operation)(values))
                for operation in FOLD_DIST_METRICS
            }

    comp = {record["fold"]: record[PRIMARY_METRIC] for record in fold_records if record["track"] == "B_comp"}
    for track in ("B_ph", "B_comp+ph"):
        deltas = [
            record[PRIMARY_METRIC] - comp[record["fold"]]
            for record in fold_records
            if record["track"] == track
        ]
        summary[f"{track}_minus_B_comp_mae"] = {
            "per_fold": {f"fold_{i}": float(deltas[i]) for i in range(5)},
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "interpretation": "negative is lower MAE and therefore an improvement over B_comp",
        }
    return summary


def immutify_dictionary(d: dict[str, Any]) -> dict[str, Any]:
    d_new: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d_new[key] = tuple(value.tolist())
        elif isinstance(value, pd.Series):
            d_new[key] = tuple(value.tolist())
        elif isinstance(value, list):
            d_new[key] = tuple(value)
        elif isinstance(value, dict):
            d_new[key] = immutify_dictionary(value)
        elif hasattr(value, "dtype"):
            d_new[key] = value.item()
        else:
            d_new[key] = value
    return dict(sorted(d_new.items(), key=lambda item: item[0]))


def hash_dictionary(d: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(immutify_dictionary(d)).encode("utf-8")).hexdigest()


def build_results_artifact(
    *,
    best_track: str,
    best_fold_outputs: dict[str, dict[str, Any]],
    fold_doc: dict[str, Any],
    dataset_info: dict[str, Any],
    ph_config: dict[str, Any],
) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for fold_key in [f"fold_{i}" for i in range(5)]:
        output = best_fold_outputs[fold_key]
        task_results[fold_key] = {
            "data": output["data"],
            "parameters": output["parameters"],
            "scores": output["scores"],
            "uncertainty": None,
        }
    user_metadata = immutify_dictionary(
        {
            "algorithm": f"{best_track} + HistGradientBoostingRegressor",
            "package_path": "fallback_constructed_to_matbench_schema",
            "matbench_package_version": MATBENCH_VERSION,
            "official_primary_metric": PRIMARY_METRIC,
            "official_regression_metric_names": list(REG_METRICS),
            "feature_tracks_evaluated": ["B_comp", "B_ph", "B_comp+ph"],
            "composition_feature_module": "composition_features_v2.py",
            "composition_n_features": len(COMP_FEATURE_NAMES),
            "persistent_homology_library": ph_config["library"],
            "persistent_homology_n_features": ph_config["n_features"],
            "persistent_homology_config_sha256": sha256_file(PH_CONFIG_PATH),
            "dataset": dataset_info,
            "validation_source_url": fold_doc["source_url"],
            "validation_metadata": fold_doc["metadata"],
            "random_state": SEED,
        }
    )
    benchmark = {
        "@module": "matbench.bench",
        "@class": "MatbenchBenchmark",
        "version": MATBENCH_VERSION,
        "tasks": {
            DATASET_NAME: {
                "@module": "matbench.task",
                "@class": "MatbenchTask",
                "benchmark_name": BENCHMARK_NAME,
                "dataset_name": DATASET_NAME,
                "results": task_results,
            }
        },
        "user_metadata": user_metadata,
        "benchmark_name": BENCHMARK_NAME,
        "datestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d %H:%M.%S"),
    }
    benchmark["hash"] = hash_dictionary(benchmark)
    return benchmark


def validate_artifact(artifact: dict[str, Any], folds: dict[str, Any]) -> None:
    artifact_without_hash = dict(artifact)
    recorded_hash = artifact_without_hash.pop("hash")
    if recorded_hash != hash_dictionary(artifact_without_hash):
        raise RuntimeError("artifact hash mismatch")
    results = artifact["tasks"][DATASET_NAME]["results"]
    if sorted(results) != [f"fold_{i}" for i in range(5)]:
        raise RuntimeError("artifact fold keys are invalid")
    for fold_key, split in folds.items():
        fold = results[fold_key]
        if set(fold) != {"data", "parameters", "scores", "uncertainty"}:
            raise RuntimeError(f"{fold_key} artifact fields are invalid")
        if list(fold["data"]) != split["test"]:
            raise RuntimeError(f"{fold_key} prediction ids/order do not match official test ids")
        if set(fold["scores"]) != set(REG_METRICS):
            raise RuntimeError(f"{fold_key} official regression metric keys are invalid")
        if not all(isinstance(value, float) and math.isfinite(value) for value in fold["data"].values()):
            raise RuntimeError(f"{fold_key} contains invalid predictions")


class SimpleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._table: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag in {"th", "td"}:
            self._in_cell = True
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            text = " ".join(data.split())
            if text:
                self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if self._in_cell and tag in {"th", "td"}:
            self._row.append(" ".join(self._cell).strip())
            self._in_cell = False
        elif self._in_row and tag == "tr":
            if self._row:
                self._table.append(self._row)
            self._in_row = False
        elif self._in_table and tag == "table":
            self.tables.append(self._table)
            self._in_table = False


def ensure_leaderboard_context() -> dict[str, Any]:
    download(LEADERBOARD_URL, LEADERBOARD_HTML_PATH)
    parser = SimpleTableParser()
    parser.feed(LEADERBOARD_HTML_PATH.read_text(errors="replace"))
    expected_header = ["algorithm", "mean mae", "std mae", "mean rmse", "max max_error"]
    for table in parser.tables:
        if table and table[0] == expected_header:
            rows = [row for row in table[1:] if len(row) == len(expected_header)]
            if not rows:
                break
            context = {
                "source_url": LEADERBOARD_URL,
                "saved_html_path": str(LEADERBOARD_HTML_PATH.relative_to(HERE)),
                "header_verbatim": expected_header,
                "entries_verbatim": [dict(zip(expected_header, row)) for row in rows],
            }
            LEADERBOARD_JSON_PATH.write_text(json.dumps(context, indent=2) + "\n")
            return context
    raise RuntimeError("could not parse full matbench_log_kvrh leaderboard table")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_dataset()
    df = load_dataset()
    ensure_ph_cache(df)
    fold_doc = ensure_official_folds(df)
    folds = fold_doc["splits"]

    print("loading composition features from reduced formulas", flush=True)
    x_comp, formulas = composition_matrix(df[INPUT_COLUMN])
    x_ph, ph_config = load_ph_matrix(df.index)
    x_by_track = {
        "B_comp": x_comp,
        "B_ph": x_ph,
        "B_comp+ph": np.concatenate([x_comp, x_ph], axis=1),
    }
    y_all = df[TARGET_COLUMN].to_numpy(dtype=float)
    id_to_pos = {mbid: pos for pos, mbid in enumerate(df.index)}

    fold_records: list[dict[str, Any]] = []
    fold_outputs_by_track: dict[str, dict[str, dict[str, Any]]] = {
        track: {} for track in x_by_track
    }
    for fold_key in [f"fold_{i}" for i in range(5)]:
        split = folds[fold_key]
        train_pos = np.asarray([id_to_pos[mbid] for mbid in split["train"]], dtype=int)
        test_pos = np.asarray([id_to_pos[mbid] for mbid in split["test"]], dtype=int)
        for track, x_all in x_by_track.items():
            model_started = time.perf_counter()
            predictions, parameters = fit_predict(
                track,
                x_all[train_pos],
                y_all[train_pos],
                x_all[test_pos],
            )
            scores = regression_scores(y_all[test_pos], predictions)
            model_seconds = time.perf_counter() - model_started
            fold_records.append(
                {
                    "track": track,
                    "fold": fold_key,
                    "n_train": int(len(train_pos)),
                    "n_test": int(len(test_pos)),
                    **scores,
                    "model_seconds": model_seconds,
                    "parameters": parameters,
                }
            )
            fold_outputs_by_track[track][fold_key] = {
                "data": {
                    mbid: float(prediction)
                    for mbid, prediction in zip(split["test"], predictions, strict=True)
                },
                "scores": scores,
                "parameters": parameters,
            }
            print(
                f"{fold_key} {track}: mae={scores['mae']:.6f} "
                f"seconds={model_seconds:.1f}",
                flush=True,
            )

    summary = summarize_scores(fold_records)
    best_track = min(
        ("B_comp", "B_ph", "B_comp+ph"),
        key=lambda track: summary[track][PRIMARY_METRIC]["mean"],
    )
    dataset_info = {
        "url": DATASET_URL,
        "local_path": str(DATASET_PATH.relative_to(HERE)),
        "size_bytes": DATASET_PATH.stat().st_size,
        "sha256": sha256_file(DATASET_PATH),
        "rows": len(df),
        "input_column": INPUT_COLUMN,
        "target_column": TARGET_COLUMN,
    }
    artifact = build_results_artifact(
        best_track=best_track,
        best_fold_outputs=fold_outputs_by_track[best_track],
        fold_doc=fold_doc,
        dataset_info=dataset_info,
        ph_config=ph_config,
    )
    validate_artifact(artifact, folds)

    pd.DataFrame(fold_records).drop(columns=["parameters"]).to_csv(SCORES_CSV_PATH, index=False)
    SCORES_JSON_PATH.write_text(json.dumps(fold_records, indent=2, sort_keys=True) + "\n")
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with gzip.open(RESULTS_PATH, "wt") as handle:
        json.dump(artifact, handle)
    with gzip.open(RESULTS_PATH, "rt") as handle:
        on_disk_artifact = json.load(handle)
    validate_artifact(on_disk_artifact, folds)
    leaderboard_context = ensure_leaderboard_context()

    run_metadata = {
        "dataset": dataset_info,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "total_model_workflow_seconds": time.perf_counter() - started,
        },
        "official_folds": {
            "source_url": fold_doc["source_url"],
            "metadata": fold_doc["metadata"],
            "n_folds": len(folds),
            "sizes": {
                fold_key: {
                    "train": len(split["train"]),
                    "test": len(split["test"]),
                }
                for fold_key, split in folds.items()
            },
        },
        "feature_tracks": {
            "B_comp": {
                "module": "composition_features_v2.py",
                "n_features": len(COMP_FEATURE_NAMES),
                "source": "structure reduced formula",
            },
            "B_ph": {
                "config_path": str(PH_CONFIG_PATH.relative_to(HERE)),
                "n_features": ph_config["n_features"],
                "source": "structure only",
            },
            "B_comp+ph": {"n_features": x_by_track["B_comp+ph"].shape[1]},
        },
        "reduced_formula_count": len(formulas),
        "random_state": SEED,
        "model_selection": "inner 90/10 split inside each official train fold; fixed HistGB grid; MAE",
        "best_track": best_track,
        "best_mean_mae": summary[best_track][PRIMARY_METRIC]["mean"],
        "results_path": str(RESULTS_PATH.relative_to(HERE)),
        "results_sha256": sha256_file(RESULTS_PATH),
        "artifact_hash": artifact["hash"],
        "artifact_validation": "passed after gzip reload: hash, official fold ids/order, fields, regression metric keys, finite predictions",
        "leaderboard": {
            "source_url": LEADERBOARD_URL,
            "entry_count": len(leaderboard_context["entries_verbatim"]),
            "context_path": str(LEADERBOARD_JSON_PATH.relative_to(HERE)),
        },
    }
    RUN_METADATA_PATH.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "run_metadata": run_metadata}, indent=2))


if __name__ == "__main__":
    main()
