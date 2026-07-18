# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
OUT = HERE / "run_outputs" / "matbench_glass"

DATASET_URL = "https://ml.materialsproject.org/projects/matbench_glass.json.gz"
VALIDATION_URL = (
    "https://raw.githubusercontent.com/materialsproject/matbench/main/"
    "matbench/matbench_v0.1_validation.json"
)
LEADERBOARD_URL = (
    "https://matbench.materialsproject.org/Leaderboards%20Per-Task/"
    "matbench_v0.1_matbench_glass/"
)

DATASET_NAME = "matbench_glass"
BENCHMARK_NAME = "matbench_v0.1"
MATBENCH_VERSION = "0.6"
INPUT_COLUMN = "composition"
TARGET_COLUMN = "gfa"
TASK_TYPE = "classification"
CLF_METRICS = ("accuracy", "balanced_accuracy", "f1", "rocauc")
FOLD_DIST_METRICS = ("mean", "max", "min", "std")
CLF_THRESH = 0.5
SEED = 42
SIZE_GUARD_BYTES = 20 * 1024 * 1024

DATASET_PATH = OUT / "matbench_glass.json.gz"
FOLDS_PATH = OUT / "official_validation_matbench_glass.json"
LEADERBOARD_HTML_PATH = OUT / "leaderboard_matbench_glass.html"
LEADERBOARD_JSON_PATH = OUT / "leaderboard_context.json"
RESULTS_PATH = OUT / "results.json.gz"
SCORES_CSV_PATH = OUT / "fold_scores.csv"
SCORES_JSON_PATH = OUT / "fold_scores.json"
SUMMARY_JSON_PATH = OUT / "model_summaries.json"
RUN_METADATA_PATH = OUT / "run_metadata.json"


sys.path.insert(0, str(HERE))
from composition_features_v2 import FEATURE_NAMES, feature_vector  # noqa: E402


def url_content_length(url: str) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "codex-matbench-glass-local/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "codex-matbench-glass-local/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def ensure_dataset() -> dict[str, Any]:
    try:
        size = url_content_length(DATASET_URL)
    except Exception as exc:
        if not DATASET_PATH.exists() or DATASET_PATH.stat().st_size >= SIZE_GUARD_BYTES:
            raise RuntimeError(f"could not determine dataset size for {DATASET_URL}") from exc
        size = DATASET_PATH.stat().st_size
    if size >= SIZE_GUARD_BYTES:
        raise RuntimeError(f"dataset size guard failed: {size} >= {SIZE_GUARD_BYTES}")
    if not DATASET_PATH.exists() or DATASET_PATH.stat().st_size != size:
        download(DATASET_URL, DATASET_PATH)
    return {
        "url": DATASET_URL,
        "content_length_bytes": size,
        "local_path": str(DATASET_PATH.relative_to(HERE)),
        "local_size_bytes": DATASET_PATH.stat().st_size,
        "size_guard_bytes": SIZE_GUARD_BYTES,
        "size_guard_passed": DATASET_PATH.stat().st_size < SIZE_GUARD_BYTES,
    }


def matbench_index(n_rows: int) -> list[str]:
    n_zeros = math.floor(math.log(n_rows, 10)) + 1
    return [f"mb-glass-{i + 1:0{n_zeros}d}" for i in range(n_rows)]


def load_dataset() -> pd.DataFrame:
    with gzip.open(DATASET_PATH, "rt") as handle:
        raw = json.load(handle)
    df = pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])
    if list(df.columns) != [INPUT_COLUMN, TARGET_COLUMN]:
        raise RuntimeError(f"unexpected dataset columns: {list(df.columns)}")
    df.index = matbench_index(len(df))
    df.index.name = "mbid"
    if len(df) != 5680:
        raise RuntimeError(f"unexpected row count: {len(df)}")
    if df[TARGET_COLUMN].dtype != bool:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(bool)
    return df


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
    expected_fold_keys = [f"fold_{i}" for i in range(5)]
    if sorted(folds) != expected_fold_keys:
        raise RuntimeError(f"unexpected fold keys: {sorted(folds)}")
    all_ids = set(df.index)
    for fold_key in expected_fold_keys:
        split = folds[fold_key]
        train_ids = split["train"]
        test_ids = split["test"]
        if len(set(train_ids).intersection(test_ids)) != 0:
            raise RuntimeError(f"{fold_key} train/test overlap")
        if set(train_ids).union(test_ids) != all_ids:
            raise RuntimeError(f"{fold_key} does not cover the full dataset")
    return fold_doc


def feature_matrix(formulas: pd.Series) -> np.ndarray:
    return np.array([feature_vector(f) for f in formulas.astype(str)], dtype=float)


def homogenize_clf_array(
    array: list[bool] | list[float] | np.ndarray,
    *,
    to_probs: bool = False,
    to_labels: bool = False,
    thresh: float = CLF_THRESH,
) -> list[bool] | list[float]:
    if sum([to_probs, to_labels]) != 1:
        raise ValueError("set exactly one of to_probs or to_labels")
    values = list(array)
    if to_probs:
        if all(isinstance(i, (bool, np.bool_)) for i in values):
            return [1.0 if bool(i) else 0.0 for i in values]
        raise TypeError("cannot convert non-bool classification values to probs")
    if all(isinstance(i, (float, np.floating)) for i in values):
        return (np.asarray(values) > thresh).tolist()
    raise TypeError("cannot convert non-float classification values to labels")


def matbench_score_array(
    true_array: list[bool] | np.ndarray,
    pred_array: list[bool] | list[float] | np.ndarray,
) -> dict[str, float]:
    # This intentionally mirrors matbench.data_ops.score_array for
    # classification. Because accuracy is computed first, float predictions are
    # thresholded before the later rocauc call.
    y_true: list[bool] | list[float] = list(true_array)
    y_pred: list[bool] | list[float] = list(pred_array)
    computed: dict[str, float] = {}
    for metric in CLF_METRICS:
        if metric == "rocauc":
            if isinstance(y_pred[0], (float, np.floating)):
                y_true = homogenize_clf_array(y_true, to_probs=True)  # type: ignore[arg-type]
        elif metric in CLF_METRICS:
            if isinstance(y_pred[0], (float, np.floating)):
                y_pred = homogenize_clf_array(y_pred, to_labels=True)  # type: ignore[assignment]

        if metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            value = f1_score(y_true, y_pred)
        elif metric == "rocauc":
            value = roc_auc_score(y_true, y_pred)
        else:
            raise AssertionError(metric)
        computed[metric] = float(value)
    return computed


def inner_select_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_inner, x_val, y_inner, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=SEED,
        stratify=y_train,
    )
    grid = [
        {"max_iter": max_iter, "max_leaf_nodes": max_leaf_nodes, "learning_rate": 0.1}
        for max_iter in (300, 600)
        for max_leaf_nodes in (31, 63)
    ]
    evaluations: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_score = -float("inf")
    for config in grid:
        model = HistGradientBoostingClassifier(
            **config,
            random_state=SEED,
        )
        model.fit(x_inner, y_inner)
        prob = model.predict_proba(x_val)[:, 1].astype(float).tolist()
        scores = matbench_score_array(y_val.tolist(), prob)
        evaluation = {**config, **{f"inner_{k}": v for k, v in scores.items()}}
        evaluations.append(evaluation)
        if scores["rocauc"] > best_score:
            best_score = scores["rocauc"]
            best_config = config
    if best_config is None:
        raise RuntimeError("no HGB config selected")
    return best_config, evaluations


def fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[list[float], dict[str, Any]]:
    if model_name == "B0_majority_class":
        model = DummyClassifier(strategy="most_frequent", random_state=SEED)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1].astype(float).tolist()
        params = {
            "model": model_name,
            "strategy": "most_frequent",
            "train_positive_fraction": float(np.mean(y_train)),
            "classes": [bool(c) for c in model.classes_.tolist()],
            "random_state": SEED,
        }
        return prob, params

    if model_name == "B1_logistic_regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=SEED),
        )
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1].astype(float).tolist()
        params = {
            "model": model_name,
            "scaler": "StandardScaler",
            "classifier": "LogisticRegression",
            "max_iter": 5000,
            "random_state": SEED,
        }
        return prob, params

    if model_name == "B2_hist_gradient_boosting":
        best_config, inner_evaluations = inner_select_hgb(x_train, y_train)
        model = HistGradientBoostingClassifier(**best_config, random_state=SEED)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1].astype(float).tolist()
        params = {
            "model": model_name,
            "classifier": "HistGradientBoostingClassifier",
            "selected_by": "inner_90_10_train_only_official_rocauc",
            "random_state": SEED,
            "best_config": best_config,
            "inner_evaluations": inner_evaluations,
        }
        return prob, params

    raise ValueError(f"unknown model: {model_name}")


def summarize_scores(fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_name in sorted({record["model"] for record in fold_records}):
        model_records = [r for r in fold_records if r["model"] == model_name]
        summary[model_name] = {}
        for metric in CLF_METRICS:
            values = np.array([r[metric] for r in model_records], dtype=float)
            summary[model_name][metric] = {
                op: float(getattr(np, op)(values)) for op in FOLD_DIST_METRICS
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
    d_hashable = immutify_dictionary(d)
    s_hashable = json.dumps(d_hashable).encode("utf-8")
    return hashlib.sha256(s_hashable).hexdigest()


def build_results_artifact(
    b2_fold_outputs: dict[str, dict[str, Any]],
    dataset_info: dict[str, Any],
    fold_doc: dict[str, Any],
) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for fold_key in [f"fold_{i}" for i in range(5)]:
        fold_output = b2_fold_outputs[fold_key]
        task_results[fold_key] = {
            "data": fold_output["data"],
            "parameters": fold_output["parameters"],
            "scores": fold_output["scores"],
            "uncertainty": None,
        }

    user_metadata = immutify_dictionary(
        {
            "algorithm": "composition_features_v2 + HistGradientBoostingClassifier",
            "package_path": "fallback_constructed_to_matbench_schema",
            "matbench_package_version": MATBENCH_VERSION,
            "matbench_package_failure": (
                "uv run --with matbench failed before import while building "
                "scikit-learn==1.0.1: ModuleNotFoundError: "
                "distutils.msvccompiler"
            ),
            "feature_module": "composition_features_v2.py",
            "n_features": len(FEATURE_NAMES),
            "dataset": dataset_info,
            "validation_source_url": fold_doc["source_url"],
            "validation_metadata": fold_doc["metadata"],
            "official_metric_names": list(CLF_METRICS),
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
        "datestamp": dt.datetime.utcnow().strftime("%Y.%m.%d %H:%M.%S"),
    }
    benchmark["hash"] = hash_dictionary(benchmark)
    return benchmark


def validate_artifact_dict(artifact: dict[str, Any], folds: dict[str, Any]) -> None:
    artifact_no_hash = dict(artifact)
    recorded_hash = artifact_no_hash.pop("hash")
    computed_hash = hash_dictionary(artifact_no_hash)
    if recorded_hash != computed_hash:
        raise RuntimeError(f"artifact hash mismatch: {recorded_hash} != {computed_hash}")
    task = artifact["tasks"][DATASET_NAME]
    results = task["results"]
    for fold_key, split in folds.items():
        fold = results[fold_key]
        for required in ("data", "parameters", "scores", "uncertainty"):
            if required not in fold:
                raise RuntimeError(f"{fold_key} missing {required}")
        if set(fold["data"]) != set(split["test"]):
            raise RuntimeError(f"{fold_key} prediction ids do not match official test ids")
        if set(fold["scores"]) != set(CLF_METRICS):
            raise RuntimeError(f"{fold_key} scores do not match official metrics")
        for value in fold["data"].values():
            if not isinstance(value, float) or value < 0.0 or value > 1.0:
                raise RuntimeError(f"{fold_key} invalid probability value: {value}")


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


def ensure_leaderboard_context() -> list[dict[str, Any]]:
    if not LEADERBOARD_HTML_PATH.exists():
        download(LEADERBOARD_URL, LEADERBOARD_HTML_PATH)
    parser = SimpleTableParser()
    parser.feed(LEADERBOARD_HTML_PATH.read_text(errors="replace"))
    leaderboard_rows: list[dict[str, Any]] = []
    for table in parser.tables:
        if not table:
            continue
        header = table[0]
        if header == ["algorithm", "mean rocauc", "std rocauc", "mean f1", "mean balanced_accuracy"]:
            for row in table[1:]:
                if len(row) != len(header):
                    continue
                entry = dict(zip(header, row))
                leaderboard_rows.append(
                    {
                        "algorithm": entry["algorithm"],
                        "mean rocauc": float(entry["mean rocauc"]),
                        "std rocauc": float(entry["std rocauc"]),
                        "mean f1": float(entry["mean f1"]),
                        "mean balanced_accuracy": float(entry["mean balanced_accuracy"]),
                        "source_url": LEADERBOARD_URL,
                    }
                )
            break
    if not leaderboard_rows:
        raise RuntimeError("could not parse leaderboard table")
    LEADERBOARD_JSON_PATH.write_text(
        json.dumps(leaderboard_rows, indent=2, sort_keys=True) + "\n"
    )
    return leaderboard_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_info = ensure_dataset()
    df = load_dataset()
    fold_doc = ensure_official_folds(df)
    folds = fold_doc["splits"]
    leaderboard_rows = ensure_leaderboard_context()

    x_all = feature_matrix(df[INPUT_COLUMN])
    y_all = df[TARGET_COLUMN].astype(bool).to_numpy()
    id_to_pos = {mbid: pos for pos, mbid in enumerate(df.index)}

    fold_records: list[dict[str, Any]] = []
    b2_fold_outputs: dict[str, dict[str, Any]] = {}
    models = [
        "B0_majority_class",
        "B1_logistic_regression",
        "B2_hist_gradient_boosting",
    ]

    for fold_key in [f"fold_{i}" for i in range(5)]:
        split = folds[fold_key]
        train_pos = np.array([id_to_pos[mbid] for mbid in split["train"]], dtype=int)
        test_pos = np.array([id_to_pos[mbid] for mbid in split["test"]], dtype=int)
        x_train = x_all[train_pos]
        y_train = y_all[train_pos]
        x_test = x_all[test_pos]
        y_test = y_all[test_pos]

        for model_name in models:
            predictions, params = fit_predict_model(model_name, x_train, y_train, x_test)
            scores = matbench_score_array(y_test.tolist(), predictions)
            record = {
                "model": model_name,
                "fold": fold_key,
                "n_train": int(len(train_pos)),
                "n_test": int(len(test_pos)),
                **scores,
                "parameters": params,
            }
            fold_records.append(record)
            if model_name == "B2_hist_gradient_boosting":
                b2_fold_outputs[fold_key] = {
                    "data": {
                        mbid: float(pred)
                        for mbid, pred in zip(split["test"], predictions, strict=True)
                    },
                    "scores": scores,
                    "parameters": params,
                }

    summary = summarize_scores(fold_records)
    artifact = build_results_artifact(b2_fold_outputs, dataset_info, fold_doc)
    validate_artifact_dict(artifact, folds)

    SCORES_CSV_PATH.write_text(
        pd.DataFrame(fold_records).drop(columns=["parameters"]).to_csv(index=False)
    )
    SCORES_JSON_PATH.write_text(json.dumps(fold_records, indent=2, sort_keys=True) + "\n")
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with gzip.open(RESULTS_PATH, "wt") as handle:
        json.dump(artifact, handle)

    run_metadata = {
        "dataset": dataset_info,
        "dataset_rows": int(len(df)),
        "target_counts": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()},
        "feature_module": "composition_features_v2.py",
        "n_features": len(FEATURE_NAMES),
        "official_metric_names": list(CLF_METRICS),
        "official_n_folds": len(folds),
        "fold_metadata": fold_doc["metadata"],
        "results_path": str(RESULTS_PATH.relative_to(HERE)),
        "scores_csv_path": str(SCORES_CSV_PATH.relative_to(HERE)),
        "scores_json_path": str(SCORES_JSON_PATH.relative_to(HERE)),
        "summary_json_path": str(SUMMARY_JSON_PATH.relative_to(HERE)),
        "leaderboard_source_url": LEADERBOARD_URL,
        "leaderboard_rows": leaderboard_rows,
        "artifact_validation": "fallback validator passed: hash, fold ids, metric names, probability bounds",
    }
    RUN_METADATA_PATH.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"summary": summary, "run_metadata": run_metadata}, indent=2))


if __name__ == "__main__":
    main()
