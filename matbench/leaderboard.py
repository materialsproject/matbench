"""Recompute classification metrics without modifying original submissions.

Balanced accuracy is verified for saved decisions as well as probabilities.
Probability ROC AUC requires the original positive-class probabilities.
"""

from collections.abc import Mapping

import numpy as np

from matbench.constants import CLF_KEY, CLF_METRICS, FOLD_DIST_METRICS, MBV01_KEY
from matbench.data_ops import METRIC_MAP, score_array
from matbench.util import RecursiveDotDict

HARD_LABEL_AUC_REASON = (
    "No usable probabilities were saved; probability ROC AUC cannot be recovered "
    "from hard labels."
)


def _prediction_format(task):
    """Check complete folds and prediction types before loading target data.

    A task must consistently save probabilities, boolean decisions, or integer
    binary decisions. In particular, a malformed probability fold cannot fall
    back to the historical balanced-accuracy path.
    """
    fold_map = task.folds_map
    if not fold_map:
        return "invalid", "No test folds."
    if task.benchmark_name == MBV01_KEY and fold_map != {
        number: f"fold_{number}" for number in range(5)
    }:
        return "invalid", "All five official test folds are required."
    expected = list(fold_map.values())
    if len(expected) != len(set(expected)) or set(task.results) != set(expected):
        return "invalid", "Missing, duplicate, or extra test folds."
    kinds = set()
    for fold_key in expected:
        fold = task.results.get(fold_key)
        if not fold or not isinstance(fold.get("data"), Mapping) or not fold["data"]:
            return "invalid", f"Missing or empty predictions in {fold_key}."
        for value in fold["data"].values():
            # Inspect before coercion: boolean/integer decisions are not
            # probabilities.
            if isinstance(value, (bool, np.bool_)):
                kinds.add("boolean")
            elif isinstance(value, (int, np.integer)):
                if value not in (0, 1):
                    return "invalid", "Integer hard labels must be 0 or 1."
                kinds.add("integer")
            elif isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    return "invalid", "Probabilities must be finite."
                if not 0 <= value <= 1:
                    return "invalid", "Probabilities must be in [0, 1]."
                kinds.add("probabilities")
            else:
                return (
                    "invalid",
                    "Predictions must be numeric probabilities "
                    "or binary hard labels.",
                )
    if len(kinds) != 1:
        return (
            "invalid",
            "Mixed probability and hard-label types (including across folds) "
            "are not allowed.",
        )
    return ("probabilities" if kinds == {"probabilities"} else "hard_labels"), None


def probability_eligibility(task):
    """Return probability-format eligibility, without loading any dataset.

    Numerical validation cannot establish whether floats were originally model
    probabilities. Casting hard labels to floats remains prohibited by the
    submission policy; genuine probability endpoints 0.0 and 1.0 are valid.
    """
    prediction_type, reason = _prediction_format(task)
    return prediction_type == "probabilities", (
        HARD_LABEL_AUC_REASON if prediction_type == "hard_labels" else reason
    )


def _load_fold_targets(task):
    """Load official test targets once, preserving the caller's loaded state."""
    was_loaded = task.df is not None
    try:
        if not was_loaded:
            task.load()
        return {
            key: task.get_test_data(number, include_target=True)[1].copy()
            for number, key in task.folds_map.items()
        }
    finally:
        if not was_loaded:
            task.unload()


def _empty_entry(task, prediction_type, reason):
    # RecursiveDotDict creates entries on missing-key access, so never ask its
    # cached aggregation to read an incomplete result (that would mutate it).
    expected = (
        [f"fold_{number}" for number in range(5)]
        if task.benchmark_name == MBV01_KEY
        else list(task.folds_map.values())
    )
    complete_cache = all(
        isinstance(task.results.get(key), Mapping)
        and isinstance(task.results[key].get("scores"), Mapping)
        and all(metric in task.results[key]["scores"] for metric in CLF_METRICS)
        for key in expected
    )
    try:
        stored_scores = task.scores if complete_cache else None
    except (KeyError, TypeError, ValueError):
        # Incomplete cached results must not prevent reporting their exclusion.
        stored_scores = None
    auc_reason = (
        HARD_LABEL_AUC_REASON if prediction_type == "hard_labels" else reason
    )
    return dict(
        eligible=False,
        ba_available=False,
        auc_available=False,
        reason=auc_reason,
        ba_reason=reason,
        auc_reason=auc_reason,
        prediction_type=prediction_type,
        scores=None,
        fold_scores=None,
        stored_scores=stored_scores,
    )


def recompute_classification_scores(task, targets_by_fold):
    """Validate IDs and compute metrics separately on each official test fold.

    Invalid formats are excluded from both metrics. Missing/extra/duplicate IDs,
    invalid targets, or undefined scores abort rather than publish a score.
    Saved hard-label decisions are used as-is; their original thresholds cannot
    be recovered. Probabilities use p > 0.5 for decision-based metrics and are
    passed unthresholded to ROC AUC. Original task results are never modified.
    """
    prediction_type, reason = _prediction_format(task)
    entry = _empty_entry(task, prediction_type, reason)
    if prediction_type == "invalid":
        return entry
    if set(targets_by_fold) != set(task.folds_map.values()):
        raise ValueError(
            f"{task.dataset_name}: official target folds do not match test folds"
        )
    eligible = prediction_type == "probabilities"
    fold_scores = {}
    seen_ids = set()
    for fold_key in task.folds_map.values():
        predictions = task.results[fold_key]["data"]
        target = targets_by_fold[fold_key]
        if not target.index.is_unique or set(predictions) != set(target.index):
            raise ValueError(
                f"{task.dataset_name}/{fold_key}: prediction IDs "
                "do not match official test IDs"
            )
        if seen_ids.intersection(target.index):
            raise ValueError(
                f"{task.dataset_name}/{fold_key}: duplicate test IDs across folds"
            )
        seen_ids.update(target.index)
        if target.isna().any() or set(target.unique()) != {False, True}:
            raise ValueError(
                f"{task.dataset_name}/{fold_key}: classification "
                "requires both binary target classes"
            )
        pred = [predictions[sample_id] for sample_id in target.index]
        if eligible:
            fold_scores[fold_key] = score_array(target.to_numpy(), pred, CLF_KEY)
        else:
            decisions = np.asarray(pred, dtype=bool)
            fold_scores[fold_key] = {
                metric: (
                    float(METRIC_MAP[metric](target.to_numpy(), decisions))
                    if metric != "rocauc"
                    else None
                )
                for metric in CLF_METRICS
            }
        if not all(
            np.isfinite(score)
            for score in fold_scores[fold_key].values()
            if score is not None
        ):
            raise ValueError(
                f"{task.dataset_name}/{fold_key}: non-finite classification score"
            )
    scores = {
        metric: (
            {
                op: float(
                    getattr(np, op)([fold[metric] for fold in fold_scores.values()])
                )
                for op in FOLD_DIST_METRICS
            }
            if metric != "rocauc" or eligible
            else None
        )
        for metric in CLF_METRICS
    }
    entry.update(
        eligible=eligible,
        ba_available=True,
        auc_available=eligible,
        scores=RecursiveDotDict(scores),
        fold_scores=fold_scores,
    )
    return entry


def prepare_classification_scores(all_data):
    """Attach verified BA and probability ROC AUC, caching only target series.

    Historical hard-label submissions undergo the same official-ID and label
    checks as probability submissions. Aggregation is an unweighted mean of
    five fold scores and population standard deviation, never pooled OOF AUC.
    """
    targets = {}
    for packet in all_data.values():
        packet["classification_scores"] = {}
        for task in packet["results"].tasks:
            if task.metadata.task_type != CLF_KEY:
                continue
            prediction_type, reason = _prediction_format(task)
            if prediction_type != "invalid":
                key = (task.benchmark_name, task.dataset_name)
                if key not in targets:
                    targets[key] = _load_fold_targets(task)
                entry = recompute_classification_scores(task, targets[key])
            else:
                entry = _empty_entry(task, prediction_type, reason)
            packet["classification_scores"][task.dataset_name] = entry
    return all_data


def classification_score_report(all_data, benchmarks_dir):
    """Produce an auditable, JSON-serializable record of both metric statuses."""
    import hashlib
    from pathlib import Path

    from matbench.metadata import mbv01_metadata

    records = []
    for packet in all_data.values():
        artifact = (
            Path(benchmarks_dir) / packet["dir_name_short"] / "results.json.gz"
        )
        source_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        for name, entry in packet["classification_scores"].items():
            records.append(
                {
                    "submission": packet["dir_name_short"],
                    "algorithm": packet["info"]["algorithm"],
                    "task": name,
                    "results_file": artifact.relative_to(benchmarks_dir).as_posix(),
                    "results_sha256": source_hash,
                    "dataset_expected_sha256": mbv01_metadata[name]["hash"],
                    **entry,
                }
            )
    return {
        "protocol": "classification-ba-probability-rocauc-v2",
        "positive_class": True,
        "homepage_table_metrics": ["balanced_accuracy", "rocauc"],
        "homepage_scaled_error_metric": "balanced_accuracy",
        "homepage_scope": (
            "Existing general-purpose inclusion rules; each metric selects "
            "its own eligible winner(s)."
        ),
        "scaled_classification_error": "(1 - balanced_accuracy) / 0.5",
        "task_order": (
            "ROC AUC available, descending ROC AUC; then balanced accuracy only, "
            "descending BA; ties by algorithm name."
        ),
        "ba_eligibility": (
            "Complete official test folds and matching sample IDs with finite "
            "probabilities in [0, 1] or consistently typed binary hard labels."
        ),
        "eligibility": (
            "Probability ROC AUC requires finite floating-point original "
            "probabilities in [0, 1] in every official fold; boolean and integer "
            "labels are excluded. Numerical checks cannot detect a cast of hard "
            "labels to floats, which the submission policy prohibits."
        ),
        "decision_rule": (
            "Probabilities: p > 0.5. Historical hard labels: saved decisions; "
            "their original thresholds may be unknown."
        ),
        "aggregation": (
            "Unweighted mean of official test-fold scores; population standard "
            "deviation (ddof=0). Not pooled out-of-fold AUC."
        ),
        "records": sorted(records, key=lambda row: (row["task"], row["submission"])),
    }
