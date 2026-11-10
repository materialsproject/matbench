"""Recompute classification leaderboard scores without modifying submissions.

Historical result files remain provenance records. Current ROC-AUC rankings use
only finite, floating-point positive-class probabilities from every test fold.
"""

import numpy as np

from matbench.constants import CLF_KEY, CLF_METRICS, FOLD_DIST_METRICS
from matbench.data_ops import score_array
from matbench.util import RecursiveDotDict


def probability_eligibility(task):
    """Return eligibility and an exclusion reason before loading any dataset."""
    for fold_key in task.folds_map.values():
        fold = task.results.get(fold_key)
        if not fold or not fold.get("data"):
            return False, f"Missing or empty predictions in {fold_key}."
        values = list(fold["data"].values())
        # Do not coerce first: bools/integers otherwise masquerade as probabilities.
        if not all(isinstance(p, (float, np.floating)) for p in values):
            return False, "Every fold must contain floating-point probabilities; hard labels are excluded."
        probabilities = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(probabilities)):
            return False, "Probabilities must be finite."
        if np.any((probabilities < 0) | (probabilities > 1)):
            return False, "Probabilities must be in [0, 1]."
    if not task.folds_map:
        return False, "No test folds."
    return True, None


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


def recompute_classification_scores(task, targets_by_fold):
    """Align predictions by sample ID and average independently scored folds.

    Invalid prediction formats are excluded. Missing/extra sample IDs, invalid
    targets, or undefined ROC-AUC abort the build rather than publish a score.
    The original task results and cached scores are never modified.
    """
    eligible, reason = probability_eligibility(task)
    entry = dict(eligible=eligible, reason=reason, scores=None,
                 fold_scores=None, stored_scores=task.scores)
    if not eligible:
        return entry
    fold_scores = {}
    for fold_key in task.folds_map.values():
        predictions = task.results[fold_key]["data"]
        target = targets_by_fold[fold_key]
        if not target.index.is_unique or set(predictions) != set(target.index):
            raise ValueError(f"{task.dataset_name}/{fold_key}: prediction IDs do not match official test IDs")
        if target.isna().any() or set(target.unique()) != {False, True}:
            raise ValueError(f"{task.dataset_name}/{fold_key}: ROC-AUC requires both binary target classes")
        pred = [predictions[sample_id] for sample_id in target.index]
        fold_scores[fold_key] = score_array(target.to_numpy(), pred, CLF_KEY)
        if not all(np.isfinite(score) for score in fold_scores[fold_key].values()):
            raise ValueError(f"{task.dataset_name}/{fold_key}: non-finite classification score")
    scores = {
        metric: {
            op: float(getattr(np, op)([fold[metric] for fold in fold_scores.values()]))
            for op in FOLD_DIST_METRICS
        }
        for metric in CLF_METRICS
    }
    entry.update(scores=RecursiveDotDict(scores), fold_scores=fold_scores)
    return entry


def prepare_classification_scores(all_data):
    """Attach current scores to documentation packets; cache only target series.

    Test targets come from the canonical Matbench splits. Scores are the
    unweighted mean of fold ROC-AUCs, with population standard deviation,
    matching Matbench's existing aggregation convention (not pooled OOF AUC).
    """
    targets = {}
    for packet in all_data.values():
        packet["classification_scores"] = {}
        for task in packet["results"].tasks:
            if task.metadata.task_type != CLF_KEY:
                continue
            eligible, reason = probability_eligibility(task)
            if eligible:
                key = (task.benchmark_name, task.dataset_name)
                if key not in targets:
                    targets[key] = _load_fold_targets(task)
                entry = recompute_classification_scores(task, targets[key])
            else:
                entry = dict(eligible=False, reason=reason, scores=None,
                             fold_scores=None, stored_scores=task.scores)
            packet["classification_scores"][task.dataset_name] = entry
    return all_data


def classification_score_report(all_data, benchmarks_dir):
    """Produce an auditable, JSON-serializable record of the current rankings."""
    import hashlib
    from pathlib import Path
    from matbench.metadata import mbv01_metadata

    records = []
    for packet in all_data.values():
        artifact = Path(benchmarks_dir) / packet["dir_name_short"] / "results.json.gz"
        source_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        for name, entry in packet["classification_scores"].items():
            records.append({
                "submission": packet["dir_name_short"],
                "algorithm": packet["info"]["algorithm"],
                "task": name,
                "results_sha256": source_hash,
                "dataset_expected_sha256": mbv01_metadata[name]["hash"],
                **entry,
            })
    return {
        "protocol": "probability-rocauc-v1",
        "positive_class": True,
        "eligibility": "Every test prediction is a finite float in [0, 1]; boolean and integer labels are excluded.",
        "aggregation": "Unweighted mean of official test-fold scores; population standard deviation (ddof=0). Not pooled out-of-fold AUC.",
        "records": sorted(records, key=lambda row: (row["task"], row["submission"])),
    }
