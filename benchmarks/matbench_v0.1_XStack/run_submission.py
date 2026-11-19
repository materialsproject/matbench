"""Reproduce the XStack Matbench v0.1 submission (results.json.gz) for the four composition tasks.

Composition-only: the only inputs are label-free descriptors computed from each row's own formula (matminer /
CBFV). No external property data of any kind is used (no DFT block, no pretrained model).

For each task and official fold, only the official training rows (inputs and targets) are used to fit. For both
candidate feature-block sets (and, for matbench_expt_gap, the expert subsets; xstack/frozen_config4.py) the experts
are tuned and cross-fitted with 5 inner folds of the training fold; the candidate with the best inner-CV score of
the simplex stack on the training rows is chosen; and for it the stack, the optional (nested-CV-gated) local
correction and the classification threshold are all chosen inside the training fold (xstack/run_final4.py,
xstack/pipeline4.py). Missing descriptor cells are filled with column medians computed inside every fit from that
fit's training rows only, inner-CV fits included (xstack/experts4.py). Test inputs are taken with
include_target=False and predictions go through task.record. Classification predictions are recorded as raw
probabilities: ROC-AUC uses those probabilities and the current Matbench label metrics use p > 0.5. The threshold
chosen inside each training fold is retained as metadata, not applied to the recorded probabilities.

The modules in ./xstack are verbatim copies of the frozen research code (see README.md).
The Matbench tasks are downloaded on first use into ./data/cache/.

Usage:  python run_submission.py [matbench_steels ...] [--out results_rerun.json.gz]
CPU-only. The compute of the original run is recorded per fold in the parameters of results.json.gz; a rerun
records the same fields for its own hardware. results_rerun.json.gz is only this script's output name.
"""
from __future__ import annotations
import gzip
import json
import os
import pathlib
import sys

import numpy as np
from matbench.bench import MatbenchBenchmark
from matbench.data_ops import score_array

from xstack.data import TASKS, load_task_frames
from xstack.run_final3 import cpu_model, EXPERT_THREADS          # host description helpers only
from xstack.run_final4 import select_and_predict

HERE = pathlib.Path(__file__).resolve().parent
PIPELINE = "XStack (Amendment 4: composition-only, train-only imputation, fold-internal candidate selection)"


def require_probability_scoring():
    """Fail before training if Matbench uses the old thresholded-AUC scorer."""
    scores = score_array([False, False, True, True], [0.1, 0.4, 0.5, 0.6], "classification")
    if not (np.isclose(scores.get("rocauc", np.nan), 1.0)
            and np.isclose(scores.get("balanced_accuracy", np.nan), 0.75)):
        raise RuntimeError(
            "Install the Matbench source revision pinned in requirements.txt before running XStack. "
            "Scoring must use raw probabilities for ROC-AUC and p > 0.5 for balanced accuracy; "
            "the PyPI matbench 0.6 scorer is not supported."
        )


def main(tasks, out_file):
    require_probability_scoring()
    cpu = cpu_model()
    mb = MatbenchBenchmark(autoload=False, subset=tasks)
    for task in mb.tasks:
        name = task.dataset_name
        task.load()
        df, folds = load_task_frames(name)          # same official folds, cached as string ids
        out_root = HERE / "rerun_cache" / name
        out_root.mkdir(parents=True, exist_ok=True)
        for k, fold in enumerate(task.folds):
            tr_in, tr_y = task.get_train_and_val_data(fold)
            te_in = task.get_test_data(fold, include_target=False)
            tr = [str(i) for i in tr_in.index]
            te = [str(i) for i in te_in.index]
            assert tr == folds[k]["train"] and te == folds[k]["test"] and not set(tr) & set(te)
            assert np.allclose(np.asarray(tr_y, dtype=float), df.loc[tr, "target"].to_numpy(dtype=float))
            select_and_predict(name, k, df, folds, out_root, print)      # training rows only; resumable
            z = np.load(out_root / f"fold{k}_predictions.npz")
            meta = json.load(open(out_root / f"fold{k}_meta.json"))
            assert [str(i) for i in z["ids"]] == te
            p = np.asarray(z["stack_local"])
            if p.shape != (len(te),) or not np.isfinite(p).all():
                raise ValueError(f"{name} fold {k}: expected one finite prediction per test sample")
            if task.metadata.task_type == "classification":
                if p.dtype.kind != "f" or np.any((p < 0.0) | (p > 1.0)):
                    raise ValueError(f"{name} fold {k}: expected raw floating-point probabilities in [0, 1]")
            p = [float(v) for v in p]
            params = {"pipeline": PIPELINE, "seed": 0,
                      "wall_time_s": round(float(meta["compute"]["selection_job"]["wall_s"]), 1), "n_workers": 1,
                      "threads_per_worker": EXPERT_THREADS, "cpu_threads_used": EXPERT_THREADS, "cpu_count": os.cpu_count(),
                      "cpu_model": cpu, "gpu": "none",
                      "expert_fit_seconds": {e: round(float(s), 1) for e, s in meta["expert_seconds"].items()},
                      "selected_candidate": meta["design_selection"]["selected"],
                      "candidate_inner_cv_scores": {r["candidate"]: round(float(r["stack_oof_score"]), 6)
                                                    for r in meta["design_selection"]["candidates"]},
                      "imputation": meta["imputation"]}
            if task.metadata.task_type == "classification":
                params["classification_threshold"] = float(meta["thresholds"]["stack_local"])
                params["classification_threshold_operator"] = ">="
            task.record(fold, p, params=params)
            print(f"{name} fold {k}: recorded ({meta['design_selection']['selected']})", flush=True)
    mb.add_metadata({"algorithm": "XStack"})
    mb.to_file(str(out_file))
    raw = pathlib.Path(out_file).read_bytes()
    if raw[:2] != bytes([0x1F, 0x8B]):     # matbench 0.6 writes plain JSON; upstream from_file expects gzip
        with gzip.GzipFile(out_file, "wb", mtime=0) as f:
            f.write(raw)
    print("wrote", out_file, "is_valid:", mb.is_valid, "is_recorded:", mb.is_recorded)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = HERE / "results_rerun.json.gz"
    if "--out" in args:
        i = args.index("--out"); out = pathlib.Path(args[i + 1]); args = args[:i] + args[i + 2:]
    main(args or TASKS, out)
