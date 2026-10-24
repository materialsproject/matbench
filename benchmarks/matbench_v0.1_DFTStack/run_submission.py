"""Reproduce the DFT-Stack Matbench v0.1 submission (results.json.gz) for the four composition tasks.

For each task and official fold, only the official training rows (inputs and targets) are used to fit:
feature blocks are label-free functions of the formula plus the disclosed DFT transfer block (Materials
Project PBE data from matminer's matbench_mp_gap / matbench_mp_e_form, matched by reduced formula);
for every candidate configuration (feature-block set; for matbench_expt_gap also the expert subset,
dftstack/frozen_config3.py) the experts are tuned and cross-fitted with 5 inner folds of the training fold,
the candidate with the best inner-CV score of the simplex stack on the training rows is chosen, and for it the
stack, the optional (nested-CV-gated) local correction and the classification threshold are all chosen inside the
training fold (dftstack/run_final3.py, dftstack/pipeline2.py). Test inputs are taken with include_target=False
and predictions go through task.record.

The modules in ./dftstack are verbatim copies of the frozen research code (see README.md).
Data (Matbench tasks, matminer DFT datasets) is downloaded on first use into ./data/cache/.

Usage:  python run_submission.py [matbench_steels ...] [--out results_rerun.json.gz]
CPU-only; compute time of the original run is given in info.json ("notes") and per fold in the recorded
parameters of results.json.gz (wall_time_s, fit_hosts, expert_fit_seconds, threads, CPU model). A rerun records the
same fields for its own hardware. results_rerun.json.gz is only this script's output name; it is not shipped.
"""
from __future__ import annotations
import gzip
import json
import os
import pathlib
import platform
import sys

import numpy as np
from matbench.bench import MatbenchBenchmark

from dftstack.data import TASKS, load_task_frames
from dftstack.run_final3 import select_and_predict, EXPERT_THREADS

HERE = pathlib.Path(__file__).resolve().parent
PIPELINE = "DFT-Stack (Amendment-3: fold-internal candidate selection, frozen Amendment-2 pipeline)"


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(tasks, out_file):
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
            p = z["stack_local"]
            if task.metadata.task_type == "classification":    # hard labels at the inner-CV balanced-accuracy threshold
                p = [bool(v) for v in (p >= meta["thresholds"]["stack_local"])]
            else:
                p = [float(v) for v in p]
            c = meta["compute"]["selection_job"]
            wall = float(c["wall_s"]) + sum(float(r["wall_s"]) for info in meta["compute"]["expert_fits"].values()
                                            for r in info["fit_info"]["runs"][:-1])
            params = {"pipeline": PIPELINE, "seed": 0, "wall_time_s": round(wall, 1), "n_workers": 1,
                      "threads_per_worker": EXPERT_THREADS, "cpu_threads_used": EXPERT_THREADS, "cpu_count": os.cpu_count(),
                      "cpu_model": cpu, "gpu": "none",
                      "expert_fit_seconds": {e: round(float(s), 1) for e, s in meta["expert_seconds"].items()},
                      "selected_candidate": meta["design_selection"]["selected"],
                      "candidate_inner_cv_scores": {r["candidate"]: round(float(r["stack_oof_score"]), 6)
                                                    for r in meta["design_selection"]["candidates"]}}
            task.record(fold, p, params=params)
            print(f"{name} fold {k}: recorded ({meta['design_selection']['selected']}, {wall:.0f} s)", flush=True)
    mb.add_metadata({"algorithm": "DFT-Stack"})
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
