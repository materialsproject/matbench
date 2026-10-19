"""Reproduce the DFT-Stack Matbench v0.1 submission (results.json.gz) for the four composition tasks.

For each task and official fold, only the official training rows (inputs and targets) are used to fit:
feature blocks are label-free functions of the formula plus the disclosed DFT transfer block (Materials
Project PBE data from matminer's matbench_mp_gap / matbench_mp_e_form, matched by reduced formula);
experts are tuned and cross-fitted with 5 inner folds of the training fold; the simplex stack, the
optional (nested-CV-gated) local correction and the classification threshold are all chosen inside the
training fold. Test inputs are taken with include_target=False and predictions go through task.record.

The modules in ./dftstack are verbatim copies of the frozen research code (see README.md).
Data (Matbench tasks, matminer DFT datasets) is downloaded on first use into ./data/cache/.

Usage:  python run_submission.py [matbench_steels ...] [--out results_rerun.json.gz]
CPU-only; compute time of the original run is given in info.json ("notes").
"""
from __future__ import annotations
import gzip
import pathlib
import sys
import time

import numpy as np
from matbench.bench import MatbenchBenchmark

from dftstack.data import TASKS, load_task_frames
from dftstack.design import TASK_TYPE
from dftstack.features_all import features
from dftstack.frozen_config2 import BLOCKS2, CFG2, N_INNER, SEED
from dftstack.pipeline2 import run_split2

HERE = pathlib.Path(__file__).resolve().parent


def main(tasks, out_file):
    mb = MatbenchBenchmark(autoload=False, subset=tasks)
    for task in mb.tasks:
        name = task.dataset_name
        tt = TASK_TYPE[name]
        task.load()
        df, folds = load_task_frames(name)          # same official folds, cached as string ids
        X_all = features(name, df, BLOCKS2)
        X_all["composition"] = df["composition"].values
        for k, fold in enumerate(task.folds):
            t0 = time.time()
            tr_in, tr_y = task.get_train_and_val_data(fold)
            te_in = task.get_test_data(fold, include_target=False)
            tr = [str(i) for i in tr_in.index]
            te = [str(i) for i in te_in.index]
            assert tr == folds[k]["train"] and te == folds[k]["test"] and not set(tr) & set(te)
            y_tr = np.asarray(tr_y, dtype=float)
            preds, meta = run_split2(tt, X_all.loc[tr], y_tr, X_all.loc[te],
                                     HERE / "rerun_cache" / name / f"fold{k}_experts",
                                     cfg=CFG2[name], n_inner=N_INNER, seed=SEED)
            p = preds["stack_local"]
            if tt == "classification":            # hard labels at the inner-CV balanced-accuracy threshold
                p = [bool(v) for v in (p >= meta["thresholds"]["stack_local"])]
            else:
                p = [float(v) for v in p]
            task.record(fold, p, params={"pipeline": "DFT-Stack (Amendment-2 frozen config)", "seed": SEED})
            print(f"{name} fold {k}: recorded in {time.time() - t0:.0f} s", flush=True)
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
