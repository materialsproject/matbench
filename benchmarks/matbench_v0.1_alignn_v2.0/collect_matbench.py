"""Record trained-fold predictions into a Matbench results file.

Reads each fold's prediction_results_test_set.csv (id,target,prediction;
ids carry .vasp) from --preds_root/mb_run_<task>_f<fold>/, maps ids back
to matbench test order via the local fold_manifest.json written by
make_matbench_dataset.py, records with task.record, prints per-fold and
task scores, and writes results.json.gz. For classification
(mp_is_metal) the trainer writes the predicted class (argmax of the two
logits, i.e. a 0.5 threshold on the metal probability) and those hard
labels are recorded as booleans.

Run locally (needs matbench). All folds of a task are checked before any
is recorded; a task with a missing fold is skipped with a warning, so
partial collections work during the campaign.
"""
import argparse
import csv
import json
import os

import pandas as pd
from matbench.bench import MatbenchBenchmark

CLF_TASKS = {"matbench_mp_is_metal"}


def short(task):
    """matbench_mp_e_form -> mp_e_form."""
    return task.replace("matbench_", "")


def load_preds(path):
    """prediction csv -> {mbid: float}."""
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if row[0] == "id":
                continue
            out[row[0].replace(".vasp", "")] = float(row[2])
    return out


def main():
    """Record all requested tasks' folds and write results.json.gz."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--preds_root", default=".",
                    help="contains mb_run_<task>_f<fold>/ dirs")
    ap.add_argument("--data_root", default=".",
                    help="contains mb_<task>_f<fold>/fold_manifest.json")
    ap.add_argument("--out", default="results.json.gz")
    args = ap.parse_args()

    mb = MatbenchBenchmark(autoload=False, subset=args.tasks)
    for task in mb.tasks:
        name = task.dataset_name
        task.load()
        paths = {}
        for fold in task.folds:
            tag = "%s_f%d" % (short(name), fold)
            pred_csv = os.path.join(args.preds_root, "mb_run_" + tag,
                                    "prediction_results_test_set.csv")
            manifest = os.path.join(args.data_root, "mb_" + tag,
                                    "fold_manifest.json")
            if not (os.path.exists(pred_csv) and os.path.exists(manifest)):
                print("MISSING fold:", tag, "- skipping task", name)
                break
            paths[fold] = (tag, pred_csv, manifest)
        if len(paths) != len(task.folds):
            continue
        for fold in task.folds:
            tag, pred_csv, manifest = paths[fold]
            test_ids = json.load(open(manifest))["test_ids"]
            preds = load_preds(pred_csv)
            missing = [i for i in test_ids if i not in preds]
            assert not missing, "%s: %d test ids without predictions" % (
                tag, len(missing))
            vals = [preds[i] for i in test_ids]
            if name in CLF_TASKS:
                vals = [v > 0.5 for v in vals]
            task.record(fold, pd.Series(vals, index=test_ids))
            fold_key = "fold_%d" % fold
            fold_scores = task.results.get(fold_key, {}).get("scores", {})
            print("recorded", tag, "| fold scores:",
                  json.dumps(dict(fold_scores), default=str))
        print(name, "task scores:",
              json.dumps(task.scores, default=str))

    mb.to_file(args.out)
    print("wrote", args.out, "| complete:", mb.is_complete,
          "| valid:", mb.is_valid)


if __name__ == "__main__":
    main()
