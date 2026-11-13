"""Build folder-mode datasets + configs for Matbench v0.1 tasks.

For a task and fold, pulls the official matbench fold data, converts
pymatgen Structures to POSCARs, and writes a train_alignn --root_dir
(id_prop.csv ordered train -> val -> test; val = last --val_frac of the
training fold) plus a per-fold config derived from --base_config with
the exact keep_data_order split lock. Test targets are never loaded:
test rows are read with get_test_data(include_target=False) and carry a
placeholder target (0.0; alternating 0/1 for classification), so any
test-set loss or score the trainer prints is meaningless. The trainer
needs a finite value per row (rows with a NaN target are dropped, which
would break the n_test lock), and its end-of-run ROC-AUC print needs
both classes present. Scoring happens only via task.record on our
predictions. Classification tasks (mp_is_metal) get 0/1 train targets,
classification=True and classification_threshold 0.5 (the trainer then
uses a 2-logit head with NLL loss) and energy_mult_natoms=False in the
config.

Run locally (needs matbench + matminer + jarvis), then tar and scp.
"""
import argparse
import json
import os
import random

from jarvis.core.atoms import pmg_to_atoms
from matbench.bench import MatbenchBenchmark

CLF_TASKS = {"matbench_mp_is_metal"}
EPOCHS = {  # scaled by dataset size per the campaign plan
    "matbench_jdft2d": 300, "matbench_phonons": 300,
    "matbench_dielectric": 300, "matbench_log_gvrh": 200,
    "matbench_log_kvrh": 200, "matbench_perovskites": 200,
    "matbench_mp_gap": 75, "matbench_mp_is_metal": 75,
    "matbench_mp_e_form": 75,
}


def short(task):
    """matbench_mp_e_form -> mp_e_form."""
    return task.replace("matbench_", "")


def main():
    """Write dataset dir + config for one task/fold (or all folds)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--fold", default="all", help="0-4 or 'all'")
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--out_root", default=".")
    ap.add_argument("--val_frac", type=float, default=0.05)
    args = ap.parse_args()

    mb = MatbenchBenchmark(autoload=False, subset=[args.task])
    task = list(mb.tasks)[0]
    task.load()
    is_clf = args.task in CLF_TASKS
    base = json.load(open(args.base_config))
    folds = task.folds if args.fold == "all" else [int(args.fold)]

    for fold in folds:
        tr_in, tr_out = task.get_train_and_val_data(fold)
        te_in = task.get_test_data(fold, include_target=False)
        # matbench returns folds in original dataset order; shuffle
        # deterministically so the val tail is an unbiased sample
        order = list(tr_in.index)
        random.Random(123).shuffle(order)
        tr_in, tr_out = tr_in.loc[order], tr_out.loc[order]
        n_all = len(tr_in)
        n_val = max(1, int(round(args.val_frac * n_all)))
        n_train = n_all - n_val
        n_test = len(te_in)

        tag = "%s_f%d" % (short(args.task), fold)
        out_dir = os.path.join(args.out_root, "mb_" + tag)
        os.makedirs(out_dir, exist_ok=True)

        def tgt(v):
            return float(bool(v)) if is_clf else float(v)

        def placeholder(k):
            return float(k % 2) if is_clf else 0.0

        test_ids = []
        with open(os.path.join(out_dir, "id_prop.csv"), "w") as f:
            for mbid in tr_in.index:
                name = "%s.vasp" % mbid
                pmg_to_atoms(tr_in[mbid]).write_poscar(
                    os.path.join(out_dir, name))
                f.write("%s,%s\n" % (name, tgt(tr_out[mbid])))
            for k, mbid in enumerate(te_in.index):
                name = "%s.vasp" % mbid
                pmg_to_atoms(te_in[mbid]).write_poscar(
                    os.path.join(out_dir, name))
                f.write("%s,%s\n" % (name, placeholder(k)))
                test_ids.append(mbid)
        json.dump({"task": args.task, "fold": fold,
                   "test_ids": test_ids},
                  open(os.path.join(out_dir, "fold_manifest.json"), "w"))

        cfg = dict(base)
        cfg["keep_data_order"] = True
        cfg["n_train"], cfg["n_val"], cfg["n_test"] = n_train, n_val, n_test
        for k in ("train_ratio", "val_ratio", "test_ratio"):
            cfg.pop(k, None)
        cfg["epochs"] = EPOCHS.get(args.task, 200)
        cfg["output_dir"] = "mb_run_" + tag
        if is_clf:
            # the trainer switches to NLL loss on a 2-logit head when
            # classification_threshold is set; the per-atom energy
            # scaling does not apply to class logits
            cfg["classification_threshold"] = 0.5
            cfg["model"] = dict(cfg["model"])
            cfg["model"]["classification"] = True
            cfg["model"]["energy_mult_natoms"] = False
        cfg_path = os.path.join(args.out_root,
                                "config_mb_%s.json" % tag)
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print("fold %d: wrote %d train+val + %d test structures to %s | "
              "lock %d/%d/%d | epochs %d | config %s"
              % (fold, n_all, n_test, out_dir, n_train, n_val, n_test,
                 cfg["epochs"], cfg_path), flush=True)


if __name__ == "__main__":
    main()
