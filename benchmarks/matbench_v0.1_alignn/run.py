"""Train ALIGNN model on MatBench dataset."""
# Ref: https://www.nature.com/articles/s41524-021-00650-1

# conda create --name matbench_test python=3.8
# conda activate matbench_test
# pip install alignn matbench dgl-cu111

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY
import os
import glob
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import pmg_to_atoms
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from collections import defaultdict


mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        "matbench_jdft2d",
        # "matbench_dielectric",
        # "matbench_phonons",
        # "matbench_perovskites",
        # "matbench_log_gvrh",
        # "matbench_log_kvrh",
        # "matbench_mp_e_form",
        # "matbench_mp_gap",
        # "matbench_mp_is_metal",
    ],
)


def train_tasks(
    mb=None, config_template="config_example.json", file_format="poscar"
):
    """Train MatBench clalssification and regression tasks."""
    for task in mb.tasks:
        task.load()
        if task.metadata.task_type == CLF_KEY:
            classification = True
        else:
            classification = False
        # Classification tasks
        if classification:
            # rocs = []
            for ii, fold in enumerate(task.folds):
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                train_df["is_metal"] = train_df["is_metal"].astype(int)
                test_df["is_metal"] = test_df["is_metal"].astype(int)
                # Name of the target property
                target = [
                    col
                    for col in train_df.columns
                    if col not in ("id", "structure", "composition")
                ][0]
                # Making sure there are spaces or parenthesis which
                # can cause issue while creating folder
                fold_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_fold_"
                    + str(ii)
                )
                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                os.chdir(fold_name)
                # ALIGNN requires the id_prop.csv file
                f = open("id_prop.csv", "w")
                for jj, j in train_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                # There is no pre-defined validation splt, so we will use
                # a portion of training set as validation set, and
                # keep test set intact
                val_df = train_df[0 : len(test_df)]
                for jj, j in val_df.iterrows():
                    # for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                n_train = len(train_df)
                n_val = len(val_df)
                n_test = len(test_df)
                config = loadjson(config_template)
                config["n_train"] = n_train
                config["n_val"] = n_val
                config["n_test"] = n_test
                config["keep_data_order"] = True
                config["batch_size"] = 32
                config["epochs"] = 40
                config["classification_threshold"] = 0.01
                fname = "config_fold_" + str(ii) + ".json"
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                cmd = (
                    "train_folder.py --root_dir "
                    + fold_name
                    + " --config "
                    + fold_name
                    + "/"
                    + fname
                    + " --file_format="
                    + file_format
                    + " --keep_data_order=True"
                    + " --classification_threshold=0.01"
                    + " --output_dir="
                    + outdir_name
                )
                print(cmd)
                os.system(cmd)
                test_csv = outdir_name + "/prediction_results_test_set.csv"
                df = pd.read_csv(test_csv)
                target_vals = df.target.values
                id_vals = df.id.values

        # Regression tasks
        # TODO: shorten the script by taking out repetitive lines
        if not classification:
            maes = []
            for ii, fold in enumerate(task.folds):
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                # Name of the target property
                target = [
                    col
                    for col in train_df.columns
                    if col not in ("id", "structure", "composition")
                ][0]
                # Making sure there are spaces or parenthesis which
                # can cause issue while creating folder
                fold_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_fold_"
                    + str(ii)
                )
                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                os.chdir(fold_name)
                # ALIGNN requires the id_prop.csv file
                f = open("id_prop.csv", "w")
                for jj, j in train_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                # There is no pre-defined validation splt, so we will use
                # a portion of training set as validation set, and
                # keep test set intact
                val_df = train_df[0 : len(test_df)]
                for jj, j in val_df.iterrows():
                    # for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                n_train = len(train_df)
                n_val = len(val_df)
                n_test = len(test_df)
                config = loadjson(config_template)
                config["n_train"] = n_train
                config["n_val"] = n_val
                config["n_test"] = n_test
                config["keep_data_order"] = True
                config["batch_size"] = 32
                config["epochs"] = 500
                fname = "config_fold_" + str(ii) + ".json"
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                cmd = (
                    "train_folder.py --root_dir "
                    + fold_name
                    + " --config "
                    + fold_name
                    + "/"
                    + fname
                    + " --file_format="
                    + file_format
                    + " --keep_data_order=True"
                    + " --output_dir="
                    + outdir_name
                )
                print(cmd)
                os.system(cmd)
                test_csv = outdir_name + "/prediction_results_test_set.csv"
                df = pd.read_csv(test_csv)
                target_vals = df.target.values
                # id_vals = df.id.values
                pred_vals = df.prediction.values
                mae = mean_absolute_error(target_vals, pred_vals)
                maes.append(mae)
                task.record(fold, pred_vals, params=config)
                print(
                    "Dataset_name, Fold, MAE=",
                    task.dataset_name,
                    fold,
                    mean_absolute_error(target_vals, pred_vals),
                )
            maes = np.array(maes)
            print(maes, np.mean(maes), np.std(maes))
            print()
            print()
            print()


def compile_results(key="matbench_phonons", regression=True):
    """Compile fold based results for each task."""
    # Some of the jobs such as mp_e_form takes a couple of
    # days to complete for each fold
    # so we compile the results as follows
    maes = []
    roc_aucs = []
    results = defaultdict()

    for i in glob.glob(key + "*/prediction_results_test_set.csv"):
        fold = int(i.split("/")[0].split("_")[-1])
        # print (i,fold)
        df = pd.read_csv(i)

        target_vals = df.target.values
        # id_vals = df.id.values
        pred_vals = df.prediction.values
        if regression:
            mae = mean_absolute_error(target_vals, pred_vals)
            maes.append(mae)
            print("MAE", fold, mae)
        if not regression:
            roc = roc_auc_score(target_vals, pred_vals)
            roc_aucs.append(roc)
            print("ROC", fold, roc)
            pred_vals = [True if i == 1 else False for i in pred_vals]
        results[fold] = pred_vals

    if regression:
        maes = np.array(maes)
        print(key, maes, np.mean(maes), np.std(maes))
    if not regression:
        roc_aucs = np.array(roc_aucs)
        print(key, roc_aucs, np.mean(roc_aucs), np.std(roc_aucs))
    return results


if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    config = loadjson(config_template)
    train_tasks(mb=mb, config_template=config_template, file_format="poscar")

    run_dir = "."
    # run_dir = "/wrk/knc6/matbench/benchmarks/matbench_v0.1_alignn"

    cwd = os.getcwd()

    os.chdir(run_dir)

    results = defaultdict()
    for task in mb.tasks:
        task.load()
        task_name = task.dataset_name
        regr = True
        if "is" in task_name:
            regr = False
        results = compile_results(task_name, regression=regr)
        for ii, fold in enumerate(task.folds):
            train_df = task.get_train_and_val_data(fold, as_type="df")
            test_df = task.get_test_data(
                fold, include_target=True, as_type="df"
            )
            pred_vals = results[fold]
            task.record(fold, pred_vals, params=config)
    os.chdir(cwd)
    mb.add_metadata({"algorithm": "ALIGNN"})
    mb.to_file("results.json.gz")
