"""Helper function for high-throughput GNN trainings."""
"""Implementation based on the template of ALIGNN."""
import matplotlib.pyplot as plt

# import numpy as np
import time
# from matformer.train import train_dgl
import glob
import os
from collections import defaultdict
import os
import argparse
import numpy as np
import pandas as pd
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson, loadjson
from sklearn.metrics import mean_absolute_error, roc_auc_score
from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY
from train_on_folder import train_for_folder

parser = argparse.ArgumentParser(
    description="Trainer"
)
parser.add_argument("--single_run", required=False, help="specific part of subset.", default=None)
parser.add_argument("--fold", required=False, help="fold.", default=None)
parser.add_argument("--checkpoint", required=False, help="fold.", default=None)
parser.add_argument("--device", required=False, help="device.", default="gpu:0")
args = vars(parser.parse_args())
print("Input of argparse:", args)

fold_to_run = int(args["fold"]) if args["fold"] is not None else None
# fold_to_run = "matbench_mp_is_metal"
checkpoint_to_use= args["checkpoint"]
device_to_use = args["device"]

if args["single_run"] is None:
    subset =[
        "matbench_jdft2d",
        "matbench_dielectric",
        "matbench_phonons",
        "matbench_perovskites",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        "matbench_mp_e_form",
        "matbench_mp_gap",
        "matbench_mp_is_metal",
    ]
else:
    subset = [args["single_run"]]

mb = MatbenchBenchmark(
    autoload=False,
    subset=subset
)


def train_tasks(
    mb=None, config_template="config_example.json", file_format="poscar", device="gpu:0"
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
                if fold_to_run is not None:
                    if fold_to_run != ii:
                        continue
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                train_df["is_metal"] = train_df["is_metal"].astype(float)
                test_df["is_metal"] = test_df["is_metal"].astype(float)
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
                # Just for testing
                # config["n_train"] = 500
                # config["n_val"] = 100
                # config["n_test"] = 100
                config["keep_data_order"] = True
                config["batch_size"] = 64
                config["epochs"] = 50
                config["classification_threshold"] = 0.01
                config["progress"] = False
                config["learning_rate"] = 0.0005
                config["criterion"] = "BCEWithLogitsLoss"
                config["dataset"] = task.dataset_name
                config["target"] = "target"  # target.replace(" ", "_")
                fname = "config_fold_" + str(ii) + ".json"
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                config["output_dir"] = outdir_name
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
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
                # os.system(cmd)
                train_for_folder(root_dir=fold_name,
                                 config_name=fold_name + "/" + fname,
                                 file_format=file_format,
                                 output_dir=outdir_name,
                                 keep_data_order=True,
                                 classification_threshold=0.01,
                                 restore_checkpoint=checkpoint_to_use,
                                 device=device
                                 )
                test_csv = outdir_name + "/prediction_results_test_set.csv"
                df = pd.read_csv(test_csv)
                target_vals = df.target.values
                id_vals = df.id.values

        # Regression tasks
        # TODO: shorten the script by taking out repetitive lines
        if not classification:
            maes = []
            for ii, fold in enumerate(task.folds):
                if fold_to_run is not None:
                    if fold_to_run != ii:
                        continue
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
                config["batch_size"] = 64
                config["epochs"] = 500
                config["dataset"] = task.dataset_name
                if task.dataset_name == "matbench_mp_gap" or task.dataset_name == "matbench_mp_e_form":
                    config["learning_rate"] = 0.0005
                config["target"] = "target"  # target.replace(" ", "_")
                fname = "config_fold_" + str(ii) + ".json"
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                config["output_dir"] = outdir_name
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
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
                # os.system(cmd)
                train_for_folder(root_dir=fold_name,
                                 config_name=fold_name + "/" + fname,
                                 file_format=file_format,
                                 output_dir=outdir_name,
                                 keep_data_order=True,
                                 restore_checkpoint=checkpoint_to_use,
                                 device=device
                                 )
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
        fold = int(os.path.split(i)[0].split("_")[-1])
        # fold = int(i.split("/")[0].split("_")[-1])
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
            # We changed the predictions to sigmoid.
            # pred_vals = [True if i == 1 else False for i in pred_vals]
        results[fold] = pred_vals

    if regression:
        maes = np.array(maes)
        print(key, maes, np.mean(maes), np.std(maes))
    if not regression:
        roc_aucs = np.array(roc_aucs)
        print(key, roc_aucs, np.mean(roc_aucs), np.std(roc_aucs))
    return results

run_training = True

if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    config = loadjson(config_template)
    if run_training:
        train_tasks(mb=mb, config_template=config_template, file_format="poscar", device=device_to_use)

    run_dir = "../matbench"

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
    mb.add_metadata({"algorithm": "Matformer"})
    mb.to_file("results.json.gz")


for key, values in mb.scores.items():
    factor = 1000.0 if key in ["matbench_mp_e_form", "matbench_mp_gap", "matbench_perovskites"] else 1.0
    if key not in ["matbench_mp_is_metal"]:
        print(key, factor*values["mae"]["mean"], factor*values["mae"]["std"])
    else:
        print(key, values["rocauc"]["mean"],  values["rocauc"]["std"])