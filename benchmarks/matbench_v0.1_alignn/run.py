"""Train ALIGNN model on MatBench dataset."""
# Ref: https://www.nature.com/articles/s41524-021-00650-1

# conda create --name matbench_test python=3.8
# conda activate matbench_test
# pip install alignn matbench dgl-cu111

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY
import os
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import pmg_to_atoms
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        #'matbench_dielectric',
        # "matbench_log_gvrh",
        #'matbench_log_kvrh',
        "matbench_jdft2d",
        #'matbench_mp_e_form',
        #'matbench_mp_gap',
        #'matbench_phonons',
        #'matbench_glass',
        #'matbench_perovskites',
    ],
)

config_template = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "config_example.json")
)
file_format = "poscar"
for task in mb.tasks:
    task.load()
    if task.metadata.task_type == CLF_KEY:
        classification = True
    else:
        classification = False
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
                + target.replace(" ", "_").replace("(", "-").replace(")", "-")
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
                + target.replace(" ", "_").replace("(", "-").replace(")", "-")
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
                + " --output_dir="
                + outdir_name
            )

            print(cmd)
            os.system(cmd)
            test_csv = outdir_name + "/prediction_results_test_set.csv"
            df = pd.read_csv(test_csv)
            target_vals = df.target.values
            id_vals = df.id.values
            pred_vals = df.prediction.values
            mae = mean_absolute_error(target_vals, pred_vals)
            maes.append(mae)
            print("MAE", mean_absolute_error(target_vals, pred_vals))
        maes = np.array(maes)
        print(maes, np.mean(maes), np.std(maes))
        print()
        print()
        print()
