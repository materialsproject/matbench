import os.path
import argparse
import pandas as pd
import tensorflow as tf
from matbench.bench import MatbenchBenchmark
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.Schnet import make_crystal_model
from sklearn.preprocessing import StandardScaler
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
import kgcnn.training.callbacks
from kgcnn.utils.devices import set_devices_gpu
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser(description='Train SchNet.')
parser.add_argument("--gpu", required=False, help="GPU index used for training.",
                    default=None, nargs="+", type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)
gpu_to_use = args["gpu"]
set_devices_gpu(gpu_to_use)

subsets_compatible = ["matbench_mp_e_form", "matbench_mp_gap", "matbench_mp_is_metal",
                      "matbench_perovskites",
                      "matbench_log_kvrh", "matbench_log_gvrh", "matbench_dielectric", "matbench_phonons",
                      "matbench_jdft2d"]
mb = MatbenchBenchmark(subset=subsets_compatible, autoload=False)

callbacks = {
    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
    "abc": lambda st, ds: np.array(st.lattice.abc),
    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
}

hyper_1 = {
    "model": {
        "module_name": "kgcnn.literature.Schnet",
        "class_name": "make_crystal_model",
        "config": {
            "name": "Schnet",
            "inputs": [
                {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64}
            },
            "interaction_args": {
                "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
            },
            "node_pooling_args": {"pooling_method": "mean"},
            "depth": 4,
            "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
            "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                         "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            "output_embedding": "graph",
            "use_output_mlp": False,
            "output_mlp": None,  # Last MLP sets output dimension if None.
        }
    },
    "training": {
        "cross_validation": None,
        "execute_folds": None,
        "fit": {
            "batch_size": 32, "epochs": 800, "validation_freq": 10, "verbose": 2,
            "callbacks": [
                {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                    "verbose": 0}
                 }
            ]
        },
        "compile": {
            "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
            "loss": None
        },
        "scaler": {
            "class_name": "StandardScaler",
            "module_name": "kgcnn.scaler.scaler",
            "config": {"with_std": True, "with_mean": True, "copy": True}
        },
        "multi_target_indices": None
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                {"map_list": {"method": "set_range_periodic", "max_distance": 5}}
            ]
        },
        "data_unit": ""
    },
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "2.1.0"
    }
}
hyper_2 = {
    "model": {
        "module_name": "kgcnn.literature.Schnet",
        "class_name": "make_crystal_model",
        "config": {
            "name": "Schnet",
            "inputs": [
                {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64}
            },
            "interaction_args": {
                "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
            },
            "node_pooling_args": {"pooling_method": "mean"},
            "depth": 4,
            "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
            "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                         "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            "output_embedding": "graph",
            "use_output_mlp": False,
            "output_mlp": None,  # Last MLP sets output dimension if None.
        }
    },
    "training": {
        "cross_validation": None,
        "execute_folds": None,
        "fit": {
            "batch_size": 64, "epochs": 600, "validation_freq": 10, "verbose": 2,
            "callbacks": [
                {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    "learning_rate_start": 1e-4, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 600,
                    "verbose": 0}
                 }
            ]
        },
        "compile": {
            "optimizer": {"class_name": "Adam", "config": {"lr": 1e-4}},
            "loss": None
        },
        "scaler": {
            "class_name": "StandardScaler",
            "module_name": "kgcnn.scaler.scaler",
            "config": {"with_std": True, "with_mean": True, "copy": True}
        },
        "multi_target_indices": None
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                {"map_list": {"method": "set_range_periodic", "max_distance": 5}}
            ]
        },
        "data_unit": ""
    },
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "2.1.0"
    }
}

hyper_all = {
    "matbench_mp_e_form": hyper_1,
    "matbench_mp_gap": hyper_2,
    "matbench_mp_is_metal": hyper_2,
    "matbench_perovskites": hyper_1,
    "matbench_log_kvrh": hyper_1,
    "matbench_log_gvrh": hyper_1,
    "matbench_dielectric": hyper_1,
    "matbench_phonons": hyper_1,
    "matbench_jdft2d": hyper_1
}

restart_training = True
remove_invalid_graphs_on_predict = False

for idx_task, task in enumerate(mb.tasks):
    task.load()
    for i, fold in enumerate(task.folds):
        hyper = deepcopy(hyper_all[task.dataset_name])

        # Define loss for either classification or regresssion
        loss = {
            "class_name": "BinaryCrossentropy", "config": {"from_logits": True}
        } if task.metadata["task_type"] == "classification" else "mean_absolute_error"
        hyper["training"]["compile"]["loss"] = loss

        if restart_training and os.path.exists(
                "%s_predictions_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i)):
            predictions = np.load(
                "%s_predictions_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i)
            )
            task.record(fold, predictions, params=hyper)
            continue

        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        data_train = CrystalDataset()
        data_train._map_callbacks(train_inputs, pd.Series(train_outputs.values), callbacks)
        print("Making graph... (this may take a while)")
        data_train.set_methods(hyper["data"]["dataset"]["methods"])
        data_train.clean(hyper["model"]["config"]["inputs"])

        y_train = np.array(data_train.get("graph_labels"))
        x_train = data_train.tensor(hyper["model"]["config"]["inputs"])

        if task.metadata["task_type"] == "classification":
            scaler = None
        else:
            scaler = StandardScaler(**hyper["training"]["scaler"]["config"])
            y_train = scaler.fit_transform(y_train)
        print(y_train.shape)

        # train and validate your model
        model = make_crystal_model(**hyper["model"]["config"])
        model.compile(
            loss=tf.keras.losses.get(hyper["training"]["compile"]["loss"]),
            optimizer=tf.keras.optimizers.get(hyper["training"]["compile"]["optimizer"])
        )
        hist = model.fit(
            x_train, y_train,
            batch_size=hyper["training"]["fit"]["batch_size"],
            epochs=hyper["training"]["fit"]["epochs"],
            verbose=hyper["training"]["fit"]["verbose"],
            callbacks=[tf.keras.utils.deserialize_keras_object(x) for x in hyper["training"]["fit"]["callbacks"]]
        )

        # Get testing data
        test_inputs = task.get_test_data(fold, include_target=False)
        data_test = CrystalDataset()
        data_test._map_callbacks(test_inputs, pd.Series(np.zeros(len(test_inputs))), callbacks)
        print("Making graph... (this may take a while)")
        data_test.set_methods(hyper["data"]["dataset"]["methods"])

        if remove_invalid_graphs_on_predict:
            removed = data_test.clean(hyper["model"]["config"]["inputs"])
            np.save(
                "%s_predictions_invalid_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i),
                removed
            )
        else:
            removed = None

        # Predict on the testing data
        x_test = data_test.tensor(hyper["model"]["config"]["inputs"])
        predictions_model = model.predict(x_test)

        if remove_invalid_graphs_on_predict:
            indices_test = [j for j in range(len(test_inputs))]
            for j in removed:
                indices_test.pop(j)
            predictions = np.expand_dims(np.zeros(len(test_inputs), dtype="float"), axis=-1)
            predictions[np.array(indices_test)] = predictions_model
        else:
            predictions = predictions_model

        if task.metadata["task_type"] == "classification":
            def np_sigmoid(x):
                return np.exp(-np.logaddexp(0, -x))
            predictions = np_sigmoid(predictions)
        else:
            predictions = scaler.inverse_transform(predictions)

        if predictions.shape[-1] == 1:
            predictions = np.squeeze(predictions, axis=-1)

        np.save(
            "%s_predictions_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i),
            predictions
        )

        # Record data!
        task.record(fold, predictions, params=hyper)

# Save your results
mb.to_file("results.json.gz")
