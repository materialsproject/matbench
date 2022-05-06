import gc
from os.path import join
from pathlib import Path
from time import time

import cloudpickle as pickle
import crabnet
import numpy as np
import pandas as pd
import torch
from ax import Data
from ax.core import Experiment, Objective, OptimizationConfig
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.storage.json_store.save import save_experiment
from crabnet.train_crabnet import get_model
from sklearn.metrics import mean_absolute_error

from matbench.bench import MatbenchBenchmark
from utils.metrics import CrabNetMetric
from utils.parameterization import correct_parameterization
from utils.search import search_space

dummy = False
metric = "crabnet_mae"

if dummy:
    n_splits = 2
    n_sobol = 2
    n_saas = 3
    num_samples = 16
    warmup_steps = 32
else:
    n_splits = 5
    # n_sobol = 2 * len(search_space.parameters)
    n_sobol = 10
    n_saas = max(100 - n_sobol, 0)
    num_samples = 256
    warmup_steps = 512

torch.manual_seed(12345)  # To always get the same Sobol points
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# create dir https://stackoverflow.com/a/273227/13697228
parameter_str = join("saas", f"sobol_{n_sobol}-saas_{n_saas}")
experiment_dir = join("experiments", parameter_str)
figure_dir = join("figures", parameter_str)
if dummy:
    experiment_dir = join(experiment_dir, "dummy")
    figure_dir = join(figure_dir, "dummy")
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(figure_dir).mkdir(parents=True, exist_ok=True)


param_names = list(search_space.parameters.keys())

# %% matbench loop
mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

task = list(mb.tasks)[0]
task.load()


def matbench_fold(fold, model_type="FULLYBAYESIAN"):
    t0 = time()
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    train_val_df = pd.DataFrame(
        {"formula": train_inputs.values, "target": train_outputs.values}
    )
    if dummy:
        train_val_df = train_val_df[:25]

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=CrabNetMetric(
                name=metric, train_val_df=train_val_df, n_splits=n_splits
            ),
            minimize=True,
        ),
    )
    # TODO: use status_quo (Arm) as default CrabNet parameters
    exp = Experiment(
        name="nested_crabnet_mae_saas",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    sobol = Models.SOBOL(exp.search_space)
    print("evaluating SOBOL points")
    for _ in range(n_sobol):
        print(_)
        trial = exp.new_trial(generator_run=sobol.gen(1))
        trial.run()
        trial.mark_completed()

    data = exp.fetch_data()
    j = -1
    new_value = np.nan
    best_so_far = np.nan
    for j in range(n_saas):
        if model_type == "FULLYBAYESIAN":
            mdl = Models.FULLYBAYESIAN(
                experiment=exp,
                data=exp.fetch_data(),
                num_samples=num_samples,  # Increasing this may result in better model fits
                warmup_steps=warmup_steps,  # Increasing this may result in better model fits
                gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
                torch_device=tkwargs["device"],
                torch_dtype=tkwargs["dtype"],
                verbose=False,  # Set to True to print stats from MCMC
                disable_progbar=True,  # Set to False to print a progress bar from MCMC
            )
        elif model_type == "GPEI":
            mdl = Models.GPEI(
                experiment=exp,
                data=exp.fetch_data(),
                torch_device=tkwargs["device"],
                torch_dtype=tkwargs["dtype"],
            )
        generator_run = mdl.gen(1)
        best_arm, _ = generator_run.best_arm_predictions
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()
        data = Data.from_multiple_data([data, trial.fetch_data()])
        new_value = trial.fetch_data().df["mean"].min()
        best_so_far = data.df["mean"].min()
        tf = time()
        print(
            f"iter{j}, BestInIter:{new_value:.3f}, BestSoFar:{best_so_far:.3f} elapsed time: {tf - t0}",
        )

    exp.fetch_data()
    best_parameters = best_arm.parameters

    experiment_fpath = join(experiment_dir, "experiment" + str(fold) + ".json")
    save_experiment(exp, experiment_fpath)

    test_pred, default_mae, test_mae, best_parameterization = get_test_results(
        task, fold, best_parameters, train_val_df
    )
    print(f"default_mae: {default_mae}")
    print(f"test_mae: {test_mae}")
    # maes.append(test_mae)  # [0.32241879861870626, ...]

    # task.record(fold, test_pred, params=best_parameterization)

    return test_pred, best_parameterization


def matbench_fold_GPEI(fold):
    test_pred, best_parameterization = matbench_fold(fold, model_type="GPEI")
    return test_pred, best_parameterization


def get_test_results(task, fold, best_parameters, train_val_df):
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    test_df = pd.DataFrame({"formula": test_inputs, "target": test_outputs})

    default_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
    )

    default_true, default_pred, default_formulas, default_sigma = default_model.predict(
        test_df
    )
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    default_mae = mean_absolute_error(default_true, default_pred)

    # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
    del default_model
    gc.collect()
    torch.cuda.empty_cache()

    best_parameterization = correct_parameterization(best_parameters)
    test_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
        verbose=False,
        **best_parameterization,
    )
    # TODO: update CrabNet predict function to allow for no target specified
    test_true, test_pred, test_formulas, test_sigma = test_model.predict(test_df)
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    test_mae = mean_absolute_error(test_true, test_pred)

    return test_pred, default_mae, test_mae, best_parameterization


# %% collection
savepath = join(figure_dir, "expt_gap_benchmark.json.gz")


def collect_results(use_saas: bool):
    if use_saas:
        parameter_str = join("saas", f"sobol_{n_sobol}-saas_{n_saas}")
    else:
        parameter_str = f"sobol_{n_sobol}-saas_{n_saas}"
    figure_dir = join("figures", parameter_str)
    if dummy:
        figure_dir = join(figure_dir, "dummy")
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    with open("jobs.pkl", "rb") as f:
        jobs = pickle.load(f)
    # concatenation
    for i, fold in enumerate(task.folds):
        test_pred, best_parameterization = jobs[i].result()
        task.record(fold, test_pred, best_parameterization)

    my_metadata = {"algorithm_version": crabnet.__version__}
    mb.add_metadata(my_metadata)
    mb.to_file(savepath)
