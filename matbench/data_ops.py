"""
Utility functions for data operations such as loading dataframes
and scoring.
"""

import logging
import math

import numpy as np
from matminer.datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from matbench.constants import (
    CLF_KEY,
    CLF_METRICS,
    MBID_KEY,
    REG_KEY,
    REG_METRICS,
)
from matbench.metadata import mbv01_metadata

logger = logging.getLogger(__name__)


def load(dataset_name, dataset_metadata=mbv01_metadata):
    """
    Load a matbench dataset into memory as a dataframe.
    This function is simply a wrapper around the matminer dataloader.

    Each matbench dataset is completely self contained.

    See https://hackingmaterials.lbl.gov/matbench/ for a list of dataset names.
    For example, "matbench_jdft2d".

    Args:
        dataset_name (str): A matbench dataset name, as defined in the
        datasets.json.

    Returns:
        (pandas.DataFrame): The dataset, containing two columns:
            - Inputs, either compositions or pymatgen structure objects.
            - Outputs, either a float (for regression) or a boolean (for
                classification).
    """

    if dataset_name not in dataset_metadata:
        raise KeyError(
            f"Dataset name {dataset_name} not recognized by matbench. "
            f"Please see https://hackingmaterials.lbl.gov/matbench for "
            f"a list of the dataset names, or choose from:"
            f"\n{list(dataset_metadata.keys())}"
        )
    logger.debug(
        f"Loading {dataset_name} into memory; please be patient as "
        f"loading many structures can take a while to serialize."
    )
    df = load_dataset(dataset_name)

    id_n_zeros = math.floor(math.log(df.shape[0], 10)) + 1
    mpcontribs_prefix = dataset_name.replace("matbench", "mb").replace("_", "-")
    df[MBID_KEY] = [f"{mpcontribs_prefix}-{i + 1:0{id_n_zeros}d}" for i in df.index]

    df = df.set_index(MBID_KEY)
    df = df[
        [
            dataset_metadata[dataset_name].input_type,
            dataset_metadata[dataset_name].target,
        ]
    ]

    return df


def score_array(true_array, pred_array, task_type):
    """
    Score an array according to multiple metrics.

    Args:
        true_array (list or np.array): The ground truth array
        pred_array (list or np.array): The predicted (test) array
        task_type (str): Either regression or classification.

    Returns:
        (dict): dictionary of the scores, according to all defined
            metrics.

    """
    computed = {}

    if task_type == REG_KEY:
        metrics = REG_METRICS
    elif task_type == CLF_KEY:
        metrics = CLF_METRICS
    else:
        raise ValueError(
            f"'task_type' must be on of {[REG_KEY, CLF_KEY]}, not '{task_type}'"
        )

    for metric in metrics:
        mfunc = METRIC_MAP[metric]
        computed[metric] = mfunc(true_array, pred_array)
    return computed


def mean_absolute_percentage_error(y_true, y_pred, threshold=1e-5):
    """Compute mean absolute percentage error, masked

    Masking is for when y_true is zero (causing a
    divide by zero error) or when y_true is very small
    (causing a massive skewing in the absolute percentage
    error).

    **Note: THIS WILL IGNORE ALL ENTRIES WHERE y_true's
    MAGNITUDE IS less than the threshold, hence the
    MAPE score is not representative of all
    entries if the truth array contains entries with
    magnitude very close to 0.**

    Args:
        y_true (np.ndarray): A 1-D array of true values
        y_pred (np.ndarray): A 1-D array of predicted values
        threshold (float): Entries with magnitude below this
            value will be ignored in the output.

    Returns:

    """
    y_true = np.asarray(y_true)
    mask = np.abs(y_true) > threshold
    y_pred = np.asarray(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.mean(np.fabs((y_true - y_pred) / y_true))


METRIC_MAP = {
    "mae": mean_absolute_error,
    "rmse": lambda true, pred: math.sqrt(mean_squared_error(true, pred)),
    "mape": mean_absolute_percentage_error,
    "max_error": max_error,
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "rocauc": roc_auc_score,
}
