import math
import logging

from matminer.datasets import load_dataset

from matbench.constants import (
    CLF_KEY,
    CLF_METRICS,
    MBID_KEY,
    METRIC_MAP,
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
    df[MBID_KEY] = [f"{mpcontribs_prefix}-{i + 1:0{id_n_zeros}d}"
                    for i in df.index]

    df = df.set_index(MBID_KEY)
    df = df[
        [
            dataset_metadata[dataset_name].input_type,
            dataset_metadata[dataset_name].target,
        ]
    ]

    return df


def score_array(true_array, pred_array, task_type):
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
