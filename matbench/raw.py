import math

from sklearn.model_selection import KFold, StratifiedKFold
from matminer.datasets import load_dataset

from matbench.constants import REG_KEY, CLF_KEY, METRIC_MAP, CLF_METRICS, REG_METRICS, MBID_KEY
from matbench.metadata import metadata
from matbench.util import RecursiveDotDict

def load(dataset_name):
    """
    Load a matbench dataset into memory as a dataframe. This function is simply a wrapper around the matminer dataloader.

    Each matbench dataset is completely self contained.

    See https://hackingmaterials.lbl.gov/matbench/ for a list of dataset names. For example, "matbench_jdft2d".

    Args:
        dataset_name (str): A matbench dataset name, as defined in the datasets.json.

    Returns:
        (pandas.DataFrame): The dataset, containing two columns:
            - Inputs, either compositions or pymatgen structure objects.
            - Outputs, either a float (for regression) or a boolean (for classification).
    """

    if dataset_name not in metadata:
        raise KeyError(
            f"Dataset name {dataset_name} not recognized by matbench. "
            f"Please see https://hackingmaterials.lbl.gov/matbench for "
            f"a list of the dataset names, or choose from:\n{list(metadata.keys())}"
        )
    print(
        f"Loading {dataset_name} into memory; please be patient as loading many "
        f"structures can take a while to serialize."
    )
    df = load_dataset(dataset_name)

    id_n_zeros = math.floor(math.log(df.shape[0], 10)) + 1
    mpcontribs_prefix = dataset_name.replace("matbench", "mb").replace("_", "-")
    df[MBID_KEY] = [f"{mpcontribs_prefix}-{i + 1:0{id_n_zeros}d}" for i in df.index]

    df = df[[MBID_KEY, metadata[dataset_name].input_type, metadata[dataset_name].target]]

    return df


# def get_kfold(problem_type):
#     """
#     Obtain the test/train split for a given problem type (regression or classfication).
#
#     Args:
#         problem_type (str): Either 'regression' or 'classification'
#
#     Returns:
#         (KFold or StratifiedKFold): A kfold cross validation object
#
#     """
#     kfold_config = validation_metadata.common
#
#     if problem_type == REG_KEY:
#         return KFold(**kfold_config)
#     elif problem_type == CLF_KEY:
#         return StratifiedKFold(**kfold_config)
#     else:
#         raise ValueError(f"'problem_type' must be one of {[REG_KEY, CLF_KEY]}, not '{problem_type}'.")


def score_array(true_array, pred_array, problem_type):
    computed = {}

    if problem_type == REG_KEY:
        metrics = REG_METRICS
    elif problem_type == CLF_KEY:
        metrics = CLF_METRICS
    else:
        raise ValueError(f"'problem_type' must be on of {[REG_KEY, CLF_KEY]}, not '{problem_type}'")

    for metric in metrics:
        mfunc = METRIC_MAP[metric]
        computed[metric] = mfunc(true_array, pred_array)
    return computed
