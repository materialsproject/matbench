import math

from matminer.datasets import load_dataset

from matbench.constants import REG_KEY, CLF_KEY, METRIC_MAP, CLF_METRICS, REG_METRICS, MBID_KEY
from matbench.metadata import mbv01_metadata

class RecursiveDotDict(dict):
    """
    Adapted from user Curt Hagenlocher from
    https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, RecursiveDotDict):
            value = RecursiveDotDict(value)
        super(RecursiveDotDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, RecursiveDotDict.MARKER)
        if found is RecursiveDotDict.MARKER:
            found = RecursiveDotDict()
            super(RecursiveDotDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


def load(dataset_name, dataset_metadata=mbv01_metadata):
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

    if dataset_name not in dataset_metadata:
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

    df = df[[MBID_KEY, dataset_metadata[dataset_name].input_type, dataset_metadata[dataset_name].target]]

    return df


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
