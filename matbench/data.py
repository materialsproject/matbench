from matminer.datasets import load_dataset

from matbench.constants import DATASETS


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

    if dataset_name not in DATASETS:
        raise KeyError(
            f"Dataset name {dataset_name} not recognized by matbench. "
            f"Please see https://hackingmaterials.lbl.gov/matbench for "
            f"a list of the dataset names, or choose from:\n{list(DATASETS.keys())}"
        )
    return load_dataset(dataset_name)


if __name__ == "__main__":
    df = load("matbench_steels")
    print(df)
    print(df.dtypes)