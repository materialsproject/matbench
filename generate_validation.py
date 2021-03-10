import os
import json

from monty.serialization import dumpfn
from sklearn.model_selection import KFold, StratifiedKFold

from matbench.raw import load
from matbench.constants import REG_KEY, CLF_KEY, MBID_KEY
from matbench.metadata import metadata

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def matbench_v01():
    """
    Generate validation json file for matbench v0.1, a benchmark
    of 13 regression and binary classification datasets which
    uses nested cross validation (random splits for regression,
    stratified for classification).

    The end result is a dictionary of the form:

    {
        "metadata": arbitrary dict metadata about this split scheme,
        "splits": {
            dataset_name_1: {
                "fold_0": {
                    "train": {index0: matbenchid0, index1: matbenchid1...}
                    "test": {index8367: matbenchid8367, ...}
                    },
                ...
                "fold_N": ...
            },
            dataset_name_2: { ... },
            ...
    }

    The "splits" key adheres to this specific format. The "metadata"
    key is freeform. Both "splits" and "metadata" must be present
    in the validation file generated.


    Returns:
        None
    """
    d = {
        "metadata": {
            "n_splits": 5,
            "random_state": 18012019,
            "shuffle": True
        },
        "splits": None
    }

    kfold_config = d["metadata"]

    splits = {}

    for ds, info in metadata.items():
        split = {}
        task_type = info.task_type

        if task_type == REG_KEY:
            kfold = KFold(**kfold_config)
        elif task_type == CLF_KEY:
            kfold = StratifiedKFold(**kfold_config)
        else:
            raise ValueError(
                f"'problem_type' must be one of {[REG_KEY, CLF_KEY]}, not '{task_type}'.")

        df = load(ds)
        print(df)

        # y is needed for stratified splits
        fold = 0
        for train_ix, test_ix in kfold.split(X=df, y=df[metadata[ds].target]):
            # indices by iloc identical to loc indices for the original matbench datasets

            train_df = df.iloc[train_ix]
            test_df = df.iloc[test_ix]

            trix = train_df.index.tolist()
            train_mbid = train_df[MBID_KEY].tolist()
            teix = test_df.index.tolist()
            test_mbid = test_df[MBID_KEY].tolist()

            ix_mbid_map_train = dict(zip(trix, train_mbid))
            ix_mbid_map_test = dict(zip(teix, test_mbid))
            split[f"fold_{fold}"] = {"train": ix_mbid_map_train, "test": ix_mbid_map_test}

        splits[ds] = split

    d["splits"] = splits

    print("Writing file...")
    dumpfn(d, "matbench/matbench_v0.1_validation.json", "w")




if __name__ == "__main__":
    matbench_v01()