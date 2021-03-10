import os
import json

from monty.serialization import dumpfn
from sklearn.model_selection import KFold, StratifiedKFold

from matbench.data_ops import load
from matbench.constants import REG_KEY, CLF_KEY, MBID_KEY, VALIDATION_METADATA_KEY, VALIDATION_SPLIT_KEY, TRAIN_KEY, TEST_KEY
from matbench.metadata import mbv01_metadata

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
                    "train": {matbenchid0: df ix 0, matbenchid1: df ix 1...}
                    "test": {matbenchid3785: df ix 3785, ...}
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
        VALIDATION_METADATA_KEY: {
            "n_splits": 5,
            "random_state": 18012019,
            "shuffle": True
        },
        VALIDATION_SPLIT_KEY: None
    }

    kfold_config = d[VALIDATION_METADATA_KEY]

    splits = {}

    for ds, info in mbv01_metadata.items():
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
        for train_ix, test_ix in kfold.split(X=df, y=df[mbv01_metadata[ds].target]):
            # indices by iloc identical to loc indices for the original matbench datasets

            train_df = df.iloc[train_ix]
            test_df = df.iloc[test_ix]

            trix = train_df.index.tolist()
            train_mbid = train_df[MBID_KEY].tolist()
            teix = test_df.index.tolist()
            test_mbid = test_df[MBID_KEY].tolist()

            ix_mbid_map_train = dict(zip(train_mbid, trix))
            ix_mbid_map_test = dict(zip(test_mbid, teix))
            split[f"fold_{fold}"] = {TRAIN_KEY: ix_mbid_map_train, TEST_KEY: ix_mbid_map_test}

        splits[ds] = split

    d["splits"] = splits

    print("Writing file...")
    dumpfn(d, "matbench/matbench_v0.1_validation.json")




if __name__ == "__main__":
    matbench_v01()