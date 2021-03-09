import os

from sklearn.model_selection import KFold, StratifiedKFold

from matbench.raw import get_kfold
from matbench.constants import REG_KEY, CLF_KEY
from matbench.metadata import metadata


def matbench_v01():
    """
    Generate validation json file for matbench v0.1, a benchmark
    of 13 regression and binary classification datasets which
    uses nested cross validation (random splits for regression,
    stratified for classification).

    Returns:
        None
    """

    d = {
        "common": {
            "n_splits": 5,
            "random_state": 18012019,
            "shuffle": True
        },
        "splits": None
    }

    kfold_config = d["common"]

    for ds, info in metadata:
        task_type = info.task_type

        if task_type == REG_KEY:
            kfold = KFold(**kfold_config)
        elif task_type == CLF_KEY:
            kfold = StratifiedKFold(**kfold_config)
        else:
            raise ValueError(
                f"'problem_type' must be one of {[REG_KEY, CLF_KEY]}, not '{task_type}'.")
