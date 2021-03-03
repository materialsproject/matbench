import numpy as np
from monty.json import MSONable
from matminer.datasets import get_all_dataset_info

from matbench.constants import DATA_KEY, PARAMS_KEY, SCORES_KEY, REG_KEY, REG_METRICS, CLF_METRICS, FOLD_DIST_METRICS
from matbench.util import RecursiveDotDict
from matbench.raw import get_kfold, load, score_array
from matbench.metadata import metadata, validation_metadata


class MatbenchTask(MSONable):
    """
    The core interface for running a Matbench task and recording its results.
    """
    FOLD_MAPPING = {i: f"fold_{i}" for i in range(validation_metadata.n_splits)}

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = load(dataset_name)
        self.info = get_all_dataset_info(dataset_name)

        self.metadata = metadata[dataset_name]
        self.kfold = get_kfold(self.metadata.task_type)
        self.split_ix = tuple([s for s in self.kfold.split(self.df)])

        self.results = RecursiveDotDict({})
        self.is_recorded = {k: False for k in self.FOLD_MAPPING.keys()}


    @property
    def scores(self):
        metric_keys = REG_METRICS if self.metadata.problem_type == REG_KEY else CLF_METRICS
        scores = {}
        if self.all_folds_recorded:
            for mk in metric_keys:
                metric = {}

                # scores for a metric among all folds
                raw_metrics_on_folds = [self.results[fk][SCORES_KEY][mk] for fk in self.FOLD_MAPPING.keys()]
                for op in FOLD_DIST_METRICS:
                    metric[op] = getattr(np, op)(raw_metrics_on_folds)
                scores[mk] = metric
        return scores

    @property
    def all_folds_recorded(self):
        return all([v for v in self.is_recorded.values()])

    def _get_data_from_df(self, ix, as_type):
        relevant_df = self.df.iloc[ix]
        if as_type == "df":
            return relevant_df
        elif as_type == "tuple":
            # training inputs, training outputs
            return relevant_df[self.metadata.input_type], relevant_df[self.metadata.target]

    def get_task_info(self):
        print(self.info)

    def get_train_and_val_data(self, fold_number, as_type="tuple"):
        """
        The training + validation data. All model tuning and hyperparameter selection must be done on this data, NOT test data.

        Args:
            fold_number:

        Returns:

        """
        ix = self.split_ix[fold_number][0]
        return self._get_data_from_df(ix, as_type)

    def get_test_data(self, fold_number, as_type="tuple"):
        """
        The test data used for recording benchmarks.

        Args:
            fold_number:

        Returns:


        """
        ix = self.split_ix[fold_number][1]
        return self._get_data_from_df(ix, as_type)

    def record(self, fold_number, predictions, params=None):
        """
        Record the test data as well as parameters about the model trained on this fold.

        Args:
            fold_number (int): The fold number, 0-4.
            predictions ([float] or [bool] or np.ndarray): A list of predictions for fold number {fold_number}

        Returns:
            None
        """
        if self.is_recorded[fold_number]:
            # todo: replace with logging critical
            raise ValueError(f"Fold number {fold_number} already recorded! Aborting...")
        else:
            fold_key = self.FOLD_MAPPING[fold_number]
            self.results[fold_key][DATA_KEY] = predictions
            self.results[fold_key][PARAMS_KEY] = params if params else {}
            self.is_recorded[fold_number] = True

            # todo: replace with logging info
            print(f"Recorded fold {fold_number} successfully.")

            truth = self._get_data_from_df(self.split_ix[fold_number][1], as_type="tuple")[1]
            self.results[fold_number][SCORES_KEY] = score_array(truth, predictions)
            print(f"Scored fold {fold_number} successfully.")

