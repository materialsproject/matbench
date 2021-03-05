import random
import copy
import json

import numpy as np
from matminer.datasets import get_all_dataset_info

from matbench.constants import DATA_KEY, PARAMS_KEY, SCORES_KEY, REG_KEY, REG_METRICS, CLF_METRICS, FOLD_DIST_METRICS
from matbench.util import RecursiveDotDict
from matbench.raw import get_kfold, load, score_array
from matbench.metadata import metadata, validation_metadata


from monty.json import MSONable


class MatbenchTask(MSONable):
    """
    The core interface for running a Matbench task and recording its results.
    """
    FOLD_MAPPING = {i: f"fold_{i}" for i in range(validation_metadata.common.n_splits)}

    def __init__(self, dataset_name, autoload=True):
        self.dataset_name = dataset_name
        self.df = None if autoload else load(self.dataset_name)
        self.info = get_all_dataset_info(dataset_name)

        self.metadata = metadata[dataset_name]
        self.kfold = get_kfold(self.metadata.task_type)
        self.folds = list(self.FOLD_MAPPING.keys())

        self.results = RecursiveDotDict({})
        self.is_recorded = {k: False for k in self.FOLD_MAPPING.keys()}

    def load(self):
        if self.df is None:
            # todo: turn into logging
            self.df = load(self.dataset_name)
        else:
            print("Dataset already loaded")

    def _check_is_loaded(self):
        if self.df is None:
            raise ValueError("Task dataset is not loaded! Run MatbenchTask.load() to load the dataset into memory.")

    @property
    def split_ix(self):
        if self.df is None:
            return None
        else:
            return tuple([s for s in self.kfold.split(X=self.df, y=self.df[self.metadata.target])])

    @property
    def scores(self):
        metric_keys = REG_METRICS if self.metadata.task_type == REG_KEY else CLF_METRICS
        scores = {}
        if self.all_folds_recorded:
            for mk in metric_keys:
                metric = {}

                # scores for a metric among all folds
                raw_metrics_on_folds = [self.results[fk][SCORES_KEY][mk] for fk in self.FOLD_MAPPING.values()]
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

    def get_train_and_val_data(self, fold_number, as_type="tuple", shuffle_seed=None):
        """
        The training + validation data. All model tuning and hyperparameter selection must be done on this data, NOT test data.

        Args:
            fold_number:

        Returns:

        """
        self._check_is_loaded()
        ix = self.split_ix[fold_number][0]

        if shuffle_seed:
            r = random.Random(shuffle_seed)
        else:
            r = random

        r.shuffle(ix)
        return self._get_data_from_df(ix, as_type)

    def get_test_data(self, fold_number, as_type="tuple", include_target=False):
        """
        The test data used for recording benchmarks.

        Args:
            fold_number:

        Returns:


        """
        self._check_is_loaded()
        ix = self.split_ix[fold_number][1]
        if include_target:
            return self._get_data_from_df(ix, as_type)
        else:
            if as_type == "tuple":
                return self._get_data_from_df(ix, as_type)[0]
            elif as_type == "df":
                return self._get_data_from_df(ix, as_type)[self.metadata.task_type]

    def record(self, fold_number, predictions, params=None):
        """
        Record the test data as well as parameters about the model trained on this fold.

        Args:
            fold_number (int): The fold number, 0-4.
            predictions ([float] or [bool] or np.ndarray): A list of predictions for fold number {fold_number}

        Returns:
            None
        """
        self._check_is_loaded()
        if self.is_recorded[fold_number]:
            # todo: replace with logging critical
            raise ValueError(f"Fold number {fold_number} already recorded! Aborting...")
        else:
            fold_key = self.FOLD_MAPPING[fold_number]

            # create map of original df index to prediction, e.g., {ix_of_original_df1: prediction1, ... etc.}
            loc_index_to_predictions = {self.split_ix[fold_number][1][i]: p for i, p in enumerate(predictions)}
            self.results[fold_key][DATA_KEY] = loc_index_to_predictions
            self.results[fold_key][PARAMS_KEY] = params if params else {}
            self.is_recorded[fold_number] = True

            # todo: replace with logging info
            print(f"Recorded fold {fold_number} successfully.")

            truth = self._get_data_from_df(self.split_ix[fold_number][1], as_type="tuple")[1]
            self.results[fold_key][SCORES_KEY] = score_array(truth, predictions, self.metadata.task_type)
            print(f"Scored fold {fold_key} successfully.")

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "dataset_name": self.dataset_name,
            "results": dict(self.results)
            }

    def to_file(self, filename):
        with open(filename, "w") as f:
            json.dump(self.as_dict(), f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):

        req_base_keys = ["dataset_name", "results"]
        for k in req_base_keys:
            if k not in d:
                raise KeyError(f"Required key '{k}' not found.")

        extra_base_keys = [k for k in d.keys() if k not in req_base_keys]
        if extra_base_keys:
            raise KeyError(f"Extra keys {extra_base_keys} not allowed.")

        dataset_name = d["dataset_name"]
        task_type = metadata[dataset_name].task_type


        req_fold_keys = list(cls.FOLD_MAPPING.keys())

        extra_fold_keys = [k for k in d["results"].keys() if k not in req_fold_keys]
        if extra_fold_keys:
            raise KeyError(f"Extra fold keys {extra_fold_keys} not allowed.")

        for fold_key in req_fold_keys:
            if fold_key not in d["results"]:
                raise KeyError(f"Required fold data for fold '{fold_key}' not found.")

            req_subfold_keys = [SCORES_KEY, DATA_KEY, PARAMS_KEY]
            extra_subfold_keys = [k for k in d["results"][fold_key] if k not in req_subfold_keys]
            if extra_subfold_keys:
                raise KeyError(f"Extra keys {extra_subfold_keys} for fold results of '{fold_key}' not allowed.")

            for subkey in req_subfold_keys:
                fold_results = d["results"][fold_key]
                if subkey not in fold_results:
                    raise KeyError(f"Required key '{subkey}' not found for fold '{fold_key}'.")

                if subkey == SCORES_KEY:
                    scores = d["results"][fold_key][subkey]
                    metrics = REG_METRICS if task_type == REG_KEY else CLF_METRICS
                    for m in metrics:
                        if m not in scores:
                            raise KeyError(f"Required score '{m}' not found for '{fold_key}'.")
                        elif not isinstance(float, scores[m]):
                            raise TypeError(f"Required score '{m}' is not float-type for '{fold_key}'!")
                    extra_metrics = [k for k in scores if k not in  metrics]
                    if extra_metrics:
                        raise KeyError(f"Extra keys {extra_metrics} for fold scores of '{fold_key}' not allowed.")

                elif subkey == DATA_KEY:
                    data = d["results"][fold_key][DATA_KEY]

                    # Ensure all the indices are present with no extras
                    req_indices = set(list(range(metadata[dataset_name].n_samples)))

                    remaining_indices = copy.deepcopy(req_indices)

                    extra_indices = {}

                    req_data_type = float if metadata[dataset_name].task_type == REG_KEY else bool
                    for ix, datum in data:
                        if ix not in req_indices:
                            extra_indices[ix] = datum
                        else:
                            if not isinstance(req_data_type, datum):
                                raise TypeError(f"Data point '{ix}: {datum}' has data type {type(datum)} while required type is {req_data_type}!")
                            remaining_indices.remove(ix)

                    if extra_indices and not remaining_indices:
                        raise ValueError(f"{len(extra_indices)} extra indices for problem {dataset_name} are not allowed (found in {fold_key}: {remaining_indices}")
                    elif not extra_indices and remaining_indices:
                        raise ValueError(f"{len(remaining_indices)} required indices for problem {dataset_name} not found for {fold_key}: {remaining_indices}")
                    elif extra_indices and remaining_indices:
                        raise ValueError(f"{len(remaining_indices)} required indices not found and {len(extra_indices)} not allowed indices found for {fold_key}!")
                    else:
                        pass

                # Params key has no required form; it is up to the model to determine it.

        return cls._from_args(dataset_name=dataset_name, results_dict=d["results"])


    @classmethod
    def _from_args(cls, dataset_name, results_dict):
        obj = cls.__init__(dataset_name, autoload=False)
        obj.results = RecursiveDotDict(results_dict)
        obj.is_recorded = {i: True for i in obj.FOLD_MAPPING.keys()}
        return obj


