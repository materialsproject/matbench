import random
import copy
import json

import numpy as np
from monty.json import MSONable
from matminer.datasets import get_all_dataset_info

from matbench.constants import REG_KEY, REG_METRICS, CLF_METRICS, FOLD_DIST_METRICS, MBV01_KEY
from matbench.util import RecursiveDotDict, MSONable2File
from matbench.data_ops import load, score_array
from matbench.metadata import mbv01_validation, mbv01_metadata


class MatbenchTask(MSONable, MSONable2File):
    """
    The core interface for running a Matbench task and recording its results.
    """
    
    _RESULTS_KEY = "results"
    _BENCHMARK_KEY = "benchmark_name"
    _DATASET_KEY = "dataset_name"
    _DATA_KEY = "data"
    _PARAMS_KEY = "parameters"
    _SCORES_KEY = "scores"

    def __init__(self, dataset_name, autoload=True, benchmark=MBV01_KEY):
        self.dataset_name = dataset_name
        self.df = load(self.dataset_name) if autoload else None
        self.info = get_all_dataset_info(dataset_name)

        # define all static data needed for this task
        # including citations, data size, as well as specific validation splits

        if benchmark == MBV01_KEY:
            self.benchmark_name = MBV01_KEY
            self.metadata = mbv01_metadata[dataset_name]
            self.validation = mbv01_validation.splits[dataset_name]
        else:
            raise ValueError(f"Only {MBV01_KEY} available. No other benchmarks defined!")


        # keeping track of folds
        self.folds_keys = list(self.validation.keys())
        self.folds_nums = list(range(len(self.folds_keys)))
        self.folds_map = dict(zip(self.folds_nums, self.folds_keys))

        # Alias for ease of use
        self.folds = self.folds_nums

        self.results = RecursiveDotDict({})
        self.is_recorded = {k: False for k in self.folds_nums}

    def load(self):
        if self.df is None:
            # todo: turn into logging
            self.df = load(self.dataset_name)
        else:
            print("Dataset already loaded")

    def _check_is_loaded(self):
        if self.df is None:
            raise ValueError("Task dataset is not loaded! Run MatbenchTask.load() to load the dataset into memory.")

    def _check_all_folds_recorded(self, msg):
        if not self.all_folds_recorded:
            raise ValueError(
                f"{msg}; folds {[f for f in self.is_recorded if not self.is_recorded[f]]} not recorded!")

    @property
    def scores(self):
        metric_keys = REG_METRICS if self.metadata.task_type == REG_KEY else CLF_METRICS
        scores = {}
        self._check_all_folds_recorded("Cannot score unless all folds are recorded!")
        for mk in metric_keys:
            metric = {}

            # scores for a metric among all folds
            raw_metrics_on_folds = [self.results[fk][self._SCORES_KEY][mk] for fk in self.folds_map.values()]
            for op in FOLD_DIST_METRICS:
                metric[op] = getattr(np, op)(raw_metrics_on_folds)
            scores[mk] = metric
        return scores

    @property
    def all_folds_recorded(self):
        return all([v for v in self.is_recorded.values()])

    def _get_data_from_df(self, ids, as_type):
        relevant_df = self.df.loc[ids]
        if as_type == "df":
            return relevant_df
        elif as_type == "tuple":
            # training inputs, training outputs
            return relevant_df[self.metadata.input_type], relevant_df[self.metadata.target]

    def get_info(self):
        print(self.info)

    def get_train_and_val_data(self, fold_number, as_type="tuple"):
        """
        The training + validation data. All model tuning and hyperparameter selection must be done on this data, NOT test data.

        Args:
            fold_number:

        Returns:

        """
        self._check_is_loaded()
        fold_key = self.folds_map[fold_number]
        ids = self.validation[fold_key].train
        return self._get_data_from_df(ids, as_type)

    def get_test_data(self, fold_number, as_type="tuple", include_target=False):
        """
        The test data used for recording benchmarks.

        Args:
            fold_number:

        Returns:


        """
        self._check_is_loaded()
        fold_key = self.folds_map[fold_number]
        ids = self.validation[fold_key].test
        if include_target:
            return self._get_data_from_df(ids, as_type)
        else:
            if as_type == "tuple":
                return self._get_data_from_df(ids, as_type)[0]
            elif as_type == "df":
                return self._get_data_from_df(ids, as_type)[self.metadata.task_type]

    def record(self, fold_number, predictions, params=None):
        """
        Record the test data as well as parameters about the model trained on this fold.

        Args:
            fold_number (int): The fold number.
            predictions ([float] or [bool] or np.ndarray): A list of predictions for fold number {fold_number}

        Returns:
            None
        """
        if self.is_recorded[fold_number]:
            # todo: replace with logging critical
            raise ValueError(f"Fold number {fold_number} already recorded! Aborting...")
        else:
            fold_key = self.folds_map[fold_number]

            # create map of original df index to prediction, e.g., {ix_of_original_df1: prediction1, ... etc.}

            split_ids = self.validation[fold_key].test
            if len(predictions) != len(split_ids):
                raise ValueError(f"Prediction outputs must be the same length as the inputs! {len(predictions)} != {len(split_ids)}")

            ids_to_predictions = {split_ids[i]: p for i, p in enumerate(predictions)}
            self.results[fold_key][self._DATA_KEY] = ids_to_predictions

            if not isinstance(params, (dict, type(None))):
                raise TypeError(f"Parameters must be stored as a dictionary, not {type(params)}!")
            self.results[fold_key][self._PARAMS_KEY] = params if params else {}
            self.is_recorded[fold_number] = True

            # todo: replace with logging info
            print(f"Recorded fold {fold_number} successfully.")

            truth = self._get_data_from_df(split_ids, as_type="tuple")[1]
            self.results[fold_key][self._SCORES_KEY] = score_array(truth, predictions, self.metadata.task_type)
            print(f"Scored fold {fold_key} successfully.")

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            self._BENCHMARK_KEY: self.benchmark_name,
            self._DATASET_KEY: self.dataset_name,
            self._RESULTS_KEY: dict(self.results)
            }

    def validate(self):
        self._check_all_folds_recorded(f"Cannot validate task {self.dataset_name} unless all folds recorded!")
        task_type = self.metadata.task_type
        
        # Check for extra fold keys
        extra_fold_keys = [k for k in self.results if k not in self.folds_keys]
        if extra_fold_keys:
            raise KeyError(f"Extra fold keys {extra_fold_keys} for task {self.dataset_name}  not allowed.")
        
        for fold_key in self.folds_keys:
            if fold_key not in self.results:
                raise KeyError(f"Required fold data for fold '{fold_key}' for task {self.dataset_name} not found.")
            
            # Check for extra or missing keys inside each fold: need params, scores, and data.
            req_subfold_keys = [self._SCORES_KEY, self._DATA_KEY, self._PARAMS_KEY]
            extra_subfold_keys = [k for k in self.results[fold_key] if k not in req_subfold_keys]
            if extra_subfold_keys:
                raise KeyError(f"Extra keys {extra_subfold_keys} for fold results of '{fold_key}' for task {self.dataset_name}  not allowed.")
            for subkey in req_subfold_keys:
                fold_results = self.results[fold_key]
                if subkey not in fold_results:
                    raise KeyError(f"Required key '{subkey}' for task {self.dataset_name}  not found for fold '{fold_key}'.")
                if subkey == self._SCORES_KEY:
                    scores = self.results[fold_key][subkey]
                    metrics = REG_METRICS if task_type == REG_KEY else CLF_METRICS
                    for m in metrics:
                        if m not in scores:
                            raise KeyError(f"Required score '{m}' for task {self.dataset_name}  not found for '{fold_key}'.")
                        elif not isinstance(scores[m], float):
                            raise TypeError(f"Required score '{m}' for task {self.dataset_name}  is not float-type for '{fold_key}'!")
                    extra_metrics = [k for k in scores if k not in metrics]
                    if extra_metrics:
                        raise KeyError(f"Extra keys {extra_metrics} for fold scores of '{fold_key}' for task {self.dataset_name} not allowed.")

                # results data indices are cast by json to be strings, so must be converted to int
                elif subkey == self._DATA_KEY:
                    fold_data = self.results[fold_key].data
                    
                    # Ensure all the indices are present with no extras for each fold
                    req_indices = set(self.validation[fold_key].test)
                    remaining_indices = copy.deepcopy(req_indices)
                    extra_indices = {}
                    req_data_type = float if self.metadata.task_type == REG_KEY else bool
                    for ix, datum in fold_data.items():
                        if ix not in req_indices:
                            extra_indices[ix] = datum
                        else:
                            if not isinstance(datum, req_data_type):
                                raise TypeError(
                                    f"Data point '{ix}: {datum}' has data type {type(datum)} while required type is {req_data_type} for task {self.dataset_name} !")
                            remaining_indices.remove(ix)

                    if extra_indices and not remaining_indices:
                        raise ValueError(
                            f"{len(extra_indices)} extra indices for problem {self.dataset_name} are not allowed (found in {fold_key}: {remaining_indices}")
                    elif not extra_indices and remaining_indices:
                        raise ValueError(
                            f"{len(remaining_indices)} required indices for problem {self.dataset_name} not found for {fold_key}: {remaining_indices}")
                    elif extra_indices and remaining_indices:
                        raise ValueError(
                            f"{len(remaining_indices)} required indices for problem {self.dataset_name} not found and {len(extra_indices)} not allowed indices found for {fold_key}!")
                    else:
                        pass
                    
                # Params key has no required form; it is up to the model to determine it.

        print(f"Data for {self.dataset_name} successfully validated.")


    @classmethod
    def from_dict(cls, d):
        req_base_keys = ["@module", "@class", cls._DATASET_KEY, cls._RESULTS_KEY, cls._BENCHMARK_KEY]
        for k in req_base_keys:
            if k not in d:
                raise KeyError(f"Required key '{k}' not found.")
        extra_base_keys = [k for k in d.keys() if k not in req_base_keys]
        if extra_base_keys:
            raise KeyError(f"Extra keys {extra_base_keys} not allowed.")
        return cls._from_args(dataset_name=d[cls._DATASET_KEY], benchmark_name=d[cls._BENCHMARK_KEY], results_dict=d[cls._RESULTS_KEY])


    @classmethod
    def _from_args(cls, dataset_name, benchmark_name, results_dict):
        obj = cls(dataset_name, autoload=False, benchmark=benchmark_name)
        obj.results = RecursiveDotDict(results_dict)
        obj.is_recorded = {i: True for i in obj.folds_nums}
        obj.validate()
        return obj


