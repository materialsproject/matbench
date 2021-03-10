import random
import copy
import json

import numpy as np
from monty.json import MSONable
from matminer.datasets import get_all_dataset_info

from matbench.constants import DATA_KEY, PARAMS_KEY, SCORES_KEY, REG_KEY, REG_METRICS, CLF_METRICS, FOLD_DIST_METRICS, MBV01_KEY
from matbench.util import RecursiveDotDict
from matbench.data_ops import load, score_array
from matbench.metadata import mbv01_validation, mbv01_metadata


class MatbenchTask(MSONable):
    """
    The core interface for running a Matbench task and recording its results.
    """
    
    RESULTS_KEY = "results"
    BENCHMARK_KEY = "benchmark_name"
    DATASET_KEY = "dataset_name"

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
            raise ValueError("Only matbnch_v0.1 available. No other benchmarks defined!")

        # keeping track of folds
        self.folds_keys = list(range(self.validation.splits))
        self.folds_nums = [f"fold_{f}" for f in self.folds_keys]
        self.folds_map = dict(zip(self.folds_nums, self.folds_keys))

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
            raw_metrics_on_folds = [self.results[fk][SCORES_KEY][mk] for fk in self.folds_map.values()]
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
        fold_key = self.folds_map[fold_number]
        ids = self.validation[fold_key].train

        if shuffle_seed:
            r = random.Random(shuffle_seed)
        else:
            r = random

        r.shuffle(ix)
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
            self.results[fold_key][DATA_KEY] = ids_to_predictions

            if not isinstance(params, (dict, type(None))):
                raise TypeError(f"Parameters must be stored as a dictionary, not {type(params)}!")
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
            self.BENCHMARK_KEY: self.benchmark_name,
            self.DATASET_KEY: self.dataset_name,
            self.RESULTS_KEY: dict(self.results)
            }

    def to_file(self, filename):
        with open(filename, "w") as f:
            json.dump(self.as_dict(), f)

    def validate(self):
        self._check_all_folds_recorded("Cannot validate unless all folds recorded!")
        rev_fold_mapping = {v: k for k, v in self.folds_map.items()}

        for fold_name, fold in self.results.items():
            fold_data = fold.data

            # Ensure all the indices are present with no extras for each fold

            fold_number = rev_fold_mapping[fold_name]
            req_indices = set(self.split_ix[fold_number][1])
            remaining_indices = copy.deepcopy(req_indices)
            extra_indices = {}
            req_data_type = float if self.metadata.task_type == REG_KEY else bool
            for ix, datum in fold_data.items():
                if ix not in req_indices:
                    extra_indices[ix] = datum
                else:
                    if not isinstance(datum, req_data_type):
                        raise TypeError(
                            f"Data point '{ix}: {datum}' has data type {type(datum)} while required type is {req_data_type}!")
                    remaining_indices.remove(ix)

            if extra_indices and not remaining_indices:
                raise ValueError(
                    f"{len(extra_indices)} extra indices for problem {self.dataset_name} are not allowed (found in {fold_name}: {remaining_indices}")
            elif not extra_indices and remaining_indices:
                raise ValueError(
                    f"{len(remaining_indices)} required indices for problem {self.dataset_name} not found for {fold_name}: {remaining_indices}")
            elif extra_indices and remaining_indices:
                raise ValueError(
                    f"{len(remaining_indices)} required indices for problem {self.dataset_name} not found and {len(extra_indices)} not allowed indices found for {fold_name}!")
            else:
                pass
        print(f"Data for {self.dataset_name} successfully validated.")


    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):

        req_base_keys = ["@module", "@class", cls.DATASET_KEY, cls.RESULTS_KEY, cls.BENCHMARK_KEY]
        for k in req_base_keys:
            if k not in d:
                raise KeyError(f"Required key '{k}' not found.")

        # Ensure the benchmark key is defined
        if d[cls.BENCHMARK_KEY] == MBV01_KEY:
            metadata = mbv01_metadata[cls.DATASET_KEY]
            validation = mbv01_validation.splits[cls.DATASET_KEY]
            fold_keys = set(validation.keys())
        else:
            raise ValueError(f"{d[cls.BENCHMARK_KEY]} not a defined benchmark! Only defined benchmark name is '{MBV01_KEY}'.")

        extra_base_keys = [k for k in d.keys() if k not in req_base_keys]
        if extra_base_keys:
            raise KeyError(f"Extra keys {extra_base_keys} not allowed.")
        dataset_name = d[cls.DATASET_KEY]
        task_type = metadata[dataset_name].task_type
        extra_fold_keys = [k for k in d[cls.RESULTS_KEY].keys() if k not in fold_keys]
        if extra_fold_keys:
            raise KeyError(f"Extra fold keys {extra_fold_keys} not allowed.")
        for fold_key in fold_keys:
            if fold_key not in d[cls.RESULTS_KEY]:
                raise KeyError(f"Required fold data for fold '{fold_key}' not found.")
            req_subfold_keys = [SCORES_KEY, DATA_KEY, PARAMS_KEY]
            extra_subfold_keys = [k for k in d[cls.RESULTS_KEY][fold_key] if k not in req_subfold_keys]
            if extra_subfold_keys:
                raise KeyError(f"Extra keys {extra_subfold_keys} for fold results of '{fold_key}' not allowed.")
            for subkey in req_subfold_keys:
                fold_results = d[cls.RESULTS_KEY][fold_key]
                if subkey not in fold_results:
                    raise KeyError(f"Required key '{subkey}' not found for fold '{fold_key}'.")
                if subkey == SCORES_KEY:
                    scores = d[cls.RESULTS_KEY][fold_key][subkey]
                    metrics = REG_METRICS if task_type == REG_KEY else CLF_METRICS
                    for m in metrics:
                        if m not in scores:
                            raise KeyError(f"Required score '{m}' not found for '{fold_key}'.")
                        elif not isinstance(scores[m], float):
                            raise TypeError(f"Required score '{m}' is not float-type for '{fold_key}'!")
                    extra_metrics = [k for k in scores if k not in metrics]
                    if extra_metrics:
                        raise KeyError(f"Extra keys {extra_metrics} for fold scores of '{fold_key}' not allowed.")

                # Data key checking is done in .validate()
                # results data indices are cast by json to be strings, so must be converted to int
                elif subkey == DATA_KEY:
                    try:
                        formatted_data = {int(k): v for k, v in d[cls.RESULTS_KEY][fold_key][DATA_KEY].items()}
                    except TypeError:
                        raise TypeError(f"Indices for {fold_key} cannot be cast to int!")
                    d[cls.RESULTS_KEY][fold_key][DATA_KEY] = formatted_data

                # Params key has no required form; it is up to the model to determine it.

        return cls._from_args(dataset_name=dataset_name, benchmark_name=d[cls.BENCHMARK_KEY], results_dict=d[cls.RESULTS_KEY])


    @classmethod
    def _from_args(cls, dataset_name, benchmark_name, results_dict):
        obj = cls(dataset_name, autoload=False, benchmark=benchmark_name)
        obj.results = RecursiveDotDict(results_dict)
        obj.is_recorded = {i: True for i in obj.folds_nums}
        obj.validate()
        return obj


