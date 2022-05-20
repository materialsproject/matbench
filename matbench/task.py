import copy
import logging

import numpy as np
from matminer.datasets import get_all_dataset_info
from matminer.featurizers.conversions import (
    StrToComposition,
    StructureToComposition,
)
from monty.json import MSONable
from scipy import stats

from matbench.constants import (
    CLF_KEY,
    CLF_METRICS,
    FOLD_DIST_METRICS,
    MBV01_KEY,
    REG_KEY,
    REG_METRICS,
)
from matbench.data_ops import load, score_array
from matbench.metadata import mbv01_metadata, mbv01_validation
from matbench.util import MSONable2File, RecursiveDotDict, immutify_dictionary

logger = logging.getLogger(__name__)


class MatbenchTask(MSONable, MSONable2File):
    """The core interface for running a Matbench task and recording its results.

    MatbenchTask handles creating training/validation and testing sets, as
    well as recording and managing all data in a consistent fashion.
    MatbenchTask also validates data according to te specifications in the
    validation file.

    MatbenchTasks have a few core methods:

    - MatbenchTask.get_train_and_val_data: Get nested cross validation data to
        be used for all training and validation.
    - MatbenchTask.get_test_data: Get test data for nested cross validation.
    - MatbenchTask.record: Record your predicted results for the test data.
    - MatbenchTask.validate: Check to make sure the data you recorded for this
        task is valid.

    You can iterate through the folds of a matbench task using .folds and
    the .get_*_data methods.

    You can load the results of a task without having to load large
    datasets themselves. However, to get training and testing data,
    you must load the datasets. Tasks loaded from files do not
    automatically load the dataset into memory; to load a dataset into memory,
    use MatbenchTask.load().

    See the full documentation online for more info and tutorials on
    using MatbenchTask.

    Attributes:
        benchmark_name (str): The name of the benchmark this task belongs to.
        df (pd.DataFrame): the dataframe of the dataset for this task
        info (str): Info about this dataset
        metadata (RecursiveDotDict): all metadata about this dataset
        validation (RecursiveDotDict): The validation specification for this
            task, including the training and testing splits for each fold.
        folds_keys ([str]): Keys of folds, fold_i for the ith fold.
        folds_nums ([int]): Values of folds, i for the ith fold.
        folds_map ({int: str}): Mapping of folds_nums to folds_keys
        folds ([int]): Alias for folds_nums
        results (RecursiveDotDict): all raw results in dict-like form.
    """

    _RESULTS_KEY = "results"
    _BENCHMARK_KEY = "benchmark_name"
    _DATASET_KEY = "dataset_name"
    _DATA_KEY = "data"
    _UNCERTAINTY_KEY = "uncertainty"
    _PARAMS_KEY = "parameters"
    _SCORES_KEY = "scores"

    def __init__(self, dataset_name, autoload=True, benchmark=MBV01_KEY):
        """
        Args:
            dataset_name (str): Name of the task. Must belong to the benchmark
                given in the 'benchmark' argument.
            autoload (bool): If True, will load the benchmark's raw data. This
                includes deserializing many large structures for some datasets,
                so loading make take some time. If False, you will need to
                run .load() before running .get_*_data() methods.
            benchmark (str): Name of the benchmark this task belongs to.
        """
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
            raise ValueError(
                f"Only {MBV01_KEY} available. No other benchmarks defined!"
            )

        # keeping track of folds
        self.folds_keys = list(self.validation.keys())
        self.folds_nums = list(range(len(self.folds_keys)))
        self.folds_map = dict(zip(self.folds_nums, self.folds_keys))

        # Alias for ease of use
        self.folds = self.folds_nums
        self.results = RecursiveDotDict({})

    def __repr__(self) -> str:
        keys = "input_type mad n_samples target task_type unit".split()

        md_str = ",\n  ".join(f"{k}={self.metadata[k]}" for k in keys)

        return (
            f"{type(self).__name__}(\n  dataset_name={self.dataset_name},\n  version"
            f"={self.benchmark_name.replace('matbench_v', '')},\n  {md_str},\n)"
        )

    def _get_data_from_df(self, ids, as_type):
        """Private function to get fold data from the task dataframe.

        Args:
            ids (list-like): List of string indices to grab from the df.
            as_type (str): either "df" or "tuple". If "df", returns the
                data as a subset of the task df. If "tuple", returns
                list-likes of the inputs and outputs as a 2-tuple.

        Returns:
            (pd.DataFrame or (list-like, list-like))

        """
        relevant_df = self.df.loc[ids]
        if as_type == "df":
            return relevant_df
        elif as_type == "tuple":
            # inputs, outputs
            return (
                relevant_df[self.metadata.input_type],
                relevant_df[self.metadata.target],
            )

    def _check_is_loaded(self):
        """Private method to check if the dataset is loaded.

        Throws error if the dataset is not loaded.

        Returns:
            None
        """
        if self.df is None:
            raise ValueError(
                "Task dataset is not loaded! Run MatbenchTask.load() to "
                "load the dataset into memory."
            )

    def _check_all_folds_recorded(self, msg):
        """Private method to check if all folds have been recorded.

        Throws error if all folds have not been recorded.

        Args:
            msg (str): Error message to be displayed.

        Returns:
            None
        """
        if not self.all_folds_recorded:
            raise ValueError(
                f"{msg}; folds "
                f"{[f for f in self.is_recorded if not self.is_recorded[f]]} "
                f"not recorded!"
            )

    @classmethod
    def from_dict(cls, d):
        """Create a MatbenchTask from a dictionary input.

        Required method from MSONable.

        Args:
            d (dict):

        Returns:
            (MatbenchTask)

        """
        req_base_keys = [
            "@module",
            "@class",
            cls._DATASET_KEY,
            cls._RESULTS_KEY,
            cls._BENCHMARK_KEY,
        ]
        for k in req_base_keys:
            if k not in d:
                raise KeyError(f"Required key '{k}' not found.")
        extra_base_keys = [k for k in d.keys() if k not in req_base_keys]
        if extra_base_keys:
            raise KeyError(f"Extra keys {extra_base_keys} not allowed.")
        return cls._from_args(
            dataset_name=d[cls._DATASET_KEY],
            benchmark_name=d[cls._BENCHMARK_KEY],
            results_dict=d[cls._RESULTS_KEY],
        )

    @classmethod
    def _from_args(cls, dataset_name, benchmark_name, results_dict):
        """Instantiate a MatbenchTask from a arguments

        Args:
            dataset_name (str): The name of the dataset/task
            benchmark_name (str): The name of the corresponding benchmark
            results_dict (dict): A formatted dictionary of raw results.

        Returns:
            (MatbenchTask)
        """
        obj = cls(dataset_name, autoload=False, benchmark=benchmark_name)
        obj.results = RecursiveDotDict(results_dict)
        obj.validate()
        return obj

    def load(self):
        """Load the dataset for this task into memory.

        Returns:
            None
        """
        if self.df is None:
            logger.info(f"Loading dataset '{self.dataset_name}'...")
            self.df = load(self.dataset_name)
            logger.info(f"Dataset '{self.dataset_name} loaded.")
        else:
            logger.info(
                f"Dataset {self.dataset_name} already loaded; "
                f"not reloading dataset."
            )

    def get_info(self):
        logger.info(self.info)

    def get_train_and_val_data(self, fold_number, as_type="tuple"):
        """
        The training + validation data. All model tuning and
        hyperparameter selection must be done on this data, NOT test data.

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
                return self._get_data_from_df(ids, as_type)[
                    [self.metadata.input_type]
                ]

    def record(self, fold_number, predictions, ci=None, std=None, params=None):
        """Record the test data as well as parameters about the model
        trained on this fold.

        Args:
            fold_number (int): The fold number.
            predictions ([float] or [bool] or np.ndarray): A list of predictions for
            fold number {fold_number}
            ci ([tuple] or [list] or np.ndarray): A list of 95% confidence
            intervals on predictions for fold number {fold_number}. By default
            None. Only one of `ci` or `std` should be specified, not both.
            std ([float] or np.ndarray): A list of prediction standard deviations
            for fold number {fold_number}. By default None. Only one of
            `ci` or `std` should be specified, not both.
            params (dict): Any free-form parameters for information
                about the algorithm on this fold. For example,
                hyperparameters determined during validation. Parameters
                must be a dictionary; dictionary types must adhere to
                the same requirements as in the MatbenchBenchmark.add_metadata
                docstring.

        Returns:
            None
        """
        if self.is_recorded[fold_number]:
            logger.error(
                f"Fold number {fold_number} already recorded! Aborting record..."
            )
        else:
            # avoid problems with json serialization
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()

            if isinstance(std, np.ndarray):
                std = std.tolist()

            if isinstance(ci, np.ndarray):
                ci = ci.tolist()

            if std is not None and ci is not None:
                raise ValueError(
                    """Both standard deviation (`std`) and confidence
                    intervals (`ci`) were specified as kwargs. Only one
                    should be specified, not both."""
                )

            fold_key = self.folds_map[fold_number]

            # create map of original df index to prediction, e.g.,
            # {ix_of_original_df1: prediction1, ... etc.}

            split_ids = self.validation[fold_key].test
            if len(predictions) != len(split_ids):
                raise ValueError(
                    f"Prediction outputs must be the same length as the "
                    f"inputs! {len(predictions)} != {len(split_ids)}"
                )

            ids_to_predictions = {split_ids[i]: p for i, p in enumerate(predictions)}
            self.results[fold_key][self._DATA_KEY] = ids_to_predictions

            if std is not None or ci is not None:
                if self.metadata["task_type"] == "classification":
                    raise ValueError(
                        "`std` and `ci` are not valid kwargs for classification "
                        + "tasks. See "
                        + "https://github.com/materialsproject/matbench/pull/99/files#issuecomment-1022662192."  # noqa: E501
                    )

                if ci is None:
                    low_p = 0.05
                    high_p = 0.95
                    # convert from two-tail to one-tail probabilities
                    # for compatibility with `ppf`
                    # https://stackoverflow.com/a/29562808/13697228
                    low_p = low_p / 2.0
                    high_p = (1 + high_p) / 2.0
                    # convert std to ci, modified from source:
                    # https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/b2f342f6606d1d667bf9583919a663adf8643efe/uncertainty_toolbox/metrics_scoring_rule.py#L187 # noqa: E501
                    pred_l = stats.norm.ppf(low_p, loc=predictions, scale=std)
                    pred_u = stats.norm.ppf(high_p, loc=predictions, scale=std)
                    ci = np.vstack((pred_l.ravel(), pred_u.ravel())).T.tolist()
                    ci = [tuple(c) for c in ci]

                if std is None:
                    # std calculated and stored iff ci is symmetric within tol
                    pred_l, pred_u = np.hsplit(np.array(ci), 2)
                    if np.allclose(-pred_l, pred_u):
                        high_p = 0.95
                        # convert from two-tail to one-tail probabilities for
                        # compatibility with `ppf`
                        # https://stackoverflow.com/a/29562808/13697228
                        high_p = (1 + high_p) / 2.0
                        std = (pred_u - pred_l) / (2 * stats.norm.ppf(high_p))
                    else:
                        std = [None] * len(ci)

                if len(ci) != len(split_ids):
                    raise ValueError(
                        f"""Confidence interval outputs (derived from standard
                         deviations if `std` was supplied) must be the same
                         length as the inputs! {len(ci)} != {len(split_ids)}"""
                    )

                ids_to_uncertainties = {
                    split_ids[i]: {"ci_lower": p[0], "ci_upper": p[1], "std": s}
                    for i, (p, s) in enumerate(zip(ci, std))
                }
                self.results[fold_key][self._UNCERTAINTY_KEY] = ids_to_uncertainties
            else:
                self.results[fold_key][self._UNCERTAINTY_KEY] = None

            if not isinstance(params, (dict, type(None))):
                raise TypeError(
                    f"Parameters must be stored as a dictionary, not {type(params)}!"
                )
            params = immutify_dictionary(params) if params else params
            self.results[fold_key][self._PARAMS_KEY] = params if params else {}
            self.is_recorded[fold_number] = True

            logger.info(
                f"Recorded fold " f"{self.dataset_name}-{fold_number} successfully."
            )

            truth = self._get_data_from_df(split_ids, as_type="tuple")[1]
            self.results[fold_key][self._SCORES_KEY] = score_array(
                truth, predictions, self.metadata.task_type
            )
            logger.debug(
                f"Scored fold '" f"{self.dataset_name}-{fold_key} successfully."
            )

    def as_dict(self):
        """Return a MatbenchTask object as a dictionary.

        Required method from MSONAble.

        Returns:
            (dict)
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            self._BENCHMARK_KEY: self.benchmark_name,
            self._DATASET_KEY: self.dataset_name,
            self._RESULTS_KEY: dict(self.results),
        }

    def validate(self):
        """Validate a task after all folds have been recorded.

        There are a few requirements for a task to be validated:
        - Data types of each predicted sample must match those
            specified by the validation procedure
        - All folds must be recorded
        - There must be no extra or missing required keys from
            the data, including indices. Every index specified in
            the validation procedure must be present in its
            correct fold, and no extras may be present.
        - Ensure consistency of the supplied uncertainty values.
            For example, if std is specified and ci is specified
            for one sample, it must be specified for all samples.
            If ci is specified but std is not, that must be
            consistent for all samples.
        Returns:

        """
        self._check_all_folds_recorded(
            f"Cannot validate task {self.dataset_name} "
            f"unless all folds recorded!"
        )
        task_type = self.metadata.task_type

        # Check for extra fold keys
        extra_fold_keys = [k for k in self.results if k not in self.folds_keys]
        if extra_fold_keys:
            raise KeyError(
                f"Extra fold keys {extra_fold_keys} for task "
                f"{self.dataset_name} not allowed."
            )

        for fold_key in self.folds_keys:
            if fold_key not in self.results:
                raise KeyError(
                    f"Required fold data for fold '{fold_key}' "
                    f"for task {self.dataset_name} not found."
                )

            # Check for extra or missing keys inside each fold:
            # need params, scores, and data.
            req_subfold_keys = [self._SCORES_KEY, self._DATA_KEY, self._PARAMS_KEY]
            extra_subfold_keys = [
                k for k in self.results[fold_key] if k not in req_subfold_keys
            ]
            if self._UNCERTAINTY_KEY in extra_subfold_keys:
                extra_subfold_keys.remove(self._UNCERTAINTY_KEY)
            if extra_subfold_keys:
                raise KeyError(
                    f"Extra keys {extra_subfold_keys} for fold results of "
                    f"'{fold_key}' for task {self.dataset_name}  not allowed."
                )
            req_subfold_keys.append(self._UNCERTAINTY_KEY)
            for subkey in req_subfold_keys:
                fold_results = self.results[fold_key]
                if (
                    subkey is not self._UNCERTAINTY_KEY
                    and subkey not in fold_results
                ):
                    raise KeyError(
                        f"Required key '{subkey}' for task {self.dataset_name} "
                        f"not found for fold '{fold_key}'."
                    )
                if subkey == self._SCORES_KEY:
                    scores = self.results[fold_key][subkey]
                    metrics = REG_METRICS if task_type == REG_KEY else CLF_METRICS
                    for m in metrics:
                        if m not in scores:
                            raise KeyError(
                                f"Required score '{m}' for task "
                                f"{self.dataset_name} "
                                f"not found for '{fold_key}'."
                            )
                        elif not isinstance(scores[m], float):
                            raise TypeError(
                                f"Required score '{m}' for task "
                                f"{self.dataset_name} "
                                f"is not float-type for '{fold_key}'!"
                            )
                    extra_metrics = [k for k in scores if k not in metrics]
                    if extra_metrics:
                        raise KeyError(
                            f"Extra keys {extra_metrics} for fold scores of "
                            f"'{fold_key}' for task {self.dataset_name} "
                            f"not allowed."
                        )

                # results data indices are cast by json to be strings,
                # so must be converted to int
                elif subkey == self._DATA_KEY:
                    fold_data = self.results[fold_key].data

                    # Ensure all the indices are present with no
                    # extras for each fold
                    req_indices = set(self.validation[fold_key].test)
                    remaining_indices = copy.deepcopy(req_indices)
                    extra_indices = {}
                    if self.metadata.task_type == REG_KEY:
                        allowed_types = (float,)
                    else:
                        allowed_types = (bool, float)

                    for ix, datum in fold_data.items():
                        if ix not in req_indices:
                            extra_indices[ix] = datum
                        else:
                            if not isinstance(datum, allowed_types):
                                raise TypeError(
                                    f"Data point '{ix}: {datum}' has data type "
                                    f"{type(datum)} while required type is "
                                    f"{allowed_types} for task "
                                    f"{self.dataset_name} !"
                                )
                            if self.metadata.task_type == CLF_KEY:
                                if isinstance(datum, float):
                                    if datum < 0 or datum > 1:
                                        raise ValueError(
                                            f"Probability estimate '{ix}': {datum}"
                                            f"for task {self.dataset_name} outside "
                                            f"of range [0, 1]."
                                        )

                            remaining_indices.remove(ix)

                    if extra_indices and not remaining_indices:
                        raise ValueError(
                            f"{len(extra_indices)} extra indices for problem "
                            f"{self.dataset_name} are not allowed (found in "
                            f"{fold_key}: {remaining_indices}"
                        )
                    elif not extra_indices and remaining_indices:
                        raise ValueError(
                            f"{len(remaining_indices)} required indices "
                            f"for problem {self.dataset_name} not "
                            f"found for {fold_key}: {remaining_indices}"
                        )
                    elif extra_indices and remaining_indices:
                        raise ValueError(
                            f"{len(remaining_indices)} required indices "
                            f"for problem {self.dataset_name} not "
                            f"found and {len(extra_indices)} not "
                            f"allowed indices found for {fold_key}!"
                        )
                    else:
                        pass

                elif subkey == self._UNCERTAINTY_KEY:
                    if self._UNCERTAINTY_KEY in self.results[fold_key].keys():
                        uncertainties = self.results[fold_key][subkey]
                    else:
                        uncertainties = None
                    if uncertainties is not None:
                        std = uncertainties["std"]
                        ci = uncertainties["ci"]

                        if all(isinstance(s, float) for s in std):
                            if any(isinstance(c, float) for c in ci):
                                if not all(isinstance(c, float) for c in ci):
                                    raise ValueError(
                                        "std specified for all samples "
                                        "but ci not specified for some."
                                    )
                        else:
                            if any(isinstance(s, float) for s in std):
                                raise ValueError(
                                    "std is specified for some, but not for all."
                                )

                        if all(isinstance(c, float) for c in ci):
                            if any(isinstance(s, float) for s in std):
                                if not all(isinstance(s, float) for s in std):
                                    raise ValueError(
                                        "ci specified for all samples "
                                        "but ci not specified for some."
                                    )
                        else:
                            if any(isinstance(c, float) for c in ci):
                                raise ValueError(
                                    "ci is specified for some, but not for all."
                                )

                # Params key has no required form;
                # it is up to the model to determine it.

        logger.debug(f"Data for {self.dataset_name} successfully validated.")

    @property
    def scores(self):
        """Comprehensive score metrics for this task.

        Gets means, maxes, mins, and more distribution stats (across folds)
        for all scoring metrics defined for this task.

        There will be different scores for classification problems and
        regression problems.

        Returns:
            (dict): A dictionary of all the scores for this task.
        """
        metric_keys = (
            REG_METRICS if self.metadata.task_type == REG_KEY else CLF_METRICS
        )
        scores = {}
        self._check_all_folds_recorded("Cannot score unless all folds are recorded!")
        for mk in metric_keys:
            metric = {}

            # scores for a metric among all folds
            raw_metrics_on_folds = [
                self.results[fk][self._SCORES_KEY][mk]
                for fk in self.folds_map.values()
            ]
            for op in FOLD_DIST_METRICS:
                metric[op] = getattr(np, op)(raw_metrics_on_folds)
            scores[mk] = metric
        return RecursiveDotDict(scores)

    @property
    def is_recorded(self):
        """Determine what folds in the task are recorded.

        Returns:
            ({int: bool}): Keys are fold numbers, values are whether the
                fold is recorded or not.
        """
        is_recorded = {}
        for fnum, fkey in self.folds_map.items():
            if self.results[fkey][self._DATA_KEY]:
                is_recorded[fnum] = True
            else:
                is_recorded[fnum] = False
        return is_recorded

    @property
    def all_folds_recorded(self):
        """Determine if all folds are recorded.

        Returns:
            (bool): True if all folds are recorded, False otherwise.
        """
        return all([v for v in self.is_recorded.values()])

    @property
    def has_polymorphs(self):
        """Determine if a task's raw data contains polymorphs.

        Returns:
            (bool) If true, contains polymorphs.
        """
        checker_key = "pmg_composition"
        self._check_is_loaded()
        if self.metadata.input_type == "composition":
            stc = StrToComposition(target_col_id=checker_key, reduce=True)
            comps = stc.featurize_dataframe(self.df, "composition")[
                checker_key
            ].values
        elif self.metadata.input_type == "structure":
            stc = StructureToComposition(target_col_id=checker_key, reduce=True)
            comps = stc.featurize_dataframe(self.df, "structure")[checker_key].values
        else:
            raise ValueError(
                "Cannot check for polymorphs without input type in "
                "(structure, composition)!"
            )

        unique_comps = set(comps)
        if len(unique_comps) != len(comps):
            return True
        else:
            return False
