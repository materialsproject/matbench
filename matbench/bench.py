"""
Core class for benchmarking.
"""

import datetime
import hashlib
import json
import logging
import pprint
import traceback

import numpy as np
import pandas as pd
from monty.json import MSONable

from matbench.constants import (
    CLF_KEY,
    COMPOSITION_KEY,
    MBV01_KEY,
    REG_KEY,
    STRUCTURE_KEY,
    VERSION,
)
from matbench.metadata import mbv01_metadata
from matbench.task import MatbenchTask
from matbench.util import MSONable2File, RecursiveDotDict, initialize_logger

logger = initialize_logger(logger_name="matbench", level=logging.INFO)


class MatbenchBenchmark(MSONable, MSONable2File):
    """The core class for benchmarking with Matbench.

    MatbenchBenchmark is capable of benchmarking and validating arbitrary
    materials science benchmarks. It is a container class for sets of
    MatbenchTasks, objects which provide predetermined sets of
    training/validation and testing data for any algorithm to benchmark
    with. MatbenchBenchmark can also give summaries of entire complex
    benchmarks, including access to individual score statistics for
    each metric.

    MatbenchBenchmark can run any benchmark as long as it has a corresponding
    benchmark name key. Matbench v0.1 ("matbench_v0.1") is the only benchmark
    currently configured for use with MatbenchBenchmark.

    MatbenchBenchmark is capable of running benchmark subsets; for example,
    only 3 of the 13 available Matbench v0.1 problems.

    See the documentation for more details.

    Attributes:
        benchmark_name (str): The benchmark name, defaults to the original
            Matbench v0.1 "matbench_v0.1". Should have an associated
            validation file in order for the MatbenchTasks to work
            correctly.
        metadata (dict): The corresponding metadata file for this benchmark,
            which defines the basic configuration for each task. See
            matbench_v0.1_validation for an example. Each dataset
            has the same required keys in order to work correctly.
        user_metadata (dict): Any metadata about the algorithm or benchmark
            that the user wants to keep as part of the benchmark file.
        tasks_map ({str: MatbenchTask}): A mapping of task name to the
            corresponding MatbenchTask object.

        <<task_names>> (MatbenchTask): Access any task obj via
            MatbenchTask.<<task_name>>. For example:

            mb = MatbenchBenchmark()
            mb.matbench_dielectric

            <<MatbenchTask object>>
    """

    _VERSION_KEY = "version"
    _BENCHMARK_KEY = "benchmark_name"
    _USER_METADATA_KEY = "user_metadata"
    _TASKS_KEY = "tasks"
    _DATESTAMP_KEY = "datestamp"
    _DATESTAMP_FMT = "%Y.%m.%d %H:%M.%S"
    _HASH_KEY = "hash"

    def __init__(self, benchmark=MBV01_KEY, autoload=False, subset=None):
        """

        Args:
            benchmark (str): The name of the benchmark. Only supported benchmark
                currently is "matbench_v0.1", though more will be added in the
                future.
            autoload (bool): If True, automatically load the dataset into memory
                For a full benchmark, this can take some time. If False, you'll
                need to load each task with .load before you can access the raw
                data.
            subset ([str]): A list of task names to use as a subset of a full
                benchmark. Only the named tasks will be contained in the class.
                Must correspond to the metadata file defined by the benchmark
                name.
        """

        if benchmark == MBV01_KEY:
            self.benchmark_name = MBV01_KEY
            self.metadata = mbv01_metadata
        else:
            raise ValueError(
                f"Only '{MBV01_KEY}' available. No other benchmarks defined!"
            )

        if subset:
            not_datasets = [k for k in subset if k not in self.metadata]
            if not_datasets:
                raise KeyError(
                    f"Some tasks in {subset} are not benchmark="
                    f"'{self.benchmark_name}' datasets! Remove {not_datasets}."
                )
            else:
                available_tasks = subset
        else:
            available_tasks = self.metadata.keys()

        self.user_metadata = {}
        self.tasks_map = RecursiveDotDict()

        for ds in available_tasks:
            self.tasks_map[ds] = MatbenchTask(
                ds, autoload=autoload, benchmark=self.benchmark_name
            )

        logger.info(
            f"Created benchmark '{benchmark}' "
            f"with {len(available_tasks)} tasks: \n"
            f"{pprint.pformat(list(available_tasks))}"
        )

    def __getattr__(self, item):
        """
        Enable MatbenchBenchmark.task_name behavior.

        Args:
            item (str): The name of the attr.

        Returns:
            The attr, if not in the metadata defined by the benchmark
            If the attr is a task name, returns that MatBenchTask object.

        """
        if item in self.metadata:
            return self.tasks_map[item]
        else:
            return self.__getattribute__(self, item)

    @classmethod
    def from_preset(cls, benchmark, preset_name, autoload=False):
        """
        The following presets are defined for each benchmark:

        benchmark: 'matbench_v0.1':
            - preset: 'structure' - Only structure problems
            - preset: 'composition' - Only composition problems
            - preset: 'regression' - Only regression problems
            - preset: 'classification' - Only classification problems
            - preset: 'all' - All problems in matbench v0.1

        Args:
            benchmark_name
            preset_name
            autoload:

        Returns:
            (MatbenchBenchmark object)

        """
        all_key = "all"
        if benchmark == MBV01_KEY:
            if preset_name == STRUCTURE_KEY:
                available_tasks = [
                    k
                    for k, v in mbv01_metadata.items()
                    if v.input_type == STRUCTURE_KEY
                ]
            elif preset_name == COMPOSITION_KEY:
                available_tasks = [
                    k
                    for k, v in mbv01_metadata.items()
                    if v.input_type == COMPOSITION_KEY
                ]
            elif preset_name == REG_KEY:
                available_tasks = [
                    k for k, v in mbv01_metadata.items() if v.task_type == REG_KEY
                ]
            elif preset_name == CLF_KEY:
                available_tasks = [
                    k for k, v in mbv01_metadata.items() if v.task_type == CLF_KEY
                ]
            elif preset_name == all_key:
                available_tasks = [k for k, v in mbv01_metadata.items()]
            else:
                valid_keys = [
                    STRUCTURE_KEY,
                    COMPOSITION_KEY,
                    CLF_KEY,
                    REG_KEY,
                    all_key,
                ]
                raise ValueError(
                    f"Preset name '{preset_name}' not recognized for "
                    f"benchmark '{MBV01_KEY}'! Select from "
                    f"{valid_keys}"
                )
        else:
            raise ValueError(
                f"Only '{MBV01_KEY}' available. No other benchmarks defined!"
            )

        return cls(benchmark=benchmark, autoload=autoload, subset=available_tasks)

    @classmethod
    def from_dict(cls, d):
        """Create a MatbenchBenchmark object from a dictionary.

        Args:
            d (dict):

        Returns:
            (MatbenchBenchmark)

        """
        required_keys = [
            "@module",
            "@class",
            cls._VERSION_KEY,
            cls._BENCHMARK_KEY,
            cls._TASKS_KEY,
            cls._USER_METADATA_KEY,
            cls._DATESTAMP_KEY,
            cls._HASH_KEY,
        ]

        missing_keys = []
        for k in required_keys:
            if k not in d:
                missing_keys.append(k)

        extra_keys = []
        for k in d:
            if k not in required_keys:
                extra_keys.append(k)

        if missing_keys and not extra_keys:
            raise ValueError(
                f"Required keys {missing_keys} for {cls.__class__.__name__} "
                f"not found!"
            )
        elif not missing_keys and extra_keys:
            raise ValueError(
                f"Extra keys {extra_keys} for {cls.__class__.__name__} " f"present!"
            )
        elif missing_keys and extra_keys:
            raise ValueError(
                f"Missing required keys {missing_keys} and extra keys "
                f"{extra_keys} present!"
            )

        # Check all tasks to make sure their benchmark name is matching in the
        # benchmark and in the tasks
        not_matching_bench = []
        for t_dict in d[cls._TASKS_KEY].values():
            if t_dict[MatbenchTask._BENCHMARK_KEY] != d[cls._BENCHMARK_KEY]:
                not_matching_bench.append(t_dict[MatbenchTask._DATASET_KEY])
        if not_matching_bench:
            raise ValueError(
                f"Tasks {not_matching_bench} do not have a benchmark name "
                f"matching the benchmark ({d[cls._BENCHMARK_KEY]})!"
            )

        # Ensure the hash is matching, i.e., the data was not modified after
        # matbench got done with it
        m_from_dict = d.pop(cls._HASH_KEY)
        m = hash_dictionary(d)
        if m != m_from_dict:
            raise ValueError(
                f"Hash of dictionary does not match it's reported value! {m} "
                f"!= {m_from_dict} . Was the data modified after saving?)"
            )

        # Check to see if any tasks have task names not matching their key
        # names in the benchmark
        not_matching_tasks = []
        for task_name, task_info in d[cls._TASKS_KEY].items():
            key_as_per_task = task_info[MatbenchTask._DATASET_KEY]
            if task_name != key_as_per_task:
                not_matching_tasks.append((task_name, key_as_per_task))
        if not_matching_tasks:
            raise ValueError(
                f"Task names in benchmark and task names in tasks not "
                f"matching: {not_matching_tasks}"
            )

        # Warn if versions are not matching
        if d[cls._VERSION_KEY] != VERSION:
            logger.warning(
                f"Warning! Versions not matching: "
                f"(data file has version {d[cls._VERSION_KEY]}, "
                f"this package is {VERSION})."
            )

        return cls._from_args(
            benchmark_name=d[cls._BENCHMARK_KEY],
            tasks_dict=d[cls._TASKS_KEY],
            user_metadata=d[cls._USER_METADATA_KEY],
        )

    @classmethod
    def _from_args(cls, benchmark_name, tasks_dict, user_metadata):
        """Create a MatbenchBenchmark object from arguments

        Args:
            benchmark_name (str): name of the benchmark
            tasks_dict (dict): formatted dict of task data
            user_metadata (dict): freeform user metadata

        Returns:
            (MatbenchBenchmark)
        """
        subset = list(tasks_dict.keys())
        obj = cls(benchmark=benchmark_name, autoload=False, subset=subset)
        obj.tasks_map = RecursiveDotDict(
            {
                t_name: MatbenchTask.from_dict(t_dict)
                for t_name, t_dict in tasks_dict.items()
            }
        )

        logger.warning(
            "To add new data to this benchmark, the "
            "benchmark must be loaded with .load(). Alternatively, "
            "load individual tasks with MatbenchTask.load()."
        )

        # MatbenchTask automatically validates files during its from_dict
        obj.user_metadata = user_metadata
        return obj

    def as_dict(self):
        """Overriden from MSONable.as_dict, get dict repr of this obj

        Returns:
            (dict)

        """
        tasksd = {mbt.dataset_name: mbt.as_dict() for mbt in self.tasks}

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            self._VERSION_KEY: VERSION,
            self._TASKS_KEY: tasksd,
            self._USER_METADATA_KEY: self.user_metadata,
            self._BENCHMARK_KEY: self.benchmark_name,
            self._DATESTAMP_KEY: datetime.datetime.utcnow().strftime(
                self._DATESTAMP_FMT
            ),
        }

        # to obtain a hash for this benchmark, immutify the dictionary
        # and then stringify it
        d[self._HASH_KEY] = hash_dictionary(d)
        return d

    def get_info(self):
        """Log info about the benchmark to the respective logging handlers.

        Returns:
            None
        """
        logger.info(self.info)

    def add_metadata(self, metadata):
        """Add freeform information about this run to the object
        (and subsequent json), accessible thru the
        'user_metadata' attr.

        Args:
            metadata (dict)

        Returns:
            None
        """

        if not isinstance(metadata, dict):
            raise TypeError("User metadata must be reducible to dict format.")
        self.user_metadata = metadata
        logger.info("User metadata added successfully!")

    def load(self):
        """Load all tasks in this benchmark.
        Returns:
            None
        """
        for t in self.tasks:
            t.load()

    def validate(self):
        """Run validation on each task in this benchmark.

        Returns:
            ({str: str}): dict of errors, if they exist

        """
        errors = {}
        for t, t_obj in self.tasks_map.items():
            try:
                t_obj.validate()
            except BaseException:
                errors[t] = traceback.format_exc()
        return errors

    @property
    def tasks(self):
        """Return the tasks as a list.

        Returns:
            ([MatbenchTask]): A list of matbench tasks in this benchmark
        """
        return self.tasks_map.values()

    @property
    def scores(self):
        """Get all score metrics for all tasks as a dictionary.

        Returns:
            (RecursiveDotDict): A nested dictionary-like object of scores
                for each task.

        """
        return RecursiveDotDict({t.dataset_name: t.scores for t in self.tasks})

    @property
    def info(self):
        """Get a formatted string of into about this benchmark and its current
        state.

        Returns:
            (str)

        """

        complete = self.is_complete
        recorded = self.is_recorded
        valid = self.is_valid

        s = ""
        s += (
            f"\nMatbench package {VERSION} running benchmark "
            f"'{self.benchmark_name}'"
        )
        s += f"\n\tis complete: {complete}"
        s += f"\n\tis recorded: {recorded}"
        s += f"\n\tis valid: {valid}"

        if not recorded:
            s += (
                "\n\n Benchmark is not fully recorded; limited information " "shown."
            )
        if not valid:
            s += "\n\n Benchmark is not valid; limited information shown."

        if not valid or not recorded:
            s += "\n\nTasks:"
            for t in self.tasks_map:
                s += f"\n\t- '{t.dataset_name}: recorded={t.all_folds_recorded}"

        if valid and recorded:
            s += "\n\nResults:\n"
            for t in self.tasks:

                if t.metadata.task_type == REG_KEY:
                    score_text = (
                        f"MAE mean: " f"{self.scores[t.dataset_name].mae.mean}"
                    )
                else:
                    score_text = (
                        f"ROCAUC mean: " f"{self.scores[t.dataset_name].rocauc.mean}"
                    )
                s += f"\n\t- '{t.dataset_name}' {score_text}"

        return s

    @property
    def is_complete(self):
        """Determine if all available tasks are included in this benchmark.

        For matbench v0.1, this means all 13 tasks are in the benchmark.

        Returns:
            (bool)

        """
        for task in self.metadata:
            if task not in self.tasks_map:
                return False
        else:
            return True

    @property
    def is_recorded(self):
        """All tasks in this benchmark (whether or not it includes all tasks in
        the benchmark set) are recorded.

        Returns:
            (bool): True if all tasks (even if only a subset of all matbench)
            for this benchmark are recorded.

        """
        return all([t.all_folds_recorded for t in self.tasks_map.values()])

    @property
    def is_valid(self):
        """Checks all tasks are recorded and valid, as per each task's
        validation procedure.

        Can take some time, especially if the tasks are not already
        loaded into memory.

        Returns:
            (bool): True if all tasks are valid
        """
        errors = self.validate()
        if errors:
            formatted_errors = pprint.pformat(errors)
            logger.critical(
                f"Benchmark has errors! " f"Errors:\n {formatted_errors}"
            )
            return False
        else:
            return True


def immutify_dictionary(d):
    """Create a frozenset-esque deterministic, unique representation of
    a nested dict.


    Args:
        d (dict): The dictionary to be immutified. Key are always strings.
            Values can be arrays of various numpy or pandas types, strings,
            numpy primitives, python native numbers, or dictionaries with
            the same format.

    Returns:
        (dict): A sorted, deterministic, unique representation of the
            dictionary.
    """
    d_new = {}
    for k, v in d.items():
        if isinstance(v, (np.ndarray, pd.Series)):
            d_new[k] = tuple(v.tolist())
        elif isinstance(v, list):
            d_new[k] = tuple(v)
        elif isinstance(v, dict):
            d_new[k] = immutify_dictionary(v)
        else:
            # convert numpy types to native
            if hasattr(v, "dtype"):
                d_new[k] = v.item()
            else:
                d_new[k] = v
    # dictionaries are ordered in python 3.6+
    return dict(sorted(d_new.items(), key=lambda item: item[0]))


def hash_dictionary(d):
    """Hash a dictionary that can be immutified with immutify_dictionary.

    Order of the keys does not matter, as dictionary becomes deterministically
    immutified. Dictionary can be nested.

    Args:
        d (dict): The dictionary to hash.

    Returns:
        (str): base16 encoded hash of the dictionary.
    """
    d_hashable = immutify_dictionary(d)
    s_hashable = json.dumps(d_hashable).encode("utf-8")
    m = hashlib.sha256(s_hashable).hexdigest()
    return m
