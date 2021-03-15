import datetime
import json
import traceback
import hashlib

import numpy as np
import pandas as pd
from monty.json import MSONable

from matbench.metadata import mbv01_metadata
from matbench.constants import STRUCTURE_KEY, COMPOSITION_KEY, REG_KEY, CLF_KEY, MBV01_KEY
from matbench.util import RecursiveDotDict, MSONable2File
from matbench.task import MatbenchTask
from matbench.version import version

'''
Core functions for benchmarking.
---


Final, bedrock truth results JSON format:

{
    "raw": {
        "matbench_steels": {fold0: {data: { matbench_id 1: prediction1, matbench_id 2: prediction2...}, params}, fold1: ...}
        "matbench_dielectric": {fold0: {data: { matbench_id 1: prediction1, matbench_id 2: prediction2...}, params}, fold1: ...}
        ...
    },
    "processed": {
        fold0: {rmse: rmse1, mae: mae1, mape: mape1, ...}
        fold1: {rmse: rmse2, mae: mae2, mape: mape2, ...}
        ...
        mean: {rmse: mean_rmse, mae: mean_mae, ...}
        std: {rmse: std_rmse, mae: std_mae, ...}
        ...
    },
    "metadata": {
        "is_full" : True if it was a full benchmark
    }
    
    
    
Usage should be something like:


mb = MatbenchBenchmark()

for task in mb.tasks:
    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        my_model.train_and_validate(train_inputs, train_outputs)
        
        test_inputs = task.get_test_data(fold)
        predictions = my_model.predict(test_inputs)
        
        task.record(predictions, params=my_model.hyperparams)  


# holds all the scoring info
mb.scores

# access the raw results of an individual test
mb.tasks.matbench_dielectric.results.fold0.data
mb.tasks.matbench_dielectric.results.fold0.params
mb.tasks.matbench_dielectric.results.fold0.scores


# access the individual scores of a fold
mb.tasks.matbench_steels.scores.fold0

'''


class MatbenchBenchmark(MSONable, MSONable2File):

    _VERSION_KEY = "version"
    _BENCHMARK_KEY = "benchmark_name"
    _USER_METADATA_KEY = "user_metadata"
    _TASKS_KEY = "tasks"
    _DATESTAMP_KEY = "datestamp"
    _DATESTAMP_FMT = "%Y.%m.%d %H:%M.%S"
    _HASH_KEY = "hash"

    def __init__(self, benchmark=MBV01_KEY, autoload=False, subset=None):

        if benchmark == MBV01_KEY:
            self.benchmark_name = MBV01_KEY
            self.metadata = mbv01_metadata
        else:
            raise ValueError(f"Only '{MBV01_KEY}' available. No other benchmarks defined!")

        if subset:
            not_datasets = [k for k in subset if k not in self.metadata]
            if not_datasets:
                raise KeyError(f"Some tasks in {subset} are not benchmark='{self.benchmark_name}' datasets! Remove {not_datasets}.")
            else:
                available_tasks = subset
        else:
            available_tasks = self.metadata.keys()

        self.user_metadata = self.metadata
        self.tasks_map = RecursiveDotDict()

        for ds in available_tasks:
            self.tasks_map[ds] = MatbenchTask(ds, autoload=autoload, benchmark=self.benchmark_name)

        self.version = version


    @property
    def tasks(self):
        return self.tasks_map.values()

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
                available_tasks = [k for k, v in mbv01_metadata.items() if v.input_type == STRUCTURE_KEY]
            elif preset_name == COMPOSITION_KEY:
                available_tasks = [k for k, v in mbv01_metadata.items() if v.input_type == COMPOSITION_KEY]
            elif preset_name == REG_KEY:
                available_tasks = [k for k, v in mbv01_metadata.items() if v.task_type == REG_KEY]
            elif preset_name == CLF_KEY:
                available_tasks = [k for k, v in mbv01_metadata.items() if v.task_type == CLF_KEY]
            elif preset_name == all_key:
                available_tasks = [k for k, v in mbv01_metadata.items()]
            else:
                raise ValueError(f"Preset name '{preset_name}' not recognized for benchmark '{MBV01_KEY}'! Select from {[STRUCTURE_KEY, COMPOSITION_KEY, CLF_KEY, REG_KEY, all_key]}")
        else:
            raise ValueError(f"Only '{MBV01_KEY}' available. No other benchmarks defined!")

        return cls(benchmark=benchmark, autoload=autoload, subset=available_tasks)

    @property
    def scores(self):
        return {t.dataset_name: t.scores for t in self.tasks_map}


    @property
    def info(self):

        complete = self.is_complete
        recorded = self.is_recorded
        valid = self.is_valid
        scores = self.scores

        s = ""
        s += f"\nMatbench package {version} running benchmark '{self.benchmark_name}'"
        s += f"\n\tis complete: {complete}"
        s += f"\n\tis recorded: {recorded}"
        s += f"\n\tis valid: {valid}"

        if not recorded:
            s += f"\n\n Benchmark is not fully recorded; limited information shown."
        if not valid:
            s += f"\n\n Benchmark is not valid; limited informatiomn shown."

        if not valid or not recorded:
            s += f"\n\nTasks:"
            for t in self.tasks_map:
                s += f"\n\t- '{t.dataset_name}: recorded={t.all_folds_recorded}"

        if valid and recorded:
            s += f"\n\nResults:\n"
            for t in self.tasks_map:
                s += f"\n\t- '{t.dataset_name}' MAE: {self.scores[t.dataset_name]['mae']['mean']}"
        return s

    def get_info(self):
        print(self.info)


    def add_metadata(self, metadata):
        """
        Add freeform information about this run to the object (and subsequent json),
        accessible thru the 'user_metadata' attr.

        Args:
            metadata (dict)

        Returns:
            None
        """

        if not isinstance(metadata, dict):
            raise TypeError("User metadata must be reducible to dict format.")
        self.user_metadata = metadata
        print("User metadata added successfully!")

    def load(self):
        """
        Load all tasks in this benchmark.
        Returns:

        """
        for t in self.tasks:
            t.load()

    @property
    def is_complete(self):
        """
        Determine if all available tasks are included in this benchmark.

        For matbench v0.1, this means all 13 tasks are in the benchmark.

        Returns:

        """
        for task in self.metadata:
            if task not in self.tasks_map:
                return False
        else:
            return True

    @property
    def is_recorded(self):
        """
        All tasks in this benchmark (whether or not it includes all tasks in the benchmark set) are recorded.

        Returns:
            (bool): True if all tasks (even if only a subset of all matbench) for this benchmark are recorded.

        """
        return all([t.all_folds_recorded for t in self.tasks_map.values()])


    @property
    def is_valid(self):
        """
        Checks all tasks are recorded and valid, as per each task's validation procedure.

        Can take some time, especially if the tasks are not already loaded into memory.

        Returns:
            (bool): True if all tasks are valid
        """
        self.load()
        if self.is_recorded:
            errors = {}
            for t, t_obj in self.tasks_map.items():
                try:
                    t_obj.validate()
                except BaseException as E:
                    errors[t] = traceback.format_exc()

            if errors:
                # todo: replace with logging
                print("Errors found")
                print(errors)
                return False
            else:
                return True
        else:
            print("Not all tasks have all folds recorded!")
            return False

    def as_dict(self):
        tasksd = {
            mbt.dataset_name: mbt.as_dict() for mbt in self.tasks
        }

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            self._VERSION_KEY: self.version,
            self._TASKS_KEY: tasksd,
            self._USER_METADATA_KEY: self.user_metadata,
            self._BENCHMARK_KEY: self.benchmark_name,
            self._DATESTAMP_KEY: datetime.datetime.utcnow().strftime(self._DATESTAMP_FMT)
        }

        # to obtain a hash for this benchmark, immutify the dictionary and then stringify it
        d[self._HASH_KEY] = hash_dictionary(d)
        return d

    @classmethod
    def from_dict(cls, d):
        required_keys = ["@module", "@class", cls._VERSION_KEY, cls._BENCHMARK_KEY, cls._TASKS_KEY, cls._USER_METADATA_KEY, cls._DATESTAMP_KEY, cls._HASH_KEY]

        missing_keys = []
        for k in required_keys:
            if k not in d:
                missing_keys.append(k)

        extra_keys = []
        for k in d:
            if k not in required_keys:
                extra_keys.append(k)

        if missing_keys and not extra_keys:
            raise ValueError(f"Required keys {missing_keys} for {cls.__class__.__name__} not found!")
        elif not missing_keys and extra_keys:
            raise ValueError(f"Extra keys {extra_keys} for {cls.__class__.__name__} present!")
        elif missing_keys and extra_keys:
            raise ValueError(f"Missing required keys {missing_keys} and extra keys {extra_keys} present!")

        # Check all tasks to make sure their benchmark name is matching in the benchmark and in the tasks
        not_matching_bench = []
        for t_dict in d[cls._TASKS_KEY].values():
            if t_dict[MatbenchTask._BENCHMARK_KEY] != d[cls._BENCHMARK_KEY]:
                not_matching_bench.append(t_dict[MatbenchTask._DATASET_KEY])
        if not_matching_bench:
            raise ValueError(f"Tasks {not_matching_bench} do not have a benchmark name matching the benchmark ({d[cls._BENCHMARK_KEY]})!")

        # Ensure the hash is matching, i.e., the data was not modified after matbench got done with it
        m_from_dict = d.pop(cls._HASH_KEY)
        m = hash_dictionary(d)
        if m != m_from_dict:
            raise ValueError(f"Hash of dictionary does not match it's reported value! {m} != {m_from_dict} . Was the data modified after saving?)")

        # Check to see if any tasks have task names not matching their key names in the benchmark
        not_matching_tasks = []
        for task_name, task_info in d[cls._TASKS_KEY].items():
            key_as_per_task = task_info[MatbenchTask._DATASET_KEY]
            if task_name != key_as_per_task:
                not_matching_tasks.append((task_name, key_as_per_task))
        if not_matching_tasks:
            raise ValueError(f"Task names in benchmark and task names in tasks not matching: {not_matching_tasks}")


        # Warn if versions are not matching
        if d[cls._VERSION_KEY] != version:
            #todo: replace with logging
            print(f"Warning! Versions not matching: (data file has version {d[cls._VERSION_KEY]}, this package is {version}).")

        return cls._from_args(benchmark_name=d[cls._BENCHMARK_KEY], tasks_dict=d[cls._TASKS_KEY], user_metadata=d[cls._USER_METADATA_KEY])

    @classmethod
    def _from_args(cls, benchmark_name, tasks_dict, user_metadata):
        subset = list(tasks_dict.keys())
        obj = cls(benchmark=benchmark_name, autoload=False, subset=subset)
        obj.tasks_map = RecursiveDotDict({t_name: MatbenchTask.from_dict(t_dict) for t_name, t_dict in tasks_dict.items()})

        # todo: change to logging
        print("To add new data to this benchmark, the benchmark must be loaded with .load().")
        # MatbenchTask automatically validates files during its from_dict
        obj.user_metadata = user_metadata
        return obj



def immutify_dictionary(d):
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
    d_hashable = immutify_dictionary(d)
    s_hashable = json.dumps(d_hashable).encode("utf-8")
    m = hashlib.sha256(s_hashable).hexdigest()
    return m
