import datetime
import json

import numpy as np
import pandas as pd
from monty.json import MSONable

from matbench.metadata import mbv01_metadata
from matbench.constants import STRUCTURE_KEY, COMPOSITION_KEY, REG_KEY, CLF_KEY, MBV01_KEY
from matbench.util import RecursiveDotDict
from matbench.task import MatbenchTask

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


class MatbenchBenchmark(MSONable):

    _VERSION_KEY = "version"
    _USER_METADATA_KEY = "user_metadata"
    _TASKS_KEY = "tasks"
    _DATESTAMP_KEY = "datestamp"
    _DATESTAMP_FMT = "yyyy.MM.dd HH:mm:ss"
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
            available_tasks = metadata.keys()

        self.user_metadata = metadata
        self.tasks = RecursiveDotDict()

        for ds in available_tasks:
            self.tasks[ds] = MatbenchTask(ds, autoload=autoload)


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
        for t in self.tasks.values():
            t.load()

    @property
    def is_complete(self):
        """
        Determine if all available tasks are included in this benchmark.

        For matbench v0.1, this means all 13 tasks are in the benchmark.

        Returns:

        """
        for task in self.metadata:
            if task not in self.tasks:
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
        return all([t.all_folds_recorded for t in self.tasks.values()])


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
            for t in self.tasks.values():
                t.validate()
        else:
            raise ValueError("Not all tasks have all folds recorded!")


    def as_dict(self):
        tasksd = {
            mbt.dataset_name: mbt.as_dict() for mbt in self.tasks
        }

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            self._VERSION_KEY: "something",
            self._TASKS_KEY: tasksd,
            self._USER_METADATA_KEY: self.user_metadata,
            self._DATESTAMP_KEY: datetime.datetime.utcnow().strftime(self._DATESTAMP_FMT)
        }

        # to obtain a hash for this benchmark, immutify the dictionary and then stringify it
        d_hashable = immutify_dictionary(d)
        s_hashable = json.dumps(d_hashable).encode("utf-8")
        m = hashlib.sha256(s_hashable).hexdigest()
        d[self._HASH_KEY] = m
        return d

    @classmethod
    def from_dict(cls, d):
        required_keys = [cls._VERSION_KEY, cls._TASKS_KEY, cls._USER_METADATA_KEY, cls._DATESTAMP_KEY, cls._HASH_KEY]



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



if __name__ == "__main__":
    import hashlib

    d = {"q": [1,2,3], "b": {"c": "12", "d": 15, "e": np.asarray([4,5,6]), "f": {"z": 12, "a": [7, 8]}}, "c": 400}
    d_same = {"q": [1,2,3], "b": {"c": "12", "d": 15, "e": np.asarray([4,5,6]), "f": {"a": [7, 8], "z": 12}}, "c": 400}
    d_different = {"q": [1,2,3], "b": {"c": "12", "d": 15, "e": np.asarray([4,5,6]), "f": {"z": 12, "a": [7, 9]}}, "c": 400}




    import pprint

    # pprint.pprint(d_immut)

    for d in (d, d_same, d_different):
        d_immut = immutify_dictionary(d)
        pprint.pprint(d_immut)
        js = json.dumps(d_immut).encode("utf-8")
        print(hashlib.sha256(js).hexdigest())




    # print(hashlib.sha256(frozenset(d_immut)))
    # print(hash(frozenset(d_immut)))
