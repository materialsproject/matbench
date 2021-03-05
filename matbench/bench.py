from matbench.metadata import metadata
from matbench.constants import STRUCTURE_KEY, COMPOSITION_KEY, REG_KEY, CLF_KEY
from matbench.util import RecursiveDotDict
from matbench.task import MatbenchTask

from monty.json import MSONable

'''
Core functions for benchmarking.
---


Final, bedrock truth results JSON format:

{
    "raw": {
        "matbench_steels": {fold0: {data: { index1: prediction1, index2: prediction2...}, params}, fold1: ...}
        "matbench_dielectric": {fold0: {data: { index1: prediction1, index2: prediction2...}, params}, fold1: ...}
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

    def __init__(self, autoload=False, subset=None):

        if subset:
            not_datasets = [k for k in subset if k not in metadata]
            if not_datasets:
                raise KeyError(f"Some tasks in {subset} are not matbench datasets! Remove {not_datasets}.")
            else:
                available_tasks = subset
        else:
            available_tasks = metadata.keys()

        self.metadata = metadata
        self.tasks = RecursiveDotDict()

        for ds in available_tasks:
            self.tasks[ds] = MatbenchTask(ds, autoload=autoload)


    @classmethod
    def from_preset(cls, preset_name, autoload=False):
        if preset_name == STRUCTURE_KEY:
            available_tasks = [k for k, v in metadata.items() if v.input_type == STRUCTURE_KEY]
        elif preset_name == COMPOSITION_KEY:
            available_tasks = [k for k, v in metadata.items() if v.input_type == COMPOSITION_KEY]
        elif preset_name == REG_KEY:
            available_tasks = [k for k, v in metadata.items() if v.task_type == REG_KEY]
        elif preset_name == CLF_KEY:
            available_tasks = [k for k, v in metadata.items() if v.task_type == CLF_KEY]
        else:
            raise ValueError(f"Preset name '{preset_name}' not recognized! Select from {[STRUCTURE_KEY, COMPOSITION_KEY, CLF_KEY, REG_KEY]}")
        return cls(autoload=autoload, subset=available_tasks)

    def add_metadata(self, metadata):
        """
        Add freeform information about this run to the object (and subsequent json)

        Args:
            metadata:

        Returns:

        """
        pass

    @property
    def is_complete(self):
        """
        All 13 available tasks are included in this benchmark.

        Returns:

        """
        pass

    @property
    def is_recorded(self):
        """
        All tasks in this benchmark (whether or not it includes all problems) are recorded.

        Returns:

        """
        pass

    def as_dict(self):
        d = {
            mbt.dataset_name for mbt in self.tasks
        }
        return d

    def from_dict(cls, d):
        pass





