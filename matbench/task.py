from matbench.constants import datasets

from matminer.datasets import get_all_dataset_info


class MatbenchTask:
    """
    The core interface for running a Matbench task and recording its results.
    """

    def __init__(self, dataset_name):
        self.df = load(dataset_name)
        self.info = get_all_dataset_info(dataset_name)

        self.metadata = datasets[dataset_name]
        self.kfold = get_kfold(self.metadata["task_type"])

        self.results = None

    @property
    def _is_recorded(self):
        pass

    def get_task_info(self):
        print(self.info)

    def get_train_and_val_data(self, fold_number):
        """
        The training + validation data. All model tuning and hyperparameter selection must be done on this data, NOT test data.

        Args:
            fold_number:

        Returns:

        """
        pass

    def get_test_data(self, fold_number):
        """
        The test data used for recording benchmarks.

        Args:
            fold_number:

        Returns:


        """
        pass

    def record(self, fold_number, predictions):
        """
        Record the test data.

        Args:
            fold_number (int): The fold number, 0-4.
            predictions ([float] or [bool] or np.ndarray): A list of predictions for fold number {fold_number}

        Returns:
            None
        """
        pass
