from matminer.datasets import get_all_dataset_info

from matbench.metadata import DATA_KEY, PARAMS_KEY
from matbench.util import RecursiveDotDict
from matbench.raw import get_kfold


class MatbenchTask:
    """
    The core interface for running a Matbench task and recording its results.
    """
    FOLD_MAPPING = {i: f"fold_{i}" for i in range(5)}

    def __init__(self, dataset_name):
        self.df = load(dataset_name)
        self.info = get_all_dataset_info(dataset_name)

        self.metadata = datasets[dataset_name]
        self.kfold = get_kfold(self.metadata.task_type)
        self.split_ix = tuple([s for s in self.kfold.split(self.df)])

        self.results = RecursiveDotDict({})
        self.scores = None

        self.is_recorded = {k: False for k in self.FOLD_MAPPING.keys()}

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

        fold_key = self.FOLD_MAPPING[fold_number]
        self.results[fold_key][DATA_KEY] = predictions
        self.results[fold_key][PARAMS_KEY] = params if params else {}
        self.is_recorded


    def score(self):
        if self._is_recorded:
            # score
            pass



 for _, test_ix in kfold.split(X=df, y=df[target]):
            if fold in fold_subset:
                logger.info("Training on fold index {}".format(fold))
                # Split, identify, and randomize test set
                test = df.iloc[test_ix].sample(frac=1)
                train = df[~df.index.isin(test.index)].sample(frac=1)
                self.fit(train, target)
                logger.info("Predicting fold index {}".format(fold))
                test = self.predict(test, ignore=ignore)
                results.append(test)
            fold += 1
        return results