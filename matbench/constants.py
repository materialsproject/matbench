import math
import os

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)

VERSION = "0.1.0"

# paths for metadata and validation splits
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MBV01_DATASET_METADATA_PATH = os.path.join(
    THIS_DIR, "matbench_v0.1_dataset_metadata.json"
)
MBV01_VALIDATION_DATA_PATH = os.path.join(THIS_DIR, "matbench_v0.1_validation.json")


MBV01_KEY = "matbench_v0.1"


# keys for validation files
VALIDATION_SPLIT_KEY = "splits"
VALIDATION_METADATA_KEY = "metadata"
TRAIN_KEY = "train"
TEST_KEY = "test"

# universal keys
MBID_KEY = "mbid"
REG_KEY = "regression"
CLF_KEY = "classification"
STRUCTURE_KEY = "structure"
COMPOSITION_KEY = "composition"


# scoring per task on a single fold
REG_METRICS = ["mae", "rmse", "mape", "max_error"]
CLF_METRICS = ["accuracy", "balanced_accuracy", "f1", "rocauc"]
METRIC_MAP = {
    "mae": mean_absolute_error,
    "rmse": lambda true, pred: math.sqrt(mean_squared_error(true, pred)),
    "mape": mean_absolute_percentage_error,
    "max_error": max_error,
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "rocauc": roc_auc_score,
}

# scoring on a single task among folds
FOLD_DIST_METRICS = ["mean", "max", "min", "std"]
