import os
import math

from sklearn.metrics import \
    mean_absolute_error, \
    mean_squared_error, \
    mean_absolute_percentage_error, \
    max_error, \
    roc_auc_score, \
    accuracy_score, \
    balanced_accuracy_score, \
    f1_score


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_JSON_PATH = os.path.join(THIS_DIR, "datasets.json")
VALIDATION_JSON_PATH = os.path.join(THIS_DIR, "validation.json")

MBID_KEY = "mbid"

REG_KEY = "regression"
CLF_KEY = "classification"


# per task keys
DATA_KEY = "data"
PARAMS_KEY = "parameters"
SCORES_KEY = "scores"

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
    "rocauc": roc_auc_score
}

# scoring on a single task among folds
FOLD_DIST_METRICS = ["mean", "max", "min", "std"]