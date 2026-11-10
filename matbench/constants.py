"""
Source of ground truth for all constants used across matbench.
"""
import os

# single place for defining version
VERSION = "0.6"

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

# threshold for converting probabilities to labels
CLF_THRESH = 0.5

# scoring on a single task among folds
FOLD_DIST_METRICS = ["mean", "max", "min", "std"]
