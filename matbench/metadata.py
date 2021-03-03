import os
import json

from matminer.datasets.utils import _load_dataset_dict

from matbench.util import RecursiveDotDict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_JSON_PATH = os.path.join(THIS_DIR, "datasets.json")
VALIDATION_JSON_PATH = os.path.join(THIS_DIR, "validation.json")

REG_KEY = "regression"
CLF_KEY = "classification"


# per task keys
DATA_KEY = "data"
PARAMS_KEY = "parameters"

MATMINER_DATASET_METADATA = _load_dataset_dict()


with open(DATASETS_JSON_PATH, "r") as f:
    metadata = json.load(f)

for d in metadata.keys():
    metadata[d].update(MATMINER_DATASET_METADATA[d])

with open(VALIDATION_JSON_PATH, "r") as f:
    validation_metadata = json.load(f)

metadata = RecurisveDotDict(metadata)
validation_metadata = RecurisveDotDict(validation_metadata)