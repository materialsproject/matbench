import json

from matminer.datasets.utils import _load_dataset_dict

from matbench.constants import DATASETS_JSON_PATH, VALIDATION_JSON_PATH
from matbench.util import RecursiveDotDict

MATMINER_DATASET_METADATA = _load_dataset_dict()

with open(DATASETS_JSON_PATH, "r") as f:
    metadata = json.load(f)

for d in metadata.keys():
    metadata[d].update(MATMINER_DATASET_METADATA[d])

with open(VALIDATION_JSON_PATH, "r") as f:
    validation_metadata = json.load(f)

metadata = RecursiveDotDict(metadata)
validation_metadata = RecursiveDotDict(validation_metadata)