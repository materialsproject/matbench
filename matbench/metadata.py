import os
import json

from matminer.datasets.utils import _load_dataset_dict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_JSON_PATH = os.path.join(THIS_DIR, "datasets.json")
VALIDATION_JSON_PATH = os.path.join(THIS_DIR, "validation.json")

REG_KEY = "regression"
CLF_KEY = "classification"

MATMINER_DATASET_METADATA = _load_dataset_dict()


class RecurisveDotDict(dict):
    """
    Adapted from user Curt Hagenlocher from
    https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, RecurisveDotDict):
            value = RecurisveDotDict(value)
        super(RecurisveDotDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, RecurisveDotDict.MARKER)
        if found is RecurisveDotDict.MARKER:
            found = RecurisveDotDict()
            super(RecurisveDotDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


with open(DATASETS_JSON_PATH, "r") as f:
    metadata = json.load(f)

for d in metadata.keys():
    metadata[d].update(MATMINER_DATASET_METADATA[d])

with open(VALIDATION_JSON_PATH, "r") as f:
    validation_metadata = json.load(f)

metadata = RecurisveDotDict(metadata)
