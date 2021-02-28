import os
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_JSON_PATH = os.path.join(THIS_DIR, "datasets.json")
VALIDATION_JSON_PATH = os.path.join(THIS_DIR, "validation.json")


with open(DATASETS_JSON_PATH, "r") as f:
    DATASETS = json.load(f)

with open(VALIDATION_JSON_PATH, "r") as f:
    VALIDATION = json.load(f)

REG_KEY = "regression"
CLF_KEY = "classification"


class RecurisveDotDict(dict):
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


datasets_metadata = RecurisveDotDict(DATASETS)


from matminer.datasets import g