import os
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_JSON_PATH = os.path.join(THIS_DIR, "datasets.json")
VALIDATION_JSON_PATH = os.path.join(THIS_DIR, "validation.json")


with open(DATASETS_JSON_PATH, "r") as f:
    DATASETS = json.load(f)


with open(VALIDATION_JSON_PATH ,"r") as f:
    VALIDATION = json.load(f)

REG_KEY = "regression"
CLF_KEY = "classification"
