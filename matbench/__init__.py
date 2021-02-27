import json


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_JSON_PATH = os.path.join(THIS_DIR, "datasets.json")


with open(DATASETS_JSON_PATH, "r") as f:
    DATASETS = json.load(f)