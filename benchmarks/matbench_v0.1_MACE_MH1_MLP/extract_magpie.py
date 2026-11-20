import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import pickle
import numpy as np

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_NAME = sys.argv[1] if len(sys.argv) > 1 else "matbench_mp_e_form"
CACHE = os.path.join(SAVE_DIR, f"magpie_{TASK_NAME}.pkl")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

if os.path.exists(CACHE):
    log(f"Already cached at {CACHE}")
    with open(CACHE, "rb") as f:
        X = pickle.load(f)
    log(f"Shape: {X.shape}")
    sys.exit(0)

pkl_path = os.path.join(SAVE_DIR, f"structs_{TASK_NAME}.pkl")
if os.path.exists(pkl_path):
    log(f"Loading structures from pickle cache...")
    with open(pkl_path, "rb") as f:
        all_structs = pickle.load(f)
    log(f"Loaded {len(all_structs)} structures")
else:
    log(f"Loading {TASK_NAME} from matbench...")
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
    task = list(mb.tasks)[0]
    task.load()
    all_structs = list(task.df["structure"])
    with open(pkl_path, "wb") as f:
        pickle.dump(all_structs, f)
    log(f"Loaded and cached {len(all_structs)} structures")

import pandas as pd
from matminer.featurizers.composition import ElementProperty

log(f"Extracting compositions from {len(all_structs)} structures...")
compositions = [s.composition for s in all_structs]

df = pd.DataFrame({"composition": compositions})

log("Magpie composition features...")
magpie = ElementProperty.from_preset("magpie")
df = magpie.featurize_dataframe(df, "composition", ignore_errors=True)

mm_cols = [c for c in df.columns if c != "composition"]
X_mm = df[mm_cols].values.astype(float)
log(f"Magpie features: {X_mm.shape}")

with open(CACHE, "wb") as f:
    pickle.dump(X_mm, f)
log(f"Saved to {CACHE}")
