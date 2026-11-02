import os
import numpy as np
from matbench.bench import MatbenchBenchmark

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_NAME = "matbench_mp_e_form"

print("Loading matbench task...")
mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
task = list(mb.tasks)[0]
task.load()

print("Loading predictions...")
data = np.load(os.path.join(SAVE_DIR, f"preds_{TASK_NAME}.npz"))

for fold in range(5):
    preds = data[f"preds_{fold}"]
    task.record(fold, preds)
    print(f"Fold {fold}: recorded {len(preds)} predictions")

print("\nMatbench scores:")
print(task.scores)

out_path = os.path.join(SAVE_DIR, f"{TASK_NAME}_combined.json.gz")
mb.to_file(out_path)
print(f"Saved to {out_path}")
