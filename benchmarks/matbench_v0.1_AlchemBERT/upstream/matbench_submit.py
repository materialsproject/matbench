import os
import pandas as pd
from matbench.bench import MatbenchBenchmark

# Path to your CSV results
results_dir = "nl_results/bert-base-cased"

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()

    task_name = task.dataset_name

    for fold in task.folds:
        task_dir = os.path.join(results_dir, f"{task_name}_{fold}_True")
        fold_dir = os.path.join(task_dir, f"version_0")

        # Find the only CSV file in the directory
        csv_files = [f for f in os.listdir(fold_dir) if f.endswith('.csv')]
        if len(csv_files) != 1:
            raise ValueError(f"Expected one CSV file in {fold_dir}, found {len(csv_files)}.")

        csv_file = os.path.join(fold_dir, csv_files[0])

        # Load predictions from the CSV (specifically the 'predictions' column)
        data = pd.read_csv(csv_file)
        if 'predictions' not in data.columns:
            raise ValueError(f"Expected 'predictions' column in {csv_file}.")

        # Extract predictions column
        predictions = data['predictions']

        # Record your loaded predictions
        task.record(fold, predictions)

# Save your results
mb.to_file("my_loaded_predictions_benchmark.json.gz")

