import os
import glob
import json
from matbench.bench import MatbenchBenchmark
from hydragnn_gfm_finetuning.utils.ensemble_utils import build_arg_parser

def run_compile(args):
    mb = MatbenchBenchmark(autoload=False, subset=args.task_names)
    for task_obj in mb.tasks:
        task_obj.load()

    for task_obj in mb.tasks:
        task_name = task_obj.dataset_name
        first_underscore_index = task_name.find('_')
        name = task_name[first_underscore_index:]
        pattern = os.path.join(args.output_dir+name, f"{task_name}_fold*_predictions.json")
        pred_files = sorted(glob.glob(pattern))
        if not pred_files:
            raise FileNotFoundError(
                f"No prediction files found for {task_name} matching:\n  {pattern}\n"
                "Run `evaluate --matbench` for each fold first."
            )
 
        fold_predictions = {}
        for path in pred_files:
            with open(path) as f:
                data = json.load(f)
            fold_predictions[data["fold_idx"]] = data["predictions"]
            print(f"  [{task_name}] Loaded fold {data['fold_idx']} "
                  f"({len(data['predictions'])} predictions): {path}")
 
        missing = set(task_obj.folds) - set(fold_predictions.keys())
        if missing:
            raise ValueError(
                f"[{task_name}] Missing predictions for folds {sorted(missing)}. "
                "Run `evaluate --matbench` for those folds before compiling."
            )
 
        for fold_idx in sorted(task_obj.folds):
            task_obj.record(fold_idx, fold_predictions[fold_idx])
            print(f"  [{task_name}] Recorded fold {fold_idx}")

    mb.add_metadata({"algorithm": "HydraGNN_GFM_FineTuning"})
    mb.to_file(args.output_file)

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_compile(args)
