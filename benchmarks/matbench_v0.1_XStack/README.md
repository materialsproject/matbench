# XStack (matbench_v0.1)

Composition-only, cross-fitted, simplex-weighted stack of composition experts, evaluated on the four Matbench v0.1
composition tasks with nested cross-validation inside each official training fold. No external data: the inputs are
label-free descriptors of each row's own formula. Missing descriptor cells are imputed with medians of the training
rows of each fit, inner-CV fits included. See `info.json` for the full description and for the earlier runs on the
official folds (two of which used external DFT data and are not part of this entry).

## Files
- `results.json.gz`: recorded benchmark (4 tasks x 5 folds; classification recorded as hard labels). The
  `parameters` of every fold carry the compute (`wall_time_s`, `n_workers`, `threads_per_worker`, `cpu_threads_used`,
  `cpu_model`, `gpu`, `expert_fit_seconds`), the per-fold selection (`selected_candidate`,
  `candidate_inner_cv_scores`) and the imputation counts.
- `info.json`: metadata, references, notes, pinned requirements.
- `run_submission.py`: reproduces the run and records every fold with `task.record`
  (`python run_submission.py [task ...]`, output `results_rerun.json.gz`, which is not shipped).
- `xstack/`: verbatim copies of the frozen research modules. The code path of this entry is `run_final4.py`
  (per-fold candidate selection) -> `features4.py` (features without any cross-row fill) -> `experts4.py`
  (train-only median imputation around the `experts2.py` pool) -> `pipeline4.py` / `pipeline2.py` / `pipeline.py`
  (tuning, cross-fitting, stack, gated correction, thresholds) with `frozen_config4.py`. `run_final3.py`,
  `frozen_config3.py`, `features_all.py`, `design.py` and `experts.py` are imported for helper functions only. No DFT
  feature builder is shipped. Docstrings refer to the analysis plan in the research repository (https://github.com/bhaskargurram-ai/matbench-composition-stack).
- `requirements.txt`: the same pinned versions as `info.json`.

## Reproduce
```
pip install -r requirements.txt   # matbench==0.6 needs --no-deps on Python 3.12, see info.json notes
python run_submission.py
```
CPU-only, about 28 hours of job time at 4 threads per job. Data are downloaded on first use into
`./data/cache/`; expert predictions are cached in `./rerun_cache/` so that an interrupted run resumes.
