# XStack (matbench_v0.1)

Composition-only, cross-fitted, simplex-weighted stack of composition experts, evaluated on the four Matbench v0.1
composition tasks with nested cross-validation inside each official training fold. No external data: the inputs are
label-free descriptors of each row's own formula. Missing descriptor cells are imputed with medians of the training
rows of each fit, inner-CV fits included. See `info.json` for the full description and for the earlier runs on the
official folds (two of which used external DFT data and are not part of this entry).

## Files
- `results.json.gz`: recorded benchmark (4 tasks x 5 folds; classification recorded as raw probabilities). The
  `parameters` of every fold carry the compute (`wall_time_s`, `n_workers`, `threads_per_worker`, `cpu_threads_used`,
  `cpu_model`, `gpu`, `expert_fit_seconds`), the per-fold selection (`selected_candidate`,
  `candidate_inner_cv_scores`) and the imputation counts. These original fold parameters are unchanged; probability
  migration provenance and the original classification thresholds are retained in `user_metadata`.
- `info.json`: metadata, references, notes, pinned requirements.
- `run_submission.py`: reproduces the run and records every fold with `task.record`, without thresholding or clipping
  classification probabilities. It checks the installed scorer before training and rejects invalid prediction arrays.
  Future reruns also record `classification_threshold` and `classification_threshold_operator` (`>=`) in the fold
  parameters; these describe the training-selected decision rule and do not override Matbench scoring.
  (`python run_submission.py [task ...]`, output `results_rerun.json.gz`, which is not shipped).
- `xstack/`: verbatim copies of the frozen research modules. The code path of this entry is `run_final4.py`
  (per-fold candidate selection) -> `features4.py` (features without any cross-row fill) -> `experts4.py`
  (train-only median imputation around the `experts2.py` pool) -> `pipeline4.py` / `pipeline2.py` / `pipeline.py`
  (tuning, cross-fitting, stack, gated correction, thresholds) with `frozen_config4.py`. `run_final3.py`,
  `frozen_config3.py`, `features_all.py`, `design.py` and `experts.py` are imported for helper functions only. No DFT
  feature builder is shipped. Docstrings refer to the analysis plan in the research repository (https://github.com/bhaskargurram-ai/matbench-composition-stack).
- `requirements.txt`: the same pinned versions as `info.json`.

## Probability export and scores

The classification probabilities are from the **same Amendment 4 run**, not a new fit or a new model selection:
[frozen probability artifact](https://github.com/bhaskargurram-ai/matbench-composition-stack/blob/b486e3fd23a03c5832721c8be182b1b94f98a441/results/final_a4/matbench_submission/probabilities/results.json.gz).
That source file is plain JSON despite its `.gz` suffix and has cached scores from the old scorer. This submission
uses its original probabilities, recomputes the classification scores with the current Matbench scorer, and writes
actual gzip. Regression predictions and scores are unchanged.

ROC-AUC uses raw probabilities. Matbench currently computes balanced accuracy (BA), accuracy and F1 using
**`p > 0.5`**; it does not read custom thresholds from fold parameters. The original model thresholds were chosen
only from training-fold out-of-fold predictions and used **`p >= threshold`**. Their BA remains documented as a
separate reference, not the cached or leaderboard BA for this probability submission.

| Task | Probability ROC-AUC | Current BA (`p > 0.5`) | Original training-selected BA (`p >= threshold`, reference) |
|---|---|---|---|
| `matbench_expt_is_metal` | 0.9850 ± 0.0018 | 0.9433 ± 0.0028 | 0.9439 ± 0.0034 |
| `matbench_glass` | 0.9608 ± 0.0071 | 0.8731 ± 0.0167 | 0.8902 ± 0.0145 |

All values are mean ± **population** standard deviation over the five official folds (`ddof=0`). The earlier
probability summary used sample standard deviations; the means and probabilities are unchanged. See `info.json`
for the full history of earlier official-test evaluations and withdrawn DFT-informed results.

## Reproduce

Use Python 3.12 or newer. From this submission directory:

```sh
python -m pip install -r requirements.txt
python run_submission.py
```

The requirements pin the official Matbench source at
`2b17f3b87c81537709cd416de90cb36a65a55706`, which includes corrected probability scoring. The PyPI `matbench==0.6`
wheel still has the old scorer; the version string alone cannot distinguish the two. The scorer check at startup
requires probability ROC-AUC and strict `p > 0.5` BA before any models are fitted.

CPU-only, about 28 hours of job time at 4 threads per job. Data are downloaded on first use into
`./data/cache/`; expert predictions are cached in `./rerun_cache/` so that an interrupted run resumes.

To validate the shipped result from the **Matbench repository root**, use a checkout containing the current
submission validator and corrected scoring code:

```sh
PYTHONPATH=. python scripts/test_submission.py --submission matbench_v0.1_XStack
```

This validates the saved probabilities and cached metrics against the official test folds without rerunning training.
