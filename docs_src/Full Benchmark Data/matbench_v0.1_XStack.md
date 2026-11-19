# matbench_v0.1: XStack

### Algorithm description: 

XStack: a cross-fitted, simplex-weighted stack of composition experts for the four Matbench v0.1 composition tasks. COMPOSITION-ONLY: the only inputs are label-free descriptors computed from each row's own formula; no external property data of any kind is used (no DFT values, no pretrained model, no data from other Matbench tasks). The only labels seen are the task's own training-fold labels. Features: matminer/pymatgen composition descriptors (Magpie and other ElementProperty presets, element fractions, Stoichiometry, ValenceOrbital, IonProperty, BandCenter, AtomicOrbitals, Meredig, Miedema, YangSolidSolution, AtomicPackingEfficiency, WenAlloys, TMetalFraction) and CBFV Oliynyk statistics. Experts: Optuna-tuned LightGBM, XGBoost, CatBoost, extra trees (two feature sets and element fractions), kernel ridge (regression) or RBF SVM (classification), k-nearest neighbours and an MLP; for matbench_expt_gap additionally hurdle experts for the zero-inflated band gap (a classifier for gap > 0 combined with a regressor fitted on the non-zero gaps). Protocol: nested cross-validation inside each official training fold T_k; the test rows are only predicted and recorded. Five inner folds give out-of-fold predictions for every expert. Chosen on T_k only: the Optuna hyperparameters; the non-negative stack weights summing to one (L1 loss for regression, Brier score for classification); an optional novelty-damped local out-of-fold residual correction, kept only if a further nested inner-CV check confirms it (kept in 0 of 20 task-folds); the classification threshold; and the feature-block set (F_min = Magpie + element fractions + 21 extra descriptors, or F_all = F_min + the remaining descriptor blocks) together with, for matbench_expt_gap, the use of the hurdle experts and of the element-fraction extra-trees expert. The last two are picked per fold from a small candidate set by the cross-fitted inner-CV score of the uncorrected stack on T_k, ties to the simpler candidate (candidates: matbench_steels: 2; matbench_expt_gap: 8; matbench_expt_is_metal: 2; matbench_glass: 2). Selected per fold: matbench_steels: fold 0 F_min|hurdle=off|et_frac=in, fold 1 F_all|hurdle=off|et_frac=in, fold 2 F_all|hurdle=off|et_frac=in, fold 3 F_min|hurdle=off|et_frac=in, fold 4 F_min|hurdle=off|et_frac=in; matbench_expt_gap: fold 0 F_all|hurdle=on|et_frac=in, fold 1 F_all|hurdle=on|et_frac=in, fold 2 F_min|hurdle=on|et_frac=in, fold 3 F_all|hurdle=on|et_frac=in, fold 4 F_all|hurdle=on|et_frac=in; matbench_expt_is_metal: fold 0 F_all|hurdle=off|et_frac=in, fold 1 F_all|hurdle=off|et_frac=in, fold 2 F_all|hurdle=off|et_frac=in, fold 3 F_all|hurdle=off|et_frac=in, fold 4 F_all|hurdle=off|et_frac=in; matbench_glass: fold 0 F_all|hurdle=off|et_frac=in, fold 1 F_all|hurdle=off|et_frac=in, fold 2 F_all|hurdle=off|et_frac=in, fold 3 F_all|hurdle=off|et_frac=in, fold 4 F_all|hurdle=off|et_frac=in. IMPUTATION: missing descriptor cells are filled with column medians computed inside every model fit from that fit's training rows only: T_k for the final refit, the inner-training part for every inner-CV fit (cross-fitting and every Optuna trial). No statistic is computed over test rows or over inner-validation rows. NOT fold-internal, disclosed: the candidate set itself and a few coarse settings (Optuna budget of 20 trials; random forest, CrabNet and MODNet dropped; CatBoost restricted to the smaller feature set; fixed SVC C and gamma) were decided in a design phase on fold 0's training rows; and earlier versions of the pipeline were scored on the official folds before this one (see notes). Classification: raw probabilities are recorded for ROC-AUC. The current Matbench scorer computes balanced accuracy, accuracy and F1 with p > 0.5. The thresholds that maximise balanced accuracy on the training-fold out-of-fold predictions are preserved as metadata; their p >= threshold decision rule is a separate reference and does not override the current scorer.

#### Notes:
CPU-only (no GPU). Every expert was fitted from scratch on one Intel(R) Core(TM) i7-14650HX workstation (5 parallel jobs x 4 threads); no cached fit of an earlier run was reused. Total active job wall time 27.6 h over the 20 task-folds; the machine slept for 14.3 h during the run, five jobs resumed unchanged and their recorded times have the sleep subtracted (raw value kept as wall_time_s_raw) (mean per fold: matbench_steels 18 min, matbench_expt_gap 141 min, matbench_expt_is_metal 79 min, matbench_glass 93 min). Per-fold compute is recorded in the parameters of every fold in results.json.gz (wall_time_s, n_workers, threads_per_worker, cpu_threads_used, cpu_model, gpu, expert_fit_seconds per expert, selected_candidate, candidate_inner_cv_scores, imputation). Cells that are missing before imputation, per task: matbench_steels: 0 cells in 312 rows x 329 columns; matbench_expt_gap: 2824 cells in 4604 rows x 329 columns; matbench_expt_is_metal: 3117 cells in 4921 rows x 329 columns; matbench_glass: 195 cells in 5680 rows x 329 columns. A descriptor column is dropped only if it is missing in every row of the task (it then has no value anywhere). PRIOR RUNS ON THE OFFICIAL FOLDS, all pre-registered, each scored once, all reported in the research repository (board metric: MAE for steels and expt_gap, balanced accuracy for expt_is_metal and glass). (1) Amendment 1 (2026-10-18; composition-only, smaller expert pool): matbench_steels 82.80; matbench_expt_gap 0.3240; matbench_expt_is_metal 0.9370; matbench_glass 0.8841. (2) Amendment 2 (2026-10-19; USED EXTERNAL DATA: Materials Project PBE DFT band gaps and formation energies matched by reduced formula; design choices fixed from fold 0's training rows): matbench_steels 80.10; matbench_expt_gap 0.2352; matbench_expt_is_metal 0.9547; matbench_glass 0.8937. (3) Amendment 3 (2026-10-24; same external DFT data, design choices selected per fold): matbench_steels 80.32; matbench_expt_gap 0.2352; matbench_expt_is_metal 0.9547; matbench_glass 0.8937. (4) A post-hoc ablation of Amendment 2 without the DFT block (2026-10-19): matbench_steels 80.06; matbench_expt_gap 0.2724; matbench_expt_is_metal 0.9425; matbench_glass 0.8922. (5) The present entry, Amendment 4 (2026-11-17; no DFT block, train-only imputation): matbench_steels 80.20; matbench_expt_gap 0.2773; matbench_expt_is_metal 0.9439; matbench_glass 0.8902. Amendments 2 and 3 are NOT composition-only and are not part of this entry: the DFT gap and metal flag are close proxies of the matbench_expt_gap and matbench_expt_is_metal targets. They were the earlier contents of this pull request and were withdrawn at the maintainers' request. Because of these earlier runs the design of this entry was not blind to test-fold results; no choice in Amendment 4 was made from test scores (its two changes are the ones the maintainers asked for). PROBABILITY EXPORT: the original Amendment 4 protocol recorded hard labels at its training-selected thresholds (p >= threshold), with BA matbench_expt_is_metal 0.9439 ± 0.0034 and matbench_glass 0.8902 ± 0.0145. This artifact now records the raw probabilities from that same run, without retraining or new model selection, and recomputes classification metrics with the corrected Matbench scorer. Probability ROC-AUC: matbench_expt_is_metal 0.9850 ± 0.0018; matbench_glass 0.9608 ± 0.0071. Current BA at p > 0.5: matbench_expt_is_metal 0.9433 ± 0.0028; matbench_glass 0.8731 ± 0.0167. All these uncertainties are population standard deviations across the five official folds (ddof=0); the original probability summary used sample standard deviations. Source: https://github.com/bhaskargurram-ai/matbench-composition-stack/blob/b486e3fd23a03c5832721c8be182b1b94f98a441/results/final_a4/matbench_submission/probabilities/results.json.gz. Original fold parameters and regression results are unchanged; provenance and original classification thresholds are preserved in user_metadata. Future run_submission.py outputs also retain the training-selected threshold and the >= operator in fold parameters. REPRODUCIBILITY: same-machine reruns are bit-identical. Across CPU architectures XGBoost (tree_method=hist) and, on the small steels task, the Optuna-tuned LightGBM parameters were found not to be bitwise reproducible in an earlier check (Windows i7 vs Linux EPYC); the other experts agreed to 1e-9. The original run used matbench==0.6 installed with --no-deps because its scikit-learn==1.0 / monty pins do not build on Python 3.12. Reproduction now requires the corrected scorer from the official source revision 2b17f3b87c81537709cd416de90cb36a65a55706, pinned in requirements.txt (Python >= 3.12), rather than the PyPI 0.6 wheel. The package version string is still 0.6, so run_submission.py checks probability ROC-AUC and strict p > 0.5 BA before training. run_submission.py plus the xstack/ package (verbatim copies of the frozen research modules) reproduce the training protocol and export raw probabilities. Research repository (analysis plan with all amendments, freeze hashes, per-fold predictions): https://github.com/bhaskargurram-ai/matbench-composition-stack. The frozen source probability artifact is plain JSON even though its filename ends in .gz, and its old cached classification scores require recomputation. The file shipped here contains recomputed classification scores and is gzip-compressed.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_XStack).

### References (in bibtex format): 

```
['@article{dunn2020matbench,\n'
 '  author = {Alexander Dunn and Qi Wang and Alex Ganose and Daniel Dopp and '
 'Anubhav Jain},\n'
 '  title = {Benchmarking materials property prediction methods: the Matbench '
 'test set and Automatminer reference algorithm},\n'
 '  journal = {npj Computational Materials},\n'
 '  year = {2020},\n'
 '  volume = {6},\n'
 '  pages = {138},\n'
 '  doi = {10.1038/s41524-020-00406-3}\n'
 '}',
 '@article{ward2018matminer,\n'
 '  author = {Logan Ward and Alexander Dunn and Alireza Faghaninia and Nils '
 'E.R. Zimmermann and Saurabh Bajaj and Qi Wang and Joseph Montoya and Jiming '
 'Chen and Kyle Bystrom and Maxwell Dylla and Kyle Chard and Mark Asta and '
 'Kristin A. Persson and G. Jeffrey Snyder and Ian Foster and Anubhav Jain},\n'
 '  title = {Matminer: An open source toolkit for materials data mining},\n'
 '  journal = {Computational Materials Science},\n'
 '  year = {2018},\n'
 '  volume = {152},\n'
 '  pages = {60--69},\n'
 '  doi = {10.1016/j.commatsci.2018.05.018}\n'
 '}']
```

### User metadata:

```
{'algorithm': 'XStack',
 'probability_migration': {'author_threshold_results': {'matbench_expt_is_metal': {'fold_0': {'balanced_accuracy': 0.9441155371587358,
                                                                                              'operator': '>=',
                                                                                              'threshold': 0.43491910426064706},
                                                                                   'fold_1': {'balanced_accuracy': 0.9421093943650334,
                                                                                              'operator': '>=',
                                                                                              'threshold': 0.3569266883619039},
                                                                                   'fold_2': {'balanced_accuracy': 0.9502561348426011,
                                                                                              'operator': '>=',
                                                                                              'threshold': 0.3898254814932184},
                                                                                   'fold_3': {'balanced_accuracy': 0.943113277699744,
                                                                                              'operator': '>=',
                                                                                              'threshold': 0.4220180111415585},
                                                                                   'fold_4': {'balanced_accuracy': 0.9400437907956705,
                                                                                              'operator': '>=',
                                                                                              'threshold': 0.46413744709748794}},
                                                        'matbench_glass': {'fold_0': {'balanced_accuracy': 0.9120424251326726,
                                                                                      'operator': '>=',
                                                                                      'threshold': 0.7669218281495153},
                                                                           'fold_1': {'balanced_accuracy': 0.8806887304474902,
                                                                                      'operator': '>=',
                                                                                      'threshold': 0.7508792179222851},
                                                                           'fold_2': {'balanced_accuracy': 0.8723065276098575,
                                                                                      'operator': '>=',
                                                                                      'threshold': 0.7005071024198032},
                                                                           'fold_3': {'balanced_accuracy': 0.9014625070149866,
                                                                                      'operator': '>=',
                                                                                      'threshold': 0.7202304523976684},
                                                                           'fold_4': {'balanced_accuracy': 0.8844532830137513,
                                                                                      'operator': '>=',
                                                                                      'threshold': 0.7680311341034171}}},
                           'author_threshold_selection': 'Maximise balanced '
                                                         'accuracy on '
                                                         'training-fold '
                                                         'out-of-fold '
                                                         'predictions; '
                                                         'reference results '
                                                         'only under the '
                                                         'current scoring '
                                                         'policy.',
                           'method': 'Raw stack_local probabilities from the '
                                     'same Amendment 4 run; no retraining. All '
                                     '20 folds match official test IDs and '
                                     'source NPZ arrays. Saved classification '
                                     'labels are exactly reproduced by the '
                                     "author's training-fold thresholds. "
                                     'Classification scores are recomputed '
                                     'with current Matbench; regression '
                                     "results and every fold's original "
                                     'parameters and uncertainty are '
                                     'unchanged.',
                           'official_decision_rule': 'p > 0.5',
                           'original_pr_commit': '6a88c55cf5c50396d6d60608d6dea19cb892c3ad',
                           'original_results_sha256': '31678d9071863959c454ed0ba8161b7d0cb6164b704c85c8884eb5a6f2fb979f',
                           'scoring_commit': '2b17f3b87c81537709cd416de90cb36a65a55706',
                           'source_artifact': {'bytes': 638265,
                                               'git_blob': '0ddd346257f443f8601cc0cf3918ed8fdcbfc755',
                                               'path': 'results/final_a4/matbench_submission/probabilities/results.json.gz',
                                               'sha256': '1530bad0f1797c2d5014f1937a891dadf20e72a57c2d0f5d5a76ec4f427409a0'},
                           'source_commit': 'b486e3fd23a03c5832721c8be182b1b94f98a441',
                           'source_repository': 'https://github.com/bhaskargurram-ai/matbench-composition-stack'}}
```

### Metadata:

| tasks recorded | 4/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✓ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['matbench @ '
            'git+https://github.com/materialsproject/matbench.git@2b17f3b87c81537709cd416de90cb36a65a55706',
            'matminer==0.10.1',
            'pymatgen==2026.9.24',
            'monty==2026.7.16',
            'scikit-learn==1.9.1',
            'numpy==2.5.4',
            'scipy==1.18.1',
            'pandas==2.3.3',
            'lightgbm==4.7.0',
            'xgboost==3.4.1',
            'catboost==1.2.10',
            'optuna==5.0.0',
            'joblib==1.6.0',
            'CBFV==1.1.0']}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 0.2570 | 0.6096 | 0.3494 | 6.3049 |
| fold_1 | 0.2759 | 0.6547 | 0.2915 | 6.4000 |
| fold_2 | 0.2980 | 0.7537 | 0.3897 | 9.9000 |
| fold_3 | 0.2746 | 0.6901 | 0.3135 | 5.7017 |
| fold_4 | 0.2807 | 0.7225 | 0.3919 | 6.7994 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2773 | 0.2980 | 0.2570 | 0.0131 |
| rmse | 0.6861 | 0.7537 | 0.6096 | 0.0505 |
| mape* | 0.3472 | 0.3919 | 0.2915 | 0.0401 |
| max_error | 7.0212 | 9.9000 | 5.7017 | 1.4817 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.339172, 'F_all|hurdle=off|et_frac=out': 0.339172, 'F_all|hurdle=on|et_frac=in': 0.293785, 'F_all|hurdle=on|et_frac=out': 0.293785, 'F_mi...` |
| fold_1 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.337188, 'F_all|hurdle=off|et_frac=out': 0.337188, 'F_all|hurdle=on|et_frac=in': 0.288946, 'F_all|hurdle=on|et_frac=out': 0.288946, 'F_mi...` |
| fold_2 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.337313, 'F_all|hurdle=off|et_frac=out': 0.337313, 'F_all|hurdle=on|et_frac=in': 0.29042, 'F_all|hurdle=on|et_frac=out': 0.29042, 'F_min|...` |
| fold_3 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.339372, 'F_all|hurdle=off|et_frac=out': 0.339372, 'F_all|hurdle=on|et_frac=in': 0.288873, 'F_all|hurdle=on|et_frac=out': 0.288873, 'F_mi...` |
| fold_4 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.338295, 'F_all|hurdle=off|et_frac=out': 0.338295, 'F_all|hurdle=on|et_frac=in': 0.27942, 'F_all|hurdle=on|et_frac=out': 0.27942, 'F_min|...` |




#### `matbench_expt_is_metal`

Probability ROC AUC below is recomputed from the original saved positive-class probabilities. Balanced accuracy, accuracy and F1 use p > 0.5. Predictions are aligned by sample ID with official test labels. Statistics are unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 | probability ROC AUC |
|------|------|------|------|------|
| fold_0 | 0.9462 | 0.9461 | 0.9447 | 0.9867 |
| fold_1 | 0.9421 | 0.9421 | 0.9415 | 0.9835 |
| fold_2 | 0.9411 | 0.9410 | 0.9402 | 0.9856 |
| fold_3 | 0.9472 | 0.9472 | 0.9469 | 0.9868 |
| fold_4 | 0.9400 | 0.9400 | 0.9396 | 0.9824 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9433 | 0.9472 | 0.9400 | 0.0028 |
| balanced accuracy | 0.9433 | 0.9472 | 0.9400 | 0.0028 |
| f1 | 0.9426 | 0.9469 | 0.9396 | 0.0028 |
| probability ROC AUC | 0.9850 | 0.9868 | 0.9824 | 0.0018 |

<details><summary>Original stored classification scores (historical)</summary>
<p>These values reproduce the original submission artifact. Older stored <code>rocauc</code> values may reflect thresholded decisions, whose AUC equals balanced accuracy; they must not be interpreted as the recomputed probability ROC AUC above.</p>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>fold</th>
      <th>accuracy</th>
      <th>balanced_accuracy</th>
      <th>f1</th>
      <th>stored rocauc (historical)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fold_0</td>
      <td>0.9462</td>
      <td>0.9461</td>
      <td>0.9447</td>
      <td>0.9867</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.9421</td>
      <td>0.9421</td>
      <td>0.9415</td>
      <td>0.9835</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.9411</td>
      <td>0.9410</td>
      <td>0.9402</td>
      <td>0.9856</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.9472</td>
      <td>0.9472</td>
      <td>0.9469</td>
      <td>0.9868</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.9400</td>
      <td>0.9400</td>
      <td>0.9396</td>
      <td>0.9824</td>
    </tr>
  </tbody>
</table><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>metric</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>accuracy</td>
      <td>0.9433</td>
      <td>0.9472</td>
      <td>0.9400</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.9433</td>
      <td>0.9472</td>
      <td>0.9400</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.9426</td>
      <td>0.9469</td>
      <td>0.9396</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.9850</td>
      <td>0.9868</td>
      <td>0.9824</td>
      <td>0.0018</td>
    </tr>
  </tbody>
</table>
</details>

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.98413, 'F_min|hurdle=off|et_frac=in': 0.982154}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, 'e...` |
| fold_1 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.983439, 'F_min|hurdle=off|et_frac=in': 0.980804}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_2 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.983111, 'F_min|hurdle=off|et_frac=in': 0.981079}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_3 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.981394, 'F_min|hurdle=off|et_frac=in': 0.979732}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_4 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.983661, 'F_min|hurdle=off|et_frac=in': 0.981931}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |




#### `matbench_glass`

Probability ROC AUC below is recomputed from the original saved positive-class probabilities. Balanced accuracy, accuracy and F1 use p > 0.5. Predictions are aligned by sample ID with official test labels. Statistics are unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 | probability ROC AUC |
|------|------|------|------|------|
| fold_0 | 0.9278 | 0.8988 | 0.9501 | 0.9723 |
| fold_1 | 0.8900 | 0.8532 | 0.9239 | 0.9557 |
| fold_2 | 0.8944 | 0.8563 | 0.9272 | 0.9516 |
| fold_3 | 0.9076 | 0.8791 | 0.9357 | 0.9637 |
| fold_4 | 0.9085 | 0.8780 | 0.9365 | 0.9606 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9056 | 0.9278 | 0.8900 | 0.0132 |
| balanced accuracy | 0.8731 | 0.8988 | 0.8532 | 0.0167 |
| f1 | 0.9347 | 0.9501 | 0.9239 | 0.0091 |
| probability ROC AUC | 0.9608 | 0.9723 | 0.9516 | 0.0071 |

<details><summary>Original stored classification scores (historical)</summary>
<p>These values reproduce the original submission artifact. Older stored <code>rocauc</code> values may reflect thresholded decisions, whose AUC equals balanced accuracy; they must not be interpreted as the recomputed probability ROC AUC above.</p>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>fold</th>
      <th>accuracy</th>
      <th>balanced_accuracy</th>
      <th>f1</th>
      <th>stored rocauc (historical)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fold_0</td>
      <td>0.9278</td>
      <td>0.8988</td>
      <td>0.9501</td>
      <td>0.9723</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.8900</td>
      <td>0.8532</td>
      <td>0.9239</td>
      <td>0.9557</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.8944</td>
      <td>0.8563</td>
      <td>0.9272</td>
      <td>0.9516</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.9076</td>
      <td>0.8791</td>
      <td>0.9357</td>
      <td>0.9637</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.9085</td>
      <td>0.8780</td>
      <td>0.9365</td>
      <td>0.9606</td>
    </tr>
  </tbody>
</table><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>metric</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>accuracy</td>
      <td>0.9056</td>
      <td>0.9278</td>
      <td>0.8900</td>
      <td>0.0132</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.8731</td>
      <td>0.8988</td>
      <td>0.8532</td>
      <td>0.0167</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.9347</td>
      <td>0.9501</td>
      <td>0.9239</td>
      <td>0.0091</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.9608</td>
      <td>0.9723</td>
      <td>0.9516</td>
      <td>0.0071</td>
    </tr>
  </tbody>
</table>
</details>

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.952235, 'F_min|hurdle=off|et_frac=in': 0.947203}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_1 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.955698, 'F_min|hurdle=off|et_frac=in': 0.952031}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_2 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.957224, 'F_min|hurdle=off|et_frac=in': 0.953512}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_3 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.954732, 'F_min|hurdle=off|et_frac=in': 0.951361}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, '...` |
| fold_4 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 0.95044, 'F_min|hurdle=off|et_frac=in': 0.945718}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4, 'e...` |




#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 93.8769 | 135.1969 | 0.0617 | 449.4449 |
| fold_1 | 67.1461 | 93.0639 | 0.0489 | 399.5136 |
| fold_2 | 80.7058 | 117.0952 | 0.0562 | 415.8957 |
| fold_3 | 85.9888 | 112.5174 | 0.0635 | 320.3493 |
| fold_4 | 73.3038 | 113.5350 | 0.0573 | 409.9922 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 80.2043 | 93.8769 | 67.1461 | 9.3679 |
| rmse | 114.2817 | 135.1969 | 93.0639 | 13.4106 |
| mape* | 0.0575 | 0.0635 | 0.0489 | 0.0051 |
| max_error | 399.0391 | 449.4449 | 320.3493 | 42.7485 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 81.253075, 'F_min|hurdle=off|et_frac=in': 79.992704}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4,...` |
| fold_1 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 86.474586, 'F_min|hurdle=off|et_frac=in': 86.857121}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4,...` |
| fold_2 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 87.994325, 'F_min|hurdle=off|et_frac=in': 87.994325}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4,...` |
| fold_3 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 88.511237, 'F_min|hurdle=off|et_frac=in': 86.393778}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4,...` |
| fold_4 | `{'candidate_inner_cv_scores': {'F_all|hurdle=off|et_frac=in': 85.999397, 'F_min|hurdle=off|et_frac=in': 85.839255}, 'cpu_count': 24, 'cpu_model': 'Intel(R) Core(TM) i7-14650HX', 'cpu_threads_used': 4,...` |




