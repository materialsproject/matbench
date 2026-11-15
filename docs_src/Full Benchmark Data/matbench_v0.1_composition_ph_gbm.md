# matbench_v0.1: CompPHGBM

### Algorithm description: 

Fixed composition plus structure descriptor (156 composition + 51 persistent-homology/geometry features) with HistGradientBoostingRegressor on matbench_log_gvrh and matbench_log_kvrh. The feature track is always B_comp+ph; outer-fold errors never select between feature tracks. Within each official training fold, select max_iter in {300,600} and max_leaf_nodes in {31,63}, learning_rate=0.1, using a seeded 90/10 inner split. Selection uses probability ROC-AUC for classification or MAE for regression. Refit on the entire official training fold. MatbenchTask.get_test_data(include_target=False) supplies inference inputs and MatbenchTask.record owns outer targets/scoring. Seed=42. Current pinned official Matbench computes probability ROC-AUC and all four regression scores including MAPE. No learned feature transform or external training data is used.

#### Notes:
Corrected replacement for PR #363. Earlier outer-fold scores for alternative tracks/models were inspected; no historical preregistration evidence or original hardware/timing record was recovered. This revised procedure fixes the declared feature track before refitting and discloses that the benchmark folds are not newly unseen holdouts. It does not claim that rerunning erases historical exposure. All submitted results are regenerated with the real pinned official API, not a handwritten JSON schema. See notebook.md, requirements.txt and validation.json for the new execution environment, measured timings, input/code hashes, and verified scores. No leaderboard-rank claim or topology-only attribution is made.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_composition_ph_gbm).

### References (in bibtex format): 

```
('@article{Dunn2020Matbench, title={Benchmarking materials property prediction '
 'methods: the Matbench test set and Automatminer reference algorithm}, '
 'author={Dunn, Alexander and Wang, Qi and Ganose, Alex and Dopp, Daniel and '
 'Jain, Anubhav}, journal={npj Computational Materials}, volume={6}, '
 'number={1}, pages={138}, year={2020}, doi={10.1038/s41524-020-00406-3}}')
```

### User metadata:

```
{'feature_track': 'B_comp+ph',
 'task_runs': {'matbench_log_gvrh': {'dataset': {'name': 'matbench_log_gvrh',
                                                 'sha256': '098af941f4c663270f1fe21abf20ffad6fb85ecbfcba5786ceac03983ac29da7',
                                                 'url': 'https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz'},
                                     'features': {'cache_hit': False,
                                                  'cache_key': '1e3fdf6b24785d20b7d8b6a1ef430710dcbc1153a9de85010d07d26f8e8d2d28',
                                                  'cache_sha256': '42fc8ce42047f54c0b07c34236d5b389ecf8c21d41a39c6857e5cbb007e98bc3',
                                                  'expansion_fallback_count': 0,
                                                  'feature_track': 'B_comp+ph',
                                                  'features': 207,
                                                  'featurization_seconds': 57.74562904200138,
                                                  'hard_cap_count': 0,
                                                  'input_sha256': '74c5bebc531f9dd85c7b7dd4fe1307881d47896c0f42df4324e6aa4a463b57f3',
                                                  'rows': 10987},
                                     'historical_test_exposure': 'The earlier '
                                                                 'submission '
                                                                 'inspected '
                                                                 'outer-fold '
                                                                 'scores for '
                                                                 'alternative '
                                                                 'feature/model '
                                                                 'tracks. This '
                                                                 'revision '
                                                                 'freezes the '
                                                                 'declared '
                                                                 'feature '
                                                                 'track before '
                                                                 'refitting; '
                                                                 'it does not '
                                                                 'turn '
                                                                 'previously '
                                                                 'observed '
                                                                 'benchmark '
                                                                 'folds into '
                                                                 'an unseen '
                                                                 'holdout. No '
                                                                 'historical '
                                                                 'preregistration '
                                                                 'evidence is '
                                                                 'available.',
                                     'protocol': 'fixed features; '
                                                 'training-only hyperparameter '
                                                 'selection',
                                     'sdk': {'git_revision': '4b7d0714849a917814115be380a64e7ace94be55',
                                             'scorer_sha256': 'f791263c675c45b6938c78d856676a6821ac6879526bc9aca7d82523303c3414',
                                             'validation_sha256': '94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d',
                                             'version': '0.6'},
                                     'source_sha256': {'composition_features.py': 'eac3fbdee730e669d405ac4fe9701e5c8832c6efd2ff11b7b797bcf74c8c5e08',
                                                       'composition_features_v2.py': '4ead01a7665d62371c633d44a5f90d5e8a047a1b49e7e0d9b2c4f80b165aa59b',
                                                       'element_table_generated.py': 'dec52d5cd2b5f14960e1cff52516ff3596232f536aeca1ce7e6bba9d97d531f5',
                                                       'ph_features.py': 'ddcb2edb8b520d1ded2145466f0cedb84653ff35bf7c5fa6ec0c091ae494d47e',
                                                       'protocol.py': '888f9b932443936ac4847f18a520f2da92357a372de417b611333d6671a57520'}},
               'matbench_log_kvrh': {'dataset': {'name': 'matbench_log_kvrh',
                                                 'sha256': '44b113ddb7e23aa18731a62c74afa7e5aa654199e0db5f951c8248a00955c9cd',
                                                 'url': 'https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz'},
                                     'features': {'cache_hit': True,
                                                  'cache_key': '1e3fdf6b24785d20b7d8b6a1ef430710dcbc1153a9de85010d07d26f8e8d2d28',
                                                  'cache_sha256': '42fc8ce42047f54c0b07c34236d5b389ecf8c21d41a39c6857e5cbb007e98bc3',
                                                  'expansion_fallback_count': 0,
                                                  'feature_track': 'B_comp+ph',
                                                  'features': 207,
                                                  'featurization_seconds': 57.74562904200138,
                                                  'hard_cap_count': 0,
                                                  'input_sha256': '74c5bebc531f9dd85c7b7dd4fe1307881d47896c0f42df4324e6aa4a463b57f3',
                                                  'rows': 10987},
                                     'historical_test_exposure': 'The earlier '
                                                                 'submission '
                                                                 'inspected '
                                                                 'outer-fold '
                                                                 'scores for '
                                                                 'alternative '
                                                                 'feature/model '
                                                                 'tracks. This '
                                                                 'revision '
                                                                 'freezes the '
                                                                 'declared '
                                                                 'feature '
                                                                 'track before '
                                                                 'refitting; '
                                                                 'it does not '
                                                                 'turn '
                                                                 'previously '
                                                                 'observed '
                                                                 'benchmark '
                                                                 'folds into '
                                                                 'an unseen '
                                                                 'holdout. No '
                                                                 'historical '
                                                                 'preregistration '
                                                                 'evidence is '
                                                                 'available.',
                                     'protocol': 'fixed features; '
                                                 'training-only hyperparameter '
                                                 'selection',
                                     'sdk': {'git_revision': '4b7d0714849a917814115be380a64e7ace94be55',
                                             'scorer_sha256': 'f791263c675c45b6938c78d856676a6821ac6879526bc9aca7d82523303c3414',
                                             'validation_sha256': '94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d',
                                             'version': '0.6'},
                                     'source_sha256': {'composition_features.py': 'eac3fbdee730e669d405ac4fe9701e5c8832c6efd2ff11b7b797bcf74c8c5e08',
                                                       'composition_features_v2.py': '4ead01a7665d62371c633d44a5f90d5e8a047a1b49e7e0d9b2c4f80b165aa59b',
                                                       'element_table_generated.py': 'dec52d5cd2b5f14960e1cff52516ff3596232f536aeca1ce7e6bba9d97d531f5',
                                                       'ph_features.py': 'ddcb2edb8b520d1ded2145466f0cedb84653ff35bf7c5fa6ec0c091ae494d47e',
                                                       'protocol.py': '888f9b932443936ac4847f18a520f2da92357a372de417b611333d6671a57520'}}}}
```

### Metadata:

| tasks recorded | 2/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'lockfile': 'requirements.txt (full tested dependency set)',
 'python': ['python==3.12.10',
            'matbench @ '
            'git+https://github.com/materialsproject/matbench.git@4b7d0714849a917814115be380a64e7ace94be55',
            'numpy==2.2.6',
            'pandas==2.3.3',
            'scikit-learn==1.7.2',
            'pymatgen==2025.10.7',
            'ripser==0.6.15',
            'persim==0.3.8',
            'pyarrow==19.0.1',
            'threadpoolctl==3.6.0']}
```

### Task data:

#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 0.0809 | 0.1184 | 0.0627 | 0.8844 |
| fold_1 | 0.0827 | 0.1228 | 0.0650 | 1.1775 |
| fold_2 | 0.0820 | 0.1196 | 0.0642 | 0.9518 |
| fold_3 | 0.0799 | 0.1173 | 0.0611 | 0.8343 |
| fold_4 | 0.0798 | 0.1179 | 0.0620 | 0.8987 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0810 | 0.0827 | 0.0798 | 0.0011 |
| rmse | 0.1192 | 0.1228 | 0.1173 | 0.0019 |
| mape* | 0.0630 | 0.0650 | 0.0611 | 0.0014 |
| max_error | 0.9494 | 1.1775 | 0.8343 | 0.1201 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_1 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 31}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_2 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 31}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_3 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_4 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 31}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 0.0641 | 0.1134 | 0.0413 | 1.7834 |
| fold_1 | 0.0634 | 0.1136 | 0.0414 | 1.4041 |
| fold_2 | 0.0617 | 0.1070 | 0.0403 | 1.1168 |
| fold_3 | 0.0674 | 0.1190 | 0.0484 | 1.2162 |
| fold_4 | 0.0634 | 0.1097 | 0.0413 | 1.3758 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0640 | 0.0674 | 0.0617 | 0.0019 |
| rmse | 0.1125 | 0.1190 | 0.1070 | 0.0041 |
| mape* | 0.0425 | 0.0484 | 0.0403 | 0.0030 |
| max_error | 1.3792 | 1.7834 | 1.1168 | 0.2278 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 31}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_1 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_2 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_3 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |
| fold_4 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B_comp+ph', 'hyperparameter_selecti...` |




