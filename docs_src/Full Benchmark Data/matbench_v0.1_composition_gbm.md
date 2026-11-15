# matbench_v0.1: CompGBM

### Algorithm description: 

Composition-only baseline with a fixed 156-dimensional static composition descriptor and HistGradientBoostingClassifier. The task is matbench_glass. Within each official training fold, select max_iter in {300,600} and max_leaf_nodes in {31,63}, learning_rate=0.1, using a seeded 90/10 inner split. Selection uses probability ROC-AUC for classification or MAE for regression. Refit on the entire official training fold. MatbenchTask.get_test_data(include_target=False) supplies inference inputs and MatbenchTask.record owns outer targets/scoring. Seed=42. Current pinned official Matbench computes probability ROC-AUC and all four regression scores including MAPE. No learned feature transform or external training data is used.

#### Notes:
Corrected replacement for PR #363. Earlier outer-fold scores for alternative tracks/models were inspected; no historical preregistration evidence or original hardware/timing record was recovered. This revised procedure fixes the declared feature track before refitting and discloses that the benchmark folds are not newly unseen holdouts. It does not claim that rerunning erases historical exposure. All submitted results are regenerated with the real pinned official API, not a handwritten JSON schema. See notebook.md, requirements.txt and validation.json for the new execution environment, measured timings, input/code hashes, and verified scores. No leaderboard-rank claim or topology-only attribution is made.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_composition_gbm).

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
{'dataset': {'name': 'matbench_glass',
             'sha256': '36beb654e2a463ee2a6572105bea0ca2961eee7c7b26a25377bff2c3b338e53a',
             'url': 'https://ml.materialsproject.org/projects/matbench_glass.json.gz'},
 'features': {'cache_hit': True,
              'cache_key': '42943df581f05018860cc57992f8afa116a6ec86870c10cccba062341c37f311',
              'cache_sha256': 'ebac79c2969fe43dc058e7a269b90bb7f1b299332955056de217d51e4bcb6012',
              'expansion_fallback_count': 0,
              'feature_track': 'B2_composition_HistGB',
              'features': 156,
              'featurization_seconds': 1.5084051250014454,
              'hard_cap_count': 0,
              'input_sha256': '0536bf8c1d34df45fa3ec5e1bc2b6b6b2ed35dd7b6adc696fbf28c606ea5080b',
              'rows': 5680},
 'historical_test_exposure': 'The earlier submission inspected outer-fold '
                             'scores for alternative feature/model tracks. '
                             'This revision freezes the declared feature track '
                             'before refitting; it does not turn previously '
                             'observed benchmark folds into an unseen holdout. '
                             'No historical preregistration evidence is '
                             'available.',
 'protocol': 'fixed features; training-only hyperparameter selection',
 'sdk': {'git_revision': '4b7d0714849a917814115be380a64e7ace94be55',
         'scorer_sha256': 'f791263c675c45b6938c78d856676a6821ac6879526bc9aca7d82523303c3414',
         'validation_sha256': '94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d',
         'version': '0.6'},
 'source_sha256': {'composition_features.py': 'eac3fbdee730e669d405ac4fe9701e5c8832c6efd2ff11b7b797bcf74c8c5e08',
                   'composition_features_v2.py': '4ead01a7665d62371c633d44a5f90d5e8a047a1b49e7e0d9b2c4f80b165aa59b',
                   'element_table_generated.py': 'dec52d5cd2b5f14960e1cff52516ff3596232f536aeca1ce7e6bba9d97d531f5',
                   'protocol.py': '7c41b05234d45623d2022554d9adc3ca5447486b254d730cdedcf5238a8780fb',
                   'run_matbench_glass.py': '2cf87bec10f6a85925641b835ca3a282ba5c059f66e0f730e916cc4c7d06e15a'}}
```

### Metadata:

| tasks recorded | 1/13 |
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

#### `matbench_glass`

Probability ROC AUC below is recomputed from the original saved positive-class probabilities. Balanced accuracy, accuracy and F1 use p > 0.5. Predictions are aligned by sample ID with official test labels. Statistics are unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 | probability ROC AUC |
|------|------|------|------|------|
| fold_0 | 0.9155 | 0.8838 | 0.9416 | 0.9662 |
| fold_1 | 0.8900 | 0.8505 | 0.9242 | 0.9522 |
| fold_2 | 0.8873 | 0.8442 | 0.9227 | 0.9426 |
| fold_3 | 0.8988 | 0.8657 | 0.9298 | 0.9513 |
| fold_4 | 0.9023 | 0.8718 | 0.9321 | 0.9528 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8988 | 0.9155 | 0.8873 | 0.0100 |
| balanced accuracy | 0.8632 | 0.8838 | 0.8442 | 0.0143 |
| f1 | 0.9301 | 0.9416 | 0.9227 | 0.0067 |
| probability ROC AUC | 0.9530 | 0.9662 | 0.9426 | 0.0076 |

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
      <td>0.9155</td>
      <td>0.8838</td>
      <td>0.9416</td>
      <td>0.9662</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.8900</td>
      <td>0.8505</td>
      <td>0.9242</td>
      <td>0.9522</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.8873</td>
      <td>0.8442</td>
      <td>0.9227</td>
      <td>0.9426</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.8988</td>
      <td>0.8657</td>
      <td>0.9298</td>
      <td>0.9513</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.9023</td>
      <td>0.8718</td>
      <td>0.9321</td>
      <td>0.9528</td>
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
      <td>0.8988</td>
      <td>0.9155</td>
      <td>0.8873</td>
      <td>0.0100</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.8632</td>
      <td>0.8838</td>
      <td>0.8442</td>
      <td>0.0143</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.9301</td>
      <td>0.9416</td>
      <td>0.9227</td>
      <td>0.0067</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.9530</td>
      <td>0.9662</td>
      <td>0.9426</td>
      <td>0.0076</td>
    </tr>
  </tbody>
</table>
</details>

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B2_composition_HistGB', 'hyperparam...` |
| fold_1 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B2_composition_HistGB', 'hyperparam...` |
| fold_2 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 31}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B2_composition_HistGB', 'hyperparam...` |
| fold_3 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 31}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B2_composition_HistGB', 'hyperparam...` |
| fold_4 | `{'best_config': {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 63}, 'feature_selection': 'fixed; no outer-score-based track selection', 'feature_track': 'B2_composition_HistGB', 'hyperparam...` |




