# matbench_v0.1: TabPFN-HYD

### Algorithm description: 

Composition-only steel yield-strength regression using an unmodified TabPFN-v2 foundation checkpoint and a conventional baseline selected within each official training fold. A 42-configuration baseline search is followed by an independent per-outer-fold inner-CV comparison of three feature sets, two TabPFN ensemble sizes, mean/median prediction summaries and four mixture weights, with baseline-only as an additional option. All five final models, contexts and mixtures are frozen before final test inference and native Matbench scoring. Built with PriorLabs-TabPFN.

#### Notes:
Only matbench_steels is submitted. Native five-fold MAE=76.59472316987244 MPa, std=12.406993875998648 MPa, all312 predictions retained. Seed1701, threefold inner CV independently for each outer fold; no pooled outer-score selection. Development pilot used only fold0 training rows; final configurations are reselected independently within every outer training fold. No official test scores/labels or public prediction files are used to train, calibrate, tune or select. Foundation gradient updates=0. The original TabPFN-v2 checkpoint and mandatory license/attribution are identified in README and source-lock. Actual runtime: Python3.12.3, PyTorch2.13.0+cu126, one A100-SXM4-80GB; conventional stage executes on CPU, TabPFN on CUDA. Saved-model replay and source validation evidence are supplied separately; public acceptance is determined by maintainers.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_TabPFN_HYD).

### References (in bibtex format): 

```
('@article{dunn2020matbench,\n'
 ' title={Benchmarking materials property prediction methods: the Matbench '
 'test set and Automatminer reference algorithm},\n'
 ' author={Dunn, Alexander and Wang, Qi and Ganose, Alex and Dopp, Daniel and '
 'Jain, Anubhav},\n'
 ' journal={npj Computational Materials}, volume={6}, pages={138}, '
 'year={2020}, doi={10.1038/s41524-020-00406-3}}\n'
 '@article{hollmann2025tabpfn,\n'
 ' title={Accurate predictions on small data with a tabular foundation '
 'model},\n'
 ' author={Hollmann, Noah and Muller, Samuel and Purucker, Lennart and '
 'Krishnakumar, Arjun and Korfer, Max and Hoo, Shi Bin and Schirrmeister, '
 'Robin Tibor and Hutter, Frank},\n'
 ' journal={Nature}, volume={637}, pages={319--326}, year={2025}, '
 'doi={10.1038/s41586-024-08328-6}}\n'
 '@article{geurts2006extremely,\n'
 ' title={Extremely randomized trees}, author={Geurts, Pierre and Ernst, '
 'Damien and Wehenkel, Louis},\n'
 ' journal={Machine Learning}, volume={63}, pages={3--42}, year={2006}, '
 'doi={10.1007/s10994-006-6226-1}}\n'
 '@article{ward2018matminer,\n'
 ' title={Matminer: An open source toolkit for materials data mining}, '
 'author={Ward, Logan and others},\n'
 ' journal={Computational Materials Science}, volume={152}, pages={60--69}, '
 'year={2018}, doi={10.1016/j.commatsci.2018.05.018}}\n'
 '@article{pedregosa2011scikit,\n'
 ' title={Scikit-learn: Machine Learning in Python}, author={Pedregosa, Fabian '
 'and others},\n'
 ' journal={Journal of Machine Learning Research}, volume={12}, '
 'pages={2825--2830}, year={2011}}\n')
```

### User metadata:

```
{'algorithm': 'TabPFN-HYD',
 'all_folds_freeze_sha256': '739f941d9e917df7e88299f4e49ffec3cbb2ff6cba064c782d6d8af816a0ffbd',
 'attribution': 'Built with PriorLabs-TabPFN',
 'authors': 'HYD',
 'foundation_gradient_updates': 0,
 'foundation_model_revision': '02a44705ff7790b50cded404bc59adb94da456a1',
 'official_source_revision': '35f88110a51adf817a53515780d99fc34f614bf8',
 'predictions_sha256': '611aa719f81efd90f6b78e06a7e9bda9beacab405078c254bada2cac6f4d97ca',
 'selection': 'Independent train-only inner CV per official outer fold. All '
              'five selections frozen before test inference and scoring.'}
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
{'python': ['numpy==2.5.4',
            'scikit-learn==1.9.1',
            'scipy==1.18.1',
            'joblib==1.6.0',
            'threadpoolctl==3.7.0',
            'matminer==0.10.1',
            'pymatgen==2026.10.24',
            'catboost==1.2.10',
            'tabpfn==9.2.0',
            'torch==2.13.0',
            'matbench @ '
            'git+https://github.com/materialsproject/matbench.git@35f88110a51adf817a53515780d99fc34f614bf8']}
```

### Task data:

#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 100.5027 | 169.0673 | 0.0638 | 889.3908 |
| fold_1 | 65.1272 | 87.8552 | 0.0472 | 378.5525 |
| fold_2 | 69.3749 | 103.2860 | 0.0495 | 423.0119 |
| fold_3 | 73.3440 | 97.1075 | 0.0539 | 322.4382 |
| fold_4 | 74.6249 | 123.0131 | 0.0569 | 565.3201 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 76.5947 | 100.5027 | 65.1272 | 12.4070 |
| rmse | 116.0658 | 169.0673 | 87.8552 | 28.8998 |
| mape* | 0.0543 | 0.0638 | 0.0472 | 0.0059 |
| max_error | 515.7427 | 889.3908 | 322.4382 | 203.3999 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'features': 'fractions', 'inner_fold_mae': [77.1372823088795, 95.37733729291094, 77.34234957713058], 'inner_mae': 83.28565639297368, 'n_estimators': 8, 'output_type': 'median', 'tabpfn_weight': 1.0}` |
| fold_1 | `{'features': 'combined', 'inner_fold_mae': [93.63461082795203, 80.76603630276534, 83.89962376182136], 'inner_mae': 86.1000902975129, 'n_estimators': 8, 'output_type': 'median', 'tabpfn_weight': 0.25}` |
| fold_2 | `{'features': 'magpie', 'inner_fold_mae': [73.03741250565366, 106.06911939211236, 71.39147425184568], 'inner_mae': 83.4993353832039, 'n_estimators': 8, 'output_type': 'median', 'tabpfn_weight': 0.5}` |
| fold_3 | `{'features': 'combined', 'inner_fold_mae': [83.48325302531849, 77.61247643071562, 85.47289188368137], 'inner_mae': 82.18954044657183, 'n_estimators': 8, 'output_type': 'median', 'tabpfn_weight': 1.0}` |
| fold_4 | `{'features': 'combined', 'inner_fold_mae': [82.91111563989477, 75.32156263923228, 81.72640913758883], 'inner_mae': 79.98636247223864, 'n_estimators': 8, 'output_type': 'median', 'tabpfn_weight': 1.0}` |




