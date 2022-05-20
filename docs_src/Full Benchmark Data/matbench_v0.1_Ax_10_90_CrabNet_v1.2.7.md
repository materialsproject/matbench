# matbench_v0.1: Ax(10/90)+CrabNet v1.2.7

### Algorithm description: 

Use Ax Bayesian adaptive design to simultaneously optimize 23 hyperparameters of CrabNet. 100 sequential design iterations were used, and parameters were chosen based on a combination of intuition and algorithm/data constraints (e.g. elemental featurizers which were missing elements contained in the dataset were removed). The first 10 iterations (for more direct comparison with SAASBO) were based on SOBOL sampling to create a rough initial model, while the remaining 90 iterations were Bayesian adaptive design iterations. For the inner loops (where hyperparameter optimization is performed), the average MAE across each of the five inner folds was used as Ax's objective to minimize. The best parameter set was then trained on all the inner fold data and used to predict on the test set (unknown during hyperparameter optimization). This is nested cross-validation, and is computationally expensive.

#### Notes:
A Jupyter notebook is provided which contains additional details about the run of the algorithm. If you decide to run this yourself, because it can take several days to run, be sure to set the `dummy` variable to True and run an initial test that it runs free of errors.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Ax_10_90_CrabNet_v1.2.7).

### References (in bibtex format): 

```
['@article{Wang2021crabnet,  author = {Wang, Anthony Yu-Tung and Kauwe, Steven '
 'K. and Murdock, Ryan J. and Sparks, Taylor D.},  year = {2021},  title = '
 '{Compositionally restricted attention-based network for materials property '
 'predictions},  pages = {77},  volume = {7},  number = {1},  doi = '
 '{10.1038/s41524-021-00545-1},  publisher = {{Nature Publishing Group}},  '
 'shortjournal = {npj Comput. Mater.},  journal = {npj Computational '
 'Materials}',
 '@article{wang_kauwe_murdock_sparks_2021, place={Cambridge}, '
 'title={Compositionally-Restricted Attention-Based Network for Materials '
 'Property Prediction}, DOI={10.26434/chemrxiv.11869026.v3}, '
 'journal={ChemRxiv}, publisher={Cambridge Open Engage}, author={Wang, Anthony '
 'and Kauwe, Steven and Murdock, Ryan and Sparks, Taylor}, year={2021}} This '
 'content is a preprint and has not been peer-reviewed.']
```

### User metadata:

```
{'algorithm_version': '1.2.7'}
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
{'python': [['ax_platform==0.2.3',
             'crabnet==1.2.7',
             'scikit_learn',
             'matbench==0.5',
             'matplotlib==3.5.0',
             'pandas==1.3.5',
             'ax-platform==0.2.3',
             'pyro-ppl==1.8.0',
             'plotly==5.5.0',
             'submitit==1.4.1',
             'cloudpickle==2.0.0']]}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3457| 0.7883| 0.3564| 6.7229 |
 | fold_1 | 0.3668| 0.8431| 0.3281| 6.4005 |
 | fold_2 | 0.3931| 1.0137| 0.4027| 11.1003 |
 | fold_3 | 0.3721| 0.8858| 0.4132| 9.3770 |
 | fold_4 | 0.3381| 0.8085| 0.4338| 7.2568 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3632 | 0.3931 | 0.3381 | 0.0196 |
| rmse | 0.8679 | 1.0137 | 0.7883 | 0.0801 |
| mape* | 0.3868 | 0.4338 | 0.3281 | 0.0388 |
| max_error | 8.1715 | 11.1003 | 6.4005 | 1.7946 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'N': 4, 'alpha': 0.48871585125853706, 'batch_size': 106, 'betas': [0.8316306312719108, 0.8958260465822976], 'bias': False, 'criterion': 'RobustL2', 'd_model': 860, 'dim_feedforward': 1411, 'dropout':...` |
| fold_1 | `{'N': 3, 'alpha': 0.6316979414711735, 'batch_size': 70, 'betas': [0.7728603241989385, 0.9438804169876437], 'bias': False, 'criterion': 'RobustL2', 'd_model': 940, 'dim_feedforward': 1702, 'dropout': 0...` |
| fold_2 | `{'N': 4, 'alpha': 0.5969894183232147, 'batch_size': 84, 'betas': [0.7123950881835376, 0.8704530737662193], 'bias': False, 'criterion': 'RobustL2', 'd_model': 726, 'dim_feedforward': 1024, 'dropout': 4...` |
| fold_3 | `{'N': 3, 'alpha': 0.6335838688405715, 'batch_size': 100, 'betas': [0.815471216688928, 0.9330437529491037], 'bias': False, 'criterion': 'RobustL2', 'd_model': 784, 'dim_feedforward': 1080, 'dropout': 0...` |
| fold_4 | `{'N': 3, 'alpha': 0.6381644715564362, 'batch_size': 103, 'betas': [0.8304237621083581, 0.90535025277763], 'bias': False, 'criterion': 'RobustL2', 'd_model': 784, 'dim_feedforward': 1487, 'dropout': 0....` |




