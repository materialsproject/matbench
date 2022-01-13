# matbench_v0.1: Ax+CrabNet v1.2.1

### Algorithm description: 

Use Ax Bayesian adaptive design to simultaneously optimize 23 hyperparameters of CrabNet. 100 sequential design iterations were used, and parameters were chosen based on a combination of intuition and algorithm/data constraints (e.g. elemental featurizers which were missing elements contained in the dataset were removed). The first 46 iterations (23*2 parameters) were based on SOBOL sampling to create a rough initial model, while the remaining 56 iterations were Bayesian adaptive design iterations. For the inner loops (where hyperparameter optimization is performed), the average MAE across each of the five inner folds was used as Ax's objective to minimize. The best parameter set was then trained on all the inner fold data and used to predict on the test set (unknown during hyperparameter optimization). This is nested cross-validation, and is computationally expensive.

#### Notes:
A Jupyter notebook is provided which contains additional details about the run of the algorithm. If you decide to run this yourself, because it can take several days to run, be sure to set the `dummy` variable to True and run an initial test that it runs free of errors.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Ax_CrabNet_v1.2.1).

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
{'algorithm_version': '1.2.1'}
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
             'crabnet==1.2.1',
             'scikit_learn==1.0.2',
             'matbench==0.5',
             'kaleido==0.2.1']]}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3465| 0.8037| 0.3899| 5.5211 |
 | fold_1 | 0.4029| 0.9289| 0.3584| 7.2696 |
 | fold_2 | 0.3599| 0.9700| 0.3834| 11.0998 |
 | fold_3 | 0.3324| 0.7836| 0.3411| 5.8159 |
 | fold_4 | 0.3412| 0.8500| 0.4276| 7.2613 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3566 | 0.4029 | 0.3324 | 0.0248 |
| rmse | 0.8673 | 0.9700 | 0.7836 | 0.0717 |
| mape* | 0.3801 | 0.4276 | 0.3411 | 0.0295 |
| max_error | 7.3935 | 11.0998 | 5.5211 | 1.9882 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'N': 4, 'alpha': 0.8790919451473411, 'batch_size': 69, 'betas': [0.5216069223062726, 0.7117768790338862], 'bias': True, 'criterion': 'RobustL1', 'd_model': 860, 'dim_feedforward': 3498, 'dropout': 0....` |
| fold_1 | `{'N': 5, 'alpha': 0.7990423841817611, 'batch_size': 165, 'betas': [0.6461252540288698, 0.7283172840513323], 'bias': False, 'criterion': 'RobustL1', 'd_model': 516, 'dim_feedforward': 2663, 'dropout': ...` |
| fold_2 | `{'N': 3, 'alpha': 0.8041617902337612, 'batch_size': 63, 'betas': [0.711095287462972, 0.9476000614084613], 'bias': False, 'criterion': 'RobustL1', 'd_model': 660, 'dim_feedforward': 3469, 'dropout': 0....` |
| fold_3 | `{'N': 3, 'alpha': 1.0, 'batch_size': 241, 'betas': [0.5591583071453617, 0.5830227398533708], 'bias': True, 'criterion': 'RobustL1', 'd_model': 940, 'dim_feedforward': 1981, 'dropout': 0.00347905979314...` |
| fold_4 | `{'N': 6, 'alpha': 0.7344910928263977, 'batch_size': 125, 'betas': [0.5574111505617741, 0.9346732886315889], 'bias': False, 'criterion': 'RobustL1', 'd_model': 288, 'dim_feedforward': 1393, 'dropout': ...` |




