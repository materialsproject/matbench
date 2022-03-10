# matbench_v0.1: Ax/SAASBO CrabNet v1.2.7

### Algorithm description: 

Recently, SAASBO has been demonstrated to be a highly effect high-dimensional Bayesian optimization scheme. Here, we use [Ax/SAASBO Bayesian adaptive design](https://ax.dev/tutorials/saasbo.html) to simultaneously optimize 23 hyperparameters of [CrabNet](https://crabnet.readthedocs.io/). `100` sequential design iterations were used, and parameters were chosen based on a combination of intuition and algorithm/data constraints (e.g. elemental featurizers which were missing elements contained in the dataset were removed). The first `10` iterations were based on SOBOL sampling to create a rough initial model, while the remaining `90` iterations were SAASBO Bayesian adaptive design iterations. For the innerloops (where hyperparameter optimization is performed), the average MAE across each of the *five inner folds* was used as Ax's objective to minimize. The best parameter set was then trained on all the inner fold data and used to predict on the test set (unknown during hyperparameter optimization). This is nested cross-validation (CV), and is computationally expensive. See [automatminer: running a benchmark](https://hackingmaterials.lbl.gov/automatminer/advanced.html#running-a-benchmark) for more information on nested CV.

#### Notes:
A Jupyter notebook is provided which contains additional details about the run of the algorithm. If you decide to run this yourself, because it can take several days to run, be sure to set the `dummy` variable to True and run an initial test to ensure it runs free of errors.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Ax_SAASBO_CrabNet_v1.2.7).

### References (in bibtex format): 

```
['@article{Wang2021crabnet,  author = {Wang, Anthony Yu-Tung and Kauwe, Steven '
 'K. and Murdock, Ryan J. and Sparks, Taylor D.},  year = {2021},  title = '
 '{Compositionally restricted attention-based network for materials property '
 'predictions},  pages = {77},  volume = {7},  number = {1},  doi = '
 '{10.1038/s41524-021-00545-1},  publisher = {{Nature Publishing Group}},  '
 'shortjournal = {npj Comput. Mater.},  journal = {npj Computational '
 'Materials}',
 '@article{erikssonHighDimensionalBayesianOptimization2021,             title '
 '= {High-{{Dimensional Bayesian Optimization}} with {{Sparse Axis-Aligned '
 'Subspaces}}}, author = {Eriksson, David and Jankowiak, Martin},             '
 'year = {2021}, month = jun, journal = {arXiv:2103.00349 [cs, stat]}, eprint '
 '= {2103.00349}, eprinttype = {arxiv}, primaryclass = {cs, stat}, '
 'archiveprefix = {arXiv}, langid = {english}, keywords = {Computer Science - '
 'Machine Learning,Statistics - Machine Learning}}']
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
{'python': [['matplotlib==3.5.0',
             'pandas==1.3.5',
             'ax-platform==0.2.3',
             'pyro-ppl==1.8.0',
             'plotly==5.5.0',
             'crabnet==1.2.5',
             'scikit-learn==1.0.2',
             'submitit==1.4.1',
             'matbench==0.5',
             'cloudpickle==2.0.0']]}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3214| 0.7561| 0.3429| 5.7689 |
 | fold_1 | 0.3385| 0.7832| 0.2888| 6.3999 |
 | fold_2 | 0.3383| 0.9170| 0.3705| 11.1001 |
 | fold_3 | 0.3327| 0.8318| 0.3375| 6.3998 |
 | fold_4 | 0.3239| 0.7733| 0.4494| 6.2801 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3310 | 0.3385 | 0.3214 | 0.0071 |
| rmse | 0.8123 | 0.9170 | 0.7561 | 0.0581 |
| mape* | 0.3578 | 0.4494 | 0.2888 | 0.0528 |
| max_error | 7.1897 | 11.1001 | 5.7689 | 1.9690 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'N': 2, 'alpha': 0.9999998976773382, 'batch_size': 32, 'betas': [0.516639075322522, 0.5264563063760717], 'bias': True, 'criterion': 'RobustL1', 'd_model': 890, 'dim_feedforward': 4096, 'dropout': 0.0...` |
| fold_1 | `{'N': 3, 'alpha': 0.8019048022306048, 'batch_size': 32, 'betas': [0.5000000000000002, 0.5000000000000008], 'bias': False, 'criterion': 'RobustL1', 'd_model': 690, 'dim_feedforward': 1179, 'dropout': 1...` |
| fold_2 | `{'N': 4, 'alpha': 0.6659722577118914, 'batch_size': 32, 'betas': [0.5000000000015171, 0.5000000000018112], 'bias': False, 'criterion': 'RobustL1', 'd_model': 1024, 'dim_feedforward': 1903, 'dropout': ...` |
| fold_3 | `{'N': 5, 'alpha': 0.6209436008996955, 'batch_size': 104, 'betas': [0.7642325868682494, 0.7978278147950777], 'bias': False, 'criterion': 'RobustL1', 'd_model': 1024, 'dim_feedforward': 2322, 'dropout':...` |
| fold_4 | `{'N': 2, 'alpha': 0.9402737238860547, 'batch_size': 32, 'betas': [0.5212575617080646, 0.9998999993348248], 'bias': False, 'criterion': 'RobustL1', 'd_model': 1024, 'dim_feedforward': 2074, 'dropout': ...` |




