# matbench_v0.1: CrabNet v1.2.1

### Algorithm description: 

Fit CrabNet with default hyperparameters to serve as a baseline for Ax+CrabNet v1.2.1.

#### Notes:
A Jupyter notebook is provided which contains additional details about the run of the algorithm.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_CrabNet_v1.2.1).

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
{'python': [['crabnet==1.2.1', 'scikit_learn==1.0.2', 'matbench==0.5']]}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3489| 0.8079| 0.4441| 5.6781 |
 | fold_1 | 0.3674| 0.8399| 0.3349| 7.0404 |
 | fold_2 | 0.4106| 1.0092| 0.4539| 10.2572 |
 | fold_3 | 0.3677| 0.8437| 0.4181| 6.1608 |
 | fold_4 | 0.3839| 0.9019| 0.4944| 7.4912 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3757 | 0.4106 | 0.3489 | 0.0207 |
| rmse | 0.8805 | 1.0092 | 0.8079 | 0.0711 |
| mape* | 0.4291 | 0.4944 | 0.3349 | 0.0531 |
| max_error | 7.3256 | 10.2572 | 5.6781 | 1.5984 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'N': 3, 'adam': False, 'alpha': 0.5, 'base_lr': 0.0001, 'betas': [0.9, 0.999], 'bias': False, 'criterion': None, 'd_model': 512, 'dim_feedforward': 2048, 'dropout': 0.1, 'elem_prop': 'mat2vec', 'emb_...` |
| fold_1 | `{'N': 3, 'adam': False, 'alpha': 0.5, 'base_lr': 0.0001, 'betas': [0.9, 0.999], 'bias': False, 'criterion': None, 'd_model': 512, 'dim_feedforward': 2048, 'dropout': 0.1, 'elem_prop': 'mat2vec', 'emb_...` |
| fold_2 | `{'N': 3, 'adam': False, 'alpha': 0.5, 'base_lr': 0.0001, 'betas': [0.9, 0.999], 'bias': False, 'criterion': None, 'd_model': 512, 'dim_feedforward': 2048, 'dropout': 0.1, 'elem_prop': 'mat2vec', 'emb_...` |
| fold_3 | `{'N': 3, 'adam': False, 'alpha': 0.5, 'base_lr': 0.0001, 'betas': [0.9, 0.999], 'bias': False, 'criterion': None, 'd_model': 512, 'dim_feedforward': 2048, 'dropout': 0.1, 'elem_prop': 'mat2vec', 'emb_...` |
| fold_4 | `{'N': 3, 'adam': False, 'alpha': 0.5, 'base_lr': 0.0001, 'betas': [0.9, 0.999], 'bias': False, 'criterion': None, 'd_model': 512, 'dim_feedforward': 2048, 'dropout': 0.1, 'elem_prop': 'mat2vec', 'emb_...` |




