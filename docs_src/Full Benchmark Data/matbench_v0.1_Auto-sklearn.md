# matbench_v0.1: AutoML-Mat

### Algorithm description: 

This algorithm is a modification of the 'AutoML Benchmark' framework from the publication Conrad2022AutoMLBench. It combines 4 AutoML tools and selects the most performant one. For this purpose, the AutoML tools are each run in a container to solve the problems of the different dependencies. This was simplified for this benchmark, so Docker is not needed. The best framework for the task was selected by hand, so only one AutoML tool is needed. Further information on the implementation can be found in the publication. More details on the used AutoML tool autosklearn can be found in Feurer2015Neur

#### Notes:
Autosklearn (sklearn>=0.24) and Matbench(sklearn>=1.0) have mutually exclusive dependencies for sklearn. In order to run the script, an environment according to the 'requirements' must be created. Installation instructions for autosklearn can be found at https://automl.github.io/auto-sklearn/master/installation.html#installation. Matbench cannot be installed via pip, but must be added via git clone. Link to GitHub from the AutoML Benchmark: https://github.com/mm-tud/automl-materials .

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Auto-sklearn).

### References (in bibtex format): 

```
['@article{Conrad2022AutoMLBench, author  = {Conrad, Felix and M{"a}lzer, '
 'Mauritz and Schwarzenberger, Michael and Wiemer, Hajo and Ihlenfeldt, '
 'Steffen}, title   = {Benchmarking AutoML for regression tasks on small '
 'tabular data in materials design}, journal = {Scientific Reports}, year    = '
 '{2022}, month   = {Nov}, day     = {11}, volume  = {12}, issn    = '
 '{2045-2322}, doi     = {10.1038/s41598-022-23327-1}, url     = '
 '{https://doi.org/10.1038/s41598-022-23327-1}}',
 '@inproceedings{feurer-neurips15a, title     = {Efficient and Robust '
 'Automated Machine Learning}, author    = {Feurer, Matthias and Klein, Aaron '
 'and Eggensperger, Katharina and Springenberg, Jost and Blum, Manuel and '
 'Hutter, Frank}, booktitle = {Advances in Neural Information Processing '
 'Systems 28 (2015)}, pages     = {2962--2970}, year      = {2015}}']
```

### User metadata:

```
{}
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
{'python': ['gxx_linux-64==12.2.0',
            'gcc_linux-64==12.2.0',
            'swig==4.1.0',
            'auto-sklearn==0.15.0',
            'numpy==1.23.4',
            'pandas==1.5.1',
            'monty==2022.4.26',
            'matminer==0.8.0',
            'jupyter==1.0.0']}
```

### Task data:

#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 97.1669| 139.5237| 0.0643| 463.0130 |
 | fold_1 | 70.6172| 97.1152| 0.0521| 399.3569 |
 | fold_2 | 83.3158| 114.2351| 0.0586| 369.3930 |
 | fold_3 | 83.7402| 106.0132| 0.0600| 270.0560 |
 | fold_4 | 76.6812| 113.4013| 0.0592| 377.1294 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 82.3043 | 97.1669 | 70.6172 | 8.8565 |
| rmse | 114.0577 | 139.5237 | 97.1152 | 14.1474 |
| mape* | 0.0588 | 0.0643 | 0.0521 | 0.0039 |
| max_error | 375.7897 | 463.0130 | 270.0560 | 62.2666 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




