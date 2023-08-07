# matbench_v0.1: GN-OA v1

### Algorithm description: 

Crystal structure prediction by combining graph network and optimization algorithm, GN-OA v1.

#### Notes:


Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_GN-OA).

### References (in bibtex format): 

```
('@article{cheng_crystal_2022,\n'
 '  doi = {10.1038/s41467-022-29241-4},\n'
 '  url = {https://www.nature.com/articles/s41467-022-29241-4},\n'
 '  year = {2022},\n'
 '  month = mar,\n'
 '  publisher = {Nature Publishing Group},\n'
 '  volume = {13},\n'
 '  number = {1},\n'
 '  pages={1492},\n'
 '  author = {Guanjian Cheng and Xin-Gao Gong and Wan-Jian Yin},\n'
 '  title = {Crystal structure prediction by combining graph network and '
 'optimization algorithm},\n'
 '  journal = {Nature Communications}\n'
 '}')
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
{'python': ['pymatgen==2020.8.3',
            'tensorflow==2.3.0',
            'megnet==1.1.8',
            'megnet==1.1.8',
            'hyperopt==0.2.4',
            'sko==0.6.1',
            'matbench==0.1.0']}
```

### Task data:

#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0245| 0.0626| 12.5887| 2.1366 |
 | fold_1 | 0.0245| 0.0616| 7.9466| 1.9405 |
 | fold_2 | 0.0249| 0.0648| 9.2633| 2.4150 |
 | fold_3 | 0.0249| 0.0632| 11.8882| 2.1705 |
 | fold_4 | 0.0250| 0.0658| 12.1946| 1.7974 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0248 | 0.0250 | 0.0245 | 0.0002 |
| rmse | 0.0636 | 0.0658 | 0.0616 | 0.0015 |
| mape* | 10.7763 | 12.5887 | 7.9466 | 1.8346 |
| max_error | 2.0920 | 2.4150 | 1.7974 | 0.2108 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




