# matbench_v0.1: RF-Regex Steels

### Algorithm description: 

The RF algorithm from sklearn is used.

#### Notes:
No special considerations required. Key is to convert the composition string properly into a table

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_RFLR).

### References (in bibtex format): 

```
''
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
{'python': ['scikit-learn==0.24.1', 'numpy==1.20.1', 'matbench==0.1.0']}
```

### Task data:

#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 97.5404| 135.2950| 0.0660| 500.0100 |
 | fold_1 | 86.2789| 120.2379| 0.0620| 422.3500 |
 | fold_2 | 79.5099| 114.1154| 0.0559| 357.8433 |
 | fold_3 | 94.5817| 128.5511| 0.0678| 328.0567 |
 | fold_4 | 95.0372| 142.2333| 0.0720| 505.2967 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 90.5896 | 97.5404 | 79.5099 | 6.7138 |
| rmse | 128.0865 | 142.2333 | 114.1154 | 10.0906 |
| mape* | 0.0647 | 0.0720 | 0.0559 | 0.0054 |
| max_error | 422.7113 | 505.2967 | 328.0567 | 72.0596 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




