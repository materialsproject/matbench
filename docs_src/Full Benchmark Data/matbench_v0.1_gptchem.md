# matbench_v0.1: gptchem

### Algorithm description: 

Language-interface (LIFT) fine-tuned ada GPT-3 model (without optimization of finetuning parameters or prompt)

#### Notes:
We use the high-level GPTRegressor and GPTClassifier APIs provided via our gptchem package. The code for running the benchmarks is available at https://github.com/kjappelbaum/gptchem-matbench.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_gptchem).

### References (in bibtex format): 

```
('@article{Jablonka_2023,\n'
 ' doi = {10.26434/chemrxiv-2023-fw8n4},\n'
 ' url = {https:doi.org/10.26434%2Fchemrxiv-2023-fw8n4},\n'
 ' year = 2023,\n'
 ' month = {feb},\n'
 '  publisher = {American Chemical Society ({ACS})},\n'
 ' author = {Kevin Maik Jablonka and Philippe Schwaller and Andres '
 'Ortega-Guerrero and Berend Smit},\n'
 ' title = {Is {GPT}-3 all you need for low-data discovery in chemistry\n'
 '}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 4/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✓ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['git+https://github.com/kjappelbaum/gptchem.git',
            'matbench==0.1.0']}
```

### Task data:

#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.4734| 1.0919| 0.4774| 7.8000 |
 | fold_1 | 0.4451| 1.0520| 0.4390| 9.3300 |
 | fold_2 | 0.4572| 1.1435| 0.4767| 11.7000 |
 | fold_3 | 0.4585| 1.0736| 0.5515| 10.4000 |
 | fold_4 | 0.4376| 1.0074| 0.6414| 7.6000 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.4544 | 0.4734 | 0.4376 | 0.0123 |
| rmse | 1.0737 | 1.1435 | 1.0074 | 0.0449 |
| mape* | 0.5172 | 0.6414 | 0.4390 | 0.0720 |
| max_error | 9.3660 | 11.7000 | 7.6000 | 1.5549 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_expt_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9025| 0.9025| 0.9012| 0.9025 |
 | fold_1 | 0.8852| 0.8851| 0.8824| 0.8851 |
 | fold_2 | 0.8994| 0.8993| 0.8961| 0.8993 |
 | fold_3 | 0.8974| 0.8974| 0.8983| 0.8974 |
 | fold_4 | 0.8984| 0.8984| 0.8984| 0.8984 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8966 | 0.9025 | 0.8852 | 0.0060 |
| balanced_accuracy | 0.8965 | 0.9025 | 0.8851 | 0.0060 |
| f1 | 0.8953 | 0.9012 | 0.8824 | 0.0066 |
| rocauc | 0.8965 | 0.9025 | 0.8851 | 0.0060 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_glass`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.8363| 0.7839| 0.8874| 0.7839 |
 | fold_1 | 0.8143| 0.7684| 0.8703| 0.7684 |
 | fold_2 | 0.8204| 0.7674| 0.8761| 0.7674 |
 | fold_3 | 0.8363| 0.7965| 0.8855| 0.7965 |
 | fold_4 | 0.8151| 0.7646| 0.8718| 0.7646 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8245 | 0.8363 | 0.8143 | 0.0099 |
| balanced_accuracy | 0.7762 | 0.7965 | 0.7646 | 0.0122 |
| f1 | 0.8782 | 0.8874 | 0.8703 | 0.0070 |
| rocauc | 0.7762 | 0.7965 | 0.7646 | 0.0122 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 162.9667| 227.5822| 0.1112| 650.8000 |
 | fold_1 | 161.0762| 270.1202| 0.1211| 1368.2000 |
 | fold_2 | 137.0645| 198.1973| 0.0947| 651.6000 |
 | fold_3 | 135.9887| 195.8941| 0.0973| 720.9000 |
 | fold_4 | 117.9177| 198.3470| 0.0880| 1139.3000 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 143.0028 | 162.9667 | 117.9177 | 16.9642 |
| rmse | 218.0282 | 270.1202 | 195.8941 | 28.5495 |
| mape* | 0.1025 | 0.1211 | 0.0880 | 0.0120 |
| max_error | 906.1600 | 1368.2000 | 650.8000 | 293.9952 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




