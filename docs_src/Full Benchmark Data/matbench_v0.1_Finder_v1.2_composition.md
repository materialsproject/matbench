# matbench_v0.1: Finder_v1.2 composition-only version

### Algorithm description: 

Formula graph self-attention network for representation-domain independent materials discovery (Finder). Formula graph is a general representation of crystal structure and chemical composition for graph neural networks (GNNs). Finder GNN can therefore be used for materials property prediction with or without crystal structure. Please see the related publication (https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202200164) and the github repository for more details (https://github.com/ihalage/Finder).

#### Notes:
An example python script with instructions to evaluate Finder algorithm on matbench suite is provided.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Finder_v1.2_composition).

### References (in bibtex format): 

```
('@article{Ihalage_2022_Adv_Sci, author = {Ihalage, Achintha and Hao, Yang}, '
 'title = {Formula Graph Self-Attention Network for Representation-Domain '
 'Independent Materials Discovery}, journal = {Advanced Science}, volume = '
 '{9}, number = {18}, pages = {2200164}, keywords = {attention, '
 'epsilon-near-zero, graph-network, machine-learning, materials-informatics}, '
 'doi = {https://doi.org/10.1002/advs.202200164}, url = '
 '{https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202200164}, eprint = '
 '{https://onlinelibrary.wiley.com/doi/pdf/10.1002/advs.202200164}, year = '
 '{2022}}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 8/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': [['spektral==1.1.0',
             'tensorflow==2.9.1',
             'pymatgen==2022.7.19',
             'matminer==0.7.8',
             'numpy==1.23.1',
             'pandas==1.4.3',
             'matplotlib==3.5.2',
             'scikit-learn==1.1.1',
             'scipy==1.8.1',
             'sparse==0.13.0',
             'protobuf==3.19.4']]}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2020| 0.6838| 0.0727| 14.8287 |
 | fold_1 | 0.2675| 1.0293| 0.0874| 19.0338 |
 | fold_2 | 0.4347| 2.9821| 0.0970| 59.0528 |
 | fold_3 | 0.3222| 2.1621| 0.0775| 46.3432 |
 | fold_4 | 0.3754| 1.7371| 0.1209| 27.8804 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3204 | 0.4347 | 0.2020 | 0.0811 |
| rmse | 1.7189 | 2.9821 | 0.6838 | 0.8172 |
| mape* | 0.0911 | 0.1209 | 0.0727 | 0.0171 |
| max_error | 33.4278 | 59.0528 | 14.8287 | 16.7770 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 37.3620| 72.8756| 24.6383| 356.9083 |
 | fold_1 | 45.0031| 109.9971| 0.3200| 696.8793 |
 | fold_2 | 63.3670| 175.1722| 0.6752| 914.8421 |
 | fold_3 | 34.3768| 68.2091| 0.3643| 385.8836 |
 | fold_4 | 59.6980| 178.1554| 0.7090| 1582.3598 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 47.9614 | 63.3670 | 34.3768 | 11.6680 |
| rmse | 120.8819 | 178.1554 | 68.2091 | 47.8021 |
| mape* | 5.3414 | 24.6383 | 0.3200 | 9.6497 |
| max_error | 787.3746 | 1582.3598 | 356.9083 | 447.8694 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0982| 0.1516| 0.0754| 1.1378 |
 | fold_1 | 0.0984| 0.1643| 0.0787| 2.3854 |
 | fold_2 | 0.0986| 0.1548| 0.0771| 1.0763 |
 | fold_3 | 0.0996| 0.1520| 0.0759| 0.9424 |
 | fold_4 | 0.1029| 0.1631| 0.0805| 1.2900 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0996 | 0.1029 | 0.0982 | 0.0018 |
| rmse | 0.1572 | 0.1643 | 0.1516 | 0.0055 |
| mape* | 0.0775 | 0.0805 | 0.0754 | 0.0019 |
| max_error | 1.3664 | 2.3854 | 0.9424 | 0.5216 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0745| 0.1425| 0.0485| 1.5642 |
 | fold_1 | 0.0756| 0.1554| 0.0496| 2.3863 |
 | fold_2 | 0.0737| 0.1420| 0.0485| 1.3227 |
 | fold_3 | 0.0806| 0.1500| 0.0556| 0.9465 |
 | fold_4 | 0.0778| 0.1555| 0.0531| 1.6076 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0764 | 0.0806 | 0.0737 | 0.0025 |
| rmse | 0.1491 | 0.1555 | 0.1420 | 0.0059 |
| mape* | 0.0511 | 0.0556 | 0.0485 | 0.0028 |
| max_error | 1.5655 | 2.3863 | 0.9465 | 0.4728 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0838| 0.2512| 0.6783| 4.2840 |
 | fold_1 | 0.0826| 0.2569| 0.4626| 6.3948 |
 | fold_2 | 0.0843| 0.2537| 0.4024| 4.1659 |
 | fold_3 | 0.0830| 0.2485| 0.5036| 5.4366 |
 | fold_4 | 0.0858| 0.2583| 0.8146| 3.8705 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0839 | 0.0858 | 0.0826 | 0.0011 |
| rmse | 0.2537 | 0.2583 | 0.2485 | 0.0036 |
| mape* | 0.5723 | 0.8146 | 0.4024 | 0.1520 |
| max_error | 4.8304 | 6.3948 | 3.8705 | 0.9462 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2291| 0.4816| 3.3802| 5.8312 |
 | fold_1 | 0.2350| 0.4943| 2.3466| 7.6477 |
 | fold_2 | 0.2326| 0.4808| 4.1827| 7.8152 |
 | fold_3 | 0.2265| 0.4720| 6.1036| 5.4306 |
 | fold_4 | 0.2306| 0.4900| 4.8680| 5.5791 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2308 | 0.2350 | 0.2265 | 0.0029 |
| rmse | 0.4837 | 0.4943 | 0.4720 | 0.0078 |
| mape* | 4.1762 | 6.1036 | 2.3466 | 1.2786 |
| max_error | 6.4608 | 7.8152 | 5.4306 | 1.0467 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.6366| 0.8655| 0.7413| 3.4641 |
 | fold_1 | 0.6773| 0.9258| 0.8507| 3.5402 |
 | fold_2 | 0.6399| 0.8800| 0.7475| 3.3632 |
 | fold_3 | 0.6415| 0.8821| 0.8200| 3.5053 |
 | fold_4 | 0.6294| 0.8620| 0.7008| 3.5391 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.6450 | 0.6773 | 0.6294 | 0.0167 |
| rmse | 0.8831 | 0.9258 | 0.8620 | 0.0227 |
| mape* | 0.7721 | 0.8507 | 0.7008 | 0.0550 |
| max_error | 3.4824 | 3.5402 | 3.3632 | 0.0658 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 50.6994| 106.3352| 0.0791| 891.8557 |
 | fold_1 | 43.3725| 90.7974| 0.0823| 1051.2485 |
 | fold_2 | 47.9669| 103.6501| 0.0802| 706.1363 |
 | fold_3 | 41.0528| 77.7973| 0.0907| 533.1135 |
 | fold_4 | 49.7836| 95.6768| 0.0916| 644.7436 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 46.5751 | 50.6994 | 41.0528 | 3.7415 |
| rmse | 94.8514 | 106.3352 | 77.7973 | 10.1711 |
| mape* | 0.0848 | 0.0916 | 0.0791 | 0.0053 |
| max_error | 765.4195 | 1051.2485 | 533.1135 | 184.2431 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




