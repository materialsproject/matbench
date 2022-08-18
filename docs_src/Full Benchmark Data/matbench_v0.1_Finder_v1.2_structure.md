# matbench_v0.1: Finder_v1.2 structure-based version

### Algorithm description: 

Formula graph self-attention network for representation-domain independent materials discovery (Finder). Formula graph is a general representation of crystal structure and chemical composition for graph neural networks (GNNs). Finder GNN can therefore be used for materials property prediction with or without crystal structure. Please see the related publication (https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202200164) and the github repository for more details (https://github.com/ihalage/Finder).

#### Notes:
An example python script with instructions to evaluate Finder algorithm on matbench suite is provided.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Finder_v1.2_structure).

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
 | fold_0 | 0.2068| 0.7005| 0.0727| 14.8493 |
 | fold_1 | 0.2879| 1.0938| 0.0903| 20.5043 |
 | fold_2 | 0.4186| 2.9374| 0.0885| 59.0606 |
 | fold_3 | 0.3187| 2.1634| 0.0740| 48.5382 |
 | fold_4 | 0.3663| 1.7113| 0.1228| 28.3808 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3197 | 0.4186 | 0.2068 | 0.0717 |
| rmse | 1.7213 | 2.9374 | 0.7005 | 0.7887 |
| mape* | 0.0897 | 0.1228 | 0.0727 | 0.0181 |
| max_error | 34.2666 | 59.0606 | 14.8493 | 16.8493 |


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
 | fold_0 | 30.4010| 59.8060| 24.0386| 307.2029 |
 | fold_1 | 48.3155| 112.6815| 0.3680| 673.9473 |
 | fold_2 | 64.6416| 177.1717| 0.7375| 916.1028 |
 | fold_3 | 38.6522| 86.1866| 0.3161| 568.6914 |
 | fold_4 | 48.6590| 164.6126| 0.5689| 1581.4571 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 46.1339 | 64.6416 | 30.4010 | 11.4644 |
| rmse | 120.0917 | 177.1717 | 59.8060 | 44.8978 |
| mape* | 5.2058 | 24.0386 | 0.3161 | 9.4176 |
| max_error | 809.4803 | 1581.4571 | 307.2029 | 432.6540 |


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
 | fold_0 | 0.0881| 0.1346| 0.0702| 0.9501 |
 | fold_1 | 0.0915| 0.1478| 0.0740| 1.4842 |
 | fold_2 | 0.0897| 0.1392| 0.0712| 0.9853 |
 | fold_3 | 0.0931| 0.1415| 0.0731| 0.9482 |
 | fold_4 | 0.0925| 0.1428| 0.0729| 0.9433 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0910 | 0.0931 | 0.0881 | 0.0018 |
| rmse | 0.1412 | 0.1478 | 0.1346 | 0.0043 |
| mape* | 0.0723 | 0.0740 | 0.0702 | 0.0014 |
| max_error | 1.0622 | 1.4842 | 0.9433 | 0.2115 |


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
 | fold_0 | 0.0671| 0.1270| 0.0447| 1.5412 |
 | fold_1 | 0.0707| 0.1402| 0.0469| 1.6242 |
 | fold_2 | 0.0640| 0.1223| 0.0429| 1.1117 |
 | fold_3 | 0.0743| 0.1353| 0.0532| 0.9727 |
 | fold_4 | 0.0703| 0.1344| 0.0474| 1.3475 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0693 | 0.0743 | 0.0640 | 0.0035 |
| rmse | 0.1318 | 0.1402 | 0.1223 | 0.0064 |
| mape* | 0.0470 | 0.0532 | 0.0429 | 0.0035 |
| max_error | 1.3194 | 1.6242 | 0.9727 | 0.2475 |


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
 | fold_0 | 0.0343| 0.1006| 0.3993| 5.3738 |
 | fold_1 | 0.0332| 0.0949| 0.2940| 4.9769 |
 | fold_2 | 0.0338| 0.0882| 0.2527| 2.2726 |
 | fold_3 | 0.0366| 0.2927| 0.2853| 45.1834 |
 | fold_4 | 0.0338| 0.0892| 0.3819| 2.0420 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0343 | 0.0366 | 0.0332 | 0.0012 |
| rmse | 0.1331 | 0.2927 | 0.0882 | 0.0799 |
| mape* | 0.3226 | 0.3993 | 0.2527 | 0.0574 |
| max_error | 11.9698 | 45.1834 | 2.0420 | 16.6622 |


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
 | fold_0 | 0.2182| 0.4971| 2.6076| 6.3889 |
 | fold_1 | 0.2213| 0.5032| 2.9288| 7.2332 |
 | fold_2 | 0.2177| 0.4878| 4.3448| 7.6676 |
 | fold_3 | 0.2194| 0.5090| 7.5606| 7.5448 |
 | fold_4 | 0.2198| 0.4975| 4.4658| 6.5257 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2193 | 0.2213 | 0.2177 | 0.0012 |
| rmse | 0.4989 | 0.5090 | 0.4878 | 0.0071 |
| mape* | 4.3815 | 7.5606 | 2.6076 | 1.7534 |
| max_error | 7.0720 | 7.6676 | 6.3889 | 0.5233 |


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
 | fold_0 | 0.0330| 0.0635| 0.0328| 0.8298 |
 | fold_1 | 0.0337| 0.0670| 0.0344| 0.8875 |
 | fold_2 | 0.0313| 0.0551| 0.0311| 0.8150 |
 | fold_3 | 0.0314| 0.0565| 0.0294| 0.7990 |
 | fold_4 | 0.0305| 0.0549| 0.0280| 0.8683 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0320 | 0.0337 | 0.0305 | 0.0012 |
| rmse | 0.0594 | 0.0670 | 0.0549 | 0.0050 |
| mape* | 0.0311 | 0.0344 | 0.0280 | 0.0023 |
| max_error | 0.8399 | 0.8875 | 0.7990 | 0.0331 |


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
 | fold_0 | 58.5674| 156.9785| 0.0794| 1706.8711 |
 | fold_1 | 43.6763| 85.9967| 0.0814| 882.3383 |
 | fold_2 | 47.4812| 109.3605| 0.0810| 850.8088 |
 | fold_3 | 55.2361| 153.8394| 0.0946| 1506.3175 |
 | fold_4 | 48.7417| 114.2167| 0.0847| 978.8324 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 50.7406 | 58.5674 | 43.6763 | 5.4036 |
| rmse | 124.0783 | 156.9785 | 85.9967 | 27.3211 |
| mape* | 0.0842 | 0.0946 | 0.0794 | 0.0055 |
| max_error | 1185.0336 | 1706.8711 | 850.8088 | 352.5301 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




