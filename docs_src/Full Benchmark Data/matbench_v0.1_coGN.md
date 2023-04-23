# matbench_v0.1: coGN

### Algorithm description: 

Connectivity optimized Graph Network

#### Notes:
We found that there is a strong interdependency between crystal preprocessing for GNNs and GNN architectures. Our model 'coGN' was optimized with respect to both aspects.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_coGN).

### References (in bibtex format): 

```
('@misc{ruff2023connectivity, title={Connectivity Optimized Nested Graph '
 'Networks for Crystal Structures}, author={Robin Ruff and Patrick Reiser and '
 'Jan Stühmer and Pascal Friederich}, year={2023}, eprint={2302.14102}, '
 'archivePrefix={arXiv}, primaryClass={cs.LG}}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 9/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✓ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['git+https://github.com/robinruff/graphlist@dcbf79e',
            'kgcnn==3.0.0']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1775| 1.1939| 0.0588| 31.1751 |
 | fold_1 | 0.2663| 1.2545| 0.0928| 19.5706 |
 | fold_2 | 0.4209| 3.1241| 0.0910| 58.7728 |
 | fold_3 | 0.2986| 2.3053| 0.0561| 50.6162 |
 | fold_4 | 0.3805| 2.3953| 0.1318| 34.7823 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3088 | 0.4209 | 0.1775 | 0.0859 |
| rmse | 2.0546 | 3.1241 | 1.1939 | 0.7353 |
| mape* | 0.0861 | 0.1318 | 0.0561 | 0.0276 |
| max_error | 38.9834 | 58.7728 | 19.5706 | 14.0173 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 22.2500| 37.9580| 16.0047| 159.0532 |
 | fold_1 | 37.2341| 93.9861| 0.3232| 511.9405 |
 | fold_2 | 60.1682| 173.0409| 0.6480| 886.4798 |
 | fold_3 | 24.3924| 46.1018| 0.2122| 300.6775 |
 | fold_4 | 41.7814| 154.7034| 0.5589| 1515.5614 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 37.1652 | 60.1682 | 22.2500 | 13.6825 |
| rmse | 101.1580 | 173.0409 | 37.9580 | 54.9748 |
| mape* | 3.5494 | 16.0047 | 0.2122 | 6.2296 |
| max_error | 674.7425 | 1515.5614 | 159.0532 | 486.6567 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0681| 0.1078| 0.0538| 0.9176 |
 | fold_1 | 0.0700| 0.1145| 0.0570| 1.0842 |
 | fold_2 | 0.0685| 0.1084| 0.0544| 0.9597 |
 | fold_3 | 0.0699| 0.1110| 0.0550| 0.9638 |
 | fold_4 | 0.0679| 0.1091| 0.0530| 0.7744 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0689 | 0.0700 | 0.0679 | 0.0009 |
| rmse | 0.1102 | 0.1145 | 0.1078 | 0.0024 |
| mape* | 0.0547 | 0.0570 | 0.0530 | 0.0013 |
| max_error | 0.9399 | 1.0842 | 0.7744 | 0.0997 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0521| 0.1067| 0.0346| 1.6521 |
 | fold_1 | 0.0547| 0.1140| 0.0365| 1.3120 |
 | fold_2 | 0.0495| 0.0998| 0.0334| 1.1084 |
 | fold_3 | 0.0579| 0.1142| 0.0410| 1.1212 |
 | fold_4 | 0.0536| 0.1064| 0.0359| 1.4031 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0535 | 0.0579 | 0.0495 | 0.0028 |
| rmse | 0.1082 | 0.1142 | 0.0998 | 0.0054 |
| mape* | 0.0363 | 0.0410 | 0.0334 | 0.0026 |
| max_error | 1.3194 | 1.6521 | 1.1084 | 0.2008 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0174| 0.0597| 0.2045| 3.8249 |
 | fold_1 | 0.0167| 0.0473| 0.1188| 2.5255 |
 | fold_2 | 0.0168| 0.0433| 0.1260| 1.4799 |
 | fold_3 | 0.0173| 0.0464| 0.1515| 1.2865 |
 | fold_4 | 0.0169| 0.0448| 0.1966| 1.6626 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0170 | 0.0174 | 0.0167 | 0.0003 |
| rmse | 0.0483 | 0.0597 | 0.0433 | 0.0059 |
| mape* | 0.1595 | 0.2045 | 0.1188 | 0.0353 |
| max_error | 2.1559 | 3.8249 | 1.2865 | 0.9358 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1532| 0.3904| 1.5736| 7.3352 |
 | fold_1 | 0.1561| 0.3928| 1.9621| 6.7683 |
 | fold_2 | 0.1570| 0.4010| 3.8426| 7.3269 |
 | fold_3 | 0.1549| 0.3909| 5.5071| 7.1401 |
 | fold_4 | 0.1583| 0.4027| 4.0927| 6.8235 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1559 | 0.1583 | 0.1532 | 0.0017 |
| rmse | 0.3956 | 0.4027 | 0.3904 | 0.0052 |
| mape* | 3.3956 | 5.5071 | 1.5736 | 1.4504 |
| max_error | 7.0788 | 7.3352 | 6.7683 | 0.2419 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9184| 0.9162| 0.9055| 0.9162 |
 | fold_1 | 0.9116| 0.9092| 0.8976| 0.9092 |
 | fold_2 | 0.9138| 0.9117| 0.9003| 0.9117 |
 | fold_3 | 0.9155| 0.9135| 0.9024| 0.9135 |
 | fold_4 | 0.9137| 0.9113| 0.9000| 0.9113 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9146 | 0.9184 | 0.9116 | 0.0023 |
| balanced_accuracy | 0.9124 | 0.9162 | 0.9092 | 0.0023 |
| f1 | 0.9012 | 0.9055 | 0.8976 | 0.0027 |
| rocauc | 0.9124 | 0.9162 | 0.9092 | 0.0023 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0273| 0.0577| 0.0272| 0.8584 |
 | fold_1 | 0.0280| 0.0616| 0.0297| 0.9449 |
 | fold_2 | 0.0256| 0.0494| 0.0254| 0.8721 |
 | fold_3 | 0.0271| 0.0538| 0.0277| 0.8234 |
 | fold_4 | 0.0266| 0.0545| 0.0240| 0.8305 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0269 | 0.0280 | 0.0256 | 0.0008 |
| rmse | 0.0554 | 0.0616 | 0.0494 | 0.0041 |
| mape* | 0.0268 | 0.0297 | 0.0240 | 0.0019 |
| max_error | 0.8659 | 0.9449 | 0.8234 | 0.0434 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 32.1211| 62.9439| 0.0581| 622.4674 |
 | fold_1 | 29.1911| 54.8274| 0.0602| 480.0777 |
 | fold_2 | 30.2587| 55.7380| 0.0661| 497.7857 |
 | fold_3 | 30.7953| 62.8746| 0.0695| 593.3290 |
 | fold_4 | 26.1921| 52.1653| 0.0547| 530.2232 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 29.7117 | 32.1211 | 26.1921 | 1.9968 |
| rmse | 57.7099 | 62.9439 | 52.1653 | 4.4047 |
| mape* | 0.0617 | 0.0695 | 0.0547 | 0.0054 |
| max_error | 544.7766 | 622.4674 | 480.0777 | 54.7706 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_are...` |




