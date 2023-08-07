# matbench_v0.1: coNGN

### Algorithm description: 

Connectivity optimized Nested Graph Network

#### Notes:
We found that there is a strong interdependency between crystal preprocessing for GNNs and GNN architectures. Our model 'coNGN' was optimized with respect to both aspects and further adds nested line graphs.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_coNGN).

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
 | fold_0 | 0.1989| 0.9752| 0.0733| 15.1038 |
 | fold_1 | 0.2949| 1.4516| 0.1033| 19.4905 |
 | fold_2 | 0.3904| 2.9550| 0.0841| 58.8654 |
 | fold_3 | 0.2875| 2.1322| 0.0577| 43.9192 |
 | fold_4 | 0.3992| 2.6535| 0.1217| 46.3053 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3142 | 0.3992 | 0.1989 | 0.0740 |
| rmse | 2.0335 | 2.9550 | 0.9752 | 0.7351 |
| mape* | 0.0880 | 0.1217 | 0.0577 | 0.0224 |
| max_error | 36.7368 | 58.8654 | 15.1038 | 16.7227 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 23.7560| 43.2821| 21.8411| 203.2730 |
 | fold_1 | 35.7004| 79.6640| 0.2947| 439.8148 |
 | fold_2 | 53.9029| 152.3112| 0.5833| 852.7897 |
 | fold_3 | 23.9486| 50.7536| 0.1900| 307.8413 |
 | fold_4 | 43.5412| 151.3720| 0.5862| 1496.9020 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 36.1698 | 53.9029 | 23.7560 | 11.5972 |
| rmse | 95.4766 | 152.3112 | 43.2821 | 47.6003 |
| mape* | 4.6990 | 21.8411 | 0.1900 | 8.5725 |
| max_error | 660.1242 | 1496.9020 | 203.2730 | 473.0052 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0662| 0.1052| 0.0523| 0.9334 |
 | fold_1 | 0.0678| 0.1094| 0.0541| 1.1760 |
 | fold_2 | 0.0663| 0.1082| 0.0541| 0.8977 |
 | fold_3 | 0.0674| 0.1084| 0.0525| 0.9771 |
 | fold_4 | 0.0672| 0.1075| 0.0526| 0.7083 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0670 | 0.0678 | 0.0662 | 0.0006 |
| rmse | 0.1078 | 0.1094 | 0.1052 | 0.0014 |
| mape* | 0.0531 | 0.0541 | 0.0523 | 0.0008 |
| max_error | 0.9385 | 1.1760 | 0.7083 | 0.1501 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0482| 0.1016| 0.0315| 1.6329 |
 | fold_1 | 0.0494| 0.1046| 0.0337| 1.3323 |
 | fold_2 | 0.0458| 0.0979| 0.0307| 1.2123 |
 | fold_3 | 0.0536| 0.1108| 0.0380| 0.9669 |
 | fold_4 | 0.0484| 0.1037| 0.0323| 1.3584 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0491 | 0.0536 | 0.0458 | 0.0026 |
| rmse | 0.1037 | 0.1108 | 0.0979 | 0.0042 |
| mape* | 0.0332 | 0.0380 | 0.0307 | 0.0026 |
| max_error | 1.3006 | 1.6329 | 0.9669 | 0.2163 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0181| 0.0584| 0.1996| 3.2378 |
 | fold_1 | 0.0178| 0.0474| 0.1399| 1.7003 |
 | fold_2 | 0.0172| 0.0470| 0.1301| 2.0011 |
 | fold_3 | 0.0182| 0.0506| 0.1838| 1.7068 |
 | fold_4 | 0.0179| 0.0478| 0.2313| 1.5277 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0178 | 0.0182 | 0.0172 | 0.0004 |
| rmse | 0.0502 | 0.0584 | 0.0470 | 0.0043 |
| mape* | 0.1769 | 0.2313 | 0.1301 | 0.0376 |
| max_error | 2.0347 | 3.2378 | 1.5277 | 0.6205 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1694| 0.4187| 1.5208| 7.3460 |
 | fold_1 | 0.1658| 0.4075| 1.9359| 6.3431 |
 | fold_2 | 0.1719| 0.4374| 4.4042| 7.0906 |
 | fold_3 | 0.1663| 0.4287| 4.3879| 7.9674 |
 | fold_4 | 0.1750| 0.4434| 4.1872| 7.0762 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1697 | 0.1750 | 0.1658 | 0.0035 |
| rmse | 0.4271 | 0.4434 | 0.4075 | 0.0129 |
| mape* | 3.2872 | 4.4042 | 1.5208 | 1.2818 |
| max_error | 7.1646 | 7.9674 | 6.3431 | 0.5226 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9122| 0.9096| 0.8981| 0.9096 |
 | fold_1 | 0.9141| 0.9123| 0.9009| 0.9123 |
 | fold_2 | 0.9098| 0.9072| 0.8953| 0.9072 |
 | fold_3 | 0.9104| 0.9085| 0.8966| 0.9085 |
 | fold_4 | 0.9089| 0.9069| 0.8949| 0.9069 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9111 | 0.9141 | 0.9089 | 0.0019 |
| balanced_accuracy | 0.9089 | 0.9123 | 0.9069 | 0.0019 |
| f1 | 0.8972 | 0.9009 | 0.8949 | 0.0022 |
| rocauc | 0.9089 | 0.9123 | 0.9069 | 0.0019 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0295| 0.0612| 0.0287| 0.8405 |
 | fold_1 | 0.0309| 0.0656| 0.0316| 0.9346 |
 | fold_2 | 0.0277| 0.0554| 0.0290| 0.7072 |
 | fold_3 | 0.0283| 0.0554| 0.0285| 0.8785 |
 | fold_4 | 0.0284| 0.0573| 0.0263| 0.8348 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0290 | 0.0309 | 0.0277 | 0.0011 |
| rmse | 0.0590 | 0.0656 | 0.0554 | 0.0039 |
| mape* | 0.0288 | 0.0316 | 0.0263 | 0.0017 |
| max_error | 0.8391 | 0.9346 | 0.7072 | 0.0750 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 33.0209| 76.5063| 0.0568| 747.0843 |
 | fold_1 | 32.6300| 69.5953| 0.0570| 617.0889 |
 | fold_2 | 26.0880| 48.5695| 0.0547| 291.2322 |
 | fold_3 | 25.3229| 43.7936| 0.0615| 335.4448 |
 | fold_4 | 27.3750| 47.2225| 0.0566| 358.0064 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 28.8874 | 33.0209 | 25.3229 | 3.2840 |
| rmse | 57.1375 | 76.5063 | 43.7936 | 13.2674 |
| mape* | 0.0573 | 0.0615 | 0.0547 | 0.0023 |
| max_error | 469.7713 | 747.0843 | 291.2322 | 179.4526 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_1 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_2 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_3 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |
| fold_4 | `{'input_block_cfg': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': 32, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area'...` |




