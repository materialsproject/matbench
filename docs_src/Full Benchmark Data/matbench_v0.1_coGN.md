# matbench_v0.1: coGN

### Algorithm description: 

Connectivity optimized Graph Network

#### Notes:


Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_coGN).

### References (in bibtex format): 

```
''
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
{'python': ['git+https://github.com/matbench-submission-coGN/CrystalGNNs']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1881| 1.2052| 0.0615| 30.4824 |
 | fold_1 | 0.2652| 1.2112| 0.0940| 19.0093 |
 | fold_2 | 0.4088| 3.1209| 0.0889| 58.6299 |
 | fold_3 | 0.2851| 2.1472| 0.0561| 45.6425 |
 | fold_4 | 0.3775| 2.2437| 0.1414| 30.4891 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3049 | 0.4088 | 0.1881 | 0.0796 |
| rmse | 1.9857 | 3.1209 | 1.2052 | 0.7198 |
| mape* | 0.0884 | 0.1414 | 0.0561 | 0.0304 |
| max_error | 36.8507 | 58.6299 | 19.0093 | 13.7911 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 20.4129| 34.0208| 16.7510| 147.5706 |
 | fold_1 | 36.8559| 88.0698| 0.2906| 482.6166 |
 | fold_2 | 58.2834| 170.5519| 0.6320| 886.2500 |
 | fold_3 | 24.6639| 44.8349| 0.2278| 285.3291 |
 | fold_4 | 38.6134| 151.8924| 0.5330| 1554.5435 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 35.7659 | 58.2834 | 20.4129 | 13.2311 |
| rmse | 97.8740 | 170.5519 | 34.0208 | 55.1118 |
| mape* | 3.6869 | 16.7510 | 0.2278 | 6.5337 |
| max_error | 671.2620 | 1554.5435 | 147.5706 | 506.9892 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0679| 0.1077| 0.0542| 0.9003 |
 | fold_1 | 0.0703| 0.1121| 0.0567| 1.0508 |
 | fold_2 | 0.0688| 0.1107| 0.0554| 0.9464 |
 | fold_3 | 0.0695| 0.1112| 0.0547| 0.9811 |
 | fold_4 | 0.0698| 0.1123| 0.0546| 0.8259 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0693 | 0.0703 | 0.0679 | 0.0008 |
| rmse | 0.1108 | 0.1123 | 0.1077 | 0.0017 |
| mape* | 0.0551 | 0.0567 | 0.0542 | 0.0009 |
| max_error | 0.9409 | 1.0508 | 0.8259 | 0.0756 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0517| 0.1052| 0.0341| 1.6432 |
 | fold_1 | 0.0532| 0.1101| 0.0351| 1.3853 |
 | fold_2 | 0.0492| 0.1031| 0.0329| 1.4786 |
 | fold_3 | 0.0582| 0.1152| 0.0414| 1.0340 |
 | fold_4 | 0.0538| 0.1079| 0.0363| 1.3281 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0532 | 0.0582 | 0.0492 | 0.0030 |
| rmse | 0.1083 | 0.1152 | 0.1031 | 0.0042 |
| mape* | 0.0360 | 0.0414 | 0.0329 | 0.0029 |
| max_error | 1.3738 | 1.6432 | 1.0340 | 0.2006 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0173| 0.0576| 0.1895| 3.2492 |
 | fold_1 | 0.0167| 0.0459| 0.1057| 2.4144 |
 | fold_2 | 0.0166| 0.0428| 0.1149| 1.7081 |
 | fold_3 | 0.0176| 0.0497| 0.1341| 1.9816 |
 | fold_4 | 0.0168| 0.0463| 0.1862| 2.4855 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0170 | 0.0176 | 0.0166 | 0.0004 |
| rmse | 0.0485 | 0.0576 | 0.0428 | 0.0051 |
| mape* | 0.1461 | 0.1895 | 0.1057 | 0.0353 |
| max_error | 2.3678 | 3.2492 | 1.7081 | 0.5248 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1538| 0.3950| 1.5448| 7.1551 |
 | fold_1 | 0.1546| 0.3971| 1.6310| 6.6974 |
 | fold_2 | 0.1600| 0.4078| 4.1607| 7.3029 |
 | fold_3 | 0.1520| 0.3923| 6.3196| 7.0559 |
 | fold_4 | 0.1589| 0.4046| 4.1557| 6.7068 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1559 | 0.1600 | 0.1520 | 0.0031 |
| rmse | 0.3994 | 0.4078 | 0.3923 | 0.0059 |
| mape* | 3.5624 | 6.3196 | 1.5448 | 1.7952 |
| max_error | 6.9836 | 7.3029 | 6.6974 | 0.2430 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9183| 0.9155| 0.9049| 0.9155 |
 | fold_1 | 0.9141| 0.9118| 0.9005| 0.9118 |
 | fold_2 | 0.9141| 0.9113| 0.9001| 0.9113 |
 | fold_3 | 0.9152| 0.9128| 0.9017| 0.9128 |
 | fold_4 | 0.9149| 0.9124| 0.9013| 0.9124 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9153 | 0.9183 | 0.9141 | 0.0016 |
| balanced_accuracy | 0.9127 | 0.9155 | 0.9113 | 0.0015 |
| f1 | 0.9017 | 0.9049 | 0.9001 | 0.0017 |
| rocauc | 0.9127 | 0.9155 | 0.9113 | 0.0015 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0269| 0.0552| 0.0268| 0.8322 |
 | fold_1 | 0.0281| 0.0629| 0.0289| 0.9489 |
 | fold_2 | 0.0265| 0.0503| 0.0276| 0.7050 |
 | fold_3 | 0.0265| 0.0503| 0.0265| 0.7964 |
 | fold_4 | 0.0268| 0.0553| 0.0255| 0.8531 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0270 | 0.0281 | 0.0265 | 0.0006 |
| rmse | 0.0548 | 0.0629 | 0.0503 | 0.0046 |
| mape* | 0.0271 | 0.0289 | 0.0255 | 0.0011 |
| max_error | 0.8271 | 0.9489 | 0.7050 | 0.0792 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 33.6455| 76.2273| 0.0593| 868.0997 |
 | fold_1 | 30.4823| 62.3384| 0.0602| 635.7188 |
 | fold_2 | 29.1343| 52.4933| 0.0622| 400.1526 |
 | fold_3 | 29.2746| 56.3612| 0.0711| 500.1160 |
 | fold_4 | 27.2320| 52.7534| 0.0566| 535.7434 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 29.9538 | 33.6455 | 27.2320 | 2.1189 |
| rmse | 60.0347 | 76.2273 | 52.4933 | 8.8419 |
| mape* | 0.0619 | 0.0711 | 0.0566 | 0.0049 |
| max_error | 587.9661 | 868.0997 | 400.1526 | 159.0433 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_1 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_2 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_3 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |
| fold_4 | `{'input_block': {'atomic_mass': True, 'atomic_radius': True, 'edge_embedding_args': {'bins_distance': 32, 'bins_voronoi_area': None, 'distance_log_base': 1.0, 'max_distance': 8.0, 'max_voronoi_area': ...` |




