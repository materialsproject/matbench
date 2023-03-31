# matbench_v0.1: DimeNet++ (kgcnn v2.1.0)

### Algorithm description: 

Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules. Adapted implementation of `kgcnn`. Original code from https://github.com/gasteigerjo/dimenet. Settings are almost similar compared to original work for QM9. We had to reduce the batch size to 16 and the maximum number of edges or neighbours to 17 due to memory issues (in addition to 5A cutoff). For angles, multi-edges and correct images are taken into account. We added a standard scaler for regression. No additional features were introduced but geometry and atom type. Training was carried out on A100-SXM with 41 GB of memory.

#### Notes:


Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_DimeNetPP_kgcnn_v2.1.0).

### References (in bibtex format): 

```
['@inproceedings{gasteiger_dimenet_2020,\n'
 'title = {Directional Message Passing for Molecular Graphs},\n'
 'author = {Gasteiger, Johannes and Gro{\\ss}, Janek and G{\\"u}nnemann, '
 'Stephan},\n'
 'booktitle={International Conference on Learning Representations (ICLR)},\n'
 'year = {2020} }',
 '@inproceedings{gasteiger_dimenetpp_2020,\n'
 'title = {Fast and Uncertainty-Aware Directional Message Passing for '
 'Non-Equilibrium Molecules},\n'
 'author = {Gasteiger, Johannes and Giri, Shankari and Margraf, Johannes T. '
 'and G{\\"u}nnemann, Stephan},\n'
 'booktitle={Machine Learning for Molecules Workshop, NeurIPS},\n'
 'year = {2020} }']
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
{'python': ['scikit-learn==0.24.1',
            'numpy==1.20.1',
            'matbench==0.1.0',
            'tensorflow==2.9.0',
            'kgcnn==2.1.0',
            'pymatgen==2022.9.8',
            'pyxtal==0.5.2',
            'networkx',
            'pandas',
            'tensorflow-addons']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2831| 1.7549| 0.0981| 34.6909 |
 | fold_1 | 0.3120| 1.2357| 0.1049| 19.2668 |
 | fold_2 | 0.4431| 3.0083| 0.0998| 58.5416 |
 | fold_3 | 0.3043| 2.1882| 0.0641| 49.1359 |
 | fold_4 | 0.3576| 1.7811| 0.1137| 28.5201 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3400 | 0.4431 | 0.2831 | 0.0570 |
| rmse | 1.9936 | 3.0083 | 1.2357 | 0.5906 |
| mape* | 0.0961 | 0.1137 | 0.0641 | 0.0169 |
| max_error | 38.0311 | 58.5416 | 19.2668 | 14.1259 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 37.6448| 59.7893| 26.8152| 240.0727 |
 | fold_1 | 52.4543| 107.0413| 0.4509| 624.3835 |
 | fold_2 | 68.4016| 187.3249| 0.6867| 1008.1589 |
 | fold_3 | 35.3625| 57.1472| 0.3687| 292.5341 |
 | fold_4 | 51.2584| 163.3720| 0.6936| 1515.0046 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 49.0243 | 68.4016 | 35.3625 | 11.9027 |
| rmse | 114.9349 | 187.3249 | 57.1472 | 52.9702 |
| mape* | 5.8030 | 26.8152 | 0.3687 | 10.5069 |
| max_error | 736.0308 | 1515.0046 | 240.0727 | 476.6514 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0786| 0.1276| 0.0615| 1.5533 |
 | fold_1 | 0.0804| 0.1304| 0.0659| 1.5549 |
 | fold_2 | 0.0781| 0.1220| 0.0625| 1.1013 |
 | fold_3 | 0.0785| 0.1195| 0.0614| 0.9208 |
 | fold_4 | 0.0806| 0.1281| 0.0635| 1.5558 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0792 | 0.0806 | 0.0781 | 0.0011 |
| rmse | 0.1255 | 0.1304 | 0.1195 | 0.0041 |
| mape* | 0.0630 | 0.0659 | 0.0614 | 0.0016 |
| max_error | 1.3372 | 1.5558 | 0.9208 | 0.2723 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0547| 0.1189| 0.0396| 1.7063 |
 | fold_1 | 0.0574| 0.1174| 0.0399| 1.5804 |
 | fold_2 | 0.0530| 0.1001| 0.0349| 1.0732 |
 | fold_3 | 0.0618| 0.1202| 0.0453| 1.1725 |
 | fold_4 | 0.0593| 0.1178| 0.0409| 1.4483 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0572 | 0.0618 | 0.0530 | 0.0032 |
| rmse | 0.1149 | 0.1202 | 0.1001 | 0.0074 |
| mape* | 0.0401 | 0.0453 | 0.0349 | 0.0033 |
| max_error | 1.3961 | 1.7063 | 1.0732 | 0.2397 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0235| 0.0725| 0.2865| 3.2321 |
 | fold_1 | 0.0236| 0.0717| 0.1561| 2.4344 |
 | fold_2 | 0.0234| 0.0640| 0.1688| 1.4689 |
 | fold_3 | 0.0241| 0.0739| 0.2090| 3.6006 |
 | fold_4 | 0.0229| 0.0655| 0.2567| 2.3014 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0235 | 0.0241 | 0.0229 | 0.0004 |
| rmse | 0.0695 | 0.0739 | 0.0640 | 0.0040 |
| mape* | 0.2154 | 0.2865 | 0.1561 | 0.0500 |
| max_error | 2.6075 | 3.6006 | 1.4689 | 0.7478 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1976| 0.4493| 2.7455| 6.5509 |
 | fold_1 | 0.2011| 0.4799| 2.1119| 14.0169 |
 | fold_2 | 0.1991| 0.4789| 4.5244| 7.5415 |
 | fold_3 | 0.1904| 0.4673| 4.9372| 9.0348 |
 | fold_4 | 0.2083| 0.4846| 4.8806| 7.6471 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1993 | 0.2083 | 0.1904 | 0.0058 |
| rmse | 0.4720 | 0.4846 | 0.4493 | 0.0127 |
| mape* | 3.8399 | 4.9372 | 2.1119 | 1.1781 |
| max_error | 8.9582 | 14.0169 | 6.5509 | 2.6502 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9111| 0.9089| 0.8972| 0.9089 |
 | fold_1 | 0.9063| 0.9038| 0.8915| 0.9038 |
 | fold_2 | 0.9022| 0.8998| 0.8869| 0.8998 |
 | fold_3 | 0.9062| 0.9043| 0.8919| 0.9043 |
 | fold_4 | 0.9014| 0.8989| 0.8858| 0.8989 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9054 | 0.9111 | 0.9014 | 0.0035 |
| balanced_accuracy | 0.9032 | 0.9089 | 0.8989 | 0.0036 |
| f1 | 0.8907 | 0.8972 | 0.8858 | 0.0041 |
| rocauc | 0.9032 | 0.9089 | 0.8989 | 0.0036 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0391| 0.0668| 0.0390| 0.8217 |
 | fold_1 | 0.0385| 0.0689| 0.0379| 0.9676 |
 | fold_2 | 0.0370| 0.0605| 0.0378| 0.7633 |
 | fold_3 | 0.0362| 0.0612| 0.0344| 0.8475 |
 | fold_4 | 0.0372| 0.0637| 0.0346| 0.9014 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0376 | 0.0391 | 0.0362 | 0.0011 |
| rmse | 0.0642 | 0.0689 | 0.0605 | 0.0032 |
| mape* | 0.0367 | 0.0390 | 0.0344 | 0.0019 |
| max_error | 0.8603 | 0.9676 | 0.7633 | 0.0697 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 39.9212| 93.3040| 0.0663| 883.7585 |
 | fold_1 | 35.3137| 60.9662| 0.0756| 413.8156 |
 | fold_2 | 37.1448| 77.1708| 0.0730| 604.9076 |
 | fold_3 | 40.0427| 94.8770| 0.0752| 1012.6802 |
 | fold_4 | 34.8869| 75.2053| 0.0768| 607.9076 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 37.4619 | 40.0427 | 34.8869 | 2.1934 |
| rmse | 80.3047 | 94.8770 | 60.9662 | 12.5789 |
| mape* | 0.0734 | 0.0768 | 0.0663 | 0.0037 |
| max_error | 704.6139 | 1012.6802 | 413.8156 | 214.8743 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_1 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_2 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_3 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |
| fold_4 | `{'data': {'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0, 'max_neighbours': 17}}, {'map_list': {'method': 'set_...` |




