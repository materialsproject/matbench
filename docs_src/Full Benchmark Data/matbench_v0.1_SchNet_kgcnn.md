# matbench_v0.1: SchNet (kgcnn)

### Algorithm description: 

A continuous-filter convolutional neural network for modeling quantum interactions - SchNet. Implementation adapted to crystals in `kgcnn`. Original code from https://github.com/atomistic-machine-learning/schnetpack .

#### Notes:


Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_SchNet_kgcnn).

### References (in bibtex format): 

```
['@article{doi:10.1063/1.5019779,author={Schütt, K.T. and Sauceda, H. E. and '
 'Kindermans, P.-J. and Tkatchenko, A. and Müller, K.-R.},title={SchNet - A '
 'deep learning architecture for molecules and materials},journal={The Journal '
 'of Chemical '
 'Physics},volume={148},number={24},pages={241722},year={2018},doi={10.1063/1.5019779},URL={https://doi.org/10.1063/1.5019779},eprint={https://doi.org/10.1063/1.5019779}}']
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
 | fold_0 | 0.1797| 0.7529| 0.0610| 14.6940 |
 | fold_1 | 0.3327| 1.5348| 0.1185| 21.6101 |
 | fold_2 | 0.4288| 3.0209| 0.0941| 58.6071 |
 | fold_3 | 0.3228| 2.2977| 0.0702| 51.8160 |
 | fold_4 | 0.3747| 1.8887| 0.1275| 28.3467 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3277 | 0.4288 | 0.1797 | 0.0829 |
| rmse | 1.8990 | 3.0209 | 0.7529 | 0.7568 |
| mape* | 0.0942 | 0.1275 | 0.0610 | 0.0260 |
| max_error | 35.0148 | 58.6071 | 14.6940 | 17.1812 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 27.5059| 53.8311| 22.7853| 409.8511 |
 | fold_1 | 49.5297| 106.4853| 0.4151| 562.8652 |
 | fold_2 | 63.6005| 185.0466| 0.6597| 1015.3435 |
 | fold_3 | 27.7970| 54.7520| 0.2490| 287.0124 |
 | fold_4 | 44.8856| 154.9782| 0.6035| 1524.9143 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 42.6637 | 63.6005 | 27.5059 | 13.7201 |
| rmse | 111.0187 | 185.0466 | 53.8311 | 52.6678 |
| mape* | 4.9425 | 22.7853 | 0.2490 | 8.9226 |
| max_error | 759.9973 | 1524.9143 | 287.0124 | 455.0775 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0769| 0.1203| 0.0615| 0.9939 |
 | fold_1 | 0.0825| 0.1313| 0.0675| 1.1584 |
 | fold_2 | 0.0772| 0.1246| 0.0624| 0.9158 |
 | fold_3 | 0.0804| 0.1261| 0.0644| 0.9228 |
 | fold_4 | 0.0812| 0.1276| 0.0641| 0.7567 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0796 | 0.0825 | 0.0769 | 0.0022 |
| rmse | 0.1260 | 0.1313 | 0.1203 | 0.0036 |
| mape* | 0.0640 | 0.0675 | 0.0615 | 0.0021 |
| max_error | 0.9495 | 1.1584 | 0.7567 | 0.1301 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0577| 0.1137| 0.0387| 1.7542 |
 | fold_1 | 0.0568| 0.1159| 0.0395| 1.4185 |
 | fold_2 | 0.0575| 0.1069| 0.0387| 1.0520 |
 | fold_3 | 0.0628| 0.1183| 0.0452| 1.2305 |
 | fold_4 | 0.0601| 0.1167| 0.0404| 1.4135 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0590 | 0.0628 | 0.0568 | 0.0022 |
| rmse | 0.1143 | 0.1183 | 0.1069 | 0.0040 |
| mape* | 0.0405 | 0.0452 | 0.0387 | 0.0024 |
| max_error | 1.3737 | 1.7542 | 1.0520 | 0.2334 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0223| 0.0581| 0.2173| 2.9568 |
 | fold_1 | 0.0212| 0.0506| 0.1738| 2.0016 |
 | fold_2 | 0.0219| 0.0523| 0.1543| 2.9990 |
 | fold_3 | 0.0221| 0.0539| 0.1662| 1.9801 |
 | fold_4 | 0.0216| 0.0495| 0.2167| 1.4672 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0218 | 0.0223 | 0.0212 | 0.0004 |
| rmse | 0.0529 | 0.0581 | 0.0495 | 0.0030 |
| mape* | 0.1856 | 0.2173 | 0.1543 | 0.0263 |
| max_error | 2.2809 | 2.9990 | 1.4672 | 0.6005 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2316| 0.5058| 2.4606| 7.3981 |
 | fold_1 | 0.2313| 0.5064| 2.2767| 9.1171 |
 | fold_2 | 0.2360| 0.5152| 3.7636| 7.1947 |
 | fold_3 | 0.2366| 0.5278| 5.7306| 7.6585 |
 | fold_4 | 0.2405| 0.5308| 5.0042| 7.4353 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2352 | 0.2405 | 0.2313 | 0.0034 |
| rmse | 0.5172 | 0.5308 | 0.5058 | 0.0105 |
| mape* | 3.8472 | 5.7306 | 2.2767 | 1.3625 |
| max_error | 7.7607 | 9.1171 | 7.1947 | 0.6940 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.8930| 0.8902| 0.8759| 0.8902 |
 | fold_1 | 0.8909| 0.8890| 0.8745| 0.8890 |
 | fold_2 | 0.8914| 0.8888| 0.8744| 0.8888 |
 | fold_3 | 0.8946| 0.8929| 0.8790| 0.8929 |
 | fold_4 | 0.8952| 0.8928| 0.8788| 0.8928 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8930 | 0.8952 | 0.8909 | 0.0017 |
| balanced_accuracy | 0.8907 | 0.8929 | 0.8888 | 0.0018 |
| f1 | 0.8765 | 0.8790 | 0.8744 | 0.0020 |
| rocauc | 0.8907 | 0.8929 | 0.8888 | 0.0018 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0348| 0.0624| 0.0336| 0.8552 |
 | fold_1 | 0.0346| 0.0645| 0.0356| 0.8765 |
 | fold_2 | 0.0336| 0.0559| 0.0348| 0.6017 |
 | fold_3 | 0.0340| 0.0584| 0.0326| 0.6391 |
 | fold_4 | 0.0338| 0.0585| 0.0318| 0.8929 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0342 | 0.0348 | 0.0336 | 0.0005 |
| rmse | 0.0599 | 0.0645 | 0.0559 | 0.0031 |
| mape* | 0.0337 | 0.0356 | 0.0318 | 0.0014 |
| max_error | 0.7731 | 0.8929 | 0.6017 | 0.1258 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 39.8691| 89.5683| 0.0746| 1034.3312 |
 | fold_1 | 41.1306| 83.3891| 0.0865| 827.0298 |
 | fold_2 | 40.1591| 86.2715| 0.0767| 731.4202 |
 | fold_3 | 38.1467| 63.9434| 0.0950| 355.3882 |
 | fold_4 | 35.5125| 61.4670| 0.0795| 607.1646 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 38.9636 | 41.1306 | 35.5125 | 1.9760 |
| rmse | 76.9279 | 89.5683 | 61.4670 | 11.8023 |
| mape* | 0.0825 | 0.0950 | 0.0746 | 0.0074 |
| max_error | 711.0668 | 1034.3312 | 355.3882 | 226.1258 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5}}], 'module_name': 'kgcnn.data.crystal'...` |




