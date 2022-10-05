# matbench_v0.1: MegNet (kgcnn v2.1.0)

### Algorithm description: 

Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals. Adapted implementation of `kgcnn`. Original code from https://github.com/materialsvirtuallab/megnet. Settings are similar compared to original work: A model depth of 3 and units for MegNet block of [64,32,32], a Set2Set encoder for node and edge embeddings, feed-forward blocks of units [64, 32], softplus activation and gauss distance expansion with cutoff of 5A and 25 bins with 0.4 sigma. We used a larger input embedding vector [64] of atom species and added the charge as input graph attributes. We trained with MAE loss and a linear learning rate scheduler from 5e-4 to 5e-6 over 1000 epochs using Adam. We added a standard scaler for regression. Training was carried out on A100-SXM with 41 GB of memory. Hyperparameter were not optimized but just copied over from training on  QM9/QM7 datasets.

#### Notes:


Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_MegNet_kgcnn_v2.1.0).

### References (in bibtex format): 

```
['@article{doi:10.1021/acs.chemmater.9b01294, author = {Chen, Chi and Ye, '
 'Weike and Zuo, Yunxing and Zheng, Chen and Ong, Shyue Ping}, title = {Graph '
 'Networks as a Universal Machine Learning Framework for Molecules and '
 'Crystals}, journal = {Chemistry of Materials}, volume = {31}, number = {9}, '
 'pages = {3564-3572}, year = {2019}, doi = {10.1021/acs.chemmater.9b01294}, '
 'URL = {https://doi.org/10.1021/acs.chemmater.9b01294}, eprint = '
 '{https://doi.org/10.1021/acs.chemmater.9b01294}}']
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
 | fold_0 | 0.2385| 1.0393| 0.0767| 20.5088 |
 | fold_1 | 0.2872| 1.2163| 0.0936| 20.4615 |
 | fold_2 | 0.4444| 3.0835| 0.0973| 59.3095 |
 | fold_3 | 0.3254| 2.2884| 0.0714| 52.2159 |
 | fold_4 | 0.3998| 2.3078| 0.1274| 47.8845 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3391 | 0.4444 | 0.2385 | 0.0745 |
| rmse | 1.9871 | 3.0835 | 1.0393 | 0.7600 |
| mape* | 0.0933 | 0.1274 | 0.0714 | 0.0197 |
| max_error | 40.0760 | 59.3095 | 20.4615 | 16.4066 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 50.6822| 91.2961| 15.5595| 511.9383 |
 | fold_1 | 54.6317| 115.4827| 0.3805| 518.5855 |
 | fold_2 | 67.3932| 184.2895| 0.6927| 1064.3459 |
 | fold_3 | 34.6872| 73.0294| 0.3497| 407.8362 |
 | fold_4 | 63.4653| 182.5356| 0.8128| 1561.5756 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 54.1719 | 67.3932 | 34.6872 | 11.4299 |
| rmse | 129.3267 | 184.2895 | 73.0294 | 46.1724 |
| mape* | 3.5591 | 15.5595 | 0.3497 | 6.0029 |
| max_error | 812.8563 | 1561.5756 | 407.8362 | 439.3213 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0848| 0.1350| 0.0659| 1.5533 |
 | fold_1 | 0.0883| 0.1396| 0.0699| 1.5549 |
 | fold_2 | 0.0867| 0.1313| 0.0690| 0.8507 |
 | fold_3 | 0.0883| 0.1342| 0.0681| 0.9500 |
 | fold_4 | 0.0876| 0.1387| 0.0681| 1.5558 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0871 | 0.0883 | 0.0848 | 0.0013 |
| rmse | 0.1358 | 0.1396 | 0.1313 | 0.0030 |
| mape* | 0.0682 | 0.0699 | 0.0659 | 0.0013 |
| max_error | 1.2929 | 1.5558 | 0.8507 | 0.3221 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0648| 0.1286| 0.0452| 1.6193 |
 | fold_1 | 0.0673| 0.1365| 0.0454| 1.8705 |
 | fold_2 | 0.0633| 0.1172| 0.0410| 1.2643 |
 | fold_3 | 0.0730| 0.1337| 0.0517| 1.2378 |
 | fold_4 | 0.0657| 0.1275| 0.0451| 1.4056 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0668 | 0.0730 | 0.0633 | 0.0034 |
| rmse | 0.1287 | 0.1365 | 0.1172 | 0.0066 |
| mape* | 0.0457 | 0.0517 | 0.0410 | 0.0034 |
| max_error | 1.4795 | 1.8705 | 1.2378 | 0.2377 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0248| 0.0711| 0.3029| 3.4758 |
 | fold_1 | 0.0250| 0.0707| 0.1600| 2.2742 |
 | fold_2 | 0.0253| 0.0679| 0.1952| 2.3452 |
 | fold_3 | 0.0257| 0.0742| 0.2225| 3.6006 |
 | fold_4 | 0.0251| 0.0668| 0.2638| 2.2808 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0252 | 0.0257 | 0.0248 | 0.0003 |
| rmse | 0.0701 | 0.0742 | 0.0668 | 0.0026 |
| mape* | 0.2289 | 0.3029 | 0.1600 | 0.0502 |
| max_error | 2.7953 | 3.6006 | 2.2742 | 0.6083 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1845| 0.4415| 2.4896| 6.8774 |
 | fold_1 | 0.1970| 0.4744| 2.0477| 7.0580 |
 | fold_2 | 0.2039| 0.5019| 4.1405| 6.9070 |
 | fold_3 | 0.1817| 0.4501| 6.2172| 7.8821 |
 | fold_4 | 0.1999| 0.4895| 5.3344| 7.0037 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1934 | 0.2039 | 0.1817 | 0.0087 |
| rmse | 0.4715 | 0.5019 | 0.4415 | 0.0229 |
| mape* | 4.0459 | 6.2172 | 2.0477 | 1.5999 |
| max_error | 7.1457 | 7.8821 | 6.8774 | 0.3739 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9072| 0.9048| 0.8926| 0.9048 |
 | fold_1 | 0.9039| 0.9013| 0.8886| 0.9013 |
 | fold_2 | 0.9063| 0.9035| 0.8912| 0.9035 |
 | fold_3 | 0.9032| 0.9008| 0.8880| 0.9008 |
 | fold_4 | 0.9029| 0.9002| 0.8873| 0.9002 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9047 | 0.9072 | 0.9029 | 0.0017 |
| balanced_accuracy | 0.9021 | 0.9048 | 0.9002 | 0.0018 |
| f1 | 0.8895 | 0.8926 | 0.8873 | 0.0020 |
| rocauc | 0.9021 | 0.9048 | 0.9002 | 0.0018 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0355| 0.0653| 0.0366| 0.8320 |
 | fold_1 | 0.0374| 0.0722| 0.0372| 1.0236 |
 | fold_2 | 0.0335| 0.0554| 0.0341| 0.7220 |
 | fold_3 | 0.0363| 0.0646| 0.0357| 0.7184 |
 | fold_4 | 0.0331| 0.0602| 0.0293| 0.8357 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0352 | 0.0374 | 0.0331 | 0.0016 |
| rmse | 0.0635 | 0.0722 | 0.0554 | 0.0056 |
| mape* | 0.0346 | 0.0372 | 0.0293 | 0.0029 |
| max_error | 0.8263 | 1.0236 | 0.7184 | 0.1110 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 31.6159| 63.3487| 0.0560| 724.7676 |
 | fold_1 | 31.0426| 61.1066| 0.0590| 589.2758 |
 | fold_2 | 25.2700| 55.0800| 0.0513| 489.6110 |
 | fold_3 | 29.7114| 64.6030| 0.0611| 774.1321 |
 | fold_4 | 26.1630| 43.2011| 0.0529| 208.9928 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 28.7606 | 31.6159 | 25.2700 | 2.5767 |
| rmse | 57.4679 | 64.6030 | 43.2011 | 7.8483 |
| mape* | 0.0561 | 0.0611 | 0.0513 | 0.0036 |
| max_error | 557.3559 | 774.1321 | 208.9928 | 200.9894 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_1 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_2 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_3 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |
| fold_4 | `{'data': {'data_unit': '', 'dataset': {'class_name': 'CrystalDataset', 'config': {}, 'methods': [{'map_list': {'method': 'set_range_periodic', 'max_distance': 5.0}}], 'module_name': 'kgcnn.data.crysta...` |




