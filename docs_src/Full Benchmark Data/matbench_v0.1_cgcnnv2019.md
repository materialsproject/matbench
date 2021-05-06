# matbench_v0.1: CGCNN v2019

### Algorithm description: 

Convolutional graph neural network, in it's original implementation as in https://github.com/txie-93/cgcnn. Utility modifications were made in order to run CGCNN without error across all structure tasks. Adapted from data originally taken from Dunn et. al 'Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm' (2020). Training was performed using one NVIDIA 1080Ti GPU using CUDA (accompanied by two Intel Xeon E5-2623 CPUs with 60GB RAM). Each outer NCV training set was split 75/25 for train/validation; thus the final split for each fold was 60% train, 20% validation, 20% test. Each model is trained in epochs of 128-structure batches by optimizing according to mean squared error loss (regression) or binary cross-entropy (classification). After each epoch, the validation loss is computed with the same scoring functions as the final evaluation: MAE for regression or ROC-AUC for classification (made negative so that higher loss represents worse performance). To prevent overfitting, the training is stopped early when the validation loss does not improve over a period of at least 500 epochs.



Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_cgcnnv2019).

### References (in bibtex format): 

```
['@article{Xie2018,\n'
 '  doi = {10.1103/physrevlett.120.145301},\n'
 '  url = {https://doi.org/10.1103/physrevlett.120.145301},\n'
 '  year = {2018},\n'
 '  month = apr,\n'
 '  publisher = {American Physical Society ({APS})},\n'
 '  volume = {120},\n'
 '  number = {14},\n'
 '  author = {Tian Xie and Jeffrey C. Grossman},\n'
 '  title = {Crystal Graph Convolutional Neural Networks for an Accurate and '
 'Interpretable Prediction of Material Properties},\n'
 '  journal = {Physical Review Letters}\n'
 '}',
 '@article{Dunn2020,\n'
 '  doi = {10.1038/s41524-020-00406-3},\n'
 '  url = {https://doi.org/10.1038/s41524-020-00406-3},\n'
 '  year = {2020},\n'
 '  month = sep,\n'
 '  publisher = {Springer Science and Business Media {LLC}},\n'
 '  volume = {6},\n'
 '  number = {1},\n'
 '  author = {Alexander Dunn and Qi Wang and Alex Ganose and Daniel Dopp and '
 'Anubhav Jain},\n'
 '  title = {Benchmarking materials property prediction methods: the Matbench '
 'test set and Automatminer reference algorithm},\n'
 '  journal = {npj Computational Materials}\n'
 '}']
```

### User metadata:

```
{'conv_to_fc.bias': 32,
 'conv_to_fc.weight': 2048,
 'convs.0.bn1.bias': 128,
 'convs.0.bn1.num_batches_tracked': 1,
 'convs.0.bn1.running_mean': 128,
 'convs.0.bn1.running_var': 128,
 'convs.0.bn1.weight': 128,
 'convs.0.bn2.bias': 64,
 'convs.0.bn2.num_batches_tracked': 1,
 'convs.0.bn2.running_mean': 64,
 'convs.0.bn2.running_var': 64,
 'convs.0.bn2.weight': 64,
 'convs.0.fc_full.bias': 128,
 'convs.0.fc_full.weight': 21632,
 'convs.1.bn1.bias': 128,
 'convs.1.bn1.num_batches_tracked': 1,
 'convs.1.bn1.running_mean': 128,
 'convs.1.bn1.running_var': 128,
 'convs.1.bn1.weight': 128,
 'convs.1.bn2.bias': 64,
 'convs.1.bn2.num_batches_tracked': 1,
 'convs.1.bn2.running_mean': 64,
 'convs.1.bn2.running_var': 64,
 'convs.1.bn2.weight': 64,
 'convs.1.fc_full.bias': 128,
 'convs.1.fc_full.weight': 21632,
 'convs.2.bn1.bias': 128,
 'convs.2.bn1.num_batches_tracked': 1,
 'convs.2.bn1.running_mean': 128,
 'convs.2.bn1.running_var': 128,
 'convs.2.bn1.weight': 128,
 'convs.2.bn2.bias': 64,
 'convs.2.bn2.num_batches_tracked': 1,
 'convs.2.bn2.running_mean': 64,
 'convs.2.bn2.running_var': 64,
 'convs.2.bn2.weight': 64,
 'convs.2.fc_full.bias': 128,
 'convs.2.fc_full.weight': 21632,
 'convs.3.bn1.bias': 128,
 'convs.3.bn1.num_batches_tracked': 1,
 'convs.3.bn1.running_mean': 128,
 'convs.3.bn1.running_var': 128,
 'convs.3.bn1.weight': 128,
 'convs.3.bn2.bias': 64,
 'convs.3.bn2.num_batches_tracked': 1,
 'convs.3.bn2.running_mean': 64,
 'convs.3.bn2.running_var': 64,
 'convs.3.bn2.weight': 64,
 'convs.3.fc_full.bias': 128,
 'convs.3.fc_full.weight': 21632,
 'embedding.bias': 64,
 'embedding.weight': 5888,
 'fc_out.bias': 1,
 'fc_out.weight': 32}
```

### Metadata:

Tasks recorded: 9 of 13 total

Benchmark is complete? False

Benchmark is structure complete? True

Benchmark is composition complete? False

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.4704| 0.9059| 0.1949| 14.6895 |
 | fold_1 | 0.5724| 1.2825| 0.2222| 20.3729 |
 | fold_2 | 0.7301| 3.0600| 0.2194| 58.9996 |
 | fold_3 | 0.6111| 2.4214| 0.2119| 53.4782 |
 | fold_4 | 0.6099| 1.8183| 0.2348| 28.6714 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.5988 | 0.7301 | 0.4704 | 0.0833 |
| rmse | 1.8976 | 3.0600 | 0.9059 | 0.7738 |
| mape* | 0.2167 | 0.2348 | 0.1949 | 0.0131 |
| max_error | 35.2423 | 58.9996 | 14.6895 | 17.7969 |


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
 | fold_0 | 34.4937| 56.8278| 33.7683| 256.0330 |
 | fold_1 | 51.1167| 98.1228| 0.5027| 407.6809 |
 | fold_2 | 69.4250| 182.5647| 0.6043| 1061.5574 |
 | fold_3 | 42.7453| 71.8811| 0.4072| 303.9963 |
 | fold_4 | 48.4396| 154.4480| 0.6338| 1516.9120 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 49.2440 | 69.4250 | 34.4937 | 11.5865 |
| rmse | 112.7689 | 182.5647 | 56.8278 | 48.2169 |
| mape* | 7.1833 | 33.7683 | 0.4072 | 13.2927 |
| max_error | 709.2359 | 1516.9120 | 256.0330 | 497.3969 |


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
 | fold_0 | 0.0870| 0.1270| 0.0680| 1.0473 |
 | fold_1 | 0.0899| 0.1384| 0.0714| 1.4520 |
 | fold_2 | 0.0887| 0.1323| 0.0699| 1.0024 |
 | fold_3 | 0.0902| 0.1344| 0.0705| 0.9712 |
 | fold_4 | 0.0918| 0.1362| 0.0712| 0.8430 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0895 | 0.0918 | 0.0870 | 0.0016 |
| rmse | 0.1337 | 0.1384 | 0.1270 | 0.0039 |
| mape* | 0.0702 | 0.0714 | 0.0680 | 0.0012 |
| max_error | 1.0632 | 1.4520 | 0.8430 | 0.2059 |


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
 | fold_0 | 0.0702| 0.1290| 0.0456| 1.7725 |
 | fold_1 | 0.0722| 0.1353| 0.0477| 1.3813 |
 | fold_2 | 0.0665| 0.1191| 0.0423| 1.1052 |
 | fold_3 | 0.0748| 0.1341| 0.0517| 1.1231 |
 | fold_4 | 0.0724| 0.1328| 0.0480| 1.5001 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0712 | 0.0748 | 0.0665 | 0.0028 |
| rmse | 0.1301 | 0.1353 | 0.1191 | 0.0059 |
| mape* | 0.0471 | 0.0517 | 0.0423 | 0.0031 |
| max_error | 1.3765 | 1.7725 | 1.1052 | 0.2490 |


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
 | fold_0 | 0.0340| 0.0714| 0.4273| 3.4254 |
 | fold_1 | 0.0340| 0.0681| 0.1934| 2.0786 |
 | fold_2 | 0.0328| 0.0756| 0.2075| 7.7205 |
 | fold_3 | 0.0332| 0.0623| 0.2258| 1.3283 |
 | fold_4 | 0.0346| 0.0633| 0.2830| 1.5782 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0337 | 0.0346 | 0.0328 | 0.0006 |
| rmse | 0.0682 | 0.0756 | 0.0623 | 0.0050 |
| mape* | 0.2674 | 0.4273 | 0.1934 | 0.0855 |
| max_error | 3.2262 | 7.7205 | 1.3283 | 2.3611 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'training_mae': 0.020714398227129675, 'training_n_samples': 79650, 'validation_mae': 0.03311641346102554, 'validation_n_samples': 26551}` |
| fold_1 | `{'training_mae': 0.020727165395357502, 'training_n_samples': 79650, 'validation_mae': 0.03387322865233633, 'validation_n_samples': 26551}` |
| fold_2 | `{'training_mae': 0.020593563770909047, 'training_n_samples': 79651, 'validation_mae': 0.033246301549184044, 'validation_n_samples': 26551}` |
| fold_3 | `{'training_mae': 0.020858287502723834, 'training_n_samples': 79651, 'validation_mae': 0.03272479726288639, 'validation_n_samples': 26551}` |
| fold_4 | `{'training_mae': 0.0220815778646356, 'training_n_samples': 79651, 'validation_mae': 0.03445024667244967, 'validation_n_samples': 26551}` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2978| 0.6753| 3.5253| 7.2169 |
 | fold_1 | 0.2939| 0.6827| 3.3933| 13.6569 |
 | fold_2 | 0.2960| 0.6653| 5.5089| 6.8339 |
 | fold_3 | 0.2947| 0.6740| 7.7018| 7.7523 |
 | fold_4 | 0.3038| 0.6884| 5.7405| 7.7166 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2972 | 0.3038 | 0.2939 | 0.0035 |
| rmse | 0.6771 | 0.6884 | 0.6653 | 0.0079 |
| mape* | 5.1740 | 7.7018 | 3.3933 | 1.5945 |
| max_error | 8.6353 | 13.6569 | 6.8339 | 2.5336 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9590| 0.9578| 0.9526| 0.9578 |
 | fold_1 | 0.9450| 0.9432| 0.9363| 0.9432 |
 | fold_2 | 0.9643| 0.9632| 0.9588| 0.9632 |
 | fold_3 | 0.9480| 0.9463| 0.9398| 0.9463 |
 | fold_4 | 0.9510| 0.9494| 0.9433| 0.9494 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9534 | 0.9643 | 0.9450 | 0.0072 |
| balanced_accuracy | 0.9520 | 0.9632 | 0.9432 | 0.0074 |
| f1 | 0.9462 | 0.9588 | 0.9363 | 0.0083 |
| rocauc | 0.9520 | 0.9632 | 0.9432 | 0.0074 |


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
 | fold_0 | 0.0456| 0.0753| 0.0483| 0.9441 |
 | fold_1 | 0.0462| 0.0735| 0.0497| 0.9923 |
 | fold_2 | 0.0448| 0.0690| 0.0466| 0.9840 |
 | fold_3 | 0.0454| 0.0714| 0.0482| 0.7688 |
 | fold_4 | 0.0442| 0.0718| 0.0419| 0.9384 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0452 | 0.0462 | 0.0442 | 0.0007 |
| rmse | 0.0722 | 0.0753 | 0.0690 | 0.0021 |
| mape* | 0.0469 | 0.0497 | 0.0419 | 0.0027 |
| max_error | 0.9255 | 0.9923 | 0.7688 | 0.0812 |


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
 | fold_0 | 81.1553| 231.3233| 0.1330| 2504.8743 |
 | fold_1 | 45.0945| 79.3798| 0.0995| 835.2144 |
 | fold_2 | 54.2563| 132.8543| 0.1081| 1667.5734 |
 | fold_3 | 56.5819| 169.7248| 0.1201| 2378.4055 |
 | fold_4 | 51.7292| 95.2267| 0.1087| 658.0856 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 57.7635 | 81.1553 | 45.0945 | 12.3109 |
| rmse | 141.7018 | 231.3233 | 79.3798 | 54.6618 |
| mape* | 0.1139 | 0.1330 | 0.0995 | 0.0116 |
| max_error | 1608.8306 | 2504.8743 | 658.0856 | 761.7071 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




