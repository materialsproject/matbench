# matbench_v0.1: MODNet (v0.1.10)

### Algorithm description: 

MODNet, the Materials Optimal Descriptor Network (v0.1.10). A feed-forward neural network, using all compatible matminer features and a relevance-redundancy based feature selection algorithm. Hyperparameter optimisation is performed with a nested grid search. Benchmark results were loaded from https://github.com/ml-evs/modnet-matbench.

#### Notes:
None

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_modnet_v0.1.10).

### References (in bibtex format): 

```
('@article{De_Breuck_2021, doi = {10.1088/1361-648x/ac1280}, url = '
 '{https://doi.org/10.1088/1361-648x/ac1280}, year = 2021, month = {jul}, '
 'publisher = {{IOP} Publishing}, volume = {33}, number = {40}, pages = '
 '{404002}, author = {Pierre-Paul De Breuck and Matthew L Evans and Gian-Marco '
 'Rignanese}, title = {Robust model benchmarking and bias-imbalance in '
 'data-driven materials science: a case study on {MODNet}}, journal = {Journal '
 'of Physics: Condensed Matter}, abstract = {As the number of novel '
 'data-driven approaches to material science continues to grow, it is crucial '
 'to perform consistent quality, reliability and applicability assessments of '
 'model performance. In this paper, we benchmark the Materials Optimal '
 'Descriptor Network (MODNet) method and architecture against the recently '
 'released MatBench v0.1, a curated test suite of materials datasets. MODNet '
 'is shown to outperform current leaders on 6 of the 13 tasks, while closely '
 'matching the current leaders on a further 2 tasks; MODNet performs '
 'particularly well when the number of samples is below 10\xa0000. Attention '
 'is paid to two topics of concern when benchmarking models. First, we '
 'encourage the reporting of a more diverse set of metrics as it leads to a '
 'more comprehensive and holistic comparison of model performance. Second, an '
 'equally important task is the uncertainty assessment of a model towards a '
 'target domain. Significant variations in validation errors can be observed, '
 'depending on the imbalance and bias in the training set (i.e., similarity '
 'between training and application space). By using an ensemble MODNet model, '
 'confidence intervals can be built and the uncertainty on individual '
 'predictions can be quantified. Imbalance and bias issues are often '
 'overlooked, and yet are important for successful real-world applications of '
 'machine learning in materials science and condensed matter.}}, '
 '@article{DeBreuck2021, doi = {10.1038/s41524-021-00552-2}, url = '
 '{https://doi.org/10.1038/s41524-021-00552-2}, year = {2021}, month = jun, '
 'publisher = {Springer Science and Business Media {LLC}}, volume = {7}, '
 'number = {1}, author = {Pierre-Paul De Breuck and Geoffroy Hautier and '
 'Gian-Marco Rignanese}, title = {Materials property prediction for limited '
 'datasets enabled by feature selection and joint learning with {MODNet}}, '
 'journal = {npj Computational Materials}}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 13/13 |
|----------------|-------------------------------------|
| complete? | ✓ | 
| composition complete? | ✓ | 
| structure complete? | ✓ | 
| regression complete? | ✓ | 
| classification complete? | ✓ | 

### Software Requirements

```
{'python': ['modnet==0.1.10', 'matbench==0.2.0']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1939| 0.7043| 0.0657| 13.9549 |
 | fold_1 | 0.2669| 1.0559| 0.0897| 19.4132 |
 | fold_2 | 0.4138| 2.9360| 0.0873| 58.9519 |
 | fold_3 | 0.2880| 2.2447| 0.0593| 52.4648 |
 | fold_4 | 0.3223| 1.6518| 0.1040| 28.0662 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2970 | 0.4138 | 0.1939 | 0.0720 |
| rmse | 1.7185 | 2.9360 | 0.7043 | 0.8039 |
| mape* | 0.0812 | 0.1040 | 0.0593 | 0.0164 |
| max_error | 34.5702 | 58.9519 | 13.9549 | 17.9539 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3272| 0.7062| 0.3510| 6.3096 |
 | fold_1 | 0.3594| 0.7340| 0.3187| 6.3544 |
 | fold_2 | 0.3845| 0.8563| 0.3841| 9.8567 |
 | fold_3 | 0.3259| 0.6888| 0.3231| 5.1081 |
 | fold_4 | 0.3382| 0.7334| 0.4075| 6.5141 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3470 | 0.3845 | 0.3259 | 0.0222 |
| rmse | 0.7437 | 0.8563 | 0.6888 | 0.0588 |
| mape* | 0.3569 | 0.4075 | 0.3187 | 0.0345 |
| max_error | 6.8286 | 9.8567 | 5.1081 | 1.5952 |


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
 | fold_0 | 0.9269| 0.9269| 0.9255| 0.9269 |
 | fold_1 | 0.9136| 0.9136| 0.9121| 0.9136 |
 | fold_2 | 0.9177| 0.9177| 0.9173| 0.9177 |
 | fold_3 | 0.9177| 0.9177| 0.9169| 0.9177 |
 | fold_4 | 0.9045| 0.9045| 0.9049| 0.9045 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9161 | 0.9269 | 0.9045 | 0.0073 |
| balanced_accuracy | 0.9161 | 0.9269 | 0.9045 | 0.0072 |
| f1 | 0.9153 | 0.9255 | 0.9049 | 0.0068 |
| rocauc | 0.9161 | 0.9269 | 0.9045 | 0.0072 |


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
 | fold_0 | 0.8759| 0.8262| 0.9153| 0.8262 |
 | fold_1 | 0.8539| 0.7783| 0.9030| 0.7783 |
 | fold_2 | 0.8565| 0.8063| 0.9016| 0.8063 |
 | fold_3 | 0.8856| 0.8402| 0.9217| 0.8402 |
 | fold_4 | 0.8662| 0.8023| 0.9102| 0.8023 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8676 | 0.8856 | 0.8539 | 0.0119 |
| balanced_accuracy | 0.8107 | 0.8402 | 0.7783 | 0.0212 |
| f1 | 0.9104 | 0.9217 | 0.9016 | 0.0075 |
| rocauc | 0.8107 | 0.8402 | 0.7783 | 0.0212 |


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
 | fold_0 | 27.5769| 49.7512| 21.3632| 243.2504 |
 | fold_1 | 27.9722| 63.3103| 0.2282| 364.1909 |
 | fold_2 | 51.3402| 142.7963| 0.6111| 845.7528 |
 | fold_3 | 26.9141| 52.8447| 0.2724| 311.7558 |
 | fold_4 | 38.8806| 152.4413| 0.4853| 1534.9797 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 34.5368 | 51.3402 | 26.9141 | 9.4959 |
| rmse | 92.2288 | 152.4413 | 49.7512 | 45.5508 |
| mape* | 4.5920 | 21.3632 | 0.2282 | 8.3868 |
| max_error | 659.9859 | 1534.9797 | 243.2504 | 486.3231 |


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
 | fold_0 | 0.0731| 0.1089| 0.0576| 0.9014 |
 | fold_1 | 0.0738| 0.1111| 0.0579| 1.1745 |
 | fold_2 | 0.0731| 0.1101| 0.0587| 0.9076 |
 | fold_3 | 0.0738| 0.1115| 0.0567| 0.9225 |
 | fold_4 | 0.0718| 0.1101| 0.0560| 0.8007 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0731 | 0.0738 | 0.0718 | 0.0007 |
| rmse | 0.1103 | 0.1115 | 0.1089 | 0.0009 |
| mape* | 0.0574 | 0.0587 | 0.0560 | 0.0009 |
| max_error | 0.9413 | 1.1745 | 0.8007 | 0.1243 |


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
 | fold_0 | 0.0536| 0.1013| 0.0356| 1.5366 |
 | fold_1 | 0.0559| 0.1079| 0.0366| 1.2998 |
 | fold_2 | 0.0510| 0.0949| 0.0340| 1.1808 |
 | fold_3 | 0.0585| 0.1126| 0.0418| 1.1355 |
 | fold_4 | 0.0549| 0.1046| 0.0370| 1.3202 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0548 | 0.0585 | 0.0510 | 0.0025 |
| rmse | 0.1043 | 0.1126 | 0.0949 | 0.0060 |
| mape* | 0.0370 | 0.0418 | 0.0340 | 0.0026 |
| max_error | 1.2946 | 1.5366 | 1.1355 | 0.1397 |


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
 | fold_0 | 0.0402| 0.0817| 0.3786| 4.0438 |
 | fold_1 | 0.0497| 0.1018| 0.3121| 4.8803 |
 | fold_2 | 0.0475| 0.0905| 0.2562| 1.6230 |
 | fold_3 | 0.0464| 0.0889| 0.3515| 1.5189 |
 | fold_4 | 0.0400| 0.0812| 0.2882| 3.3787 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0448 | 0.0497 | 0.0400 | 0.0039 |
| rmse | 0.0888 | 0.1018 | 0.0812 | 0.0075 |
| mape* | 0.3173 | 0.3786 | 0.2562 | 0.0436 |
| max_error | 3.0889 | 4.8803 | 1.5189 | 1.3281 |


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
 | fold_0 | 0.2147| 0.4441| 2.8966| 5.0558 |
 | fold_1 | 0.2161| 0.4484| 2.6899| 6.2874 |
 | fold_2 | 0.2165| 0.4433| 4.1912| 7.5685 |
 | fold_3 | 0.2309| 0.4705| 4.6749| 6.9325 |
 | fold_4 | 0.2211| 0.4564| 4.9590| 4.9406 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2199 | 0.2309 | 0.2147 | 0.0059 |
| rmse | 0.4525 | 0.4705 | 0.4433 | 0.0101 |
| mape* | 3.8823 | 4.9590 | 2.6899 | 0.9248 |
| max_error | 6.1570 | 7.5685 | 4.9406 | 1.0299 |


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
 | fold_0 | 0.8515| 0.8415| 0.8175| 0.8415 |
 | fold_1 | 0.8824| 0.8709| 0.8526| 0.8709 |
 | fold_2 | 0.5650| 0.5000| 0.0000| 0.5000 |
 | fold_3 | 0.8575| 0.8447| 0.8200| 0.8447 |
 | fold_4 | 0.8588| 0.8453| 0.8203| 0.8453 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8031 | 0.8824 | 0.5650 | 0.1195 |
| balanced_accuracy | 0.7805 | 0.8709 | 0.5000 | 0.1406 |
| f1 | 0.6621 | 0.8526 | 0.0000 | 0.3313 |
| rocauc | 0.7805 | 0.8709 | 0.5000 | 0.1406 |


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
 | fold_0 | 0.0932| 0.1304| 0.0970| 0.8705 |
 | fold_1 | 0.0939| 0.1283| 0.1058| 1.0063 |
 | fold_2 | 0.0861| 0.1216| 0.0939| 0.9432 |
 | fold_3 | 0.0892| 0.1274| 0.0995| 0.8501 |
 | fold_4 | 0.0914| 0.1310| 0.0894| 1.1780 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0908 | 0.0939 | 0.0861 | 0.0028 |
| rmse | 0.1277 | 0.1310 | 0.1216 | 0.0033 |
| mape* | 0.0971 | 0.1058 | 0.0894 | 0.0055 |
| max_error | 0.9696 | 1.1780 | 0.8501 | 0.1180 |


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
 | fold_0 | 40.2218| 99.9366| 0.0661| 1031.8168 |
 | fold_1 | 41.1190| 83.0600| 0.0680| 721.2376 |
 | fold_2 | 38.8526| 70.0409| 0.0705| 452.0254 |
 | fold_3 | 37.1039| 78.3636| 0.0710| 662.8152 |
 | fold_4 | 36.4648| 59.7092| 0.0665| 342.3226 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 38.7524 | 41.1190 | 36.4648 | 1.7732 |
| rmse | 78.2220 | 99.9366 | 59.7092 | 13.4507 |
| mape* | 0.0684 | 0.0710 | 0.0661 | 0.0020 |
| max_error | 642.0435 | 1031.8168 | 342.3226 | 238.5648 |


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
 | fold_0 | 112.2905| 189.8130| 0.0707| 931.3261 |
 | fold_1 | 81.9908| 115.9188| 0.0604| 404.5644 |
 | fold_2 | 99.3739| 139.4921| 0.0699| 411.7195 |
 | fold_3 | 93.2877| 152.1443| 0.0672| 827.5305 |
 | fold_4 | 94.1265| 152.3995| 0.0709| 672.9292 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 96.2139 | 112.2905 | 81.9908 | 9.8352 |
| rmse | 149.9535 | 189.8130 | 115.9188 | 23.9473 |
| mape* | 0.0678 | 0.0709 | 0.0604 | 0.0039 |
| max_error | 649.6139 | 931.3261 | 404.5644 | 213.6365 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




