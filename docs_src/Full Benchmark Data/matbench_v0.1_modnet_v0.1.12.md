# matbench_v0.1: MODNet (v0.1.12)

### Algorithm description: 

[MODNet](https://github.com/ppdebreuck/modnet/releases), the Materials Optimal Descriptor Network (v0.1.12). A feed-forward neural network, using all compatible matminer features and a relevance-redundancy based feature selection algorithm. Hyperparameter optimisation is performed with a nested grid search for the 9 smaller tasks, and with a genetic algorithm for the 4 larger tasks (`matbench_perovskites`, `matbench_mp_gap`, `matbench_mp_is_metal`, `matbench_mp_eform`. Benchmark results were loaded from https://github.com/ml-evs/modnet-matbench/releases/tag/v0.4.0, archived at [10.5281/zenodo.5109941](https://doi.org/10.5281/zenodo.5109941). This latest benchmark uses an improved GA-based hyperparameter optimization.

#### Notes:
None

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_modnet_v0.1.12).

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
{'python': ['modnet==0.1.12', 'matbench==0.2.0']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1691| 0.6273| 0.0541| 14.3880 |
 | fold_1 | 0.2410| 1.0270| 0.0786| 18.1817 |
 | fold_2 | 0.3899| 2.9174| 0.0759| 59.1179 |
 | fold_3 | 0.2775| 2.2353| 0.0535| 52.1521 |
 | fold_4 | 0.2781| 1.6090| 0.0762| 28.0821 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2711 | 0.3899 | 0.1691 | 0.0714 |
| rmse | 1.6832 | 2.9174 | 0.6273 | 0.8221 |
| mape* | 0.0677 | 0.0786 | 0.0535 | 0.0113 |
| max_error | 34.3844 | 59.1179 | 14.3880 | 18.0529 |


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
 | fold_0 | 0.3121| 0.7028| 0.3201| 5.7101 |
 | fold_1 | 0.3388| 0.7483| 0.3066| 6.8526 |
 | fold_2 | 0.3763| 0.8917| 0.4073| 9.8955 |
 | fold_3 | 0.3121| 0.7313| 0.3327| 6.0927 |
 | fold_4 | 0.3241| 0.7684| 0.3777| 6.9757 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3327 | 0.3763 | 0.3121 | 0.0239 |
| rmse | 0.7685 | 0.8917 | 0.7028 | 0.0653 |
| mape* | 0.3489 | 0.4073 | 0.3066 | 0.0377 |
| max_error | 7.1053 | 9.8955 | 5.7101 | 1.4723 |


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
 | fold_0 | 0.9789| 0.9743| 0.9851| 0.9743 |
 | fold_1 | 0.9701| 0.9618| 0.9790| 0.9618 |
 | fold_2 | 0.9639| 0.9539| 0.9747| 0.9539 |
 | fold_3 | 0.9621| 0.9545| 0.9733| 0.9545 |
 | fold_4 | 0.9710| 0.9570| 0.9798| 0.9570 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9692 | 0.9789 | 0.9621 | 0.0059 |
| balanced_accuracy | 0.9603 | 0.9743 | 0.9539 | 0.0075 |
| f1 | 0.9784 | 0.9851 | 0.9733 | 0.0042 |
| rocauc | 0.9603 | 0.9743 | 0.9539 | 0.0075 |


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
 | fold_0 | 25.5515| 61.1616| 18.6554| 487.7418 |
 | fold_1 | 30.4506| 74.1756| 0.1995| 366.3580 |
 | fold_2 | 45.1925| 134.5285| 0.5392| 871.3962 |
 | fold_3 | 26.9801| 58.2340| 0.2126| 318.7579 |
 | fold_4 | 37.7845| 155.5663| 0.4803| 1564.8245 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 33.1918 | 45.1925 | 25.5515 | 7.3428 |
| rmse | 96.7332 | 155.5663 | 58.2340 | 40.3638 |
| mape* | 4.0174 | 18.6554 | 0.1995 | 7.3203 |
| max_error | 721.8157 | 1564.8245 | 318.7579 | 464.0333 |


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
 | fold_0 | 0.9169| 0.9135| 0.9028| 0.9135 |
 | fold_1 | 0.9030| 0.8995| 0.8867| 0.8995 |
 | fold_2 | 0.9131| 0.9096| 0.8984| 0.9096 |
 | fold_3 | 0.8874| 0.8849| 0.8699| 0.8849 |
 | fold_4 | 0.9140| 0.9116| 0.9003| 0.9116 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9069 | 0.9169 | 0.8874 | 0.0108 |
| balanced_accuracy | 0.9038 | 0.9135 | 0.8849 | 0.0106 |
| f1 | 0.8916 | 0.9028 | 0.8699 | 0.0122 |
| rocauc | 0.9038 | 0.9135 | 0.8849 | 0.0106 |


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
 | fold_0 | 34.7662| 87.4531| 0.0580| 1079.1280 |
 | fold_1 | 36.3582| 75.3959| 0.0631| 640.3050 |
 | fold_2 | 36.5373| 71.7215| 0.0636| 575.7557 |
 | fold_3 | 32.1725| 61.8200| 0.0658| 456.9764 |
 | fold_4 | 31.5413| 53.9441| 0.0592| 396.9667 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 34.2751 | 36.5373 | 31.5413 | 2.0781 |
| rmse | 70.0669 | 87.4531 | 53.9441 | 11.5011 |
| mape* | 0.0619 | 0.0658 | 0.0580 | 0.0029 |
| max_error | 629.8264 | 1079.1280 | 396.9667 | 240.4189 |


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
 | fold_0 | 103.5846| 185.4940| 0.0659| 1121.0504 |
 | fold_1 | 72.3160| 96.1504| 0.0534| 394.0216 |
 | fold_2 | 80.5134| 117.5962| 0.0562| 452.9860 |
 | fold_3 | 81.6766| 133.7020| 0.0587| 711.4582 |
 | fold_4 | 100.7231| 190.9186| 0.0777| 932.3040 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 87.7627 | 103.5846 | 72.3160 | 12.2188 |
| rmse | 144.7722 | 190.9186 | 96.1504 | 37.4511 |
| mape* | 0.0624 | 0.0777 | 0.0534 | 0.0087 |
| max_error | 722.3641 | 1121.0504 | 394.0216 | 276.9541 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




