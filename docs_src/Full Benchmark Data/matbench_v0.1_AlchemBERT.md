# matbench_v0.1: AlchemBERT

### Algorithm description: 

Using natural language to describe a material structure or chemical formula, then employing BERT for regression or classification. Currently, only a very simple description method has been tested, and there is room for further improvement.

#### Notes:
See the local README.md and RESULTS.md for source provenance, preserved historical scores, validation, and unresolved reproduction requirements. Official source: https://gitee.com/liuxiaotong15/alchemBERT (audited snapshot 3ff872f6510cc88e96061844a2ca41d0c6041c8f; not established as the historical run commit).

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_AlchemBERT).

### References (in bibtex format): 

```
('@article{liu2024alchembert, title={AlchemBERT: Exploring Lightweight '
 'Language Models for Materials Informatics}, author={Liu, X and Liu, X and '
 'Wen, X}, year={2024}, journal={ChemRxiv}, note={This content is a preprint '
 'and has not been peer-reviewed}, doi={10.26434/chemrxiv-2024-r4dnl}, '
 'url={https://chemrxiv.org/engage/chemrxiv/article-details/67540a28085116a133a62b85}}')
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
{'python': ['fire==0.6.0',
            'lightning==2.2.3',
            'torch==2.2.1+cu118',
            'transformers==4.40.1']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2382| 0.7001| 0.0827| 14.5058 |
 | fold_1 | 0.3108| 1.0866| 0.0937| 19.6935 |
 | fold_2 | 0.4507| 2.9621| 0.0939| 59.0578 |
 | fold_3 | 0.3400| 2.3205| 0.0772| 53.5122 |
 | fold_4 | 0.3329| 1.6259| 0.0951| 28.1464 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3345 | 0.4507 | 0.2382 | 0.0684 |
| rmse | 1.7390 | 2.9621 | 0.7001 | 0.8187 |
| mape* | 0.0885 | 0.0951 | 0.0772 | 0.0072 |
| max_error | 34.9831 | 59.0578 | 14.5058 | 18.0153 |


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
 | fold_0 | 0.3629| 0.8566| 0.3681| 6.6809 |
 | fold_1 | 0.3745| 0.8236| 0.3275| 7.3186 |
 | fold_2 | 0.4128| 0.9444| 0.3934| 9.8583 |
 | fold_3 | 0.3667| 0.8410| 0.3970| 7.1650 |
 | fold_4 | 0.3560| 0.7973| 0.4508| 4.7611 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3746 | 0.4128 | 0.3560 | 0.0200 |
| rmse | 0.8526 | 0.9444 | 0.7973 | 0.0500 |
| mape* | 0.3873 | 0.4508 | 0.3275 | 0.0403 |
| max_error | 7.1568 | 9.8583 | 4.7611 | 1.6303 |


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
 | fold_0 | 0.8975| 0.8974| 0.8942| 0.8974 |
 | fold_1 | 0.8943| 0.8942| 0.8923| 0.8942 |
 | fold_2 | 0.9146| 0.9146| 0.9138| 0.9146 |
 | fold_3 | 0.8730| 0.8728| 0.8674| 0.8728 |
 | fold_4 | 0.8791| 0.8793| 0.8859| 0.8793 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8917 | 0.9146 | 0.8730 | 0.0147 |
| balanced_accuracy | 0.8917 | 0.9146 | 0.8728 | 0.0147 |
| f1 | 0.8907 | 0.9138 | 0.8674 | 0.0149 |
| rocauc | 0.8917 | 0.9146 | 0.8728 | 0.0147 |


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
 | fold_0 | 0.8319| 0.7610| 0.8870| 0.7610 |
 | fold_1 | 0.8099| 0.7572| 0.8683| 0.7572 |
 | fold_2 | 0.8204| 0.7719| 0.8753| 0.7719 |
 | fold_3 | 0.8319| 0.7862| 0.8832| 0.7862 |
 | fold_4 | 0.8231| 0.7746| 0.8772| 0.7746 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8234 | 0.8319 | 0.8099 | 0.0082 |
| balanced_accuracy | 0.7702 | 0.7862 | 0.7572 | 0.0103 |
| f1 | 0.8782 | 0.8870 | 0.8683 | 0.0065 |
| rocauc | 0.7702 | 0.7862 | 0.7572 | 0.0103 |


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
 | fold_0 | 34.1746| 56.0409| 39.2753| 270.3499 |
 | fold_1 | 45.2502| 110.5916| 0.3004| 656.7112 |
 | fold_2 | 65.8499| 181.9562| 0.6656| 1061.2964 |
 | fold_3 | 36.2086| 70.5072| 0.3012| 321.2864 |
 | fold_4 | 45.9758| 160.2051| 0.5384| 1523.8740 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 45.4918 | 65.8499 | 34.1746 | 11.2161 |
| rmse | 115.8602 | 181.9562 | 56.0409 | 48.9860 |
| mape* | 8.2162 | 39.2753 | 0.3004 | 15.5302 |
| max_error | 766.7036 | 1523.8740 | 270.3499 | 472.6423 |


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
 | fold_0 | 0.1026| 0.1530| 0.0828| 0.9980 |
 | fold_1 | 0.1003| 0.1494| 0.0790| 1.3257 |
 | fold_2 | 0.0963| 0.1452| 0.0771| 1.0373 |
 | fold_3 | 0.0994| 0.1455| 0.0781| 0.9601 |
 | fold_4 | 0.1005| 0.1478| 0.0789| 0.8616 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0998 | 0.1026 | 0.0963 | 0.0021 |
| rmse | 0.1482 | 0.1530 | 0.1452 | 0.0028 |
| mape* | 0.0792 | 0.0828 | 0.0771 | 0.0019 |
| max_error | 1.0365 | 1.3257 | 0.8616 | 0.1559 |


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
 | fold_0 | 0.0774| 0.1382| 0.0523| 1.6712 |
 | fold_1 | 0.0746| 0.1339| 0.0490| 1.7205 |
 | fold_2 | 0.0744| 0.1282| 0.0495| 1.0924 |
 | fold_3 | 0.0779| 0.1344| 0.0566| 1.0314 |
 | fold_4 | 0.0783| 0.1389| 0.0527| 1.4256 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0765 | 0.0783 | 0.0744 | 0.0017 |
| rmse | 0.1347 | 0.1389 | 0.1282 | 0.0038 |
| mape* | 0.0520 | 0.0566 | 0.0490 | 0.0027 |
| max_error | 1.3882 | 1.7205 | 1.0314 | 0.2852 |


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
 | fold_0 | 0.0647| 0.1433| 0.5567| 4.1352 |
 | fold_1 | 0.0625| 0.1405| 0.3425| 3.2707 |
 | fold_2 | 0.0603| 0.1307| 0.3087| 2.4875 |
 | fold_3 | 0.0617| 0.1361| 0.3705| 2.5468 |
 | fold_4 | 0.0604| 0.1302| 0.4014| 2.7424 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0619 | 0.0647 | 0.0603 | 0.0016 |
| rmse | 0.1362 | 0.1433 | 0.1302 | 0.0052 |
| mape* | 0.3960 | 0.5567 | 0.3087 | 0.0860 |
| max_error | 3.0365 | 4.1352 | 2.4875 | 0.6148 |


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
 | fold_0 | 0.2520| 0.5737| 2.6522| 6.7687 |
 | fold_1 | 0.2559| 0.5855| 3.1212| 7.8313 |
 | fold_2 | 0.2538| 0.5745| 4.9337| 7.5589 |
 | fold_3 | 0.2516| 0.5724| 7.7941| 7.3476 |
 | fold_4 | 0.2584| 0.5863| 7.7557| 6.6484 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2543 | 0.2584 | 0.2516 | 0.0025 |
| rmse | 0.5785 | 0.5863 | 0.5724 | 0.0061 |
| mape* | 5.2514 | 7.7941 | 2.6522 | 2.1969 |
| max_error | 7.2310 | 7.8313 | 6.6484 | 0.4549 |


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
 | fold_0 | 0.8954| 0.8907| 0.8767| 0.8907 |
 | fold_1 | 0.8925| 0.8843| 0.8692| 0.8843 |
 | fold_2 | 0.8892| 0.8809| 0.8651| 0.8809 |
 | fold_3 | 0.8902| 0.8836| 0.8684| 0.8836 |
 | fold_4 | 0.8944| 0.8906| 0.8765| 0.8906 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8924 | 0.8954 | 0.8892 | 0.0024 |
| balanced_accuracy | 0.8860 | 0.8907 | 0.8809 | 0.0040 |
| f1 | 0.8712 | 0.8767 | 0.8651 | 0.0046 |
| rocauc | 0.8860 | 0.8907 | 0.8809 | 0.0040 |


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
 | fold_0 | 0.0450| 0.0760| 0.0481| 0.8169 |
 | fold_1 | 0.0460| 0.0784| 0.0489| 0.9537 |
 | fold_2 | 0.0415| 0.0647| 0.0426| 0.7388 |
 | fold_3 | 0.0414| 0.0674| 0.0423| 0.7883 |
 | fold_4 | 0.0421| 0.0716| 0.0390| 0.8760 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0432 | 0.0460 | 0.0414 | 0.0019 |
| rmse | 0.0716 | 0.0784 | 0.0647 | 0.0051 |
| mape* | 0.0442 | 0.0489 | 0.0390 | 0.0037 |
| max_error | 0.8347 | 0.9537 | 0.7388 | 0.0742 |


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
 | fold_0 | 44.6338| 177.5047| 0.0630| 2487.0572 |
 | fold_1 | 44.4785| 94.2692| 0.0788| 771.3554 |
 | fold_2 | 41.8527| 96.5528| 0.0677| 961.9971 |
 | fold_3 | 49.5034| 182.7023| 0.0764| 2486.9284 |
 | fold_4 | 44.4688| 100.3057| 0.0748| 932.5536 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 44.9874 | 49.5034 | 41.8527 | 2.4849 |
| rmse | 130.2669 | 182.7023 | 94.2692 | 40.7702 |
| mape* | 0.0721 | 0.0788 | 0.0630 | 0.0059 |
| max_error | 1527.9783 | 2487.0572 | 771.3554 | 785.7177 |


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
 | fold_0 | 132.6632| 191.6179| 0.0879| 674.3864 |
 | fold_1 | 131.0388| 180.4992| 0.0938| 692.3776 |
 | fold_2 | 146.1359| 200.8733| 0.1032| 574.9950 |
 | fold_3 | 130.3132| 165.3391| 0.0896| 405.5804 |
 | fold_4 | 124.4095| 159.1095| 0.0905| 387.1785 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 132.9121 | 146.1359 | 124.4095 | 7.1762 |
| rmse | 179.4878 | 200.8733 | 159.1095 | 15.6267 |
| mape* | 0.0930 | 0.1032 | 0.0879 | 0.0054 |
| max_error | 546.9036 | 692.3776 | 387.1785 | 129.3739 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




