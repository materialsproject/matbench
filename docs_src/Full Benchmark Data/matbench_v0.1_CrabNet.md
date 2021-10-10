# matbench_v0.1: CrabNet

### Algorithm description: 

Compositionally restricted attention-based network for materials property predictions. See github page for more information: https://github.com/anthony-wang/CrabNet.



Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_CrabNet).

### References (in bibtex format): 

```
('@article{Wang2021crabnet,\n'
 ' author = {Wang, Anthony Yu-Tung and Kauwe, Steven K. and Murdock, Ryan J. '
 'and Sparks, Taylor D.},\n'
 ' year = {2021},\n'
 ' title = {Compositionally restricted attention-based network for materials '
 'property predictions},\n'
 ' pages = {77},\n'
 ' volume = {7},\n'
 ' number = {1},\n'
 ' doi = {10.1038/s41524-021-00545-1},\n'
 ' publisher = {{Nature Publishing Group}},\n'
 ' shortjournal = {npj Comput. Mater.},\n'
 ' journal = {npj Computational Materials}\n'
 ' }')
```

### User metadata:

```
{}
```

### Metadata:

Tasks recorded: 10 of 13 total

Benchmark is complete? False

Benchmark is structure complete? False

Benchmark is composition complete? False

Benchmark is regression complete? True

Benchmark is classification complete? False

### Software Requirements

```
'See GitHub page for CrabNet, CrabNet version: be89e92.'
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2147| 0.6794| 0.0733| 14.7263 |
 | fold_1 | 0.3048| 1.1243| 0.0989| 19.2249 |
 | fold_2 | 0.4376| 2.9443| 0.0925| 59.1583 |
 | fold_3 | 0.3402| 2.3061| 0.0797| 53.8845 |
 | fold_4 | 0.3195| 1.5900| 0.0942| 27.8634 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3234 | 0.4376 | 0.2147 | 0.0714 |
| rmse | 1.7288 | 2.9443 | 0.6794 | 0.8120 |
| mape* | 0.0877 | 0.0989 | 0.0733 | 0.0096 |
| max_error | 34.9715 | 59.1583 | 14.7263 | 18.1717 |


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
 | fold_0 | 0.3476| 0.8404| 0.3974| 6.6728 |
 | fold_1 | 0.3434| 0.8214| 0.2866| 6.3943 |
 | fold_2 | 0.3473| 0.8680| 0.3421| 9.1598 |
 | fold_3 | 0.3329| 0.8518| 0.3553| 9.8002 |
 | fold_4 | 0.3602| 0.8702| 0.4349| 7.6012 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3463 | 0.3602 | 0.3329 | 0.0088 |
| rmse | 0.8504 | 0.8702 | 0.8214 | 0.0181 |
| mape* | 0.3633 | 0.4349 | 0.2866 | 0.0504 |
| max_error | 7.9256 | 9.8002 | 6.3943 | 1.3459 |


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
 | fold_0 | 36.0753| 71.1404| 24.8117| 394.7442 |
 | fold_1 | 45.8800| 107.0134| 0.3347| 669.9718 |
 | fold_2 | 67.1110| 192.8415| 0.6296| 1039.2952 |
 | fold_3 | 31.6798| 65.1904| 0.2653| 319.1235 |
 | fold_4 | 47.3058| 163.8581| 0.5401| 1532.0118 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 45.6104 | 67.1110 | 31.6798 | 12.2491 |
| rmse | 120.0088 | 192.8415 | 65.1904 | 50.5756 |
| mape* | 5.3163 | 24.8117 | 0.2653 | 9.7486 |
| max_error | 791.0293 | 1532.0118 | 319.1235 | 448.3487 |


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
 | fold_0 | 0.0994| 0.1538| 0.0787| 1.4432 |
 | fold_1 | 0.0994| 0.1648| 0.0794| 2.4220 |
 | fold_2 | 0.1020| 0.1594| 0.0813| 1.0792 |
 | fold_3 | 0.1034| 0.1607| 0.0783| 1.0056 |
 | fold_4 | 0.1031| 0.1633| 0.0810| 1.5313 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1014 | 0.1034 | 0.0994 | 0.0017 |
| rmse | 0.1604 | 0.1648 | 0.1538 | 0.0038 |
| mape* | 0.0797 | 0.0813 | 0.0783 | 0.0012 |
| max_error | 1.4963 | 2.4220 | 1.0056 | 0.5051 |


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
 | fold_0 | 0.0748| 0.1449| 0.0509| 1.6732 |
 | fold_1 | 0.0780| 0.1549| 0.0525| 1.6914 |
 | fold_2 | 0.0698| 0.1344| 0.0463| 1.3116 |
 | fold_3 | 0.0793| 0.1508| 0.0571| 1.0620 |
 | fold_4 | 0.0773| 0.1506| 0.0532| 1.8430 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0758 | 0.0793 | 0.0698 | 0.0034 |
| rmse | 0.1471 | 0.1549 | 0.1344 | 0.0071 |
| mape* | 0.0520 | 0.0571 | 0.0463 | 0.0035 |
| max_error | 1.5162 | 1.8430 | 1.0620 | 0.2864 |


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
 | fold_0 | 0.4080| 0.5445| 0.4861| 2.3726 |
 | fold_1 | 0.4160| 0.5515| 0.5261| 2.1724 |
 | fold_2 | 0.4034| 0.5363| 0.4858| 2.0999 |
 | fold_3 | 0.4096| 0.5428| 0.5270| 2.2336 |
 | fold_4 | 0.3953| 0.5310| 0.4611| 2.2192 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.4065 | 0.4160 | 0.3953 | 0.0069 |
| rmse | 0.5412 | 0.5515 | 0.5310 | 0.0070 |
| mape* | 0.4972 | 0.5270 | 0.4611 | 0.0256 |
| max_error | 2.2195 | 2.3726 | 2.0999 | 0.0896 |


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
 | fold_0 | 60.8044| 155.2771| 0.0881| 1452.7562 |
 | fold_1 | 58.1439| 143.0602| 0.0915| 1207.7800 |
 | fold_2 | 60.2413| 165.1000| 0.0869| 1445.4633 |
 | fold_3 | 47.7603| 114.5270| 0.0895| 894.9224 |
 | fold_4 | 48.6072| 113.9230| 0.0871| 1124.2209 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 55.1114 | 60.8044 | 47.7603 | 5.7317 |
| rmse | 138.3775 | 165.1000 | 113.9230 | 20.9212 |
| mape* | 0.0886 | 0.0915 | 0.0869 | 0.0017 |
| max_error | 1225.0285 | 1452.7562 | 894.9224 | 209.7051 |


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
 | fold_0 | 116.2240| 176.5695| 0.0774| 576.3912 |
 | fold_1 | 88.0920| 117.7789| 0.0632| 387.1094 |
 | fold_2 | 108.1233| 153.4745| 0.0717| 485.5283 |
 | fold_3 | 137.4903| 192.2622| 0.0932| 549.5977 |
 | fold_4 | 86.6503| 124.9355| 0.0654| 386.2023 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 107.3160 | 137.4903 | 86.6503 | 18.9057 |
| rmse | 153.0041 | 192.2622 | 117.7789 | 28.7243 |
| mape* | 0.0742 | 0.0932 | 0.0632 | 0.0107 |
| max_error | 476.9658 | 576.3912 | 386.2023 | 79.4309 |


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
 | fold_0 | 0.2653| 0.5814| 5.4032| 6.8675 |
 | fold_1 | 0.2613| 0.5811| 2.9969| 7.9829 |
 | fold_2 | 0.2648| 0.5903| 5.3833| 7.7856 |
 | fold_3 | 0.2658| 0.5954| 10.1488| 7.9675 |
 | fold_4 | 0.2704| 0.6006| 5.8835| 6.8672 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2655 | 0.2704 | 0.2613 | 0.0029 |
| rmse | 0.5898 | 0.6006 | 0.5811 | 0.0077 |
| mape* | 5.9631 | 10.1488 | 2.9969 | 2.3227 |
| max_error | 7.4941 | 7.9829 | 6.8672 | 0.5165 |


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
 | fold_0 | 0.0853| 0.2492| 0.5075| 4.2164 |
 | fold_1 | 0.0857| 0.2613| 0.4542| 6.3774 |
 | fold_2 | 0.0879| 0.2587| 0.4088| 4.0334 |
 | fold_3 | 0.0854| 0.2499| 0.5596| 6.2383 |
 | fold_4 | 0.0865| 0.2532| 0.4764| 3.9335 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0862 | 0.0879 | 0.0853 | 0.0010 |
| rmse | 0.2544 | 0.2613 | 0.2492 | 0.0048 |
| mape* | 0.4813 | 0.5596 | 0.4088 | 0.0507 |
| max_error | 4.9598 | 6.3774 | 3.9335 | 1.1053 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




