# matbench_v0.1: DeeperGATGNN

### Algorithm description: 

Scalable deeper graph neural networks for high-performance materials property prediction (https://www.cell.com/patterns/pdfExtended/S2666-3899(22)00076-9). We propose a scalable global graph attention neural network model DeeperGATGNN with differentiable group normalization (DGN) and skip connections for high-performance materials property prediction. Our model not only achieved state-of-the art results on benchmark dataset, but also is the most scalable one in terms of graph convolution layers, which allows us to train very deep networks (e.g., >30 layers) without significant performance degradation. Source code link: https://github.com/usccolumbia/deeperGATGNN

#### Notes:
Check your PyTorch and CUDA versions for installing appropriate version of torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, and torch-geometric. Our code can be run on multiple GPUs. If you face any issues, create a new issue in our github repository or email me at omee.sadman@gmail.com

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_DeeperGATGNN).

### References (in bibtex format): 

```
('@article{omee2022scalable,\n'
 ' title={Scalable deeper graph neural networks for high-performance materials '
 'property prediction},\n'
 ' author={Omee, Sadman Sadeed and Louis, Steph-Yves and Fu, Nihang and Wei, '
 'Lai and Dey, Sourin and Dong, Rongzhi and Li, Qinyang and Hu, Jianjun},\n'
 ' journal={Patterns},\n'
 ' year={2022},\n'
 ' publisher={Elsevier}\n'
 '}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 8/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['torch==1.10.0+cu113',
            'torch-scatter==2.0.9',
            'torch-sparse==0.6.13',
            'torch-spline-conv==1.2.1',
            'torch-cluster==1.6.0',
            'torch-geometric==2.0.4',
            'pymatgen==2023.3.23',
            'ase==3.22.1',
            'dscribe==1.2.2',
            'hyperopt==0.2.5',
            'joblib==1.2.0',
            'matplotlib==3.7.1',
            'numpy==1.21.0',
            'pickle5==0.0.11',
            'ray==1.11.0',
            'scikit-learn==1.2.2',
            'scipy==1.10.1',
            'tensorboardX==2.6']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2203| 0.6949| 0.0780| 14.5437 |
 | fold_1 | 0.2732| 1.0361| 0.0893| 18.9965 |
 | fold_2 | 0.4473| 2.9482| 0.1005| 58.7139 |
 | fold_3 | 0.3267| 2.0680| 0.0762| 43.2260 |
 | fold_4 | 0.4097| 2.1864| 0.1378| 34.6313 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3355 | 0.4473 | 0.2203 | 0.0839 |
| rmse | 1.7867 | 2.9482 | 0.6949 | 0.8177 |
| mape* | 0.0964 | 0.1378 | 0.0762 | 0.0225 |
| max_error | 34.0223 | 58.7139 | 14.5437 | 16.1242 |


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
 | fold_0 | 31.2373| 54.7965| 39.3128| 201.5965 |
 | fold_1 | 43.4485| 101.0501| 0.3128| 505.6367 |
 | fold_2 | 66.2654| 164.7194| 0.7606| 885.5912 |
 | fold_3 | 39.1329| 90.9378| 0.3762| 502.8997 |
 | fold_4 | 47.2216| 155.5921| 0.6308| 1468.3439 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 45.4611 | 66.2654 | 31.2373 | 11.6819 |
| rmse | 113.4192 | 164.7194 | 54.7965 | 41.2439 |
| mape* | 8.2786 | 39.3128 | 0.3128 | 15.5180 |
| max_error | 712.8136 | 1468.3439 | 201.5965 | 435.6621 |


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
 | fold_0 | 0.0890| 0.1342| 0.0697| 1.0543 |
 | fold_1 | 0.0916| 0.1451| 0.0732| 1.4002 |
 | fold_2 | 0.0865| 0.1326| 0.0690| 0.9744 |
 | fold_3 | 0.0939| 0.1436| 0.0729| 0.9796 |
 | fold_4 | 0.0902| 0.1408| 0.0724| 1.0748 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0903 | 0.0939 | 0.0865 | 0.0025 |
| rmse | 0.1393 | 0.1451 | 0.1326 | 0.0050 |
| mape* | 0.0714 | 0.0732 | 0.0690 | 0.0017 |
| max_error | 1.0967 | 1.4002 | 0.9744 | 0.1569 |


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
 | fold_0 | 0.0678| 0.1335| 0.0446| 1.7755 |
 | fold_1 | 0.0688| 0.1320| 0.0454| 1.5375 |
 | fold_2 | 0.0665| 0.1264| 0.0442| 1.1731 |
 | fold_3 | 0.0752| 0.1374| 0.0532| 0.9629 |
 | fold_4 | 0.0707| 0.1327| 0.0476| 1.3972 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0698 | 0.0752 | 0.0665 | 0.0030 |
| rmse | 0.1324 | 0.1374 | 0.1264 | 0.0035 |
| mape* | 0.0470 | 0.0532 | 0.0442 | 0.0033 |
| max_error | 1.3692 | 1.7755 | 0.9629 | 0.2820 |


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
 | fold_0 | 0.0348| 0.0803| 0.4272| 3.0043 |
 | fold_1 | 0.0304| 0.0765| 0.2247| 2.8463 |
 | fold_2 | 0.0374| 0.0769| 0.2690| 3.4390 |
 | fold_3 | 0.0326| 0.0727| 0.2908| 2.1084 |
 | fold_4 | 0.0346| 0.0734| 0.2950| 1.9758 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0340 | 0.0374 | 0.0304 | 0.0023 |
| rmse | 0.0759 | 0.0803 | 0.0727 | 0.0027 |
| mape* | 0.3013 | 0.4272 | 0.2247 | 0.0677 |
| max_error | 2.6748 | 3.4390 | 1.9758 | 0.5534 |


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
 | fold_0 | 0.1765| 0.4271| 1.6708| 6.8089 |
 | fold_1 | 0.1666| 0.4181| 2.1018| 7.3731 |
 | fold_2 | 0.1734| 0.4180| 4.6320| 7.3691 |
 | fold_3 | 0.1650| 0.4090| 5.9843| 6.7121 |
 | fold_4 | 0.1655| 0.4153| 2.9621| 7.4794 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1694 | 0.1765 | 0.1650 | 0.0047 |
| rmse | 0.4175 | 0.4271 | 0.4090 | 0.0058 |
| mape* | 3.4702 | 5.9843 | 1.6708 | 1.6149 |
| max_error | 7.1485 | 7.4794 | 6.7121 | 0.3207 |


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
 | fold_0 | 0.0280| 0.0569| 0.0272| 0.8604 |
 | fold_1 | 0.0306| 0.0628| 0.0305| 1.0267 |
 | fold_2 | 0.0274| 0.0521| 0.0267| 0.8454 |
 | fold_3 | 0.0290| 0.0536| 0.0283| 0.7928 |
 | fold_4 | 0.0291| 0.0583| 0.0254| 0.9152 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0288 | 0.0306 | 0.0274 | 0.0011 |
| rmse | 0.0568 | 0.0628 | 0.0521 | 0.0038 |
| mape* | 0.0276 | 0.0305 | 0.0254 | 0.0017 |
| max_error | 0.8881 | 1.0267 | 0.7928 | 0.0795 |


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
 | fold_0 | 34.9378| 94.4311| 0.0567| 1134.0565 |
 | fold_1 | 31.8727| 66.0292| 0.0532| 583.8024 |
 | fold_2 | 32.4509| 75.4627| 0.0555| 708.0555 |
 | fold_3 | 34.7909| 91.8744| 0.0620| 1111.3660 |
 | fold_4 | 31.5530| 66.4401| 0.0554| 522.2902 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 33.1211 | 34.9378 | 31.5530 | 1.4530 |
| rmse | 78.8475 | 94.4311 | 66.0292 | 12.1841 |
| mape* | 0.0566 | 0.0620 | 0.0532 | 0.0029 |
| max_error | 811.9141 | 1134.0565 | 522.2902 | 260.8259 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




