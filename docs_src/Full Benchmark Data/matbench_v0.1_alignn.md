# matbench_v0.1: ALIGNN

### Algorithm description: 

The Atomistic Line Graph Neural Network (https://doi.org/10.1038/s41524-021-00650-1) introduces a new graph convolution layer that explicitly models both two and three body interactions in atomistic systems. This is achieved by composing two edge-gated graph convolution layers, the first applied to the atomistic line graph L(g) (representing triplet interactions) and the second applied to the atomistic bond graph g (representing pair interactions). The atomistic graph g consists of a node for each atom i (with atom/node representations hi), and one edge for each atom pair within a cutoff radius (with bond/pair representations eij). The atomistic line graph L(g) represents relationships between atom triplets: it has nodes corresponding to bonds (sharing representations eij with those in g) and edges corresponding to bond angles (with angle/triplet representations tijk).The line graph convolution updates the triplet representations and the pair representations; the direct graph convolution further updates the pair representations and the atom representations.

#### Notes:
None

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_alignn).

### References (in bibtex format): 

```
('@article{choudhary2021atomistic,title={Atomistic Line Graph Neural Network '
 'for improved materials property predictions},author={Choudhary, Kamal and '
 'DeCost, Brian},journal={npj Computational '
 'Materials},volume={7},number={1},pages={1--8},year={2021},publisher={Nature '
 'Publishing Group}}')
```

### User metadata:

```
{'algorithm': 'ALIGNN'}
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
{'python': ['absl-py==1.0.0',
            'alignn==2021.12.27',
            'astunparse==1.6.3',
            'attrs==21.3.0',
            'black==21.12b0',
            'cachetools==4.2.4',
            'certifi==2021.10.8',
            'charset-normalizer==2.0.9',
            'click==8.0.3',
            'cloudpickle==2.0.0',
            'cycler==0.11.0',
            'decorator==5.1.0',
            'dgl==0.6.1',
            'dgl-cu111==0.6.1',
            'dm-tree==0.1.6',
            'flake8==4.0.1',
            'flatbuffers==2.0',
            'fonttools==4.28.5',
            'future==0.18.2',
            'gast==0.4.0',
            'google-auth==2.3.3',
            'google-auth-oauthlib==0.4.6',
            'google-pasta==0.2.0',
            'grpcio==1.43.0',
            'h5py==3.6.0',
            'idna==3.3',
            'importlib-metadata==4.10.0',
            'importlib-resources==5.4.0',
            'jarvis-tools==2021.12.16',
            'joblib==1.1.0',
            'jsonschema==4.3.2',
            'julia==0.5.6',
            'keras==2.7.0',
            'Keras-Preprocessing==1.1.2',
            'kiwisolver==1.3.2',
            'libclang==12.0.0',
            'Markdown==3.3.6',
            'matbench==0.5',
            'matminer==0.6.5',
            'matplotlib==3.5.1',
            'mccabe==0.6.1',
            'modnet==0.1.11',
            'monty==2021.8.17',
            'mpmath==1.2.1',
            'mypy-extensions==0.4.3',
            'networkx==2.6.3',
            'numpy==1.21.5',
            'oauthlib==3.1.1',
            'opt-einsum==3.3.0',
            'packaging==21.3',
            'palettable==3.3.0',
            'pandas==1.3.5',
            'pathspec==0.9.0',
            'Pillow==8.4.0',
            'Pint==0.18',
            'platformdirs==2.4.0',
            'plotly==5.5.0',
            'protobuf==3.19.1',
            'pyasn1==0.4.8',
            'pyasn1-modules==0.2.8',
            'pycodestyle==2.8.0',
            'pydantic==1.8.2',
            'pydocstyle==6.1.1',
            'pyflakes==2.4.0',
            'pymatgen==2020.8.13',
            'pymongo==4.0.1',
            'pyparsing==2.4.7',
            'pyrsistent==0.18.0',
            'python-dateutil==2.8.2',
            'pytorch-ignite==0.4.7',
            'pytz==2021.3',
            'requests==2.26.0',
            'requests-oauthlib==1.3.0',
            'rsa==4.8',
            'ruamel.yaml==0.17.19',
            'ruamel.yaml.clib==0.2.6',
            'scikit-learn==0.23.2',
            'scipy==1.7.3',
            'six==1.16.0',
            'snowballstemmer==2.2.0',
            'spglib==1.16.3',
            'sympy==1.9',
            'tabulate==0.8.9',
            'tenacity==8.0.1',
            'tensorboard==2.7.0',
            'tensorboard-data-server==0.6.1',
            'tensorboard-plugin-wit==1.8.0',
            'tensorflow==2.7.0',
            'tensorflow-estimator==2.7.0',
            'tensorflow-io-gcs-filesystem==0.23.1',
            'tensorflow-probability==0.15.0',
            'termcolor==1.1.0',
            'threadpoolctl==3.0.0',
            'tomli==1.2.3',
            'toolz==0.11.2',
            'torch==1.10.1',
            'tqdm==4.62.3',
            'typing_extensions==4.0.1',
            'uncertainties==3.1.6',
            'urllib3==1.26.7',
            'Werkzeug==2.0.2',
            'wrapt==1.13.3',
            'xmltodict==0.12.0',
            'zipp==3.6.0']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1974| 0.6847| 0.0700| 13.6699 |
 | fold_1 | 0.3552| 1.4949| 0.1212| 23.3832 |
 | fold_2 | 0.4551| 3.1409| 0.1035| 58.7285 |
 | fold_3 | 0.3164| 2.2413| 0.0725| 51.3719 |
 | fold_4 | 0.4005| 2.2637| 0.1258| 36.5440 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3449 | 0.4551 | 0.1974 | 0.0871 |
| rmse | 1.9651 | 3.1409 | 0.6847 | 0.8257 |
| mape* | 0.0986 | 0.1258 | 0.0700 | 0.0236 |
| max_error | 36.7395 | 58.7285 | 13.6699 | 16.7825 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 37.0106| 85.2042| 23.4970| 649.7056 |
 | fold_1 | 45.3752| 121.0872| 0.4163| 767.6072 |
 | fold_2 | 58.3624| 172.0326| 0.6282| 973.0735 |
 | fold_3 | 31.9625| 55.3213| 0.3025| 275.8517 |
 | fold_4 | 44.4110| 153.4614| 0.5874| 1519.7424 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 43.4244 | 58.3624 | 31.9625 | 8.9491 |
| rmse | 117.4213 | 172.0326 | 55.3213 | 42.8697 |
| mape* | 5.0863 | 23.4970 | 0.3025 | 9.2061 |
| max_error | 837.1961 | 1519.7424 | 275.8517 | 409.7401 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0709| 0.1093| 0.0560| 0.9401 |
 | fold_1 | 0.0725| 0.1167| 0.0587| 1.1324 |
 | fold_2 | 0.0712| 0.1122| 0.0566| 0.7799 |
 | fold_3 | 0.0710| 0.1102| 0.0558| 0.8718 |
 | fold_4 | 0.0719| 0.1133| 0.0563| 0.7814 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0715 | 0.0725 | 0.0709 | 0.0006 |
| rmse | 0.1123 | 0.1167 | 0.1093 | 0.0026 |
| mape* | 0.0567 | 0.0587 | 0.0558 | 0.0011 |
| max_error | 0.9011 | 1.1324 | 0.7799 | 0.1303 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0552| 0.1062| 0.0361| 1.6438 |
 | fold_1 | 0.0572| 0.1158| 0.0375| 1.3470 |
 | fold_2 | 0.0531| 0.1017| 0.0351| 1.1254 |
 | fold_3 | 0.0615| 0.1187| 0.0442| 1.1145 |
 | fold_4 | 0.0569| 0.1105| 0.0379| 1.3937 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0568 | 0.0615 | 0.0531 | 0.0028 |
| rmse | 0.1106 | 0.1187 | 0.1017 | 0.0062 |
| mape* | 0.0382 | 0.0442 | 0.0351 | 0.0032 |
| max_error | 1.3249 | 1.6438 | 1.1145 | 0.1955 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0218| 0.0647| 0.1437| 3.5487 |
 | fold_1 | 0.0220| 0.0534| 0.1299| 2.9160 |
 | fold_2 | 0.0209| 0.0494| 0.1434| 2.1189 |
 | fold_3 | 0.0219| 0.0546| 0.1762| 1.6654 |
 | fold_4 | 0.0210| 0.0499| 0.2528| 1.4116 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0215 | 0.0220 | 0.0209 | 0.0005 |
| rmse | 0.0544 | 0.0647 | 0.0494 | 0.0055 |
| mape* | 0.1692 | 0.2528 | 0.1299 | 0.0445 |
| max_error | 2.3321 | 3.5487 | 1.4116 | 0.7948 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1860| 0.4624| 2.3206| 6.6263 |
 | fold_1 | 0.1852| 0.4622| 2.5314| 7.4756 |
 | fold_2 | 0.1901| 0.4729| 4.2501| 6.2931 |
 | fold_3 | 0.1812| 0.4497| 5.1591| 6.9986 |
 | fold_4 | 0.1880| 0.4703| 4.8122| 6.4513 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1861 | 0.1901 | 0.1812 | 0.0030 |
| rmse | 0.4635 | 0.4729 | 0.4497 | 0.0081 |
| mape* | 3.8147 | 5.1591 | 2.3206 | 1.1723 |
| max_error | 6.7690 | 7.4756 | 6.2931 | 0.4242 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9173| 0.9156| 0.9047| 0.9156 |
 | fold_1 | 0.9135| 0.9117| 0.9003| 0.9117 |
 | fold_2 | 0.9146| 0.9125| 0.9013| 0.9125 |
 | fold_3 | 0.9135| 0.9117| 0.9003| 0.9117 |
 | fold_4 | 0.9147| 0.9123| 0.9012| 0.9123 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9147 | 0.9173 | 0.9135 | 0.0014 |
| balanced_accuracy | 0.9128 | 0.9156 | 0.9117 | 0.0015 |
| f1 | 0.9015 | 0.9047 | 0.9003 | 0.0016 |
| rocauc | 0.9128 | 0.9156 | 0.9117 | 0.0015 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0293| 0.0577| 0.0292| 0.8306 |
 | fold_1 | 0.0301| 0.0622| 0.0291| 0.9028 |
 | fold_2 | 0.0276| 0.0509| 0.0274| 0.8358 |
 | fold_3 | 0.0286| 0.0532| 0.0276| 0.7984 |
 | fold_4 | 0.0282| 0.0558| 0.0253| 0.8666 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0288 | 0.0301 | 0.0276 | 0.0009 |
| rmse | 0.0559 | 0.0622 | 0.0509 | 0.0039 |
| mape* | 0.0277 | 0.0292 | 0.0253 | 0.0014 |
| max_error | 0.8468 | 0.9028 | 0.7984 | 0.0354 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 33.4099| 57.5251| 0.0623| 394.6285 |
 | fold_1 | 28.0306| 44.1719| 0.0571| 277.0146 |
 | fold_2 | 29.4772| 53.9163| 0.0575| 300.2450 |
 | fold_3 | 29.4931| 62.0127| 0.0571| 615.3466 |
 | fold_4 | 27.2814| 49.8790| 0.0521| 528.2511 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 29.5385 | 33.4099 | 27.2814 | 2.1148 |
| rmse | 53.5010 | 62.0127 | 44.1719 | 6.1476 |
| mape* | 0.0572 | 0.0623 | 0.0521 | 0.0032 |
| max_error | 423.0972 | 615.3466 | 277.0146 | 130.5836 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 2, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': 'user_data', 'epochs': 3, 'filename': 'sample', 'id_tag': 'jid', 'keep_data_...` |




