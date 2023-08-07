# matbench_v0.1: Matformer

### Algorithm description: 

Periodic Graph Transformers for Crystal Material Property Prediction. In this work, we propose a transformer architecture, known as Matformer, for periodic graph representation learning. Our Matformer is designed to be invariant to periodicity and can capture repeating patterns explicitly. In particular, Matformer encodes periodic patterns by efficient use of geometric distances between the same atoms in neighboring cells.

#### Notes:
This version has not been submitted by the original authors and has a modified training script, since the original version is not capable to train on the official matbench. All adjusted files are attached. The model and the parameters are from the original github repository (https://github.com/YKQ98/Matformer), which itself builds on the code basis of ALIGNN.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_matformer).

### References (in bibtex format): 

```
['@inproceedings{yan2022periodic,\n'
 '  title={Periodic Graph Transformers for Crystal Material Property '
 'Prediction},\n'
 '  author={Keqiang Yan and Yi Liu and Yuchao Lin and Shuiwang Ji},\n'
 '  booktitle={The 36th Annual Conference on Neural Information Processing '
 'Systems},\n'
 '  year={2022}\n'
 '}',
 '@misc{yan2022periodicArXiv,\n'
 '      title={Periodic Graph Transformers for Crystal Material Property '
 'Prediction}, \n'
 '      author={Keqiang Yan and Yi Liu and Yuchao Lin and Shuiwang Ji},\n'
 '      year={2022},\n'
 '      eprint={2209.11807},\n'
 '      archivePrefix={arXiv},\n'
 '      primaryClass={cs.LG}\n'
 '}']
```

### User metadata:

```
{'algorithm': 'Matformer'}
```

### Metadata:

| tasks recorded | 1/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['torch',
            'torchvision',
            'torchaudio',
            'torch_geometric',
            'git+https://github.com/YKQ98/Matformer']}
```

### Task data:

#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.8670| 0.8503| 0.8252| 0.8503 |
 | fold_1 | 0.8584| 0.8403| 0.8116| 0.8403 |
 | fold_2 | 0.8710| 0.8553| 0.8321| 0.8553 |
 | fold_3 | 0.7829| 0.7518| 0.6728| 0.7518 |
 | fold_4 | 0.7908| 0.7609| 0.6883| 0.7609 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8340 | 0.8710 | 0.7829 | 0.0388 |
| balanced_accuracy | 0.8117 | 0.8553 | 0.7518 | 0.0455 |
| f1 | 0.7660 | 0.8321 | 0.6728 | 0.0702 |
| rocauc | 0.8117 | 0.8553 | 0.7518 | 0.0455 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |




