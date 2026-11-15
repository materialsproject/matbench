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

**Probability ROC AUC: unavailable.** No usable probabilities were saved, so probability ROC AUC cannot be recovered. Balanced accuracy, accuracy and F1 below are recomputed from saved hard-label decisions aligned by sample ID with official test labels. The original decision threshold may be unknown; these historical decisions have not been reevaluated at a common threshold. Statistics use unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 |
|------|------|------|------|
| fold_0 | 0.8670 | 0.8503 | 0.8252 |
| fold_1 | 0.8584 | 0.8403 | 0.8116 |
| fold_2 | 0.8710 | 0.8553 | 0.8321 |
| fold_3 | 0.7829 | 0.7518 | 0.6728 |
| fold_4 | 0.7908 | 0.7609 | 0.6883 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8340 | 0.8710 | 0.7829 | 0.0388 |
| balanced accuracy | 0.8117 | 0.8553 | 0.7518 | 0.0455 |
| f1 | 0.7660 | 0.8321 | 0.6728 | 0.0702 |

<details><summary>Original stored classification scores (historical)</summary>
<p>These values reproduce the original submission artifact. Older stored <code>rocauc</code> values may reflect thresholded decisions, whose AUC equals balanced accuracy; they must not be interpreted as the recomputed probability ROC AUC above.</p>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>fold</th>
      <th>accuracy</th>
      <th>balanced_accuracy</th>
      <th>f1</th>
      <th>stored rocauc (historical)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fold_0</td>
      <td>0.8670</td>
      <td>0.8503</td>
      <td>0.8252</td>
      <td>0.8503</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.8584</td>
      <td>0.8403</td>
      <td>0.8116</td>
      <td>0.8403</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.8710</td>
      <td>0.8553</td>
      <td>0.8321</td>
      <td>0.8553</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.7829</td>
      <td>0.7518</td>
      <td>0.6728</td>
      <td>0.7518</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.7908</td>
      <td>0.7609</td>
      <td>0.6883</td>
      <td>0.7609</td>
    </tr>
  </tbody>
</table><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>metric</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>accuracy</td>
      <td>0.8340</td>
      <td>0.8710</td>
      <td>0.7829</td>
      <td>0.0388</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.8117</td>
      <td>0.8553</td>
      <td>0.7518</td>
      <td>0.0455</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.7660</td>
      <td>0.8321</td>
      <td>0.6728</td>
      <td>0.0702</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.8117</td>
      <td>0.8553</td>
      <td>0.7518</td>
      <td>0.0455</td>
    </tr>
  </tbody>
</table>
</details>

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_1 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_2 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_3 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |
| fold_4 | `{'atom_features': 'cgcnn', 'batch_size': 64, 'classification_threshold': None, 'criterion': 'mse', 'cutoff': 8.0, 'dataset': None, 'distributed': False, 'epochs': 500, 'filename': 'sample', 'id_tag': ...` |




