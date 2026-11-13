# matbench_v0.1: AlchemBERT

### Algorithm description: 

AlchemBERT fine-tunes BERT-base-cased with a scalar prediction head on natural-language descriptions of materials. Composition tasks use chemical formulas or alloy fractions; structure tasks additionally describe lattice, symmetry, atomic sites and local environments. This submission contains the natural-language-input results for all 13 Matbench v0.1 tasks and five official folds, from the work published in Cell Reports Physical Science (2025).

#### Notes:
Metrics for all 13 tasks and 65 folds are recomputed from the unchanged archived predictions using current Matbench scoring, including probability-based classification ROC-AUC. This is a scoring correction, not model retraining. See README.md, RESULTS.md, RESCORING.md, INPUT_AUDIT.md, PAPER_EVIDENCE.md and RUNNING.md for the score comparison, source and input audits, training/recording tools and remaining historical provenance limits. Official source: https://gitee.com/liuxiaotong15/alchemBERT (audited snapshot 3ff872f6510cc88e96061844a2ca41d0c6041c8f; not established as the historical run commit). Historical requirements below are retained; tested new CPU tools use requirements-rerun-cpu.txt.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_AlchemBERT).

### References (in bibtex format): 

```
('@article{wang2025alchembert, title={Exploring lightweight language models '
 'for materials informatics with {AlchemBERT}}, author={Wang, Yuhang and Liu, '
 'Xiaotong and Chen, Haotian and Yang, Tao and Liu, Xingchen and Wen, '
 'Xiaodong}, journal={Cell Reports Physical Science}, volume={6}, number={8}, '
 'pages={102724}, year={2025}, doi={10.1016/j.xcrp.2025.102724}, '
 'url={https://doi.org/10.1016/j.xcrp.2025.102724}}')
```

### User metadata:

```
{'score_migration': {'historical_training_reproduced': False,
                     'label_sources': {'matbench_dielectric': {'filename': 'matbench_dielectric.json.gz',
                                                               'kind': 'official_matminer_dataset',
                                                               'sha256': '83befa09bc2ec2f4b6143afc413157827a90e5e2e42c1eb507ccfa01bf26a1d6',
                                                               'url': 'https://ml.materialsproject.org/projects/matbench_dielectric.json.gz',
                                                               'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476067'},
                                       'matbench_expt_gap': {'filename': 'matbench_expt_gap.json.gz',
                                                             'kind': 'official_matminer_dataset',
                                                             'sha256': '783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2',
                                                             'url': 'https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz',
                                                             'verified_public_mirror': None},
                                       'matbench_expt_is_metal': {'filename': 'matbench_expt_is_metal.json.gz',
                                                                  'kind': 'official_matminer_dataset',
                                                                  'sha256': '8f2a4f9bacdcbc5c2c73615629ee7986f09d39bed40ba7db52b61b2889730887',
                                                                  'url': 'https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz',
                                                                  'verified_public_mirror': None},
                                       'matbench_glass': {'filename': 'matbench_glass.json.gz',
                                                          'kind': 'official_matminer_dataset',
                                                          'sha256': '36beb654e2a463ee2a6572105bea0ca2961eee7c7b26a25377bff2c3b338e53a',
                                                          'url': 'https://ml.materialsproject.org/projects/matbench_glass.json.gz',
                                                          'verified_public_mirror': 'https://ndownloader.figshare.com/files/17494637'},
                                       'matbench_jdft2d': {'filename': 'matbench_jdft2d.json.gz',
                                                           'kind': 'official_matminer_dataset',
                                                           'sha256': '26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf',
                                                           'url': 'https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz',
                                                           'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476046'},
                                       'matbench_log_gvrh': {'filename': 'matbench_log_gvrh.json.gz',
                                                             'kind': 'official_matminer_dataset',
                                                             'sha256': '098af941f4c663270f1fe21abf20ffad6fb85ecbfcba5786ceac03983ac29da7',
                                                             'url': 'https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz',
                                                             'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476064'},
                                       'matbench_log_kvrh': {'filename': 'matbench_log_kvrh.json.gz',
                                                             'kind': 'official_matminer_dataset',
                                                             'sha256': '44b113ddb7e23aa18731a62c74afa7e5aa654199e0db5f951c8248a00955c9cd',
                                                             'url': 'https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz',
                                                             'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476061'},
                                       'matbench_mp_e_form': {'filename': 'matbench_mp_e_form.json.gz',
                                                              'kind': 'official_matminer_dataset',
                                                              'sha256': 'dedcb1d4ba2e3e50dbdd45ba5bc647a00e9c2bcf8f8bf556dc8e92caa39eb21f',
                                                              'url': 'https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz',
                                                              'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476028'},
                                       'matbench_mp_gap': {'filename': 'matbench_mp_gap.json.gz',
                                                           'kind': 'official_matminer_dataset',
                                                           'sha256': '58b65746bd88329986ed66031a2ac1369c7c522f7bc9f9081528e07097c2c057',
                                                           'url': 'https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz',
                                                           'verified_public_mirror': 'https://ndownloader.figshare.com/files/17084741'},
                                       'matbench_mp_is_metal': {'filename': 'matbench_mp_is_metal.json.gz',
                                                                'kind': 'official_matminer_dataset',
                                                                'sha256': '9a028ed5750a4c76ca36e9f3c8d48fe0bf3fb21b76ec2289e58ae7048d527919',
                                                                'url': 'https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz',
                                                                'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476034'},
                                       'matbench_perovskites': {'filename': 'matbench_perovskites.json.gz',
                                                                'kind': 'official_matminer_dataset',
                                                                'sha256': '4641e2417f8ec8b50096d2230864468dfa08278dc9d257c327f65d0305278483',
                                                                'url': 'https://ml.materialsproject.org/projects/matbench_perovskites.json.gz',
                                                                'verified_public_mirror': 'https://ndownloader.figshare.com/files/17494805'},
                                       'matbench_phonons': {'filename': 'matbench_phonons.json.gz',
                                                            'kind': 'official_matminer_dataset',
                                                            'sha256': '4db551f21ec5f577e6202725f10e34dfc509aa7df3a6bdaac497da7f6dbbb9b3',
                                                            'url': 'https://ml.materialsproject.org/projects/matbench_phonons.json.gz',
                                                            'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476037'},
                                       'matbench_steels': {'filename': 'matbench_steels.json.gz',
                                                           'kind': 'official_matminer_dataset',
                                                           'sha256': '473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252',
                                                           'url': 'https://ml.materialsproject.org/projects/matbench_steels.json.gz',
                                                           'verified_public_mirror': 'https://ndownloader.figshare.com/files/17476040'}},
                     'migration_script_sha256': 'bb38059ccf7512225ab3db21e532ec9c9f6cbb600e26453eaa720db171395cdb',
                     'official_spec_sha256': {'matbench_v0.1_dataset_metadata.json': 'bd8d2962406a1829bda984c38c36b4f3ce6acc20e46d34a42d3ba5294a1df787',
                                              'matbench_v0.1_validation.json': '94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d'},
                     'original_artifact_sha256': 'a0bb054a3a32b09041940aa90a7adc4e63a538811af430194057736ea5bd3daf',
                     'original_content_hash': '0f438bf88da6ab50a84d493696018caa976ae785cf0c3fafc01524432b6d847f',
                     'original_datestamp': '2024.12.09 08:47.31',
                     'original_matbench_version': '0.6',
                     'predictions_parameters_and_uncertainties_unchanged': True,
                     'purpose': 'Recompute cached scores from unchanged '
                                'historical predictions',
                     'rescoring_report_sha256': 'f7672cf5e2f1f44f4912369324bbb9171f64a6b63b2d2b60896c514bac90f437',
                     'retrained': False,
                     'scorer_upstream_commit': '2f7512ef40935b5d642127887fc4daf42236a2b7',
                     'software': {'data_ops_sha256': 'f791263c675c45b6938c78d856676a6821ac6879526bc9aca7d82523303c3414',
                                  'ijson': '3.6.0',
                                  'matbench': '0.6',
                                  'numpy': '2.5.4',
                                  'rescore_script_sha256': '7b12d855c2d3b909369516db0f2aa31103917b75f9d2449bd55913ea52a160c9',
                                  'scikit-learn': '1.9.1'}}}
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
|------|------|------|------|------|
| fold_0 | 0.2382 | 0.7001 | 0.0827 | 14.5058 |
| fold_1 | 0.3108 | 1.0866 | 0.0937 | 19.6935 |
| fold_2 | 0.4507 | 2.9621 | 0.0939 | 59.0578 |
| fold_3 | 0.3400 | 2.3205 | 0.0772 | 53.5122 |
| fold_4 | 0.3329 | 1.6259 | 0.0951 | 28.1464 |

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
|------|------|------|------|------|
| fold_0 | 0.3629 | 0.8566 | 0.3681 | 6.6809 |
| fold_1 | 0.3745 | 0.8236 | 0.3275 | 7.3186 |
| fold_2 | 0.4128 | 0.9444 | 0.3934 | 9.8583 |
| fold_3 | 0.3667 | 0.8410 | 0.3970 | 7.1650 |
| fold_4 | 0.3560 | 0.7973 | 0.4508 | 4.7611 |

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

Probability ROC AUC below is recomputed from the original saved positive-class probabilities. Balanced accuracy, accuracy and F1 use p > 0.5. Predictions are aligned by sample ID with official test labels. Statistics are unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 | probability ROC AUC |
|------|------|------|------|------|
| fold_0 | 0.8975 | 0.8974 | 0.8942 | 0.9607 |
| fold_1 | 0.8943 | 0.8942 | 0.8923 | 0.9598 |
| fold_2 | 0.9146 | 0.9146 | 0.9138 | 0.9647 |
| fold_3 | 0.8730 | 0.8728 | 0.8674 | 0.9543 |
| fold_4 | 0.8791 | 0.8793 | 0.8859 | 0.9584 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8917 | 0.9146 | 0.8730 | 0.0147 |
| balanced accuracy | 0.8917 | 0.9146 | 0.8728 | 0.0147 |
| f1 | 0.8907 | 0.9138 | 0.8674 | 0.0149 |
| probability ROC AUC | 0.9596 | 0.9647 | 0.9543 | 0.0034 |

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
      <td>0.8975</td>
      <td>0.8974</td>
      <td>0.8942</td>
      <td>0.9607</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.8943</td>
      <td>0.8942</td>
      <td>0.8923</td>
      <td>0.9598</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.9146</td>
      <td>0.9146</td>
      <td>0.9138</td>
      <td>0.9647</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.8730</td>
      <td>0.8728</td>
      <td>0.8674</td>
      <td>0.9543</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.8791</td>
      <td>0.8793</td>
      <td>0.8859</td>
      <td>0.9584</td>
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
      <td>0.8917</td>
      <td>0.9146</td>
      <td>0.8730</td>
      <td>0.0147</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.8917</td>
      <td>0.9146</td>
      <td>0.8728</td>
      <td>0.0147</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.8907</td>
      <td>0.9138</td>
      <td>0.8674</td>
      <td>0.0149</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.9596</td>
      <td>0.9647</td>
      <td>0.9543</td>
      <td>0.0034</td>
    </tr>
  </tbody>
</table>
</details>

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_glass`

Probability ROC AUC below is recomputed from the original saved positive-class probabilities. Balanced accuracy, accuracy and F1 use p > 0.5. Predictions are aligned by sample ID with official test labels. Statistics are unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 | probability ROC AUC |
|------|------|------|------|------|
| fold_0 | 0.8319 | 0.7610 | 0.8870 | 0.9011 |
| fold_1 | 0.8099 | 0.7572 | 0.8683 | 0.8734 |
| fold_2 | 0.8204 | 0.7719 | 0.8753 | 0.8660 |
| fold_3 | 0.8319 | 0.7862 | 0.8832 | 0.9002 |
| fold_4 | 0.8231 | 0.7746 | 0.8772 | 0.8757 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8234 | 0.8319 | 0.8099 | 0.0082 |
| balanced accuracy | 0.7702 | 0.7862 | 0.7572 | 0.0103 |
| f1 | 0.8782 | 0.8870 | 0.8683 | 0.0065 |
| probability ROC AUC | 0.8833 | 0.9011 | 0.8660 | 0.0146 |

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
      <td>0.8319</td>
      <td>0.7610</td>
      <td>0.8870</td>
      <td>0.9011</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.8099</td>
      <td>0.7572</td>
      <td>0.8683</td>
      <td>0.8734</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.8204</td>
      <td>0.7719</td>
      <td>0.8753</td>
      <td>0.8660</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.8319</td>
      <td>0.7862</td>
      <td>0.8832</td>
      <td>0.9002</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.8231</td>
      <td>0.7746</td>
      <td>0.8772</td>
      <td>0.8757</td>
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
      <td>0.8234</td>
      <td>0.8319</td>
      <td>0.8099</td>
      <td>0.0082</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.7702</td>
      <td>0.7862</td>
      <td>0.7572</td>
      <td>0.0103</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.8782</td>
      <td>0.8870</td>
      <td>0.8683</td>
      <td>0.0065</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.8833</td>
      <td>0.9011</td>
      <td>0.8660</td>
      <td>0.0146</td>
    </tr>
  </tbody>
</table>
</details>

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
|------|------|------|------|------|
| fold_0 | 34.1746 | 56.0409 | 39.2753 | 270.3499 |
| fold_1 | 45.2502 | 110.5916 | 0.3004 | 656.7112 |
| fold_2 | 65.8499 | 181.9562 | 0.6656 | 1061.2964 |
| fold_3 | 36.2086 | 70.5072 | 0.3012 | 321.2864 |
| fold_4 | 45.9758 | 160.2051 | 0.5384 | 1523.8740 |

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
|------|------|------|------|------|
| fold_0 | 0.1026 | 0.1530 | 0.0828 | 0.9980 |
| fold_1 | 0.1003 | 0.1494 | 0.0790 | 1.3257 |
| fold_2 | 0.0963 | 0.1452 | 0.0771 | 1.0373 |
| fold_3 | 0.0994 | 0.1455 | 0.0781 | 0.9601 |
| fold_4 | 0.1005 | 0.1478 | 0.0789 | 0.8616 |

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
|------|------|------|------|------|
| fold_0 | 0.0774 | 0.1382 | 0.0523 | 1.6712 |
| fold_1 | 0.0746 | 0.1339 | 0.0490 | 1.7205 |
| fold_2 | 0.0744 | 0.1282 | 0.0495 | 1.0924 |
| fold_3 | 0.0779 | 0.1344 | 0.0566 | 1.0314 |
| fold_4 | 0.0783 | 0.1389 | 0.0527 | 1.4256 |

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
|------|------|------|------|------|
| fold_0 | 0.0647 | 0.1433 | 0.5567 | 4.1352 |
| fold_1 | 0.0625 | 0.1405 | 0.3425 | 3.2707 |
| fold_2 | 0.0603 | 0.1307 | 0.3087 | 2.4875 |
| fold_3 | 0.0617 | 0.1361 | 0.3705 | 2.5468 |
| fold_4 | 0.0604 | 0.1302 | 0.4014 | 2.7424 |

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
|------|------|------|------|------|
| fold_0 | 0.2520 | 0.5737 | 2.6522 | 6.7687 |
| fold_1 | 0.2559 | 0.5855 | 3.1212 | 7.8313 |
| fold_2 | 0.2538 | 0.5745 | 4.9337 | 7.5589 |
| fold_3 | 0.2516 | 0.5724 | 7.7941 | 7.3476 |
| fold_4 | 0.2584 | 0.5863 | 7.7557 | 6.6484 |

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

Probability ROC AUC below is recomputed from the original saved positive-class probabilities. Balanced accuracy, accuracy and F1 use p > 0.5. Predictions are aligned by sample ID with official test labels. Statistics are unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 | probability ROC AUC |
|------|------|------|------|------|
| fold_0 | 0.8954 | 0.8907 | 0.8767 | 0.9575 |
| fold_1 | 0.8925 | 0.8843 | 0.8692 | 0.9530 |
| fold_2 | 0.8892 | 0.8809 | 0.8651 | 0.9544 |
| fold_3 | 0.8902 | 0.8836 | 0.8684 | 0.9542 |
| fold_4 | 0.8944 | 0.8906 | 0.8765 | 0.9566 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8924 | 0.8954 | 0.8892 | 0.0024 |
| balanced accuracy | 0.8860 | 0.8907 | 0.8809 | 0.0040 |
| f1 | 0.8712 | 0.8767 | 0.8651 | 0.0046 |
| probability ROC AUC | 0.9551 | 0.9575 | 0.9530 | 0.0017 |

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
      <td>0.8954</td>
      <td>0.8907</td>
      <td>0.8767</td>
      <td>0.9575</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.8925</td>
      <td>0.8843</td>
      <td>0.8692</td>
      <td>0.9530</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.8892</td>
      <td>0.8809</td>
      <td>0.8651</td>
      <td>0.9544</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.8902</td>
      <td>0.8836</td>
      <td>0.8684</td>
      <td>0.9542</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.8944</td>
      <td>0.8906</td>
      <td>0.8765</td>
      <td>0.9566</td>
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
      <td>0.8924</td>
      <td>0.8954</td>
      <td>0.8892</td>
      <td>0.0024</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.8860</td>
      <td>0.8907</td>
      <td>0.8809</td>
      <td>0.0040</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.8712</td>
      <td>0.8767</td>
      <td>0.8651</td>
      <td>0.0046</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.9551</td>
      <td>0.9575</td>
      <td>0.9530</td>
      <td>0.0017</td>
    </tr>
  </tbody>
</table>
</details>

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
|------|------|------|------|------|
| fold_0 | 0.0450 | 0.0760 | 0.0481 | 0.8169 |
| fold_1 | 0.0460 | 0.0784 | 0.0489 | 0.9537 |
| fold_2 | 0.0415 | 0.0647 | 0.0426 | 0.7388 |
| fold_3 | 0.0414 | 0.0674 | 0.0423 | 0.7883 |
| fold_4 | 0.0421 | 0.0716 | 0.0390 | 0.8760 |

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
|------|------|------|------|------|
| fold_0 | 44.6338 | 177.5047 | 0.0630 | 2487.0572 |
| fold_1 | 44.4785 | 94.2692 | 0.0788 | 771.3554 |
| fold_2 | 41.8527 | 96.5528 | 0.0677 | 961.9971 |
| fold_3 | 49.5034 | 182.7023 | 0.0764 | 2486.9284 |
| fold_4 | 44.4688 | 100.3057 | 0.0748 | 932.5536 |

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
|------|------|------|------|------|
| fold_0 | 132.6632 | 191.6179 | 0.0879 | 674.3864 |
| fold_1 | 131.0388 | 180.4992 | 0.0938 | 692.3776 |
| fold_2 | 146.1359 | 200.8733 | 0.1032 | 574.9950 |
| fold_3 | 130.3132 | 165.3391 | 0.0896 | 405.5804 |
| fold_4 | 124.4095 | 159.1095 | 0.0905 | 387.1785 |

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




