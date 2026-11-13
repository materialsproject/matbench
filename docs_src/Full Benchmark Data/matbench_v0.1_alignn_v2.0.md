# matbench_v0.1: ALIGNN 2.0

### Algorithm description: 

ALIGNN 2.0 is a dependency-free, pure-PyTorch reimplementation of the Atomistic Line Graph Neural Network (ALIGNN). Each crystal is converted into an atom graph and its line graph (bond-angle graph); four edge-gated ALIGNN layers alternate between the two graphs and are followed by four graph convolutions on the atom graph, and the mean-pooled representation feeds a single linear output head. The k-nearest-neighbor graph (12 neighbors within 8 A, full line graph) is used for all tasks, as recommended for property prediction in the ALIGNN 2.0 paper. All nine structure-based Matbench v0.1 tasks (jdft2d, phonons, dielectric, log_gvrh, log_kvrh, perovskites, mp_gap, mp_is_metal, mp_e_form) were run with ONE fixed configuration, set on 2026-09-02 before any Matbench fold was trained: the published ALIGNN training settings (4 ALIGNN + 4 GCN layers, batch size 64, AdamW with weight decay 1e-5, one-cycle learning-rate schedule with peak 1e-3) with the hidden dimension raised from 256 to 768, L1 loss, no exponential moving average, random seed 123. The width of 768 comes from earlier JARVIS-Leaderboard work, where it was chosen on validation data of JARVIS-DFT bulk (dft_3d) band-gap and energy tasks; no Matbench data or splits and no exfoliation-energy data entered that choice. The only per-task setting is the number of epochs, fixed a priori as a compute budget by dataset size (300 epochs for tasks under 5k samples, 200 for 5k-20k, 75 for over 100k) and identical for all folds of a task; it was never tuned on performance. For each of the five official folds, 5% of the training fold (a seeded random slice) is held out as a validation set. Training uses only the training fold, the evaluated checkpoint is the one with the lowest validation loss, and the test rows are only passed through the model after training, with those weights. mp_is_metal is trained as a two-class classifier: setting classification_threshold makes the trainer use a 2-logit head with log-softmax output and negative log-likelihood loss (the config's criterion field does not apply), with energy_mult_natoms=false. Its predictions are hard labels, the argmax class (equivalent to a 0.5 threshold on the metal probability); no threshold tuning was applied. The results.json.gz was produced with matbench's own MatbenchBenchmark API (task.record on each fold).

#### Notes:
Structure-based tasks only (9 of 13); the four composition-only tasks are not applicable, since ALIGNN requires a crystal structure as input. Code: ALIGNN (https://github.com/atomgptlab/alignn), pure-torch model alignn_atomwise_pure with neighbor_strategy pure_torch. The reported runs used a copy of commit ce5334303e30043ddd3e7ccd702b467bfc0d1d7b of https://github.com/jae-hlee/alignn (version string 2026.5.20); its config.py, data.py, models/alignn_atomwise_pure.py, models/utils.py, utils.py, graphs.py and torch_graph_builder.py are byte-identical to that commit. Three files in the copy that ran were older versions, none of which changes single-GPU training of these models: train.py without a 3-line change to the writer for multi-output prediction files (used only by models with more than one output), train_alignn.py without the multi-node launch code (every fold ran on one GPU), and pure_lmdb_dataset.py without two later fixes (opening the LMDB lazily inside each data-loader worker, which the runs' two-worker loaders did without, and how per-structure stress or extra features are copied to atoms, which does not apply to these datasets). Data pipeline: make_matbench_dataset.py pulls each fold through the matbench API (task.get_train_and_val_data / task.get_test_data with include_target=False), converts pymatgen structures to POSCAR files, and writes a folder-mode dataset with a fixed train/val/test order recorded in fold_manifest.json; test rows carry placeholder targets, and the per-fold config locks n_train/n_val/n_test to that order (keep_data_order=true). The reported results were produced with an earlier version of this script that wrote the real test targets into the test rows. The trainer used them only for diagnostic output (for regression, the target range, MAD and a mean-predictor baseline before training; the test loss or ROC-AUC after training; and the target columns of the test result files) and never for training, checkpoint selection or the predictions. Training: alignn.train_alignn on one NVIDIA GB10 GPU per fold; measured wall time per fold: 0.4 h for jdft2d, 0.7 h for phonons, 3.6-4.4 h for dielectric, log_gvrh, log_kvrh and perovskites, 42 h for mp_gap and mp_is_metal, and 50-51 h for mp_e_form. All reported folds were trained on the same cluster (JHU atomgptlab, GB10 nodes). As a reproducibility check the five mp_gap folds and the five mp_is_metal folds were additionally retrained on a second cluster (JHU Skipjack, NVIDIA A100/H100) with identical data and configuration, using a later copy of the same code (it has no git metadata; the line positions of its loss and classification code match commit 71fb200 of the same repository, 2026-08-17); for mp_gap the two runs of a fold differed by up to 1.5% in test MAE (+1.4, +1.5, +0.2, +0.0, -0.4%, second cluster minus first), with no consistent direction, i.e. run-to-run variation between the two setups comparable to the fold-to-fold spread. For mp_is_metal the second cluster gave a mean ROCAUC of 0.9032 against the reported 0.9042, differing fold by fold by -0.2 to +0.4%, again with no consistent direction; ALIGNN 2.0 scores below the original ALIGNN (0.9128) on this task on both clusters. Disclosure: per-fold test errors were visible in the training logs during the campaign. The configuration and the epoch budgets were fixed before the first fold was trained; the only later change was turning off energy_mult_natoms for mp_is_metal after its first attempt crashed. Three decisions were made while test errors were visible: (1) after phonons fold 0 (test MAE 41.6 against validation 17.1), width-256 fallback configurations were prepared, to be chosen on validation, but they were never run; (2) the mp_is_metal test score, below the original ALIGNN, prompted a check of other decision thresholds on the validation predictions of one fold (balanced accuracy 0.9056 at 0.5, 0.9063 at the best threshold, 0.54), which was rejected, so all predictions use 0.5; (3) the rule to report every task from one cluster was adopted after both clusters' mp_gap folds had been scored (atomgptlab 0.1757, Skipjack 0.1766, same rank) and before the reported atomgptlab mp_is_metal folds were run. Collection: collect_matbench.py maps each fold's predictions back to matbench test order via fold_manifest.json and records them with task.record. Classification note: for mp_is_metal the model config sets energy_mult_natoms=false (the per-atom energy scaling does not apply to class logits); everything else is identical to the regression configuration. matbench's classifier metrics were computed from the hard 0/1 predictions.

Compatibility note: this original, unchanged submission predates the updated classification checks. Its mp_is_metal hard-label predictions are displayed as balanced accuracy only; the historical hard-label ROC-AUC equals BA and is not probability ROC-AUC. The authors' source and evaluation disclosures remain documented in their PR reply: https://github.com/materialsproject/matbench/pull/370#issuecomment-6408222455.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_alignn_v2.0).

### References (in bibtex format): 

```
('@article{lee2026alignn2,\n'
 '  title   = {ALIGNN 2.0: A Unified Line-Graph Neural Network Framework for '
 'Materials Screening, Force Fields, Inverse Design, Spectroscopy, and '
 'Microscopy},\n'
 '  author  = {Lee, Jaehyung and Campbell, Charles Rhys and Ajith, Akshaya and '
 'Kalinin, Sergei V. and Wolverton, Christopher and Choudhary, Kamal},\n'
 '  journal = {arXiv preprint arXiv:2609.19487},\n'
 '  year    = {2026},\n'
 '  url     = {https://arxiv.org/abs/2609.19487}\n'
 '}\n'
 '@article{choudhary2021alignn,\n'
 '  title   = {Atomistic Line Graph Neural Network for improved materials '
 'property predictions},\n'
 '  author  = {Choudhary, Kamal and DeCost, Brian},\n'
 '  journal = {npj Computational Materials},\n'
 '  volume  = {7},\n'
 '  number  = {1},\n'
 '  pages   = {185},\n'
 '  year    = {2021},\n'
 '  doi     = {10.1038/s41524-021-00650-1}\n'
 '}\n'
 '@article{choudhary2024leaderboard,\n'
 '  title   = {JARVIS-Leaderboard: a large scale benchmark of materials design '
 'methods},\n'
 '  author  = {Choudhary, Kamal and Wines, Daniel and Li, Kang and Garrity, '
 'Kevin F and others},\n'
 '  journal = {npj Computational Materials},\n'
 '  volume  = {10},\n'
 '  number  = {1},\n'
 '  pages   = {93},\n'
 '  year    = {2024}\n'
 '}')
```

### User metadata:

```
{}
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
{'python': ['alignn==2026.5.20 (source checkout of '
            'https://github.com/jae-hlee/alignn at commit '
            'ce5334303e30043ddd3e7ccd702b467bfc0d1d7b; see notes)',
            'torch==2.13.0+cu130',
            'numpy==2.2.6',
            'pydantic==2.13.4',
            'jarvis-tools==2026.6.12',
            'pymatgen==2025.10.7',
            'matminer==0.9.3',
            'matbench==0.6']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 0.1551 | 0.6326 | 0.0519 | 14.4391 |
| fold_1 | 0.2655 | 1.1144 | 0.0852 | 19.0955 |
| fold_2 | 0.4152 | 3.0683 | 0.0869 | 58.5310 |
| fold_3 | 0.2763 | 2.0934 | 0.0578 | 43.7645 |
| fold_4 | 0.3156 | 1.6208 | 0.0924 | 28.1688 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2856 | 0.4152 | 0.1551 | 0.0839 |
| rmse | 1.7059 | 3.0683 | 0.6326 | 0.8385 |
| mape* | 0.0748 | 0.0924 | 0.0519 | 0.0166 |
| max_error | 32.7998 | 58.5310 | 14.4391 | 16.3015 |

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
| fold_0 | 31.7231 | 59.5838 | 24.1739 | 413.1756 |
| fold_1 | 42.0477 | 109.4779 | 0.2885 | 682.4940 |
| fold_2 | 64.5557 | 190.7991 | 0.6127 | 1065.0940 |
| fold_3 | 25.7457 | 60.5177 | 0.2028 | 484.7882 |
| fold_4 | 49.7634 | 168.5382 | 0.4983 | 1566.0496 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 42.7671 | 64.5557 | 25.7457 | 13.6814 |
| rmse | 117.7833 | 190.7991 | 59.5838 | 54.1163 |
| mape* | 5.1553 | 24.1739 | 0.2028 | 9.5104 |
| max_error | 842.3203 | 1566.0496 | 413.1756 | 426.8374 |

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
| fold_0 | 0.0689 | 0.1086 | 0.0551 | 1.0166 |
| fold_1 | 0.0739 | 0.1201 | 0.0589 | 1.2785 |
| fold_2 | 0.0697 | 0.1119 | 0.0566 | 0.8873 |
| fold_3 | 0.0689 | 0.1090 | 0.0538 | 0.9790 |
| fold_4 | 0.0697 | 0.1112 | 0.0557 | 0.7618 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0702 | 0.0739 | 0.0689 | 0.0019 |
| rmse | 0.1121 | 0.1201 | 0.1086 | 0.0041 |
| mape* | 0.0560 | 0.0589 | 0.0538 | 0.0017 |
| max_error | 0.9846 | 1.2785 | 0.7618 | 0.1712 |

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
| fold_0 | 0.0529 | 0.1064 | 0.0353 | 1.7660 |
| fold_1 | 0.0551 | 0.1127 | 0.0364 | 1.3631 |
| fold_2 | 0.0503 | 0.0962 | 0.0334 | 1.1535 |
| fold_3 | 0.0579 | 0.1144 | 0.0417 | 1.1389 |
| fold_4 | 0.0569 | 0.1101 | 0.0385 | 1.3935 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0546 | 0.0579 | 0.0503 | 0.0027 |
| rmse | 0.1080 | 0.1144 | 0.0962 | 0.0064 |
| mape* | 0.0370 | 0.0417 | 0.0334 | 0.0028 |
| max_error | 1.3630 | 1.7660 | 1.1389 | 0.2269 |

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
| fold_0 | 0.0193 | 0.0547 | 0.1599 | 3.0471 |
| fold_1 | 0.0187 | 0.0564 | 0.1157 | 3.3008 |
| fold_2 | 0.0187 | 0.0456 | 0.1281 | 2.3305 |
| fold_3 | 0.0193 | 0.0486 | 0.1488 | 1.4822 |
| fold_4 | 0.0186 | 0.0443 | 0.2201 | 1.4082 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0189 | 0.0193 | 0.0186 | 0.0003 |
| rmse | 0.0499 | 0.0564 | 0.0443 | 0.0048 |
| mape* | 0.1545 | 0.2201 | 0.1157 | 0.0362 |
| max_error | 2.3138 | 3.3008 | 1.4082 | 0.7777 |

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
| fold_0 | 0.1735 | 0.4600 | 2.4015 | 7.0229 |
| fold_1 | 0.1734 | 0.4570 | 2.7574 | 6.8448 |
| fold_2 | 0.1757 | 0.4665 | 4.7279 | 6.4931 |
| fold_3 | 0.1736 | 0.4672 | 5.8713 | 7.4742 |
| fold_4 | 0.1822 | 0.4871 | 4.5587 | 6.5031 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1757 | 0.1822 | 0.1734 | 0.0034 |
| rmse | 0.4676 | 0.4871 | 0.4570 | 0.0105 |
| mape* | 4.0634 | 5.8713 | 2.4015 | 1.2979 |
| max_error | 6.8676 | 7.4742 | 6.4931 | 0.3649 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




#### `matbench_mp_is_metal`

**Probability ROC AUC: unavailable.** No usable probabilities were saved, so probability ROC AUC cannot be recovered. Balanced accuracy, accuracy and F1 below are recomputed from saved hard-label decisions aligned by sample ID with official test labels. The original decision threshold may be unknown; these historical decisions have not been reevaluated at a common threshold. Statistics use unweighted five-fold means and population standard deviations (ddof=0).

[Classification score audit and file provenance (JSON)](../static/classification_scores.json)

###### Fold scores

| fold | accuracy | balanced accuracy | f1 |
|------|------|------|------|
| fold_0 | 0.9111 | 0.9070 | 0.8955 |
| fold_1 | 0.9045 | 0.9014 | 0.8889 |
| fold_2 | 0.9098 | 0.9072 | 0.8954 |
| fold_3 | 0.9069 | 0.9016 | 0.8895 |
| fold_4 | 0.9056 | 0.9038 | 0.8913 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9076 | 0.9111 | 0.9045 | 0.0025 |
| balanced accuracy | 0.9042 | 0.9072 | 0.9014 | 0.0025 |
| f1 | 0.8921 | 0.8955 | 0.8889 | 0.0028 |

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
      <td>0.9111</td>
      <td>0.9070</td>
      <td>0.8955</td>
      <td>0.9070</td>
    </tr>
    <tr>
      <td>fold_1</td>
      <td>0.9045</td>
      <td>0.9014</td>
      <td>0.8889</td>
      <td>0.9014</td>
    </tr>
    <tr>
      <td>fold_2</td>
      <td>0.9098</td>
      <td>0.9072</td>
      <td>0.8954</td>
      <td>0.9072</td>
    </tr>
    <tr>
      <td>fold_3</td>
      <td>0.9069</td>
      <td>0.9016</td>
      <td>0.8895</td>
      <td>0.9016</td>
    </tr>
    <tr>
      <td>fold_4</td>
      <td>0.9056</td>
      <td>0.9038</td>
      <td>0.8913</td>
      <td>0.9038</td>
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
      <td>0.9076</td>
      <td>0.9111</td>
      <td>0.9045</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <td>balanced_accuracy</td>
      <td>0.9042</td>
      <td>0.9072</td>
      <td>0.9014</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <td>f1</td>
      <td>0.8921</td>
      <td>0.8955</td>
      <td>0.8889</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <td>stored rocauc (historical)</td>
      <td>0.9042</td>
      <td>0.9072</td>
      <td>0.9014</td>
      <td>0.0025</td>
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
| fold_0 | 0.0277 | 0.0559 | 0.0266 | 0.8409 |
| fold_1 | 0.0286 | 0.0615 | 0.0280 | 0.9401 |
| fold_2 | 0.0261 | 0.0479 | 0.0251 | 0.8296 |
| fold_3 | 0.0274 | 0.0505 | 0.0267 | 0.6440 |
| fold_4 | 0.0267 | 0.0527 | 0.0238 | 0.8819 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0273 | 0.0286 | 0.0261 | 0.0008 |
| rmse | 0.0537 | 0.0615 | 0.0479 | 0.0047 |
| mape* | 0.0260 | 0.0280 | 0.0238 | 0.0015 |
| max_error | 0.8273 | 0.9401 | 0.6440 | 0.0995 |

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
| fold_0 | 41.6024 | 132.6636 | 0.0508 | 1182.4331 |
| fold_1 | 26.8450 | 66.1223 | 0.0469 | 591.1618 |
| fold_2 | 36.9527 | 122.0121 | 0.0512 | 1178.1951 |
| fold_3 | 28.6047 | 94.2112 | 0.0478 | 975.6275 |
| fold_4 | 22.4017 | 38.2055 | 0.0432 | 239.4086 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 31.2813 | 41.6024 | 22.4017 | 6.9912 |
| rmse | 90.6429 | 132.6636 | 38.2055 | 34.9669 |
| mape* | 0.0480 | 0.0512 | 0.0432 | 0.0029 |
| max_error | 833.3652 | 1182.4331 | 239.4086 | 366.7265 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




