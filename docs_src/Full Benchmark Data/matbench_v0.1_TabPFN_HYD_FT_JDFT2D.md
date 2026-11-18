# matbench_v0.1: TabPFN-HYD-FT

### Algorithm description: 

TabPFN-HYD-FT genuinely trains per official outer TRAIN. Composition-group nested three-fold CV independently selects analytic composition260/all370 features, historical TabPFN fine-tuning/MLP schedules and inference alternatives, including positive mixtures of a trained model with the original frozen prior. The selected learned model is refit on the entire own TRAIN; the selected mixture, if any, is fixed by inner CV. All five final fits freeze before fresh-process replay and all 636 held-out predictions. Built with PriorLabs-TabPFN. Retrospective design exposure and the initial official result are disclosed; this rerun does not erase that history.

#### Notes:
Only matbench_jdft2d, not all 13 tasks. Corrected native five-fold MAE=29.99157698166102 meV/atom, fold std=9.44184383860221. The original result28.733328891325755 is retained in history and immutable initial commit340be4d210abbc5367fea241f7e496818f98c171; it had fold0 design-selection bias risk. Corrected protocol SHAfe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b was frozen before corrected GPU execution; all 1206 historical candidate alternatives are selected only inside each own outer TRAIN. This corrected result replaces the original regardless of score direction. Previously observed labels/results are not claimed unseen. The original public master was interrupted by course catalog resolution before a later GPU submission, then host-resumed with identical frozen public scientific functions; all 21 stages, outputSHA, genuine updates, final model freezes, fresh-process replays and 636 predictions completed. The default single-machine CUDA backend was not separately fully rerun. See REVIEW_RESPONSE.md and validation.json for limits and actual compute. No own outerTEST feedback enters fitting or configuration selection in this corrected run.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_TabPFN_HYD_FT_JDFT2D).

### References (in bibtex format): 

```
('@article{dunn2020matbench,\n'
 ' title={Benchmarking materials property prediction methods: the Matbench '
 'test set and Automatminer reference algorithm},\n'
 ' author={Dunn, Alexander and Wang, Qi and Ganose, Alex and Dopp, Daniel and '
 'Jain, Anubhav},\n'
 ' journal={npj Computational Materials}, volume={6}, pages={138}, '
 'year={2020}, doi={10.1038/s41524-020-00406-3}}\n'
 '@article{hollmann2025tabpfn,\n'
 ' title={Accurate predictions on small data with a tabular foundation '
 'model},\n'
 ' author={Hollmann, Noah and Muller, Samuel and Purucker, Lennart and '
 'Krishnakumar, Arjun and Korfer, Max and Hoo, Shi Bin and Schirrmeister, '
 'Robin Tibor and Hutter, Frank},\n'
 ' journal={Nature}, volume={637}, pages={319--326}, year={2025}, '
 'doi={10.1038/s41586-024-08328-6}}\n'
 '@article{ward2018matminer,\n'
 ' title={Matminer: An open source toolkit for materials data mining}, '
 'author={Ward, Logan and others},\n'
 ' journal={Computational Materials Science}, volume={152}, pages={60--69}, '
 'year={2018}, doi={10.1016/j.commatsci.2018.05.018}}\n'
 '@article{pedregosa2011scikit,\n'
 ' title={Scikit-learn: Machine Learning in Python}, author={Pedregosa, Fabian '
 'and others},\n'
 ' journal={Journal of Machine Learning Research}, volume={12}, '
 'pages={2825--2830}, year={2011}}\n')
```

### User metadata:

```
{'algorithm': 'TabPFN-HYD-FT',
 'attribution': 'Built with PriorLabs-TabPFN',
 'design_exposure_disclosed': True,
 'limitation': 'Retrospective correction after an original official result and '
               'fold0 development; the historical exposure is not erased. All '
               'historical choices are reselected inside each outer TRAIN; '
               'this corrected run replaces the original regardless of score.',
 'model_freeze_sha256': '2d2a44e6422f01ab131bb17e25c3bab4566bb3a8aee139903b1fef3d7bc553e3',
 'protocol_sha256': 'fe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b',
 'review_correction': True}
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
{'python': ['numpy==2.5.4',
            'scikit-learn==1.9.1',
            'scipy==1.18.1',
            'joblib==1.6.0',
            'threadpoolctl==3.7.0',
            'tabpfn==9.2.0',
            'torch==2.13.0',
            'matminer==0.10.1',
            'pymatgen==2026.10.24',
            'matbench @ '
            'git+https://github.com/materialsproject/matbench.git@35f88110a51adf817a53515780d99fc34f614bf8']}
```

### Task data:

#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 19.0785 | 44.4494 | 20.6294 | 367.0488 |
| fold_1 | 27.7549 | 67.0729 | 0.2010 | 341.8347 |
| fold_2 | 46.0301 | 145.6959 | 0.5254 | 888.5764 |
| fold_3 | 23.0592 | 50.8069 | 0.1572 | 319.5028 |
| fold_4 | 34.0352 | 154.5484 | 0.4083 | 1568.0982 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 29.9916 | 46.0301 | 19.0785 | 9.4418 |
| rmse | 92.5147 | 154.5484 | 44.4494 | 47.6938 |
| mape* | 4.3843 | 20.6294 | 0.1572 | 8.1237 |
| max_error | 697.0122 | 1568.0982 | 319.5028 | 484.3610 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'protocol_sha256': 'fe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b', 'real_gradient_training': True, 'selected': {'baseline_key': None, 'key': 'ft-all370-s4-e4-median', 'kind': 'fin...` |
| fold_1 | `{'protocol_sha256': 'fe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b', 'real_gradient_training': True, 'selected': {'baseline_key': 'frozen-all370-e4-median', 'key': 'ft-all370-s0-e4-...` |
| fold_2 | `{'protocol_sha256': 'fe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b', 'real_gradient_training': True, 'selected': {'baseline_key': 'frozen-all370-e8-median', 'key': 'ft-all370-s1-e4-...` |
| fold_3 | `{'protocol_sha256': 'fe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b', 'real_gradient_training': True, 'selected': {'baseline_key': 'frozen-all370-e4-median', 'key': 'ft-all370-s4-e4-...` |
| fold_4 | `{'protocol_sha256': 'fe0c7bb3246799c7704ba439a3da77a55d8075e1179aaeccb452f17a706bb63b', 'real_gradient_training': True, 'selected': {'baseline_key': 'frozen-all370-e8-median', 'key': 'ft-all370-s1-e4-...` |




