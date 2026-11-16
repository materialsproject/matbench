# matbench_v0.1: Atomic-AffQRF

### Algorithm description: 

Atomic-Affine-Bagged-QRF: Bayesian-bootstrap nonnegative affine MAE combination of 32 ExtraTrees conditional mean/median predictors, with coefficient sum at most two and free intercept. Five training-only inner folds; fixed 16 configurations, 600 inner trees and 900 final trees; reference seed 417.

#### Notes:
Composition-only. Reference seed 417 reported; seed repeats 0-6 give 75.77 +/- 0.33 MPa mean fold MAE (all in all_seed_scores.csv). The official folds were reused during method development, so selection across runs may be optimistic; each run fits only on outer-training labels. Exact rebuild requires CPython 3.13.16 and requirements.txt; reproduce_entry.py and the notebook rebuild all forests from raw inputs and check the fold dictionaries against results.json.gz exactly.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_Atomic_Affine_QRF).

### References (in bibtex format): 

```
['@article{meinshausen2006quantile, author={Nicolai Meinshausen}, '
 'title={Quantile Regression Forests}, journal={Journal of Machine Learning '
 'Research}, volume={7}, pages={983--999}, year={2006}, '
 'url={https://jmlr.org/papers/v7/meinshausen06a.html}}',
 '@article{geurts2006extremely, author={Pierre Geurts and Damien Ernst and '
 'Louis Wehenkel}, title={Extremely randomized trees}, journal={Machine '
 'Learning}, volume={63}, pages={3--42}, year={2006}}',
 '@article{rubin1981bayesian, author={Donald B. Rubin}, title={The Bayesian '
 'Bootstrap}, journal={The Annals of Statistics}, volume={9}, '
 'pages={130--134}, year={1981}, doi={10.1214/aos/1176345338}}',
 '@article{dunn2020matbench, author={Alexander Dunn and Qi Wang and Alex '
 'Ganose and Daniel Dopp and Anubhav Jain}, title={Benchmarking materials '
 'property prediction methods: the Matbench test set and Automatminer '
 'reference algorithm}, journal={npj Computational Materials}, volume={6}, '
 'year={2020}, doi={10.1038/s41524-020-00406-3}}']
```

### User metadata:

```
{'algorithm': 'Atomic-Affine-Bagged-QRF',
 'application_scope': 'Retrospective composition benchmark; no prospective or '
                      'physical qualification claim.',
 'protocol': {'ablation_rule': 'bayesian_shift enforces coefficient sum=1 but '
                               'retains unrestricted intercept; isolates '
                               'intercept from scale calibration. Original '
                               'Bayesian-simplex control kept for every '
                               'matched seed.',
              'ablations': ['bayesian_shift'],
              'bootstrap_draws': 128,
              'bootstrap_seed': 19072026,
              'centering': 'Subtract outer-training median response from '
                           'targets and every OOF predictor; scale by 100 MPa '
                           'for LP conditioning. Undo centering exactly at '
                           'prediction.',
              'claim_scope': 'Numerical exact-official-fold comparison; not '
                             'prospective independence or accepted leaderboard '
                             'record. No physical qualification, no realized '
                             'saving.',
              'coefficient_sum_bounds': [0.0, 2.0],
              'frozen_utc': '2026-10-05T01:23:44.230185+00:00',
              'name': 'Atomic-Affine-Bagged-QRF',
              'primary': 'bayesian_affine',
              'primary_rule': 'For every Bayesian-bootstrap Exp(1) normalized '
                              'draw, fit nonnegative coefficients with sum <= '
                              '2 and an unrestricted intercept to minimize '
                              'weighted OOF absolute loss. Average all 128 '
                              'coefficient vectors and intercepts. Same 32 '
                              'cached mean/median training-only predictors as '
                              'the existing method. No additional '
                              'hyperparameter selection.',
              'reference_seed': 417,
              'refits': 'Reuse previously fully reproduced training-only OOFs '
                        'and final components, train any newly active missing '
                        'configuration at unchanged 900-tree settings.',
              'repeats': [0, 1, 2, 3, 4],
              'scope': 'Public retrospective benchmark development after all '
                       'bagged-simplex outcomes known. Affine/shift outcomes '
                       'not computed or viewed at freeze. All declared seeds '
                       'and policies will be reported, with seed417 reference '
                       'retained regardless of repeats or ablation.',
              'verified': False},
 'verified': False}
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
{'python': ['asttokens==3.0.2',
            'attrs==26.1.0',
            'bibtexparser==1.4.4',
            'certifi==2026.10.23',
            'charset-normalizer==3.5.2',
            'cloudpickle==3.1.2',
            'comm==0.2.3',
            'contourpy==1.4.0',
            'cycler==0.12.1',
            'debugpy==1.8.22',
            'dnspython==2.9.2',
            'executing==2.2.1',
            'fastjsonschema==2.22.2',
            'fonttools==4.66.1',
            'idna==3.20',
            'ipykernel==7.4.0',
            'ipython==9.17.1',
            'ipython-pygments-lexers==1.1.1',
            'jedi==0.20.0',
            'joblib==1.6.0',
            'jsonschema==4.26.0',
            'jsonschema-specifications==2025.9.1',
            'jupyter-client==8.10.0',
            'jupyter-core==5.9.1',
            'kiwisolver==1.5.1',
            'lxml==6.1.3',
            'matbench @ '
            'https://github.com/materialsproject/matbench/archive/936176db18ca4cd7b38cbd957c017a5bac770c6b.zip',
            'matminer==0.10.1',
            'matplotlib==3.11.2',
            'matplotlib-inline==0.2.2',
            'monty==2026.7.16',
            'mpmath==1.3.0',
            'narwhals==2.26.0',
            'nbclient==0.11.0',
            'nbformat==5.11.1',
            'nest-asyncio2==1.7.3',
            'networkx==3.7',
            'numpy==2.5.3',
            'orjson==3.12.0',
            'packaging==26.3',
            'palettable==3.3.3',
            'pandas==2.3.3',
            'parso==0.8.7',
            'pexpect==4.9.0',
            'pillow==13.0.0',
            'platformdirs==4.12.3',
            'plotly==7.1.0',
            'prompt-toolkit==3.0.53',
            'psutil==7.2.2',
            'ptyprocess==0.7.0',
            'pure-eval==0.2.4',
            'pygments==2.21.0',
            'pymatgen==2026.9.24',
            'pymatgen-core==2026.10.2',
            'pymongo==4.18.3',
            'pyparsing==3.3.3',
            'python-dateutil==2.9.0.post0',
            'pytz==2026.5',
            'pyzmq==27.2.0',
            'referencing==0.37.0',
            'requests==2.34.2',
            'rpds-py==2026.9.1',
            'ruamel-yaml==0.19.1',
            'scikit-learn==1.9.1',
            'scipy==1.18.1',
            'six==1.17.0',
            'spglib==2.7.0',
            'stack-data==0.6.3',
            'sympy==1.14.0',
            'tabulate==0.10.0',
            'threadpoolctl==3.7.0',
            'tornado==6.6.1',
            'tqdm==4.70.1',
            'traitlets==5.16.1',
            'typing-extensions==4.16.0',
            'tzdata==2026.5',
            'uncertainties==3.2.3',
            'urllib3==2.8.0',
            'wcwidth==0.9.2']}
```

### Task data:

#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 90.3882 | 132.5024 | 0.0603 | 423.5474 |
| fold_1 | 67.5943 | 91.3604 | 0.0487 | 361.8044 |
| fold_2 | 73.3462 | 100.8045 | 0.0518 | 392.2737 |
| fold_3 | 77.1833 | 102.3620 | 0.0568 | 325.8486 |
| fold_4 | 69.2008 | 110.3919 | 0.0532 | 413.9772 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 75.5426 | 90.3882 | 67.5943 | 8.1400 |
| rmse | 107.4842 | 132.5024 | 91.3604 | 13.8939 |
| mape* | 0.0541 | 0.0603 | 0.0487 | 0.0040 |
| max_error | 383.4902 | 423.5474 | 325.8486 | 35.7883 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'bootstrap_draws': 128, 'bootstrap_seed': 19072026, 'coefficient_sum_bounds': [0.0, 2.0], 'coefficients': [0.0, 0.02243291050051692, 0.002309874498129588, 0.022944536797549483, 0.002291690942512453, ...` |
| fold_1 | `{'bootstrap_draws': 128, 'bootstrap_seed': 19072026, 'coefficient_sum_bounds': [0.0, 2.0], 'coefficients': [0.0, 0.0012087244624669595, 0.0011517542903182391, 0.04333101451967719, 0.0, 0.0, 0.00116672...` |
| fold_2 | `{'bootstrap_draws': 128, 'bootstrap_seed': 19072026, 'coefficient_sum_bounds': [0.0, 2.0], 'coefficients': [0.0023182580015400378, 0.0041676106336732905, 0.0, 0.029336548168602188, 0.0, 0.013716251354...` |
| fold_3 | `{'bootstrap_draws': 128, 'bootstrap_seed': 19072026, 'coefficient_sum_bounds': [0.0, 2.0], 'coefficients': [0.0, 0.0013646311274760992, 0.0, 0.00010968371507522326, 0.0, 0.07931648130273139, 0.0005649...` |
| fold_4 | `{'bootstrap_draws': 128, 'bootstrap_seed': 19072026, 'coefficient_sum_bounds': [0.0, 2.0], 'coefficients': [0.0, 0.007878705703486663, 0.035506255599324875, 0.04152400306336014, 0.00322223502339584, 0...` |




