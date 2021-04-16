# matbench_v0.1 matbench_expt_gap

## Individual Task Leaderboard for `matbench_expt_gap`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [AMMExpress v2020](/Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2019) | **0.4161** | 0.0194 | 0.9918 | 12.7533 | 
| [Dummy](/Full%20Benchmark%20Data/matbench_v0.1_dummy) | **1.1435** | 0.0310 | 1.4438 | 10.7354 | 
| [Random](/Full%20Benchmark%20Data/matbench_v0.1_random) | **5.0980** | 0.2398 | 6.0066 | 11.6885 | 


<iframe src="/static/task_matbench_v0.1_matbench_expt_gap.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting experimental band gap from composition alone. Retrieved from Zhuo et al. supplementary information. Deduplicated according to composition, removing compositions with reported band gaps spanning more than a 0.1eV range; remaining compositions were assigned values based on the closest experimental value to the mean experimental value for that composition among all reports. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 4604

Task type: regression

Input type: composition

##### Dataset columns

- composition: Chemical formula.
- gap expt: Target variable. Experimentally measured gap, in eV.


##### Dataset reference

 `Y. Zhuo, A. Masouri Tehrani, J. Brgoch (2018) Predicting the Band Gaps of Inorganic Solids by Machine Learning J. Phys. Chem. Lett. 2018, 9, 7, 1668-1673 https:doi.org/10.1021/acs.jpclett.8b00124.`

### Metadata

```
{'bibtex_refs': ['@Article{Dunn2020,\n'
                 'author={Dunn, Alexander\n'
                 'and Wang, Qi\n'
                 'and Ganose, Alex\n'
                 'and Dopp, Daniel\n'
                 'and Jain, Anubhav},\n'
                 'title={Benchmarking materials property prediction methods: '
                 'the Matbench test set and Automatminer reference '
                 'algorithm},\n'
                 'journal={npj Computational Materials},\n'
                 'year={2020},\n'
                 'month={Sep},\n'
                 'day={15},\n'
                 'volume={6},\n'
                 'number={1},\n'
                 'pages={138},\n'
                 'abstract={We present a benchmark test suite and an automated '
                 'machine learning procedure for evaluating supervised machine '
                 'learning (ML) models for predicting properties of inorganic '
                 'bulk materials. The test suite, Matbench, is a set of '
                 '13{\\thinspace}ML tasks that range in size from 312 to 132k '
                 'samples and contain data from 10 density functional '
                 'theory-derived and experimental sources. Tasks include '
                 'predicting optical, thermal, electronic, thermodynamic, '
                 "tensile, and elastic properties given a material's "
                 'composition and/or crystal structure. The reference '
                 'algorithm, Automatminer, is a highly-extensible, fully '
                 'automated ML pipeline for predicting materials properties '
                 'from materials primitives (such as composition and crystal '
                 'structure) without user intervention or hyperparameter '
                 'tuning. We test Automatminer on the Matbench test suite and '
                 'compare its predictive power with state-of-the-art crystal '
                 'graph neural networks and a traditional descriptor-based '
                 'Random Forest model. We find Automatminer achieves the best '
                 'performance on 8 of 13 tasks in the benchmark. We also show '
                 'our test suite is capable of exposing predictive advantages '
                 'of each algorithm---namely, that crystal graph methods '
                 'appear to outperform traditional machine learning methods '
                 'given {\\textasciitilde}104 or greater data points. We '
                 'encourage evaluating materials ML algorithms on the Matbench '
                 'benchmark and comparing them against the latest version of '
                 'Automatminer.},\n'
                 'issn={2057-3960},\n'
                 'doi={10.1038/s41524-020-00406-3},\n'
                 'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                 '}\n',
                 '@article{doi:10.1021/acs.jpclett.8b00124,\n'
                 'author = {Zhuo, Ya and Mansouri Tehrani, Aria and Brgoch, '
                 'Jakoah},\n'
                 'title = {Predicting the Band Gaps of Inorganic Solids by '
                 'Machine Learning},\n'
                 'journal = {The Journal of Physical Chemistry Letters},\n'
                 'volume = {9},\n'
                 'number = {7},\n'
                 'pages = {1668-1673},\n'
                 'year = {2018},\n'
                 'doi = {10.1021/acs.jpclett.8b00124},\n'
                 'note ={PMID: 29532658},\n'
                 'eprint = {\n'
                 'https://doi.org/10.1021/acs.jpclett.8b00124\n'
                 '\n'
                 '}}'],
 'columns': {'composition': 'Chemical formula.',
             'gap expt': 'Target variable. Experimentally measured gap, in '
                         'eV.'},
 'description': 'Matbench v0.1 test dataset for predicting experimental band '
                'gap from composition alone. Retrieved from Zhuo et al. '
                'supplementary information. Deduplicated according to '
                'composition, removing compositions with reported band gaps '
                'spanning more than a 0.1eV range; remaining compositions were '
                'assigned values based on the closest experimental value to '
                'the mean experimental value for that composition among all '
                'reports. For benchmarking w/ nested cross validation, the '
                'order of the dataset must be identical to the retrieved data; '
                'refer to the Automatminer/Matbench publication for more '
                'details.',
 'file_type': 'json.gz',
 'hash': '783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2',
 'input_type': 'composition',
 'mad': 1.1432002429044061,
 'n_samples': 4604,
 'num_entries': 4604,
 'reference': 'Y. Zhuo, A. Masouri Tehrani, J. Brgoch (2018) Predicting the '
              'Band Gaps of Inorganic Solids by Machine Learning J. Phys. '
              'Chem. Lett. 2018, 9, 7, 1668-1673 '
              'https:doi.org/10.1021/acs.jpclett.8b00124.',
 'target': 'gap expt',
 'task_type': 'regression',
 'unit': 'eV',
 'url': 'https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz'}
```

