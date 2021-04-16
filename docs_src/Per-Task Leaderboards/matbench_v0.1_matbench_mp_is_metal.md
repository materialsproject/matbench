# matbench_v0.1 matbench_mp_is_metal

## Individual Task Leaderboard for `matbench_mp_is_metal`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean rocauc | std rocauc | mean f1 | mean balanced_accuracy |
|------|------|------|------|------|
| [AMMExpress v2020](/Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020) | **0.9093** | 0.0008 | 0.8981 | 0.9093 | 
| [Dummy](/Full%20Benchmark%20Data/matbench_v0.1_dummy) | **0.5012** | 0.0043 | 0.4353 | 0.5012 | 


<iframe src="/static/task_matbench_v0.1_matbench_mp_is_metal.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting DFT metallicity from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 106113

Task type: classification

Input type: structure

##### Dataset columns

- is_metal: Target variable. 1 if the compound is a metal, 0 if the compound is not a metal. Metallicity determined with pymatgen
- structure: Pymatgen Structure of the material.


##### Dataset reference

 `A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323`

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
                 '@article{Jain2013,\n'
                 'author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, '
                 'Geoffroy and Chen, Wei and Richards, William Davidson and '
                 'Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and '
                 'Skinner, David and Ceder, Gerbrand and Persson, Kristin '
                 'a.},\n'
                 'doi = {10.1063/1.4812323},\n'
                 'issn = {2166532X},\n'
                 'journal = {APL Materials},\n'
                 'number = {1},\n'
                 'pages = {011002},\n'
                 'title = {{The Materials Project: A materials genome approach '
                 'to accelerating materials innovation}},\n'
                 'url = '
                 '{http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\\&Agg=doi},\n'
                 'volume = {1},\n'
                 'year = {2013}\n'
                 '}'],
 'columns': {'is_metal': 'Target variable. 1 if the compound is a metal, 0 if '
                         'the compound is not a metal. Metallicity determined '
                         'with pymatgen',
             'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting DFT metallicity '
                'from structure. Adapted from Materials Project database. '
                'Removed entries having a formation energy (or energy above '
                'the convex hull) more than 150meV and those containing noble '
                'gases. Retrieved April 2, 2019. For benchmarking w/ nested '
                'cross validation, the order of the dataset must be identical '
                'to the retrieved data; refer to the Automatminer/Matbench '
                'publication for more details.',
 'file_type': 'json.gz',
 'frac_true': 0.43492314796490533,
 'hash': '9a028ed5750a4c76ca36e9f3c8d48fe0bf3fb21b76ec2289e58ae7048d527919',
 'input_type': 'structure',
 'n_samples': 106113,
 'num_entries': 106113,
 'reference': 'A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. '
              'Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson '
              '(*=equal contributions)\n'
              'The Materials Project: A materials genome approach to '
              'accelerating materials innovation\n'
              'APL Materials, 2013, 1(1), 011002.\n'
              'doi:10.1063/1.4812323',
 'target': 'is_metal',
 'task_type': 'classification',
 'unit': None,
 'url': 'https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz'}
```

