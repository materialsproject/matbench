# matbench_v0.1 matbench_steels

## Individual Task Leaderboard for `matbench_steels`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [AMMExpress v2020](/Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020) | **97.4929** | 13.7919 | 154.0161 | 1142.9223 | 
| [Dummy](/Full%20Benchmark%20Data/matbench_v0.1_dummy) | **229.7445** | 9.6958 | 301.2211 | 1088.0568 | 


<iframe src="static/task_matbench_v0.1_matbench_steels.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting steel yield strengths from chemical composition alone. Retrieved from Citrine informatics. Deduplicated. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 312

Task type: regression

Input type: composition

##### Dataset columns

- composition: Chemical formula.
- yield strength: Target variable. Experimentally measured steel yield strengths, in MPa.


##### Dataset reference

 `https://citrination.com/datasets/153092/`

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
                 '@misc{Citrine Informatics,\n'
                 'title = {Mechanical properties of some steels},\n'
                 'howpublished = '
                 '{\\url{https://citrination.com/datasets/153092/},\n'
                 '}'],
 'columns': {'composition': 'Chemical formula.',
             'yield strength': 'Target variable. Experimentally measured steel '
                               'yield strengths, in MPa.'},
 'description': 'Matbench v0.1 test dataset for predicting steel yield '
                'strengths from chemical composition alone. Retrieved from '
                'Citrine informatics. Deduplicated. For benchmarking w/ nested '
                'cross validation, the order of the dataset must be identical '
                'to the retrieved data; refer to the Automatminer/Matbench '
                'publication for more details.',
 'file_type': 'json.gz',
 'hash': '473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252',
 'input_type': 'composition',
 'mad': 229.37426857330706,
 'n_samples': 312,
 'num_entries': 312,
 'reference': 'https://citrination.com/datasets/153092/',
 'target': 'yield strength',
 'task_type': 'regression',
 'unit': 'MPa',
 'url': 'https://ml.materialsproject.org/projects/matbench_steels.json.gz'}
```

