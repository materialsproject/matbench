# matbench_v0.1 matbench_dielectric

## Individual Task Leaderboard for `matbench_dielectric`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [MODNet (v0.1.12)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.12.md) | **0.2711** | 0.0714 | 1.6832 | 59.1179 | 
| [MODNet (v0.1.10)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.10.md) | **0.2970** | 0.0720 | 1.7185 | 58.9519 | 
| [coGN](../Full%20Benchmark%20Data/matbench_v0.1_coGN.md) | **0.3088** | 0.0859 | 2.0546 | 58.7728 | 
| [AMMExpress v2020](../Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.3150** | 0.0672 | 1.7202 | 59.0112 | 
| [Finder_v1.2 structure-based version](../Full%20Benchmark%20Data/matbench_v0.1_Finder_v1.2_structure.md) | **0.3197** | 0.0717 | 1.7213 | 59.0606 | 
| [Finder_v1.2 composition-only version](../Full%20Benchmark%20Data/matbench_v0.1_Finder_v1.2_composition.md) | **0.3204** | 0.0811 | 1.7189 | 59.0528 | 
| [CrabNet](../Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **0.3234** | 0.0714 | 1.7288 | 59.1583 | 
| [SchNet (kgcnn v2.1.0)](../Full%20Benchmark%20Data/matbench_v0.1_SchNet_kgcnn_v2.1.0.md) | **0.3277** | 0.0829 | 1.8990 | 58.6071 | 
| [MegNet (kgcnn v2.1.0)](../Full%20Benchmark%20Data/matbench_v0.1_MegNet_kgcnn_v2.1.0.md) | **0.3391** | 0.0745 | 1.9871 | 59.3095 | 
| [DimeNet++ (kgcnn v2.1.0)](../Full%20Benchmark%20Data/matbench_v0.1_DimeNetPP_kgcnn_v2.1.0.md) | **0.3400** | 0.0570 | 1.9936 | 58.5416 | 
| [ALIGNN](../Full%20Benchmark%20Data/matbench_v0.1_alignn.md) | **0.3449** | 0.0871 | 1.9651 | 58.7285 | 
| [RF-SCM/Magpie](../Full%20Benchmark%20Data/matbench_v0.1_rf.md) | **0.4196** | 0.0750 | 1.8538 | 59.1201 | 
| [CGCNN v2019](../Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **0.5988** | 0.0833 | 1.8976 | 58.9996 | 
| [Dummy](../Full%20Benchmark%20Data/matbench_v0.1_dummy.md) | **0.8088** | 0.0718 | 1.9728 | 59.6653 | 


<iframe src="../../static/task_matbench_v0.1_matbench_dielectric.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting refractive index from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having refractive indices less than 1 and those containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 4764

Task type: regression

Input type: structure

##### Dataset columns

- n: Target variable. Refractive index (unitless).
- structure: Pymatgen Structure of the material.


##### Dataset reference

 `Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
High-throughput screening of inorganic compounds for the discovery
of novel dielectric and optical materials. Sci. Data 4, 160134 (2017).`

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
                 '}',
                 '@article{Petousis2017,\n'
                 'author={Petousis, Ioannis and Mrdjenovich, David and '
                 'Ballouz, Eric\n'
                 'and Liu, Miao and Winston, Donald and Chen, Wei and Graf, '
                 'Tanja\n'
                 'and Schladt, Thomas D. and Persson, Kristin A. and Prinz, '
                 'Fritz B.},\n'
                 'title={High-throughput screening of inorganic compounds for '
                 'the\n'
                 'discovery of novel dielectric and optical materials},\n'
                 'journal={Scientific Data},\n'
                 'year={2017},\n'
                 'month={Jan},\n'
                 'day={31},\n'
                 'publisher={The Author(s)},\n'
                 'volume={4},\n'
                 'pages={160134},\n'
                 'note={Data Descriptor},\n'
                 'url={http://dx.doi.org/10.1038/sdata.2016.134}\n'
                 '}'],
 'columns': {'n': 'Target variable. Refractive index (unitless).',
             'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting refractive index '
                'from structure. Adapted from Materials Project database. '
                'Removed entries having a formation energy (or energy above '
                'the convex hull) more than 150meV and those having refractive '
                'indices less than 1 and those containing noble gases. '
                'Retrieved April 2, 2019. For benchmarking w/ nested cross '
                'validation, the order of the dataset must be identical to the '
                'retrieved data; refer to the Automatminer/Matbench '
                'publication for more details.',
 'file_type': 'json.gz',
 'hash': '83befa09bc2ec2f4b6143afc413157827a90e5e2e42c1eb507ccfa01bf26a1d6',
 'input_type': 'structure',
 'mad': 0.808534704217072,
 'n_samples': 4764,
 'num_entries': 4764,
 'reference': 'Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, '
              'D.,\n'
              'Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. '
              'B.\n'
              'High-throughput screening of inorganic compounds for the '
              'discovery\n'
              'of novel dielectric and optical materials. Sci. Data 4, 160134 '
              '(2017).',
 'target': 'n',
 'task_type': 'regression',
 'unit': 'unitless',
 'url': 'https://ml.materialsproject.org/projects/matbench_dielectric.json.gz'}
```

