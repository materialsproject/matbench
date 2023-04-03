# matbench_v0.1 matbench_mp_gap

## Individual Task Leaderboard for `matbench_mp_gap`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [coGN](../Full%20Benchmark%20Data/matbench_v0.1_coGN.md) | **0.1559** | 0.0031 | 0.3994 | 7.3029 | 
| [ALIGNN](../Full%20Benchmark%20Data/matbench_v0.1_alignn.md) | **0.1861** | 0.0030 | 0.4635 | 7.4756 | 
| [MegNet (kgcnn v2.1.0)](../Full%20Benchmark%20Data/matbench_v0.1_MegNet_kgcnn_v2.1.0.md) | **0.1934** | 0.0087 | 0.4715 | 7.8821 | 
| [DimeNet++ (kgcnn v2.1.0)](../Full%20Benchmark%20Data/matbench_v0.1_DimeNetPP_kgcnn_v2.1.0.md) | **0.1993** | 0.0058 | 0.4720 | 14.0169 | 
| [Finder_v1.2 structure-based version](../Full%20Benchmark%20Data/matbench_v0.1_Finder_v1.2_structure.md) | **0.2193** | 0.0012 | 0.4989 | 7.6676 | 
| [MODNet (v0.1.10)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.10.md) | **0.2199** | 0.0059 | 0.4525 | 7.5685 | 
| [MODNet (v0.1.12)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.12.md) | **0.2199** | 0.0059 | 0.4525 | 7.5685 | 
| [Finder_v1.2 composition-only version](../Full%20Benchmark%20Data/matbench_v0.1_Finder_v1.2_composition.md) | **0.2308** | 0.0029 | 0.4837 | 7.8152 | 
| [SchNet (kgcnn v2.1.0)](../Full%20Benchmark%20Data/matbench_v0.1_SchNet_kgcnn_v2.1.0.md) | **0.2352** | 0.0034 | 0.5172 | 9.1171 | 
| [CrabNet](../Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **0.2655** | 0.0029 | 0.5898 | 7.9829 | 
| [AMMExpress v2020](../Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.2824** | 0.0061 | 0.5611 | 6.9105 | 
| [CGCNN v2019](../Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **0.2972** | 0.0035 | 0.6771 | 13.6569 | 
| [RF-SCM/Magpie](../Full%20Benchmark%20Data/matbench_v0.1_rf.md) | **0.3452** | 0.0033 | 0.6125 | 7.0601 | 
| [Dummy](../Full%20Benchmark%20Data/matbench_v0.1_dummy.md) | **1.3272** | 0.0060 | 1.5989 | 8.5092 | 


<iframe src="../../static/task_matbench_v0.1_matbench_mp_gap.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting DFT PBE band gap from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 106113

Task type: regression

Input type: structure

##### Dataset columns

- gap pbe: Target variable. The band gap as calculated by PBE DFT from the Materials Project, in eV.
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
 'columns': {'gap pbe': 'Target variable. The band gap as calculated by PBE '
                        'DFT from the Materials Project, in eV.',
             'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting DFT PBE band gap '
                'from structure. Adapted from Materials Project database. '
                'Removed entries having a formation energy (or energy above '
                'the convex hull) more than 150meV and those containing noble '
                'gases. Retrieved April 2, 2019. For benchmarking w/ nested '
                'cross validation, the order of the dataset must be identical '
                'to the retrieved data; refer to the Automatminer/Matbench '
                'publication for more details.',
 'file_type': 'json.gz',
 'hash': '58b65746bd88329986ed66031a2ac1369c7c522f7bc9f9081528e07097c2c057',
 'input_type': 'structure',
 'mad': 1.3271449960162496,
 'n_samples': 106113,
 'num_entries': 106113,
 'reference': 'A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. '
              'Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson '
              '(*=equal contributions)\n'
              'The Materials Project: A materials genome approach to '
              'accelerating materials innovation\n'
              'APL Materials, 2013, 1(1), 011002.\n'
              'doi:10.1063/1.4812323',
 'target': 'gap pbe',
 'task_type': 'regression',
 'unit': 'eV',
 'url': 'https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz'}
```

