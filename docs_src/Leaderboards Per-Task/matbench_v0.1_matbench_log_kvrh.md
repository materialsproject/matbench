# matbench_v0.1 matbench_log_kvrh

## Individual Task Leaderboard for `matbench_log_kvrh`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [MODNet (v0.1.12)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.12.md) | **0.0548** | 0.0025 | 0.1043 | 1.5366 | 
| [MODNet (v0.1.10)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.10.md) | **0.0548** | 0.0025 | 0.1043 | 1.5366 | 
| [ALIGNN](../Full%20Benchmark%20Data/matbench_v0.1_alignn.md) | **0.0568** | 0.0028 | 0.1106 | 1.6438 | 
| [SchNet (kgcnn)](../Full%20Benchmark%20Data/matbench_v0.1_SchNet_kgcnn.md) | **0.0590** | 0.0022 | 0.1143 | 1.7542 | 
| [AMMExpress v2020](../Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.0647** | 0.0015 | 0.1183 | 1.4823 | 
| [Finder_v1.2 structure-based version](../Full%20Benchmark%20Data/matbench_v0.1_Finder_v1.2_structure.md) | **0.0693** | 0.0035 | 0.1318 | 1.6242 | 
| [CGCNN v2019](../Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **0.0712** | 0.0028 | 0.1301 | 1.7725 | 
| [CrabNet](../Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **0.0758** | 0.0034 | 0.1471 | 1.8430 | 
| [Finder_v1.2 composition-only version](../Full%20Benchmark%20Data/matbench_v0.1_Finder_v1.2_composition.md) | **0.0764** | 0.0025 | 0.1491 | 2.3863 | 
| [RF-SCM/Magpie](../Full%20Benchmark%20Data/matbench_v0.1_rf.md) | **0.0820** | 0.0027 | 0.1454 | 1.7642 | 
| [Dummy](../Full%20Benchmark%20Data/matbench_v0.1_dummy.md) | **0.2897** | 0.0043 | 0.3693 | 1.8822 | 


<iframe src="../../static/task_matbench_v0.1_matbench_log_kvrh.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting DFT log10 VRH-average bulk modulus from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having negative G_Voigt, G_Reuss, G_VRH, K_Voigt, K_Reuss, or K_VRH and those failing G_Reuss <= G_VRH <= G_Voigt or K_Reuss <= K_VRH <= K_Voigt and those containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 10987

Task type: regression

Input type: structure

##### Dataset columns

- log10(K_VRH): Target variable. Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa.
- structure: Pymatgen Structure of the material.


##### Dataset reference

 `Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting
the complete elastic properties of inorganic crystalline compounds",
Scientific Data volume 2, Article number: 150009 (2015)`

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
                 '@Article{deJong2015,\n'
                 'author={de Jong, Maarten and Chen, Wei and Angsten, Thomas\n'
                 'and Jain, Anubhav and Notestine, Randy and Gamst, Anthony\n'
                 'and Sluiter, Marcel and Krishna Ande, Chaitanya\n'
                 'and van der Zwaag, Sybrand and Plata, Jose J. and Toher, '
                 'Cormac\n'
                 'and Curtarolo, Stefano and Ceder, Gerbrand and Persson, '
                 'Kristin A.\n'
                 'and Asta, Mark},\n'
                 'title={Charting the complete elastic properties\n'
                 'of inorganic crystalline compounds},\n'
                 'journal={Scientific Data},\n'
                 'year={2015},\n'
                 'month={Mar},\n'
                 'day={17},\n'
                 'publisher={The Author(s)},\n'
                 'volume={2},\n'
                 'pages={150009},\n'
                 'note={Data Descriptor},\n'
                 'url={http://dx.doi.org/10.1038/sdata.2015.9}\n'
                 '}'],
 'columns': {'log10(K_VRH)': 'Target variable. Base 10 logarithm of the DFT '
                             'Voigt-Reuss-Hill average bulk moduli in GPa.',
             'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting DFT log10 '
                'VRH-average bulk modulus from structure. Adapted from '
                'Materials Project database. Removed entries having a '
                'formation energy (or energy above the convex hull) more than '
                '150meV and those having negative G_Voigt, G_Reuss, G_VRH, '
                'K_Voigt, K_Reuss, or K_VRH and those failing G_Reuss <= G_VRH '
                '<= G_Voigt or K_Reuss <= K_VRH <= K_Voigt and those '
                'containing noble gases. Retrieved April 2, 2019. For '
                'benchmarking w/ nested cross validation, the order of the '
                'dataset must be identical to the retrieved data; refer to the '
                'Automatminer/Matbench publication for more details.',
 'file_type': 'json.gz',
 'hash': '44b113ddb7e23aa18731a62c74afa7e5aa654199e0db5f951c8248a00955c9cd',
 'input_type': 'structure',
 'mad': 0.2896736342937069,
 'n_samples': 10987,
 'num_entries': 10987,
 'reference': 'Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., '
              'Gamst,\n'
              'A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., '
              'Toher,\n'
              'C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., '
              '"Charting\n'
              'the complete elastic properties of inorganic crystalline '
              'compounds",\n'
              'Scientific Data volume 2, Article number: 150009 (2015)',
 'target': 'log10(K_VRH)',
 'task_type': 'regression',
 'unit': 'log10(GPa)',
 'url': 'https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz'}
```

