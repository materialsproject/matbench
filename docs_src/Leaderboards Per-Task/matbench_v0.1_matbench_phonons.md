# matbench_v0.1 matbench_phonons

## Individual Task Leaderboard for `matbench_phonons`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean mae | std mae | mean rmse | max max_error |
|------|------|------|------|------|
| [ALIGNN](../Full%20Benchmark%20Data/matbench_v0.1_alignn.md) | **29.5385** | 2.1148 | 53.5010 | 615.3466 | 
| [MODNet (v0.1.12)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.12.md) | **34.2751** | 2.0781 | 70.0669 | 1079.1280 | 
| [MODNet (v0.1.10)](../Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.10.md) | **38.7524** | 1.7732 | 78.2220 | 1031.8168 | 
| [CrabNet](../Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **55.1114** | 5.7317 | 138.3775 | 1452.7562 | 
| [AMMExpress v2020](../Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **56.1706** | 6.7981 | 109.7048 | 1151.5570 | 
| [CGCNN v2019](../Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **57.7635** | 12.3109 | 141.7018 | 2504.8743 | 
| [RF-SCM/Magpie](../Full%20Benchmark%20Data/matbench_v0.1_rf.md) | **67.6126** | 8.9900 | 146.2764 | 2024.7301 | 
| [Dummy](../Full%20Benchmark%20Data/matbench_v0.1_dummy.md) | **323.9822** | 17.7269 | 492.1533 | 3062.3450 | 


<iframe src="../../static/task_matbench_v0.1_matbench_phonons.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting vibration properties from crystal structure. Original data retrieved from Petretto et al. Original calculations done via ABINIT in the harmonic approximation based on density functional perturbation theory. Removed entries having a formation energy (or energy above the convex hull) more than 150meV. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 1265

Task type: regression

Input type: structure

##### Dataset columns

- last phdos peak: Target variable. Frequency of the highest frequency optical phonon mode peak, in units of 1/cm; ; may be used as an estimation of dominant longitudinal optical phonon frequency.
- structure: Pymatgen Structure of the material.


##### Dataset visualizations


<iframe src="../../static/pymatviz_matbench_v0.1_matbench_phonons_elements.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>


<iframe src="../../static/pymatviz_matbench_v0.1_matbench_phonons_composition_by_crystal_system.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>


<iframe src="../../static/pymatviz_matbench_v0.1_matbench_phonons_spacegroup_sunburst.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

##### Dataset reference

 `Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. Sci. Data 5:180065 doi: 10.1038/sdata.2018.65 (2018).
Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. (2018). figshare. Collection.`

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
                 '@Article{Petretto2018,\n'
                 'author={Petretto, Guido\n'
                 'and Dwaraknath, Shyam\n'
                 'and P.C. Miranda, Henrique\n'
                 'and Winston, Donald\n'
                 'and Giantomassi, Matteo\n'
                 'and van Setten, Michiel J.\n'
                 'and Gonze, Xavier\n'
                 'and Persson, Kristin A.\n'
                 'and Hautier, Geoffroy\n'
                 'and Rignanese, Gian-Marco},\n'
                 'title={High-throughput density-functional perturbation '
                 'theory phonons for inorganic materials},\n'
                 'journal={Scientific Data},\n'
                 'year={2018},\n'
                 'month={May},\n'
                 'day={01},\n'
                 'publisher={The Author(s)},\n'
                 'volume={5},\n'
                 'pages={180065},\n'
                 'note={Data Descriptor},\n'
                 'url={http://dx.doi.org/10.1038/sdata.2018.65}\n'
                 '}',
                 '@misc{petretto_dwaraknath_miranda_winston_giantomassi_rignanese_van '
                 'setten_gonze_persson_hautier_2018, title={High-throughput '
                 'Density-Functional Perturbation Theory phonons for inorganic '
                 'materials}, '
                 'url={https://figshare.com/collections/High-throughput_Density-Functional_Perturbation_Theory_phonons_for_inorganic_materials/3938023/1}, '
                 'DOI={10.6084/m9.figshare.c.3938023.v1}, abstractNote={The '
                 'knowledge of the vibrational properties of a material is of '
                 'key importance to understand physical phenomena such as '
                 'thermal conductivity, superconductivity, and '
                 'ferroelectricity among others. However, detailed '
                 'experimental phonon spectra are available only for a limited '
                 'number of materials which hinders the large-scale analysis '
                 'of vibrational properties and their derived quantities. In '
                 'this work, we perform ab initio calculations of the full '
                 'phonon dispersion and vibrational density of states for 1521 '
                 'semiconductor compounds in the harmonic approximation based '
                 'on density functional perturbation theory. The data is '
                 'collected along with derived dielectric and thermodynamic '
                 'properties. We present the procedure used to obtain the '
                 'results, the details of the provided database and a '
                 'validation based on the comparison with experimental data.}, '
                 'publisher={figshare}, author={Petretto, Guido and '
                 'Dwaraknath, Shyam and Miranda, Henrique P. C. and Winston, '
                 'Donald and Giantomassi, Matteo and Rignanese, Gian-Marco and '
                 'Van Setten, Michiel J. and Gonze, Xavier and Persson, '
                 'Kristin A and Hautier, Geoffroy}, year={2018}, month={Apr}}'],
 'columns': {'last phdos peak': 'Target variable. Frequency of the highest '
                                'frequency optical phonon mode peak, in units '
                                'of 1/cm; ; may be used as an estimation of '
                                'dominant longitudinal optical phonon '
                                'frequency.',
             'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting vibration '
                'properties from crystal structure. Original data retrieved '
                'from Petretto et al. Original calculations done via ABINIT in '
                'the harmonic approximation based on density functional '
                'perturbation theory. Removed entries having a formation '
                'energy (or energy above the convex hull) more than 150meV. '
                'For benchmarking w/ nested cross validation, the order of the '
                'dataset must be identical to the retrieved data; refer to the '
                'Automatminer/Matbench publication for more details.',
 'file_type': 'json.gz',
 'hash': '4db551f21ec5f577e6202725f10e34dfc509aa7df3a6bdaac497da7f6dbbb9b3',
 'input_type': 'structure',
 'mad': 323.78696979348734,
 'n_samples': 1265,
 'num_entries': 1265,
 'reference': 'Petretto, G. et al. High-throughput density functional '
              'perturbation theory phonons for inorganic materials. Sci. Data '
              '5:180065 doi: 10.1038/sdata.2018.65 (2018).\n'
              'Petretto, G. et al. High-throughput density functional '
              'perturbation theory phonons for inorganic materials. (2018). '
              'figshare. Collection.',
 'target': 'last phdos peak',
 'task_type': 'regression',
 'unit': 'cm^-1',
 'url': 'https://ml.materialsproject.org/projects/matbench_phonons.json.gz'}
```

