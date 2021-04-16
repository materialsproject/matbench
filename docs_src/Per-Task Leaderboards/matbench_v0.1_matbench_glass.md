# matbench_v0.1 matbench_glass

## Individual Task Leaderboard for `matbench_glass`

_Leaderboard for an individual task. Algorithms shown here may include both general purpose and specialized algorithms (i.e., algorithms which are only valid for a subset of tasks in the benchmark._

### Leaderboard

| algorithm | mean rocauc | std rocauc | mean f1 | mean balanced_accuracy |
|------|------|------|------|------|
| [AMMExpress v2020](/Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020) | **0.8607** | 0.0199 | 0.9043 | 0.8607 | 
| [Dummy](/Full%20Benchmark%20Data/matbench_v0.1_dummy) | **0.5005** | 0.0178 | 0.7127 | 0.5005 | 


<iframe src="static/task_matbench_v0.1_matbench_glass.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

### Dataset info

##### Description

Matbench v0.1 test dataset for predicting full bulk metallic glass formation ability from chemical formula. Retrieved from "Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ a volume of the Landolt– Börnstein collection. Deduplicated according to composition, ensuring no compositions were reported as both GFA and not GFA (i.e., all reports agreed on the classification designation). For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.

Number of samples: 5680

Task type: classification

Input type: composition

##### Dataset columns

- composition: Chemical formula.
- gfa: Target variable. Glass forming ability: 1 means glass forming and corresponds to amorphous, 0 means non full glass forming.


##### Dataset reference

 `Y. Kawazoe, T. Masumoto, A.-P. Tsai, J.-Z. Yu, T. Aihara Jr. (1997) Y. Kawazoe, J.-Z. Yu, A.-P. Tsai, T. Masumoto (ed.) SpringerMaterials
Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys · 1 Introduction Landolt-Börnstein - Group III Condensed Matter 37A (Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys) https://www.springer.com/gp/book/9783540605072 (Springer-Verlag Berlin Heidelberg © 1997) Accessed: 03-09-2019`

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
                 '@Misc{LandoltBornstein1997:sm_lbs_978-3-540-47679-5_2,\n'
                 'author="Kawazoe, Y.\n'
                 'and Masumoto, T.\n'
                 'and Tsai, A.-P.\n'
                 'and Yu, J.-Z.\n'
                 'and Aihara Jr., T.",\n'
                 'editor="Kawazoe, Y.\n'
                 'and Yu, J.-Z.\n'
                 'and Tsai, A.-P.\n'
                 'and Masumoto, T.",\n'
                 'title="Nonequilibrium Phase Diagrams of Ternary Amorphous '
                 'Alloys {\\textperiodcentered} 1 Introduction: Datasheet from '
                 'Landolt-B{\\"o}rnstein - Group III Condensed Matter '
                 '{\\textperiodcentered} Volume 37A: ``Nonequilibrium Phase '
                 "Diagrams of Ternary Amorphous Alloys'' in SpringerMaterials "
                 '(https://dx.doi.org/10.1007/10510374{\\_}2)",\n'
                 'publisher="Springer-Verlag Berlin Heidelberg",\n'
                 'note="Copyright 1997 Springer-Verlag Berlin Heidelberg",\n'
                 'note="Part of SpringerMaterials",\n'
                 'note="accessed 2018-10-23",\n'
                 'doi="10.1007/10510374_2",\n'
                 'url="https://materials.springer.com/lb/docs/sm_lbs_978-3-540-47679-5_2"\n'
                 '}',
                 '@Article{Ward2016,\n'
                 'author={Ward, Logan\n'
                 'and Agrawal, Ankit\n'
                 'and Choudhary, Alok\n'
                 'and Wolverton, Christopher},\n'
                 'title={A general-purpose machine learning framework for '
                 'predicting properties of inorganic materials},\n'
                 'journal={Npj Computational Materials},\n'
                 'year={2016},\n'
                 'month={Aug},\n'
                 'day={26},\n'
                 'publisher={The Author(s)},\n'
                 'volume={2},\n'
                 'pages={16028},\n'
                 'note={Article},\n'
                 'url={http://dx.doi.org/10.1038/npjcompumats.2016.28}\n'
                 '}'],
 'columns': {'composition': 'Chemical formula.',
             'gfa': 'Target variable. Glass forming ability: 1 means glass '
                    'forming and corresponds to amorphous, 0 means non full '
                    'glass forming.'},
 'description': 'Matbench v0.1 test dataset for predicting full bulk metallic '
                'glass formation ability from chemical formula. Retrieved from '
                '"Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ '
                'a volume of the Landolt– Börnstein collection. Deduplicated '
                'according to composition, ensuring no compositions were '
                'reported as both GFA and not GFA (i.e., all reports agreed on '
                'the classification designation). For benchmarking w/ nested '
                'cross validation, the order of the dataset must be identical '
                'to the retrieved data; refer to the Automatminer/Matbench '
                'publication for more details.',
 'file_type': 'json.gz',
 'frac_true': 0.710387323943662,
 'hash': '36beb654e2a463ee2a6572105bea0ca2961eee7c7b26a25377bff2c3b338e53a',
 'input_type': 'composition',
 'n_samples': 5680,
 'num_entries': 5680,
 'reference': 'Y. Kawazoe, T. Masumoto, A.-P. Tsai, J.-Z. Yu, T. Aihara Jr. '
              '(1997) Y. Kawazoe, J.-Z. Yu, A.-P. Tsai, T. Masumoto (ed.) '
              'SpringerMaterials\n'
              'Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys · 1 '
              'Introduction Landolt-Börnstein - Group III Condensed Matter 37A '
              '(Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys) '
              'https://www.springer.com/gp/book/9783540605072 (Springer-Verlag '
              'Berlin Heidelberg © 1997) Accessed: 03-09-2019',
 'target': 'gfa',
 'task_type': 'classification',
 'unit': None,
 'url': 'https://ml.materialsproject.org/projects/matbench_glass.json.gz'}
```

