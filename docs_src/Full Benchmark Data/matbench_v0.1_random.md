# matbench_v0.1: Random

### Algorithm description: 

Totally random.

No notes.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_random).

### References (in bibtex format): 

```
Random
```

### User metadata:

```
{'matbench_dielectric': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                         'author={Dunn, Alexander\n'
                                         'and Wang, Qi\n'
                                         'and Ganose, Alex\n'
                                         'and Dopp, Daniel\n'
                                         'and Jain, Anubhav},\n'
                                         'title={Benchmarking materials '
                                         'property prediction methods: the '
                                         'Matbench test set and Automatminer '
                                         'reference algorithm},\n'
                                         'journal={npj Computational '
                                         'Materials},\n'
                                         'year={2020},\n'
                                         'month={Sep},\n'
                                         'day={15},\n'
                                         'volume={6},\n'
                                         'number={1},\n'
                                         'pages={138},\n'
                                         'abstract={We present a benchmark '
                                         'test suite and an automated machine '
                                         'learning procedure for evaluating '
                                         'supervised machine learning (ML) '
                                         'models for predicting properties of '
                                         'inorganic bulk materials. The test '
                                         'suite, Matbench, is a set of '
                                         '13{\\thinspace}ML tasks that range '
                                         'in size from 312 to 132k samples and '
                                         'contain data from 10 density '
                                         'functional theory-derived and '
                                         'experimental sources. Tasks include '
                                         'predicting optical, thermal, '
                                         'electronic, thermodynamic, tensile, '
                                         'and elastic properties given a '
                                         "material's composition and/or "
                                         'crystal structure. The reference '
                                         'algorithm, Automatminer, is a '
                                         'highly-extensible, fully automated '
                                         'ML pipeline for predicting materials '
                                         'properties from materials primitives '
                                         '(such as composition and crystal '
                                         'structure) without user intervention '
                                         'or hyperparameter tuning. We test '
                                         'Automatminer on the Matbench test '
                                         'suite and compare its predictive '
                                         'power with state-of-the-art crystal '
                                         'graph neural networks and a '
                                         'traditional descriptor-based Random '
                                         'Forest model. We find Automatminer '
                                         'achieves the best performance on 8 '
                                         'of 13 tasks in the benchmark. We '
                                         'also show our test suite is capable '
                                         'of exposing predictive advantages of '
                                         'each algorithm---namely, that '
                                         'crystal graph methods appear to '
                                         'outperform traditional machine '
                                         'learning methods given '
                                         '{\\textasciitilde}104 or greater '
                                         'data points. We encourage evaluating '
                                         'materials ML algorithms on the '
                                         'Matbench benchmark and comparing '
                                         'them against the latest version of '
                                         'Automatminer.},\n'
                                         'issn={2057-3960},\n'
                                         'doi={10.1038/s41524-020-00406-3},\n'
                                         'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                         '}\n',
                                         '@article{Jain2013,\n'
                                         'author = {Jain, Anubhav and Ong, '
                                         'Shyue Ping and Hautier, Geoffroy and '
                                         'Chen, Wei and Richards, William '
                                         'Davidson and Dacek, Stephen and '
                                         'Cholia, Shreyas and Gunter, Dan and '
                                         'Skinner, David and Ceder, Gerbrand '
                                         'and Persson, Kristin a.},\n'
                                         'doi = {10.1063/1.4812323},\n'
                                         'issn = {2166532X},\n'
                                         'journal = {APL Materials},\n'
                                         'number = {1},\n'
                                         'pages = {011002},\n'
                                         'title = {{The Materials Project: A '
                                         'materials genome approach to '
                                         'accelerating materials '
                                         'innovation}},\n'
                                         'url = '
                                         '{http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\\&Agg=doi},\n'
                                         'volume = {1},\n'
                                         'year = {2013}\n'
                                         '}',
                                         '@article{Petousis2017,\n'
                                         'author={Petousis, Ioannis and '
                                         'Mrdjenovich, David and Ballouz, '
                                         'Eric\n'
                                         'and Liu, Miao and Winston, Donald '
                                         'and Chen, Wei and Graf, Tanja\n'
                                         'and Schladt, Thomas D. and Persson, '
                                         'Kristin A. and Prinz, Fritz B.},\n'
                                         'title={High-throughput screening of '
                                         'inorganic compounds for the\n'
                                         'discovery of novel dielectric and '
                                         'optical materials},\n'
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
                         'columns': {'n': 'Target variable. Refractive index '
                                          '(unitless).',
                                     'structure': 'Pymatgen Structure of the '
                                                  'material.'},
                         'description': 'Matbench v0.1 test dataset for '
                                        'predicting refractive index from '
                                        'structure. Adapted from Materials '
                                        'Project database. Removed entries '
                                        'having a formation energy (or energy '
                                        'above the convex hull) more than '
                                        '150meV and those having refractive '
                                        'indices less than 1 and those '
                                        'containing noble gases. Retrieved '
                                        'April 2, 2019. For benchmarking w/ '
                                        'nested cross validation, the order of '
                                        'the dataset must be identical to the '
                                        'retrieved data; refer to the '
                                        'Automatminer/Matbench publication for '
                                        'more details.',
                         'file_type': 'json.gz',
                         'hash': '83befa09bc2ec2f4b6143afc413157827a90e5e2e42c1eb507ccfa01bf26a1d6',
                         'input_type': 'structure',
                         'n_samples': 4764,
                         'num_entries': 4764,
                         'reference': 'Petousis, I., Mrdjenovich, D., Ballouz, '
                                      'E., Liu, M., Winston, D.,\n'
                                      'Chen, W., Graf, T., Schladt, T. D., '
                                      'Persson, K. A. & Prinz, F. B.\n'
                                      'High-throughput screening of inorganic '
                                      'compounds for the discovery\n'
                                      'of novel dielectric and optical '
                                      'materials. Sci. Data 4, 160134 (2017).',
                         'target': 'n',
                         'task_type': 'regression',
                         'url': 'https://ml.materialsproject.org/projects/matbench_dielectric.json.gz'},
 'matbench_expt_gap': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                       'author={Dunn, Alexander\n'
                                       'and Wang, Qi\n'
                                       'and Ganose, Alex\n'
                                       'and Dopp, Daniel\n'
                                       'and Jain, Anubhav},\n'
                                       'title={Benchmarking materials property '
                                       'prediction methods: the Matbench test '
                                       'set and Automatminer reference '
                                       'algorithm},\n'
                                       'journal={npj Computational '
                                       'Materials},\n'
                                       'year={2020},\n'
                                       'month={Sep},\n'
                                       'day={15},\n'
                                       'volume={6},\n'
                                       'number={1},\n'
                                       'pages={138},\n'
                                       'abstract={We present a benchmark test '
                                       'suite and an automated machine '
                                       'learning procedure for evaluating '
                                       'supervised machine learning (ML) '
                                       'models for predicting properties of '
                                       'inorganic bulk materials. The test '
                                       'suite, Matbench, is a set of '
                                       '13{\\thinspace}ML tasks that range in '
                                       'size from 312 to 132k samples and '
                                       'contain data from 10 density '
                                       'functional theory-derived and '
                                       'experimental sources. Tasks include '
                                       'predicting optical, thermal, '
                                       'electronic, thermodynamic, tensile, '
                                       'and elastic properties given a '
                                       "material's composition and/or crystal "
                                       'structure. The reference algorithm, '
                                       'Automatminer, is a highly-extensible, '
                                       'fully automated ML pipeline for '
                                       'predicting materials properties from '
                                       'materials primitives (such as '
                                       'composition and crystal structure) '
                                       'without user intervention or '
                                       'hyperparameter tuning. We test '
                                       'Automatminer on the Matbench test '
                                       'suite and compare its predictive power '
                                       'with state-of-the-art crystal graph '
                                       'neural networks and a traditional '
                                       'descriptor-based Random Forest model. '
                                       'We find Automatminer achieves the best '
                                       'performance on 8 of 13 tasks in the '
                                       'benchmark. We also show our test suite '
                                       'is capable of exposing predictive '
                                       'advantages of each algorithm---namely, '
                                       'that crystal graph methods appear to '
                                       'outperform traditional machine '
                                       'learning methods given '
                                       '{\\textasciitilde}104 or greater data '
                                       'points. We encourage evaluating '
                                       'materials ML algorithms on the '
                                       'Matbench benchmark and comparing them '
                                       'against the latest version of '
                                       'Automatminer.},\n'
                                       'issn={2057-3960},\n'
                                       'doi={10.1038/s41524-020-00406-3},\n'
                                       'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                       '}\n',
                                       '@article{doi:10.1021/acs.jpclett.8b00124,\n'
                                       'author = {Zhuo, Ya and Mansouri '
                                       'Tehrani, Aria and Brgoch, Jakoah},\n'
                                       'title = {Predicting the Band Gaps of '
                                       'Inorganic Solids by Machine '
                                       'Learning},\n'
                                       'journal = {The Journal of Physical '
                                       'Chemistry Letters},\n'
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
                                   'gap expt': 'Target variable. '
                                               'Experimentally measured gap, '
                                               'in eV.'},
                       'description': 'Matbench v0.1 test dataset for '
                                      'predicting experimental band gap from '
                                      'composition alone. Retrieved from Zhuo '
                                      'et al. supplementary information. '
                                      'Deduplicated according to composition, '
                                      'removing compositions with reported '
                                      'band gaps spanning more than a 0.1eV '
                                      'range; remaining compositions were '
                                      'assigned values based on the closest '
                                      'experimental value to the mean '
                                      'experimental value for that composition '
                                      'among all reports. For benchmarking w/ '
                                      'nested cross validation, the order of '
                                      'the dataset must be identical to the '
                                      'retrieved data; refer to the '
                                      'Automatminer/Matbench publication for '
                                      'more details.',
                       'file_type': 'json.gz',
                       'hash': '783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2',
                       'input_type': 'composition',
                       'n_samples': 4604,
                       'num_entries': 4604,
                       'reference': 'Y. Zhuo, A. Masouri Tehrani, J. Brgoch '
                                    '(2018) Predicting the Band Gaps of '
                                    'Inorganic Solids by Machine Learning J. '
                                    'Phys. Chem. Lett. 2018, 9, 7, 1668-1673 '
                                    'https:doi.org/10.1021/acs.jpclett.8b00124.',
                       'target': 'gap expt',
                       'task_type': 'regression',
                       'url': 'https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz'},
 'matbench_expt_is_metal': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                            'author={Dunn, Alexander\n'
                                            'and Wang, Qi\n'
                                            'and Ganose, Alex\n'
                                            'and Dopp, Daniel\n'
                                            'and Jain, Anubhav},\n'
                                            'title={Benchmarking materials '
                                            'property prediction methods: the '
                                            'Matbench test set and '
                                            'Automatminer reference '
                                            'algorithm},\n'
                                            'journal={npj Computational '
                                            'Materials},\n'
                                            'year={2020},\n'
                                            'month={Sep},\n'
                                            'day={15},\n'
                                            'volume={6},\n'
                                            'number={1},\n'
                                            'pages={138},\n'
                                            'abstract={We present a benchmark '
                                            'test suite and an automated '
                                            'machine learning procedure for '
                                            'evaluating supervised machine '
                                            'learning (ML) models for '
                                            'predicting properties of '
                                            'inorganic bulk materials. The '
                                            'test suite, Matbench, is a set of '
                                            '13{\\thinspace}ML tasks that '
                                            'range in size from 312 to 132k '
                                            'samples and contain data from 10 '
                                            'density functional theory-derived '
                                            'and experimental sources. Tasks '
                                            'include predicting optical, '
                                            'thermal, electronic, '
                                            'thermodynamic, tensile, and '
                                            'elastic properties given a '
                                            "material's composition and/or "
                                            'crystal structure. The reference '
                                            'algorithm, Automatminer, is a '
                                            'highly-extensible, fully '
                                            'automated ML pipeline for '
                                            'predicting materials properties '
                                            'from materials primitives (such '
                                            'as composition and crystal '
                                            'structure) without user '
                                            'intervention or hyperparameter '
                                            'tuning. We test Automatminer on '
                                            'the Matbench test suite and '
                                            'compare its predictive power with '
                                            'state-of-the-art crystal graph '
                                            'neural networks and a traditional '
                                            'descriptor-based Random Forest '
                                            'model. We find Automatminer '
                                            'achieves the best performance on '
                                            '8 of 13 tasks in the benchmark. '
                                            'We also show our test suite is '
                                            'capable of exposing predictive '
                                            'advantages of each '
                                            'algorithm---namely, that crystal '
                                            'graph methods appear to '
                                            'outperform traditional machine '
                                            'learning methods given '
                                            '{\\textasciitilde}104 or greater '
                                            'data points. We encourage '
                                            'evaluating materials ML '
                                            'algorithms on the Matbench '
                                            'benchmark and comparing them '
                                            'against the latest version of '
                                            'Automatminer.},\n'
                                            'issn={2057-3960},\n'
                                            'doi={10.1038/s41524-020-00406-3},\n'
                                            'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                            '}\n',
                                            '@article{doi:10.1021/acs.jpclett.8b00124,\n'
                                            'author = {Zhuo, Ya and Mansouri '
                                            'Tehrani, Aria and Brgoch, '
                                            'Jakoah},\n'
                                            'title = {Predicting the Band Gaps '
                                            'of Inorganic Solids by Machine '
                                            'Learning},\n'
                                            'journal = {The Journal of '
                                            'Physical Chemistry Letters},\n'
                                            'volume = {9},\n'
                                            'number = {7},\n'
                                            'pages = {1668-1673},\n'
                                            'year = {2018},\n'
                                            'doi = '
                                            '{10.1021/acs.jpclett.8b00124},\n'
                                            'note ={PMID: 29532658},\n'
                                            'eprint = {\n'
                                            'https://doi.org/10.1021/acs.jpclett.8b00124\n'
                                            '\n'
                                            '}}'],
                            'columns': {'composition': 'Chemical formula.',
                                        'is_metal': 'Target variable. 1 if is '
                                                    'a metal, 0 if nonmetal.'},
                            'description': 'Matbench v0.1 test dataset for '
                                           'classifying metallicity from '
                                           'composition alone. Retrieved from '
                                           'Zhuo et al. supplementary '
                                           'information. Deduplicated '
                                           'according to composition, ensuring '
                                           'no conflicting reports were '
                                           'entered for any compositions '
                                           '(i.e., no reported compositions '
                                           'were both metal and nonmetal). For '
                                           'benchmarking w/ nested cross '
                                           'validation, the order of the '
                                           'dataset must be identical to the '
                                           'retrieved data; refer to the '
                                           'Automatminer/Matbench publication '
                                           'for more details.',
                            'file_type': 'json.gz',
                            'hash': '8f2a4f9bacdcbc5c2c73615629ee7986f09d39bed40ba7db52b61b2889730887',
                            'input_type': 'composition',
                            'n_samples': 4921,
                            'num_entries': 4921,
                            'reference': 'Y. Zhuo, A. Masouri Tehrani, J. '
                                         'Brgoch (2018) Predicting the Band '
                                         'Gaps of Inorganic Solids by Machine '
                                         'Learning J. Phys. Chem. Lett. 2018, '
                                         '9, 7, 1668-1673 \n'
                                         ' '
                                         'https//:doi.org/10.1021/acs.jpclett.8b00124.',
                            'target': 'is_metal',
                            'task_type': 'classification',
                            'url': 'https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz'},
 'matbench_glass': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                    'author={Dunn, Alexander\n'
                                    'and Wang, Qi\n'
                                    'and Ganose, Alex\n'
                                    'and Dopp, Daniel\n'
                                    'and Jain, Anubhav},\n'
                                    'title={Benchmarking materials property '
                                    'prediction methods: the Matbench test set '
                                    'and Automatminer reference algorithm},\n'
                                    'journal={npj Computational Materials},\n'
                                    'year={2020},\n'
                                    'month={Sep},\n'
                                    'day={15},\n'
                                    'volume={6},\n'
                                    'number={1},\n'
                                    'pages={138},\n'
                                    'abstract={We present a benchmark test '
                                    'suite and an automated machine learning '
                                    'procedure for evaluating supervised '
                                    'machine learning (ML) models for '
                                    'predicting properties of inorganic bulk '
                                    'materials. The test suite, Matbench, is a '
                                    'set of 13{\\thinspace}ML tasks that range '
                                    'in size from 312 to 132k samples and '
                                    'contain data from 10 density functional '
                                    'theory-derived and experimental sources. '
                                    'Tasks include predicting optical, '
                                    'thermal, electronic, thermodynamic, '
                                    'tensile, and elastic properties given a '
                                    "material's composition and/or crystal "
                                    'structure. The reference algorithm, '
                                    'Automatminer, is a highly-extensible, '
                                    'fully automated ML pipeline for '
                                    'predicting materials properties from '
                                    'materials primitives (such as composition '
                                    'and crystal structure) without user '
                                    'intervention or hyperparameter tuning. We '
                                    'test Automatminer on the Matbench test '
                                    'suite and compare its predictive power '
                                    'with state-of-the-art crystal graph '
                                    'neural networks and a traditional '
                                    'descriptor-based Random Forest model. We '
                                    'find Automatminer achieves the best '
                                    'performance on 8 of 13 tasks in the '
                                    'benchmark. We also show our test suite is '
                                    'capable of exposing predictive advantages '
                                    'of each algorithm---namely, that crystal '
                                    'graph methods appear to outperform '
                                    'traditional machine learning methods '
                                    'given {\\textasciitilde}104 or greater '
                                    'data points. We encourage evaluating '
                                    'materials ML algorithms on the Matbench '
                                    'benchmark and comparing them against the '
                                    'latest version of Automatminer.},\n'
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
                                    'title="Nonequilibrium Phase Diagrams of '
                                    'Ternary Amorphous Alloys '
                                    '{\\textperiodcentered} 1 Introduction: '
                                    'Datasheet from Landolt-B{\\"o}rnstein - '
                                    'Group III Condensed Matter '
                                    '{\\textperiodcentered} Volume 37A: '
                                    '``Nonequilibrium Phase Diagrams of '
                                    "Ternary Amorphous Alloys'' in "
                                    'SpringerMaterials '
                                    '(https://dx.doi.org/10.1007/10510374{\\_}2)",\n'
                                    'publisher="Springer-Verlag Berlin '
                                    'Heidelberg",\n'
                                    'note="Copyright 1997 Springer-Verlag '
                                    'Berlin Heidelberg",\n'
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
                                    'title={A general-purpose machine learning '
                                    'framework for predicting properties of '
                                    'inorganic materials},\n'
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
                                'gfa': 'Target variable. Glass forming '
                                       'ability: 1 means glass forming and '
                                       'corresponds to amorphous, 0 means non '
                                       'full glass forming.'},
                    'description': 'Matbench v0.1 test dataset for predicting '
                                   'full bulk metallic glass formation ability '
                                   'from chemical formula. Retrieved from '
                                   '"Nonequilibrium Phase Diagrams of Ternary '
                                   'Amorphous Alloys,’ a volume of the '
                                   'Landolt– Börnstein collection. '
                                   'Deduplicated according to composition, '
                                   'ensuring no compositions were reported as '
                                   'both GFA and not GFA (i.e., all reports '
                                   'agreed on the classification designation). '
                                   'For benchmarking w/ nested cross '
                                   'validation, the order of the dataset must '
                                   'be identical to the retrieved data; refer '
                                   'to the Automatminer/Matbench publication '
                                   'for more details.',
                    'file_type': 'json.gz',
                    'hash': '36beb654e2a463ee2a6572105bea0ca2961eee7c7b26a25377bff2c3b338e53a',
                    'input_type': 'composition',
                    'n_samples': 5680,
                    'num_entries': 5680,
                    'reference': 'Y. Kawazoe, T. Masumoto, A.-P. Tsai, J.-Z. '
                                 'Yu, T. Aihara Jr. (1997) Y. Kawazoe, J.-Z. '
                                 'Yu, A.-P. Tsai, T. Masumoto (ed.) '
                                 'SpringerMaterials\n'
                                 'Nonequilibrium Phase Diagrams of Ternary '
                                 'Amorphous Alloys · 1 Introduction '
                                 'Landolt-Börnstein - Group III Condensed '
                                 'Matter 37A (Nonequilibrium Phase Diagrams of '
                                 'Ternary Amorphous Alloys) '
                                 'https://www.springer.com/gp/book/9783540605072 '
                                 '(Springer-Verlag Berlin Heidelberg © 1997) '
                                 'Accessed: 03-09-2019',
                    'target': 'gfa',
                    'task_type': 'classification',
                    'url': 'https://ml.materialsproject.org/projects/matbench_glass.json.gz'},
 'matbench_jdft2d': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                     'author={Dunn, Alexander\n'
                                     'and Wang, Qi\n'
                                     'and Ganose, Alex\n'
                                     'and Dopp, Daniel\n'
                                     'and Jain, Anubhav},\n'
                                     'title={Benchmarking materials property '
                                     'prediction methods: the Matbench test '
                                     'set and Automatminer reference '
                                     'algorithm},\n'
                                     'journal={npj Computational Materials},\n'
                                     'year={2020},\n'
                                     'month={Sep},\n'
                                     'day={15},\n'
                                     'volume={6},\n'
                                     'number={1},\n'
                                     'pages={138},\n'
                                     'abstract={We present a benchmark test '
                                     'suite and an automated machine learning '
                                     'procedure for evaluating supervised '
                                     'machine learning (ML) models for '
                                     'predicting properties of inorganic bulk '
                                     'materials. The test suite, Matbench, is '
                                     'a set of 13{\\thinspace}ML tasks that '
                                     'range in size from 312 to 132k samples '
                                     'and contain data from 10 density '
                                     'functional theory-derived and '
                                     'experimental sources. Tasks include '
                                     'predicting optical, thermal, electronic, '
                                     'thermodynamic, tensile, and elastic '
                                     "properties given a material's "
                                     'composition and/or crystal structure. '
                                     'The reference algorithm, Automatminer, '
                                     'is a highly-extensible, fully automated '
                                     'ML pipeline for predicting materials '
                                     'properties from materials primitives '
                                     '(such as composition and crystal '
                                     'structure) without user intervention or '
                                     'hyperparameter tuning. We test '
                                     'Automatminer on the Matbench test suite '
                                     'and compare its predictive power with '
                                     'state-of-the-art crystal graph neural '
                                     'networks and a traditional '
                                     'descriptor-based Random Forest model. We '
                                     'find Automatminer achieves the best '
                                     'performance on 8 of 13 tasks in the '
                                     'benchmark. We also show our test suite '
                                     'is capable of exposing predictive '
                                     'advantages of each algorithm---namely, '
                                     'that crystal graph methods appear to '
                                     'outperform traditional machine learning '
                                     'methods given {\\textasciitilde}104 or '
                                     'greater data points. We encourage '
                                     'evaluating materials ML algorithms on '
                                     'the Matbench benchmark and comparing '
                                     'them against the latest version of '
                                     'Automatminer.},\n'
                                     'issn={2057-3960},\n'
                                     'doi={10.1038/s41524-020-00406-3},\n'
                                     'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                     '}\n',
                                     '@Article{Choudhary2017,\n'
                                     'author={Choudhary, Kamal\n'
                                     'and Kalish, Irina\n'
                                     'and Beams, Ryan\n'
                                     'and Tavazza, Francesca},\n'
                                     'title={High-throughput Identification '
                                     'and Characterization of Two-dimensional '
                                     'Materials using Density functional '
                                     'theory},\n'
                                     'journal={Scientific Reports},\n'
                                     'year={2017},\n'
                                     'volume={7},\n'
                                     'number={1},\n'
                                     'pages={5179},\n'
                                     'abstract={We introduce a simple '
                                     'criterion to identify two-dimensional '
                                     '(2D) materials based on the comparison '
                                     'between experimental lattice constants '
                                     'and lattice constants mainly obtained '
                                     'from Materials-Project (MP) density '
                                     'functional theory (DFT) calculation '
                                     'repository. Specifically, if the '
                                     'relative difference between the two '
                                     'lattice constants for a specific '
                                     'material is greater than or equal to 5%, '
                                     'we predict them to be good candidates '
                                     'for 2D materials. We have predicted at '
                                     'least 1356 such 2D materials. For all '
                                     'the systems satisfying our criterion, we '
                                     'manually create single layer systems and '
                                     'calculate their energetics, structural, '
                                     'electronic, and elastic properties for '
                                     'both the bulk and the single layer '
                                     'cases. Currently the database consists '
                                     'of 1012 bulk and 430 single layer '
                                     'materials, of which 371 systems are '
                                     'common to bulk and single layer. The '
                                     'rest of calculations are underway. To '
                                     'validate our criterion, we calculated '
                                     'the exfoliation energy of the suggested '
                                     'layered materials, and we found that in '
                                     '88.9% of the cases the currently '
                                     'accepted criterion for exfoliation was '
                                     'satisfied. Also, using molybdenum '
                                     'telluride as a test case, we performed '
                                     'X-ray diffraction and Raman scattering '
                                     'experiments to benchmark our '
                                     'calculations and understand their '
                                     'applicability and limitations. The data '
                                     'is publicly available at the website '
                                     'http://www.ctcms.nist.gov/{\t'
                                     'extasciitilde}knc6/JVASP.html.},\n'
                                     'issn={2045-2322},\n'
                                     'doi={10.1038/s41598-017-05402-0},\n'
                                     'url={https://doi.org/10.1038/s41598-017-05402-0}\n'
                                     '}',
                                     '@misc{choudhary__2018, '
                                     'title={jdft_2d-7-7-2018.json}, '
                                     'url={https://figshare.com/articles/jdft_2d-7-7-2018_json/6815705/1}, '
                                     'DOI={10.6084/m9.figshare.6815705.v1}, '
                                     'abstractNote={2D materials}, '
                                     'publisher={figshare}, author={choudhary, '
                                     'kamal and '
                                     'https://orcid.org/0000-0001-9737-8074}, '
                                     'year={2018}, month={Jul}}'],
                     'columns': {'exfoliation_en': 'Target variable. '
                                                   'Exfoliation energy (meV).',
                                 'structure': 'Pymatgen Structure of the '
                                              'material.'},
                     'description': 'Matbench v0.1 test dataset for predicting '
                                    'exfoliation energies from crystal '
                                    'structure (computed with the OptB88vdW '
                                    'and TBmBJ functionals). Adapted from the '
                                    'JARVIS DFT database. For benchmarking w/ '
                                    'nested cross validation, the order of the '
                                    'dataset must be identical to the '
                                    'retrieved data; refer to the '
                                    'Automatminer/Matbench publication for '
                                    'more details.',
                     'file_type': 'json.gz',
                     'hash': '26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf',
                     'input_type': 'structure',
                     'n_samples': 636,
                     'num_entries': 636,
                     'reference': '2D Dataset discussed in:\n'
                                  'High-throughput Identification and '
                                  'Characterization of Two dimensional '
                                  'Materials using Density functional theory '
                                  'Kamal Choudhary, Irina Kalish, Ryan Beams & '
                                  'Francesca Tavazza Scientific Reports volume '
                                  '7, Article number: 5179 (2017)\n'
                                  'Original 2D Data file sourced from:\n'
                                  'choudhary, kamal; '
                                  'https://orcid.org/0000-0001-9737-8074 '
                                  '(2018): jdft_2d-7-7-2018.json. figshare. '
                                  'Dataset.',
                     'target': 'exfoliation_en',
                     'task_type': 'regression',
                     'url': 'https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz'},
 'matbench_log_gvrh': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                       'author={Dunn, Alexander\n'
                                       'and Wang, Qi\n'
                                       'and Ganose, Alex\n'
                                       'and Dopp, Daniel\n'
                                       'and Jain, Anubhav},\n'
                                       'title={Benchmarking materials property '
                                       'prediction methods: the Matbench test '
                                       'set and Automatminer reference '
                                       'algorithm},\n'
                                       'journal={npj Computational '
                                       'Materials},\n'
                                       'year={2020},\n'
                                       'month={Sep},\n'
                                       'day={15},\n'
                                       'volume={6},\n'
                                       'number={1},\n'
                                       'pages={138},\n'
                                       'abstract={We present a benchmark test '
                                       'suite and an automated machine '
                                       'learning procedure for evaluating '
                                       'supervised machine learning (ML) '
                                       'models for predicting properties of '
                                       'inorganic bulk materials. The test '
                                       'suite, Matbench, is a set of '
                                       '13{\\thinspace}ML tasks that range in '
                                       'size from 312 to 132k samples and '
                                       'contain data from 10 density '
                                       'functional theory-derived and '
                                       'experimental sources. Tasks include '
                                       'predicting optical, thermal, '
                                       'electronic, thermodynamic, tensile, '
                                       'and elastic properties given a '
                                       "material's composition and/or crystal "
                                       'structure. The reference algorithm, '
                                       'Automatminer, is a highly-extensible, '
                                       'fully automated ML pipeline for '
                                       'predicting materials properties from '
                                       'materials primitives (such as '
                                       'composition and crystal structure) '
                                       'without user intervention or '
                                       'hyperparameter tuning. We test '
                                       'Automatminer on the Matbench test '
                                       'suite and compare its predictive power '
                                       'with state-of-the-art crystal graph '
                                       'neural networks and a traditional '
                                       'descriptor-based Random Forest model. '
                                       'We find Automatminer achieves the best '
                                       'performance on 8 of 13 tasks in the '
                                       'benchmark. We also show our test suite '
                                       'is capable of exposing predictive '
                                       'advantages of each algorithm---namely, '
                                       'that crystal graph methods appear to '
                                       'outperform traditional machine '
                                       'learning methods given '
                                       '{\\textasciitilde}104 or greater data '
                                       'points. We encourage evaluating '
                                       'materials ML algorithms on the '
                                       'Matbench benchmark and comparing them '
                                       'against the latest version of '
                                       'Automatminer.},\n'
                                       'issn={2057-3960},\n'
                                       'doi={10.1038/s41524-020-00406-3},\n'
                                       'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                       '}\n',
                                       '@Article{deJong2015,\n'
                                       'author={de Jong, Maarten and Chen, Wei '
                                       'and Angsten, Thomas\n'
                                       'and Jain, Anubhav and Notestine, Randy '
                                       'and Gamst, Anthony\n'
                                       'and Sluiter, Marcel and Krishna Ande, '
                                       'Chaitanya\n'
                                       'and van der Zwaag, Sybrand and Plata, '
                                       'Jose J. and Toher, Cormac\n'
                                       'and Curtarolo, Stefano and Ceder, '
                                       'Gerbrand and Persson, Kristin A.\n'
                                       'and Asta, Mark},\n'
                                       'title={Charting the complete elastic '
                                       'properties\n'
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
                       'columns': {'log10(G_VRH)': 'Target variable. Base 10 '
                                                   'logarithm of the DFT '
                                                   'Voigt-Reuss-Hill average '
                                                   'shear moduli in GPa',
                                   'structure': 'Pymatgen Structure of the '
                                                'material.'},
                       'description': 'Matbench v0.1 test dataset for '
                                      'predicting DFT log10 VRH-average shear '
                                      'modulus from structure. Adapted from '
                                      'Materials Project database. Removed '
                                      'entries having a formation energy (or '
                                      'energy above the convex hull) more than '
                                      '150meV and those having negative '
                                      'G_Voigt, G_Reuss, G_VRH, K_Voigt, '
                                      'K_Reuss, or K_VRH and those failing '
                                      'G_Reuss <= G_VRH <= G_Voigt or K_Reuss '
                                      '<= K_VRH <= K_Voigt and those '
                                      'containing noble gases. Retrieved April '
                                      '2, 2019. For benchmarking w/ nested '
                                      'cross validation, the order of the '
                                      'dataset must be identical to the '
                                      'retrieved data; refer to the '
                                      'Automatminer/Matbench publication for '
                                      'more details.',
                       'file_type': 'json.gz',
                       'hash': '098af941f4c663270f1fe21abf20ffad6fb85ecbfcba5786ceac03983ac29da7',
                       'input_type': 'structure',
                       'n_samples': 10987,
                       'num_entries': 10987,
                       'reference': 'Jong, M. De, Chen, W., Angsten, T., Jain, '
                                    'A., Notestine, R., Gamst,\n'
                                    'A., Sluiter, M., Ande, C. K., Zwaag, S. '
                                    'Van Der, Plata, J. J., Toher,\n'
                                    'C., Curtarolo, S., Ceder, G., Persson, K. '
                                    'and Asta, M., "Charting\n'
                                    'the complete elastic properties of '
                                    'inorganic crystalline compounds",\n'
                                    'Scientific Data volume 2, Article number: '
                                    '150009 (2015)',
                       'target': 'log10(G_VRH)',
                       'task_type': 'regression',
                       'url': 'https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz'},
 'matbench_log_kvrh': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                       'author={Dunn, Alexander\n'
                                       'and Wang, Qi\n'
                                       'and Ganose, Alex\n'
                                       'and Dopp, Daniel\n'
                                       'and Jain, Anubhav},\n'
                                       'title={Benchmarking materials property '
                                       'prediction methods: the Matbench test '
                                       'set and Automatminer reference '
                                       'algorithm},\n'
                                       'journal={npj Computational '
                                       'Materials},\n'
                                       'year={2020},\n'
                                       'month={Sep},\n'
                                       'day={15},\n'
                                       'volume={6},\n'
                                       'number={1},\n'
                                       'pages={138},\n'
                                       'abstract={We present a benchmark test '
                                       'suite and an automated machine '
                                       'learning procedure for evaluating '
                                       'supervised machine learning (ML) '
                                       'models for predicting properties of '
                                       'inorganic bulk materials. The test '
                                       'suite, Matbench, is a set of '
                                       '13{\\thinspace}ML tasks that range in '
                                       'size from 312 to 132k samples and '
                                       'contain data from 10 density '
                                       'functional theory-derived and '
                                       'experimental sources. Tasks include '
                                       'predicting optical, thermal, '
                                       'electronic, thermodynamic, tensile, '
                                       'and elastic properties given a '
                                       "material's composition and/or crystal "
                                       'structure. The reference algorithm, '
                                       'Automatminer, is a highly-extensible, '
                                       'fully automated ML pipeline for '
                                       'predicting materials properties from '
                                       'materials primitives (such as '
                                       'composition and crystal structure) '
                                       'without user intervention or '
                                       'hyperparameter tuning. We test '
                                       'Automatminer on the Matbench test '
                                       'suite and compare its predictive power '
                                       'with state-of-the-art crystal graph '
                                       'neural networks and a traditional '
                                       'descriptor-based Random Forest model. '
                                       'We find Automatminer achieves the best '
                                       'performance on 8 of 13 tasks in the '
                                       'benchmark. We also show our test suite '
                                       'is capable of exposing predictive '
                                       'advantages of each algorithm---namely, '
                                       'that crystal graph methods appear to '
                                       'outperform traditional machine '
                                       'learning methods given '
                                       '{\\textasciitilde}104 or greater data '
                                       'points. We encourage evaluating '
                                       'materials ML algorithms on the '
                                       'Matbench benchmark and comparing them '
                                       'against the latest version of '
                                       'Automatminer.},\n'
                                       'issn={2057-3960},\n'
                                       'doi={10.1038/s41524-020-00406-3},\n'
                                       'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                       '}\n',
                                       '@Article{deJong2015,\n'
                                       'author={de Jong, Maarten and Chen, Wei '
                                       'and Angsten, Thomas\n'
                                       'and Jain, Anubhav and Notestine, Randy '
                                       'and Gamst, Anthony\n'
                                       'and Sluiter, Marcel and Krishna Ande, '
                                       'Chaitanya\n'
                                       'and van der Zwaag, Sybrand and Plata, '
                                       'Jose J. and Toher, Cormac\n'
                                       'and Curtarolo, Stefano and Ceder, '
                                       'Gerbrand and Persson, Kristin A.\n'
                                       'and Asta, Mark},\n'
                                       'title={Charting the complete elastic '
                                       'properties\n'
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
                       'columns': {'log10(K_VRH)': 'Target variable. Base 10 '
                                                   'logarithm of the DFT '
                                                   'Voigt-Reuss-Hill average '
                                                   'bulk moduli in GPa.',
                                   'structure': 'Pymatgen Structure of the '
                                                'material.'},
                       'description': 'Matbench v0.1 test dataset for '
                                      'predicting DFT log10 VRH-average bulk '
                                      'modulus from structure. Adapted from '
                                      'Materials Project database. Removed '
                                      'entries having a formation energy (or '
                                      'energy above the convex hull) more than '
                                      '150meV and those having negative '
                                      'G_Voigt, G_Reuss, G_VRH, K_Voigt, '
                                      'K_Reuss, or K_VRH and those failing '
                                      'G_Reuss <= G_VRH <= G_Voigt or K_Reuss '
                                      '<= K_VRH <= K_Voigt and those '
                                      'containing noble gases. Retrieved April '
                                      '2, 2019. For benchmarking w/ nested '
                                      'cross validation, the order of the '
                                      'dataset must be identical to the '
                                      'retrieved data; refer to the '
                                      'Automatminer/Matbench publication for '
                                      'more details.',
                       'file_type': 'json.gz',
                       'hash': '44b113ddb7e23aa18731a62c74afa7e5aa654199e0db5f951c8248a00955c9cd',
                       'input_type': 'structure',
                       'n_samples': 10987,
                       'num_entries': 10987,
                       'reference': 'Jong, M. De, Chen, W., Angsten, T., Jain, '
                                    'A., Notestine, R., Gamst,\n'
                                    'A., Sluiter, M., Ande, C. K., Zwaag, S. '
                                    'Van Der, Plata, J. J., Toher,\n'
                                    'C., Curtarolo, S., Ceder, G., Persson, K. '
                                    'and Asta, M., "Charting\n'
                                    'the complete elastic properties of '
                                    'inorganic crystalline compounds",\n'
                                    'Scientific Data volume 2, Article number: '
                                    '150009 (2015)',
                       'target': 'log10(K_VRH)',
                       'task_type': 'regression',
                       'url': 'https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz'},
 'matbench_mp_e_form': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                        'author={Dunn, Alexander\n'
                                        'and Wang, Qi\n'
                                        'and Ganose, Alex\n'
                                        'and Dopp, Daniel\n'
                                        'and Jain, Anubhav},\n'
                                        'title={Benchmarking materials '
                                        'property prediction methods: the '
                                        'Matbench test set and Automatminer '
                                        'reference algorithm},\n'
                                        'journal={npj Computational '
                                        'Materials},\n'
                                        'year={2020},\n'
                                        'month={Sep},\n'
                                        'day={15},\n'
                                        'volume={6},\n'
                                        'number={1},\n'
                                        'pages={138},\n'
                                        'abstract={We present a benchmark test '
                                        'suite and an automated machine '
                                        'learning procedure for evaluating '
                                        'supervised machine learning (ML) '
                                        'models for predicting properties of '
                                        'inorganic bulk materials. The test '
                                        'suite, Matbench, is a set of '
                                        '13{\\thinspace}ML tasks that range in '
                                        'size from 312 to 132k samples and '
                                        'contain data from 10 density '
                                        'functional theory-derived and '
                                        'experimental sources. Tasks include '
                                        'predicting optical, thermal, '
                                        'electronic, thermodynamic, tensile, '
                                        'and elastic properties given a '
                                        "material's composition and/or crystal "
                                        'structure. The reference algorithm, '
                                        'Automatminer, is a highly-extensible, '
                                        'fully automated ML pipeline for '
                                        'predicting materials properties from '
                                        'materials primitives (such as '
                                        'composition and crystal structure) '
                                        'without user intervention or '
                                        'hyperparameter tuning. We test '
                                        'Automatminer on the Matbench test '
                                        'suite and compare its predictive '
                                        'power with state-of-the-art crystal '
                                        'graph neural networks and a '
                                        'traditional descriptor-based Random '
                                        'Forest model. We find Automatminer '
                                        'achieves the best performance on 8 of '
                                        '13 tasks in the benchmark. We also '
                                        'show our test suite is capable of '
                                        'exposing predictive advantages of '
                                        'each algorithm---namely, that crystal '
                                        'graph methods appear to outperform '
                                        'traditional machine learning methods '
                                        'given {\\textasciitilde}104 or '
                                        'greater data points. We encourage '
                                        'evaluating materials ML algorithms on '
                                        'the Matbench benchmark and comparing '
                                        'them against the latest version of '
                                        'Automatminer.},\n'
                                        'issn={2057-3960},\n'
                                        'doi={10.1038/s41524-020-00406-3},\n'
                                        'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                        '}\n',
                                        '@article{Jain2013,\n'
                                        'author = {Jain, Anubhav and Ong, '
                                        'Shyue Ping and Hautier, Geoffroy and '
                                        'Chen, Wei and Richards, William '
                                        'Davidson and Dacek, Stephen and '
                                        'Cholia, Shreyas and Gunter, Dan and '
                                        'Skinner, David and Ceder, Gerbrand '
                                        'and Persson, Kristin a.},\n'
                                        'doi = {10.1063/1.4812323},\n'
                                        'issn = {2166532X},\n'
                                        'journal = {APL Materials},\n'
                                        'number = {1},\n'
                                        'pages = {011002},\n'
                                        'title = {{The Materials Project: A '
                                        'materials genome approach to '
                                        'accelerating materials innovation}},\n'
                                        'url = '
                                        '{http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\\&Agg=doi},\n'
                                        'volume = {1},\n'
                                        'year = {2013}\n'
                                        '}'],
                        'columns': {'e_form': 'Target variable. Formation '
                                              'energy in eV as calculated by '
                                              'the Materials Project.',
                                    'structure': 'Pymatgen Structure of the '
                                                 'material.'},
                        'description': 'Matbench v0.1 test dataset for '
                                       'predicting DFT formation energy from '
                                       'structure. Adapted from Materials '
                                       'Project database. Removed entries '
                                       'having formation energy more than '
                                       '3.0eV and those containing noble '
                                       'gases. Retrieved April 2, 2019. For '
                                       'benchmarking w/ nested cross '
                                       'validation, the order of the dataset '
                                       'must be identical to the retrieved '
                                       'data; refer to the '
                                       'Automatminer/Matbench publication for '
                                       'more details.',
                        'file_type': 'json.gz',
                        'hash': 'dedcb1d4ba2e3e50dbdd45ba5bc647a00e9c2bcf8f8bf556dc8e92caa39eb21f',
                        'input_type': 'structure',
                        'n_samples': 132752,
                        'num_entries': 132752,
                        'reference': 'A. Jain*, S.P. Ong*, G. Hautier, W. '
                                     'Chen, W.D. Richards, S. Dacek, S. '
                                     'Cholia, D. Gunter, D. Skinner, G. Ceder, '
                                     'K.A. Persson (*=equal contributions)\n'
                                     'The Materials Project: A materials '
                                     'genome approach to accelerating '
                                     'materials innovation\n'
                                     'APL Materials, 2013, 1(1), 011002.\n'
                                     'doi:10.1063/1.4812323',
                        'target': 'e_form',
                        'task_type': 'regression',
                        'url': 'https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz'},
 'matbench_mp_gap': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                     'author={Dunn, Alexander\n'
                                     'and Wang, Qi\n'
                                     'and Ganose, Alex\n'
                                     'and Dopp, Daniel\n'
                                     'and Jain, Anubhav},\n'
                                     'title={Benchmarking materials property '
                                     'prediction methods: the Matbench test '
                                     'set and Automatminer reference '
                                     'algorithm},\n'
                                     'journal={npj Computational Materials},\n'
                                     'year={2020},\n'
                                     'month={Sep},\n'
                                     'day={15},\n'
                                     'volume={6},\n'
                                     'number={1},\n'
                                     'pages={138},\n'
                                     'abstract={We present a benchmark test '
                                     'suite and an automated machine learning '
                                     'procedure for evaluating supervised '
                                     'machine learning (ML) models for '
                                     'predicting properties of inorganic bulk '
                                     'materials. The test suite, Matbench, is '
                                     'a set of 13{\\thinspace}ML tasks that '
                                     'range in size from 312 to 132k samples '
                                     'and contain data from 10 density '
                                     'functional theory-derived and '
                                     'experimental sources. Tasks include '
                                     'predicting optical, thermal, electronic, '
                                     'thermodynamic, tensile, and elastic '
                                     "properties given a material's "
                                     'composition and/or crystal structure. '
                                     'The reference algorithm, Automatminer, '
                                     'is a highly-extensible, fully automated '
                                     'ML pipeline for predicting materials '
                                     'properties from materials primitives '
                                     '(such as composition and crystal '
                                     'structure) without user intervention or '
                                     'hyperparameter tuning. We test '
                                     'Automatminer on the Matbench test suite '
                                     'and compare its predictive power with '
                                     'state-of-the-art crystal graph neural '
                                     'networks and a traditional '
                                     'descriptor-based Random Forest model. We '
                                     'find Automatminer achieves the best '
                                     'performance on 8 of 13 tasks in the '
                                     'benchmark. We also show our test suite '
                                     'is capable of exposing predictive '
                                     'advantages of each algorithm---namely, '
                                     'that crystal graph methods appear to '
                                     'outperform traditional machine learning '
                                     'methods given {\\textasciitilde}104 or '
                                     'greater data points. We encourage '
                                     'evaluating materials ML algorithms on '
                                     'the Matbench benchmark and comparing '
                                     'them against the latest version of '
                                     'Automatminer.},\n'
                                     'issn={2057-3960},\n'
                                     'doi={10.1038/s41524-020-00406-3},\n'
                                     'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                     '}\n',
                                     '@article{Jain2013,\n'
                                     'author = {Jain, Anubhav and Ong, Shyue '
                                     'Ping and Hautier, Geoffroy and Chen, Wei '
                                     'and Richards, William Davidson and '
                                     'Dacek, Stephen and Cholia, Shreyas and '
                                     'Gunter, Dan and Skinner, David and '
                                     'Ceder, Gerbrand and Persson, Kristin '
                                     'a.},\n'
                                     'doi = {10.1063/1.4812323},\n'
                                     'issn = {2166532X},\n'
                                     'journal = {APL Materials},\n'
                                     'number = {1},\n'
                                     'pages = {011002},\n'
                                     'title = {{The Materials Project: A '
                                     'materials genome approach to '
                                     'accelerating materials innovation}},\n'
                                     'url = '
                                     '{http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\\&Agg=doi},\n'
                                     'volume = {1},\n'
                                     'year = {2013}\n'
                                     '}'],
                     'columns': {'gap pbe': 'Target variable. The band gap as '
                                            'calculated by PBE DFT from the '
                                            'Materials Project, in eV.',
                                 'structure': 'Pymatgen Structure of the '
                                              'material.'},
                     'description': 'Matbench v0.1 test dataset for predicting '
                                    'DFT PBE band gap from structure. Adapted '
                                    'from Materials Project database. Removed '
                                    'entries having a formation energy (or '
                                    'energy above the convex hull) more than '
                                    '150meV and those containing noble gases. '
                                    'Retrieved April 2, 2019. For benchmarking '
                                    'w/ nested cross validation, the order of '
                                    'the dataset must be identical to the '
                                    'retrieved data; refer to the '
                                    'Automatminer/Matbench publication for '
                                    'more details.',
                     'file_type': 'json.gz',
                     'hash': '58b65746bd88329986ed66031a2ac1369c7c522f7bc9f9081528e07097c2c057',
                     'input_type': 'structure',
                     'n_samples': 106113,
                     'num_entries': 106113,
                     'reference': 'A. Jain*, S.P. Ong*, G. Hautier, W. Chen, '
                                  'W.D. Richards, S. Dacek, S. Cholia, D. '
                                  'Gunter, D. Skinner, G. Ceder, K.A. Persson '
                                  '(*=equal contributions)\n'
                                  'The Materials Project: A materials genome '
                                  'approach to accelerating materials '
                                  'innovation\n'
                                  'APL Materials, 2013, 1(1), 011002.\n'
                                  'doi:10.1063/1.4812323',
                     'target': 'gap pbe',
                     'task_type': 'regression',
                     'url': 'https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz'},
 'matbench_mp_is_metal': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                          'author={Dunn, Alexander\n'
                                          'and Wang, Qi\n'
                                          'and Ganose, Alex\n'
                                          'and Dopp, Daniel\n'
                                          'and Jain, Anubhav},\n'
                                          'title={Benchmarking materials '
                                          'property prediction methods: the '
                                          'Matbench test set and Automatminer '
                                          'reference algorithm},\n'
                                          'journal={npj Computational '
                                          'Materials},\n'
                                          'year={2020},\n'
                                          'month={Sep},\n'
                                          'day={15},\n'
                                          'volume={6},\n'
                                          'number={1},\n'
                                          'pages={138},\n'
                                          'abstract={We present a benchmark '
                                          'test suite and an automated machine '
                                          'learning procedure for evaluating '
                                          'supervised machine learning (ML) '
                                          'models for predicting properties of '
                                          'inorganic bulk materials. The test '
                                          'suite, Matbench, is a set of '
                                          '13{\\thinspace}ML tasks that range '
                                          'in size from 312 to 132k samples '
                                          'and contain data from 10 density '
                                          'functional theory-derived and '
                                          'experimental sources. Tasks include '
                                          'predicting optical, thermal, '
                                          'electronic, thermodynamic, tensile, '
                                          'and elastic properties given a '
                                          "material's composition and/or "
                                          'crystal structure. The reference '
                                          'algorithm, Automatminer, is a '
                                          'highly-extensible, fully automated '
                                          'ML pipeline for predicting '
                                          'materials properties from materials '
                                          'primitives (such as composition and '
                                          'crystal structure) without user '
                                          'intervention or hyperparameter '
                                          'tuning. We test Automatminer on the '
                                          'Matbench test suite and compare its '
                                          'predictive power with '
                                          'state-of-the-art crystal graph '
                                          'neural networks and a traditional '
                                          'descriptor-based Random Forest '
                                          'model. We find Automatminer '
                                          'achieves the best performance on 8 '
                                          'of 13 tasks in the benchmark. We '
                                          'also show our test suite is capable '
                                          'of exposing predictive advantages '
                                          'of each algorithm---namely, that '
                                          'crystal graph methods appear to '
                                          'outperform traditional machine '
                                          'learning methods given '
                                          '{\\textasciitilde}104 or greater '
                                          'data points. We encourage '
                                          'evaluating materials ML algorithms '
                                          'on the Matbench benchmark and '
                                          'comparing them against the latest '
                                          'version of Automatminer.},\n'
                                          'issn={2057-3960},\n'
                                          'doi={10.1038/s41524-020-00406-3},\n'
                                          'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                          '}\n',
                                          '@article{Jain2013,\n'
                                          'author = {Jain, Anubhav and Ong, '
                                          'Shyue Ping and Hautier, Geoffroy '
                                          'and Chen, Wei and Richards, William '
                                          'Davidson and Dacek, Stephen and '
                                          'Cholia, Shreyas and Gunter, Dan and '
                                          'Skinner, David and Ceder, Gerbrand '
                                          'and Persson, Kristin a.},\n'
                                          'doi = {10.1063/1.4812323},\n'
                                          'issn = {2166532X},\n'
                                          'journal = {APL Materials},\n'
                                          'number = {1},\n'
                                          'pages = {011002},\n'
                                          'title = {{The Materials Project: A '
                                          'materials genome approach to '
                                          'accelerating materials '
                                          'innovation}},\n'
                                          'url = '
                                          '{http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\\&Agg=doi},\n'
                                          'volume = {1},\n'
                                          'year = {2013}\n'
                                          '}'],
                          'columns': {'is_metal': 'Target variable. 1 if the '
                                                  'compound is a metal, 0 if '
                                                  'the compound is not a '
                                                  'metal. Metallicity '
                                                  'determined with pymatgen',
                                      'structure': 'Pymatgen Structure of the '
                                                   'material.'},
                          'description': 'Matbench v0.1 test dataset for '
                                         'predicting DFT metallicity from '
                                         'structure. Adapted from Materials '
                                         'Project database. Removed entries '
                                         'having a formation energy (or energy '
                                         'above the convex hull) more than '
                                         '150meV and those containing noble '
                                         'gases. Retrieved April 2, 2019. For '
                                         'benchmarking w/ nested cross '
                                         'validation, the order of the dataset '
                                         'must be identical to the retrieved '
                                         'data; refer to the '
                                         'Automatminer/Matbench publication '
                                         'for more details.',
                          'file_type': 'json.gz',
                          'hash': '9a028ed5750a4c76ca36e9f3c8d48fe0bf3fb21b76ec2289e58ae7048d527919',
                          'input_type': 'structure',
                          'n_samples': 106113,
                          'num_entries': 106113,
                          'reference': 'A. Jain*, S.P. Ong*, G. Hautier, W. '
                                       'Chen, W.D. Richards, S. Dacek, S. '
                                       'Cholia, D. Gunter, D. Skinner, G. '
                                       'Ceder, K.A. Persson (*=equal '
                                       'contributions)\n'
                                       'The Materials Project: A materials '
                                       'genome approach to accelerating '
                                       'materials innovation\n'
                                       'APL Materials, 2013, 1(1), 011002.\n'
                                       'doi:10.1063/1.4812323',
                          'target': 'is_metal',
                          'task_type': 'classification',
                          'url': 'https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz'},
 'matbench_perovskites': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                          'author={Dunn, Alexander\n'
                                          'and Wang, Qi\n'
                                          'and Ganose, Alex\n'
                                          'and Dopp, Daniel\n'
                                          'and Jain, Anubhav},\n'
                                          'title={Benchmarking materials '
                                          'property prediction methods: the '
                                          'Matbench test set and Automatminer '
                                          'reference algorithm},\n'
                                          'journal={npj Computational '
                                          'Materials},\n'
                                          'year={2020},\n'
                                          'month={Sep},\n'
                                          'day={15},\n'
                                          'volume={6},\n'
                                          'number={1},\n'
                                          'pages={138},\n'
                                          'abstract={We present a benchmark '
                                          'test suite and an automated machine '
                                          'learning procedure for evaluating '
                                          'supervised machine learning (ML) '
                                          'models for predicting properties of '
                                          'inorganic bulk materials. The test '
                                          'suite, Matbench, is a set of '
                                          '13{\\thinspace}ML tasks that range '
                                          'in size from 312 to 132k samples '
                                          'and contain data from 10 density '
                                          'functional theory-derived and '
                                          'experimental sources. Tasks include '
                                          'predicting optical, thermal, '
                                          'electronic, thermodynamic, tensile, '
                                          'and elastic properties given a '
                                          "material's composition and/or "
                                          'crystal structure. The reference '
                                          'algorithm, Automatminer, is a '
                                          'highly-extensible, fully automated '
                                          'ML pipeline for predicting '
                                          'materials properties from materials '
                                          'primitives (such as composition and '
                                          'crystal structure) without user '
                                          'intervention or hyperparameter '
                                          'tuning. We test Automatminer on the '
                                          'Matbench test suite and compare its '
                                          'predictive power with '
                                          'state-of-the-art crystal graph '
                                          'neural networks and a traditional '
                                          'descriptor-based Random Forest '
                                          'model. We find Automatminer '
                                          'achieves the best performance on 8 '
                                          'of 13 tasks in the benchmark. We '
                                          'also show our test suite is capable '
                                          'of exposing predictive advantages '
                                          'of each algorithm---namely, that '
                                          'crystal graph methods appear to '
                                          'outperform traditional machine '
                                          'learning methods given '
                                          '{\\textasciitilde}104 or greater '
                                          'data points. We encourage '
                                          'evaluating materials ML algorithms '
                                          'on the Matbench benchmark and '
                                          'comparing them against the latest '
                                          'version of Automatminer.},\n'
                                          'issn={2057-3960},\n'
                                          'doi={10.1038/s41524-020-00406-3},\n'
                                          'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                          '}\n',
                                          '@Article{C2EE22341D,\n'
                                          'author ="Castelli, Ivano E. and '
                                          'Landis, David D. and Thygesen, '
                                          'Kristian S. and Dahl, Søren and '
                                          'Chorkendorff, Ib and Jaramillo, '
                                          'Thomas F. and Jacobsen, Karsten '
                                          'W.",\n'
                                          'title  ="New cubic perovskites for '
                                          'one- and two-photon water splitting '
                                          'using the computational materials '
                                          'repository",\n'
                                          'journal  ="Energy Environ. Sci.",\n'
                                          'year  ="2012",\n'
                                          'volume  ="5",\n'
                                          'issue  ="10",\n'
                                          'pages  ="9034-9043",\n'
                                          'publisher  ="The Royal Society of '
                                          'Chemistry",\n'
                                          'doi  ="10.1039/C2EE22341D",\n'
                                          'url  '
                                          '="http://dx.doi.org/10.1039/C2EE22341D",\n'
                                          'abstract  ="A new efficient '
                                          'photoelectrochemical cell (PEC) is '
                                          'one of the possible solutions to '
                                          'the energy and climate problems of '
                                          'our time. Such a device requires '
                                          'development of new semiconducting '
                                          'materials with tailored properties '
                                          'with respect to stability and light '
                                          'absorption. Here we perform '
                                          'computational screening of around '
                                          '19\u2009000 oxides{,} '
                                          'oxynitrides{,} oxysulfides{,} '
                                          'oxyfluorides{,} and '
                                          'oxyfluoronitrides in the cubic '
                                          'perovskite structure with PEC '
                                          'applications in mind. We address '
                                          'three main applications: light '
                                          'absorbers for one- and two-photon '
                                          'water splitting and high-stability '
                                          'transparent shields to protect '
                                          'against corrosion. We end up with '
                                          '20{,} 12{,} and 15 different '
                                          'combinations of oxides{,} '
                                          'oxynitrides and oxyfluorides{,} '
                                          'respectively{,} inviting further '
                                          'experimental investigation."}'],
                          'columns': {'e_form': 'Target variable. Heat of '
                                                'formation of the entire '
                                                '5-atom perovskite cell, in eV '
                                                'as calculated by RPBE '
                                                'GGA-DFT. Note the reference '
                                                'state for oxygen was computed '
                                                "from oxygen's chemical "
                                                'potential in water vapor, not '
                                                'as oxygen molecules, to '
                                                'reflect the application which '
                                                'these perovskites were '
                                                'studied for.',
                                      'structure': 'Pymatgen Structure of the '
                                                   'material.'},
                          'description': 'Matbench v0.1 test dataset for '
                                         'predicting formation energy from '
                                         'crystal structure. Adapted from an '
                                         'original dataset generated by '
                                         'Castelli et al. For benchmarking w/ '
                                         'nested cross validation, the order '
                                         'of the dataset must be identical to '
                                         'the retrieved data; refer to the '
                                         'Automatminer/Matbench publication '
                                         'for more details.',
                          'file_type': 'json.gz',
                          'hash': '4641e2417f8ec8b50096d2230864468dfa08278dc9d257c327f65d0305278483',
                          'input_type': 'structure',
                          'n_samples': 18928,
                          'num_entries': 18928,
                          'reference': 'Ivano E. Castelli, David D. Landis, '
                                       'Kristian S. Thygesen, Søren Dahl, Ib '
                                       'Chorkendorff, Thomas F. Jaramillo and '
                                       'Karsten W. Jacobsen (2012) New cubic '
                                       'perovskites for one- and two-photon '
                                       'water splitting using the '
                                       'computational materials repository. '
                                       'Energy Environ. Sci., 2012,5, '
                                       '9034-9043 '
                                       'https://doi.org/10.1039/C2EE22341D',
                          'target': 'e_form',
                          'task_type': 'regression',
                          'url': 'https://ml.materialsproject.org/projects/matbench_perovskites.json.gz'},
 'matbench_phonons': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                      'author={Dunn, Alexander\n'
                                      'and Wang, Qi\n'
                                      'and Ganose, Alex\n'
                                      'and Dopp, Daniel\n'
                                      'and Jain, Anubhav},\n'
                                      'title={Benchmarking materials property '
                                      'prediction methods: the Matbench test '
                                      'set and Automatminer reference '
                                      'algorithm},\n'
                                      'journal={npj Computational Materials},\n'
                                      'year={2020},\n'
                                      'month={Sep},\n'
                                      'day={15},\n'
                                      'volume={6},\n'
                                      'number={1},\n'
                                      'pages={138},\n'
                                      'abstract={We present a benchmark test '
                                      'suite and an automated machine learning '
                                      'procedure for evaluating supervised '
                                      'machine learning (ML) models for '
                                      'predicting properties of inorganic bulk '
                                      'materials. The test suite, Matbench, is '
                                      'a set of 13{\\thinspace}ML tasks that '
                                      'range in size from 312 to 132k samples '
                                      'and contain data from 10 density '
                                      'functional theory-derived and '
                                      'experimental sources. Tasks include '
                                      'predicting optical, thermal, '
                                      'electronic, thermodynamic, tensile, and '
                                      "elastic properties given a material's "
                                      'composition and/or crystal structure. '
                                      'The reference algorithm, Automatminer, '
                                      'is a highly-extensible, fully automated '
                                      'ML pipeline for predicting materials '
                                      'properties from materials primitives '
                                      '(such as composition and crystal '
                                      'structure) without user intervention or '
                                      'hyperparameter tuning. We test '
                                      'Automatminer on the Matbench test suite '
                                      'and compare its predictive power with '
                                      'state-of-the-art crystal graph neural '
                                      'networks and a traditional '
                                      'descriptor-based Random Forest model. '
                                      'We find Automatminer achieves the best '
                                      'performance on 8 of 13 tasks in the '
                                      'benchmark. We also show our test suite '
                                      'is capable of exposing predictive '
                                      'advantages of each algorithm---namely, '
                                      'that crystal graph methods appear to '
                                      'outperform traditional machine learning '
                                      'methods given {\\textasciitilde}104 or '
                                      'greater data points. We encourage '
                                      'evaluating materials ML algorithms on '
                                      'the Matbench benchmark and comparing '
                                      'them against the latest version of '
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
                                      'title={High-throughput '
                                      'density-functional perturbation theory '
                                      'phonons for inorganic materials},\n'
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
                                      'setten_gonze_persson_hautier_2018, '
                                      'title={High-throughput '
                                      'Density-Functional Perturbation Theory '
                                      'phonons for inorganic materials}, '
                                      'url={https://figshare.com/collections/High-throughput_Density-Functional_Perturbation_Theory_phonons_for_inorganic_materials/3938023/1}, '
                                      'DOI={10.6084/m9.figshare.c.3938023.v1}, '
                                      'abstractNote={The knowledge of the '
                                      'vibrational properties of a material is '
                                      'of key importance to understand '
                                      'physical phenomena such as thermal '
                                      'conductivity, superconductivity, and '
                                      'ferroelectricity among others. However, '
                                      'detailed experimental phonon spectra '
                                      'are available only for a limited number '
                                      'of materials which hinders the '
                                      'large-scale analysis of vibrational '
                                      'properties and their derived '
                                      'quantities. In this work, we perform ab '
                                      'initio calculations of the full phonon '
                                      'dispersion and vibrational density of '
                                      'states for 1521 semiconductor compounds '
                                      'in the harmonic approximation based on '
                                      'density functional perturbation theory. '
                                      'The data is collected along with '
                                      'derived dielectric and thermodynamic '
                                      'properties. We present the procedure '
                                      'used to obtain the results, the details '
                                      'of the provided database and a '
                                      'validation based on the comparison with '
                                      'experimental data.}, '
                                      'publisher={figshare}, author={Petretto, '
                                      'Guido and Dwaraknath, Shyam and '
                                      'Miranda, Henrique P. C. and Winston, '
                                      'Donald and Giantomassi, Matteo and '
                                      'Rignanese, Gian-Marco and Van Setten, '
                                      'Michiel J. and Gonze, Xavier and '
                                      'Persson, Kristin A and Hautier, '
                                      'Geoffroy}, year={2018}, month={Apr}}'],
                      'columns': {'last phdos peak': 'Target variable. '
                                                     'Frequency of the highest '
                                                     'frequency optical phonon '
                                                     'mode peak, in units of '
                                                     '1/cm; ; may be used as '
                                                     'an estimation of '
                                                     'dominant longitudinal '
                                                     'optical phonon '
                                                     'frequency.',
                                  'structure': 'Pymatgen Structure of the '
                                               'material.'},
                      'description': 'Matbench v0.1 test dataset for '
                                     'predicting vibration properties from '
                                     'crystal structure. Original data '
                                     'retrieved from Petretto et al. Original '
                                     'calculations done via ABINIT in the '
                                     'harmonic approximation based on density '
                                     'functional perturbation theory. Removed '
                                     'entries having a formation energy (or '
                                     'energy above the convex hull) more than '
                                     '150meV. For benchmarking w/ nested cross '
                                     'validation, the order of the dataset '
                                     'must be identical to the retrieved data; '
                                     'refer to the Automatminer/Matbench '
                                     'publication for more details.',
                      'file_type': 'json.gz',
                      'hash': '4db551f21ec5f577e6202725f10e34dfc509aa7df3a6bdaac497da7f6dbbb9b3',
                      'input_type': 'structure',
                      'n_samples': 1265,
                      'num_entries': 1265,
                      'reference': 'Petretto, G. et al. High-throughput '
                                   'density functional perturbation theory '
                                   'phonons for inorganic materials. Sci. Data '
                                   '5:180065 doi: 10.1038/sdata.2018.65 '
                                   '(2018).\n'
                                   'Petretto, G. et al. High-throughput '
                                   'density functional perturbation theory '
                                   'phonons for inorganic materials. (2018). '
                                   'figshare. Collection.',
                      'target': 'last phdos peak',
                      'task_type': 'regression',
                      'url': 'https://ml.materialsproject.org/projects/matbench_phonons.json.gz'},
 'matbench_steels': {'bibtex_refs': ['@Article{Dunn2020,\n'
                                     'author={Dunn, Alexander\n'
                                     'and Wang, Qi\n'
                                     'and Ganose, Alex\n'
                                     'and Dopp, Daniel\n'
                                     'and Jain, Anubhav},\n'
                                     'title={Benchmarking materials property '
                                     'prediction methods: the Matbench test '
                                     'set and Automatminer reference '
                                     'algorithm},\n'
                                     'journal={npj Computational Materials},\n'
                                     'year={2020},\n'
                                     'month={Sep},\n'
                                     'day={15},\n'
                                     'volume={6},\n'
                                     'number={1},\n'
                                     'pages={138},\n'
                                     'abstract={We present a benchmark test '
                                     'suite and an automated machine learning '
                                     'procedure for evaluating supervised '
                                     'machine learning (ML) models for '
                                     'predicting properties of inorganic bulk '
                                     'materials. The test suite, Matbench, is '
                                     'a set of 13{\\thinspace}ML tasks that '
                                     'range in size from 312 to 132k samples '
                                     'and contain data from 10 density '
                                     'functional theory-derived and '
                                     'experimental sources. Tasks include '
                                     'predicting optical, thermal, electronic, '
                                     'thermodynamic, tensile, and elastic '
                                     "properties given a material's "
                                     'composition and/or crystal structure. '
                                     'The reference algorithm, Automatminer, '
                                     'is a highly-extensible, fully automated '
                                     'ML pipeline for predicting materials '
                                     'properties from materials primitives '
                                     '(such as composition and crystal '
                                     'structure) without user intervention or '
                                     'hyperparameter tuning. We test '
                                     'Automatminer on the Matbench test suite '
                                     'and compare its predictive power with '
                                     'state-of-the-art crystal graph neural '
                                     'networks and a traditional '
                                     'descriptor-based Random Forest model. We '
                                     'find Automatminer achieves the best '
                                     'performance on 8 of 13 tasks in the '
                                     'benchmark. We also show our test suite '
                                     'is capable of exposing predictive '
                                     'advantages of each algorithm---namely, '
                                     'that crystal graph methods appear to '
                                     'outperform traditional machine learning '
                                     'methods given {\\textasciitilde}104 or '
                                     'greater data points. We encourage '
                                     'evaluating materials ML algorithms on '
                                     'the Matbench benchmark and comparing '
                                     'them against the latest version of '
                                     'Automatminer.},\n'
                                     'issn={2057-3960},\n'
                                     'doi={10.1038/s41524-020-00406-3},\n'
                                     'url={https://doi.org/10.1038/s41524-020-00406-3}\n'
                                     '}\n',
                                     '@misc{Citrine Informatics,\n'
                                     'title = {Mechanical properties of some '
                                     'steels},\n'
                                     'howpublished = '
                                     '{\\url{https://citrination.com/datasets/153092/},\n'
                                     '}'],
                     'columns': {'composition': 'Chemical formula.',
                                 'yield strength': 'Target variable. '
                                                   'Experimentally measured '
                                                   'steel yield strengths, in '
                                                   'MPa.'},
                     'description': 'Matbench v0.1 test dataset for predicting '
                                    'steel yield strengths from chemical '
                                    'composition alone. Retrieved from Citrine '
                                    'informatics. Deduplicated. For '
                                    'benchmarking w/ nested cross validation, '
                                    'the order of the dataset must be '
                                    'identical to the retrieved data; refer to '
                                    'the Automatminer/Matbench publication for '
                                    'more details.',
                     'file_type': 'json.gz',
                     'hash': '473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252',
                     'input_type': 'composition',
                     'n_samples': 312,
                     'num_entries': 312,
                     'reference': 'https://citrination.com/datasets/153092/',
                     'target': 'yield strength',
                     'task_type': 'regression',
                     'url': 'https://ml.materialsproject.org/projects/matbench_steels.json.gz'}}
```

### Metadata:

Tasks recorded: 13 of 13 total

Benchmark is complete? True

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 29.7909| 34.4495| 14.4945| 59.9166 |
 | fold_1 | 29.7190| 34.3804| 14.4187| 60.1932 |
 | fold_2 | 26.5076| 30.7072| 12.9281| 53.9821 |
 | fold_3 | 29.7641| 34.4020| 14.5676| 60.1966 |
 | fold_4 | 29.6901| 34.3323| 14.4379| 59.9926 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 29.0944 | 29.7909 | 26.5076 | 1.2938 |
| rmse | 33.6543 | 34.4495 | 30.7072 | 1.4740 |
| mape* | 14.1694 | 14.5676 | 12.9281 | 0.6228 |
| max_error | 58.8562 | 60.1966 | 53.9821 | 2.4396 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 5.1763| 6.1110| 13625441014167962.0000| 11.6227 |
 | fold_1 | 5.1436| 6.0772| 13810429964644752.0000| 11.6137 |
 | fold_2 | 4.6299| 5.4703| 12259552001352658.0000| 10.4897 |
 | fold_3 | 5.2499| 6.1483| 14348907152943880.0000| 11.6221 |
 | fold_4 | 5.2903| 6.2265| 15276180519639252.0000| 11.6885 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 5.0980 | 5.2903 | 4.6299 | 0.2398 |
| rmse | 6.0066 | 6.2265 | 5.4703 | 0.2727 |
| mape* | 13864102130549700.0000 | 15276180519639252.0000 | 12259552001352658.0000 | 986247659935790.7500 |
| max_error | 11.4073 | 11.6885 | 10.4897 | 0.4596 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_expt_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.4904| 0.4905| 0.5107| 0.4905 |
 | fold_1 | 0.4746| 0.4748| 0.4956| 0.4748 |
 | fold_2 | 0.4766| 0.4768| 0.4976| 0.4768 |
 | fold_3 | 0.5051| 0.5053| 0.5249| 0.5053 |
 | fold_4 | 0.5051| 0.5053| 0.5249| 0.5053 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.4903 | 0.5051 | 0.4746 | 0.0132 |
| balanced_accuracy | 0.4905 | 0.5053 | 0.4748 | 0.0132 |
| f1 | 0.5107 | 0.5249 | 0.4956 | 0.0127 |
| rocauc | 0.4905 | 0.5053 | 0.4748 | 0.0132 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_glass`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.5246| 0.5142| 0.6170| 0.5142 |
 | fold_1 | 0.5018| 0.4864| 0.5986| 0.4864 |
 | fold_2 | 0.5282| 0.5185| 0.6199| 0.5185 |
 | fold_3 | 0.4982| 0.4821| 0.5957| 0.4821 |
 | fold_4 | 0.4771| 0.4564| 0.5787| 0.4564 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.5060 | 0.5282 | 0.4771 | 0.0187 |
| balanced_accuracy | 0.4915 | 0.5185 | 0.4564 | 0.0227 |
| f1 | 0.6020 | 0.6199 | 0.5787 | 0.0151 |
| rocauc | 0.4915 | 0.5185 | 0.4564 | 0.0227 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 1.9807| 2.3772| 6236081301418.7900| 7.0580 |
 | fold_1 | 1.9764| 2.3723| 7492035787402.2129| 6.8782 |
 | fold_2 | 1.9803| 2.3769| 7242730732646.4551| 7.0237 |
 | fold_3 | 1.9799| 2.3763| 7343189759663.5918| 6.9474 |
 | fold_4 | 1.9820| 2.3795| 6631518929028.7598| 6.9178 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 1.9799 | 1.9820 | 1.9764 | 0.0019 |
| rmse | 2.3764 | 2.3795 | 2.3723 | 0.0023 |
| mape* | 6989111302031.9629 | 7492035787402.2129 | 6236081301418.7900 | 476980899991.2849 |
| max_error | 6.9650 | 7.0580 | 6.8782 | 0.0666 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 662.8352| 800.2585| 22.1865| 1517.1273 |
 | fold_1 | 658.8068| 799.4757| 9.7367| 1497.7263 |
 | fold_2 | 661.6956| 795.2903| 12.0728| 1532.9113 |
 | fold_3 | 656.8728| 802.8514| 9.8176| 1500.2219 |
 | fold_4 | 484.0870| 575.4212| 9.6424| 1229.7022 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 624.8595 | 662.8352 | 484.0870 | 70.4176 |
| rmse | 754.6594 | 802.8514 | 575.4212 | 89.6520 |
| mape* | 12.6912 | 22.1865 | 9.6424 | 4.8337 |
| max_error | 1455.5378 | 1532.9113 | 1229.7022 | 113.6294 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.7488| 0.8935| 12968684140533.3398| 2.2552 |
 | fold_1 | 0.7567| 0.8965| 26158506009539.2930| 2.7078 |
 | fold_2 | 0.7458| 0.8929| 14369930256302.7109| 2.5169 |
 | fold_3 | 0.7535| 0.8922| 17257620777276.5000| 2.4211 |
 | fold_4 | 0.7467| 0.8860| 4541885118479.4883| 2.2460 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.7503 | 0.7567 | 0.7458 | 0.0042 |
| rmse | 0.8922 | 0.8965 | 0.8860 | 0.0034 |
| mape* | 15059325260426.2656 | 26158506009539.2930 | 4541885118479.4883 | 6978350942510.9336 |
| max_error | 2.4294 | 2.7078 | 2.2460 | 0.1728 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.8401| 1.0191| 10062434398435.7734| 2.4510 |
 | fold_1 | 0.8315| 1.0106| 6855080726655.0957| 2.7539 |
 | fold_2 | 0.8309| 1.0128| 5144639056488.2217| 2.4563 |
 | fold_3 | 0.8409| 1.0151| 2416033931702.7036| 2.4628 |
 | fold_4 | 0.8252| 1.0039| 1547244178802.9026| 2.4978 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.8337 | 0.8409 | 0.8252 | 0.0060 |
| rmse | 1.0123 | 1.0191 | 1.0039 | 0.0051 |
| mape* | 5205086458416.9395 | 10062434398435.7734 | 1547244178802.9026 | 3081512230166.4482 |
| max_error | 2.5244 | 2.7539 | 2.4510 | 0.1159 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 4.0239| 4.8413| 9342751963003018.0000| 9.7212 |
 | fold_1 | 4.0403| 4.8517| 9530024251770120.0000| 9.7206 |
 | fold_2 | 4.0352| 4.8529| 9476134144859454.0000| 9.7206 |
 | fold_3 | 4.0323| 4.8456| 9514523573590888.0000| 9.7204 |
 | fold_4 | 3.8420| 4.6214| 9017068673849420.0000| 9.3264 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 3.9947 | 4.0403 | 3.8420 | 0.0766 |
| rmse | 4.8026 | 4.8529 | 4.6214 | 0.0907 |
| mape* | 9376100521414580.0000 | 9530024251770120.0000 | 9017068673849420.0000 | 191246529837308.3438 |
| max_error | 9.6418 | 9.7212 | 9.3264 | 0.1577 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.4983| 0.4985| 0.4645| 0.4985 |
 | fold_1 | 0.4950| 0.4952| 0.4610| 0.4952 |
 | fold_2 | 0.5019| 0.5022| 0.4684| 0.5022 |
 | fold_3 | 0.4980| 0.4982| 0.4642| 0.4982 |
 | fold_4 | 0.5033| 0.5036| 0.4698| 0.5036 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.4993 | 0.5033 | 0.4950 | 0.0030 |
| balanced_accuracy | 0.4995 | 0.5036 | 0.4952 | 0.0030 |
| f1 | 0.4656 | 0.4698 | 0.4610 | 0.0032 |
| rocauc | 0.4995 | 0.5036 | 0.4952 | 0.0030 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 1.6611| 2.0069| 17109693350693.6953| 5.4014 |
 | fold_1 | 1.6644| 2.0097| 10570407314187.8945| 5.1303 |
 | fold_2 | 1.6084| 1.9349| 170695202621.1840| 5.0545 |
 | fold_3 | 1.6490| 1.9913| 3795252908753.8677| 4.9331 |
 | fold_4 | 1.6644| 2.0050| 10725781603644.2207| 5.0949 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 1.6494 | 1.6644 | 1.6084 | 0.0213 |
| rmse | 1.9896 | 2.0097 | 1.9349 | 0.0281 |
| mape* | 8474366075980.1719 | 17109693350693.6953 | 170695202621.1840 | 5913986606286.2617 |
| max_error | 5.1228 | 5.4014 | 4.9331 | 0.1543 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 1404.6727| 1714.4002| 4.4786| 3412.1421 |
 | fold_1 | 1442.7943| 1743.3703| 4.3567| 3312.8239 |
 | fold_2 | 1446.0539| 1743.9673| 4.6925| 3362.4479 |
 | fold_3 | 1460.5342| 1745.6360| 4.5293| 3357.7323 |
 | fold_4 | 1456.9003| 1748.4453| 4.6200| 3490.7322 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 1442.1911 | 1460.5342 | 1404.6727 | 19.8784 |
| rmse | 1739.1638 | 1748.4453 | 1714.4002 | 12.5063 |
| mape* | 4.5354 | 4.6925 | 4.3567 | 0.1158 |
| max_error | 3387.1757 | 3490.7322 | 3312.8239 | 60.5869 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 488.9729| 591.9607| 0.3684| 1389.1260 |
 | fold_1 | 540.4269| 646.7450| 0.4131| 1327.3824 |
 | fold_2 | 504.0011| 606.4482| 0.3853| 1361.8915 |
 | fold_3 | 491.5035| 603.6104| 0.3741| 1272.9824 |
 | fold_4 | 548.5354| 651.1520| 0.4201| 1306.9824 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 514.6879 | 548.5354 | 488.9729 | 24.9845 |
| rmse | 619.9833 | 651.1520 | 591.9607 | 24.1835 |
| mape* | 0.3922 | 0.4201 | 0.3684 | 0.0208 |
| max_error | 1331.6729 | 1389.1260 | 1272.9824 | 40.7103 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_1 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_2 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_3 | `{'some_param': 1, 'other_param': 12.39348}` |
| fold_4 | `{'some_param': 1, 'other_param': 12.39348}` |




