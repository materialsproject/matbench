# matbench_v0.1: RF-SCM/Magpie

### Algorithm description: 

A random forest using features from the Sine Coulomb Matrix and MagPie featurization algorthims. Sine Coulomb Matrix creates structural features based on Coulombic interactions inside a periodic boundary condition (i.e., for crystalline materials with known structure). MagPie features are weighted elemental features based on elemental data such as electronegativity, melting point, and electron affinity. Algorithms were run inside of the Automatminer v1.0.3.20191111 framework for convenience, though no auto-featurization or AutoML were run. Data cleaning dropped features with more than 1% nan samples, imputing missing samples using the mean of the training data. No feature reduction was performed. Both featurization techniques were applied to structure problems, only MagPie features were applied to problems without structure.

No hyperparameter tuning was performed on the RF, as a large, constant number of trees were used in constructing each fold's model; the entire training+validation set was used as training data for the RF.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_rf).

### References (in bibtex format): 

```
['@article{Dunn2020,\n'
 '  doi = {10.1038/s41524-020-00406-3},\n'
 '  url = {https://doi.org/10.1038/s41524-020-00406-3},\n'
 '  year = {2020},\n'
 '  month = sep,\n'
 '  publisher = {Springer Science and Business Media {LLC}},\n'
 '  volume = {6},\n'
 '  number = {1},\n'
 '  author = {Alexander Dunn and Qi Wang and Alex Ganose and Daniel Dopp and '
 'Anubhav Jain},\n'
 '  title = {Benchmarking materials property prediction methods: the Matbench '
 'test set and Automatminer reference algorithm},\n'
 '  journal = {npj Computational Materials}\n'
 '}',
 '@article{Breiman2001,\n'
 '  doi = {10.1023/a:1010933404324},\n'
 '  url = {https://doi.org/10.1023/a:1010933404324},\n'
 '  year = {2001},\n'
 '  publisher = {Springer Science and Business Media {LLC}},\n'
 '  volume = {45},\n'
 '  number = {1},\n'
 '  pages = {5--32},\n'
 '  author = {Leo Breiman},\n'
 '  journal = {Machine Learning}\n'
 '}',
 '@article{Ward2016,\n'
 '  doi = {10.1038/npjcompumats.2016.28},\n'
 '  url = {https://doi.org/10.1038/npjcompumats.2016.28},\n'
 '  year = {2016},\n'
 '  month = aug,\n'
 '  publisher = {Springer Science and Business Media {LLC}},\n'
 '  volume = {2},\n'
 '  number = {1},\n'
 '  author = {Logan Ward and Ankit Agrawal and Alok Choudhary and Christopher '
 'Wolverton},\n'
 '  title = {A general-purpose machine learning framework for predicting '
 'properties of inorganic materials},\n'
 '  journal = {npj Computational Materials}\n'
 '}',
 '@article {QUA:QUA24917,author = {Faber, Felix and Lindmaa, Alexander and von '
 'Lilienfeld, O. Anatole and Armiento, Rickard},title = {Crystal structure '
 'representations for machine learning models of formation energies},journal = '
 '{International Journal of Quantum Chemistry},volume = {115},number = '
 '{16},issn = {1097-461X},url = {http://dx.doi.org/10.1002/qua.24917},doi = '
 '{10.1002/qua.24917},pages = {1094--1101},keywords = {machine learning, '
 'formation energies, representations, crystal structure, periodic '
 'systems},year = {2015},}']
```

### User metadata:

```
{'__deepcopy__': {},
 '__getstate__': {},
 '_ipython_canary_method_should_not_exist_': {'__deepcopy__': {},
                                              '__getstate__': {}},
 'autofeaturizer_kwargs': {'n_jobs': 10, 'preset': 'debug'},
 'best_pipeline': 'RandomForestRegressor(bootstrap=true, criterion=mse, '
                  'max_depth=null,\n'
                  '           max_features=auto, max_leaf_nodes=null,\n'
                  '           min_impurity_decrease=0.0, '
                  'min_impurity_split=null,\n'
                  '           min_samples_leaf=1, min_samples_split=2,\n'
                  '           min_weight_fraction_leaf=0.0, n_estimators=500, '
                  'n_jobs=null,\n'
                  '           oob_score=false, random_state=null, verbose=0, '
                  'warm_start=false)',
 'cleaner_kwargs': {'feature_na_method': 'mean',
                    'max_na_frac': 0.01,
                    'na_method_fit': 'drop',
                    'na_method_transform': 'mean'},
 'features_all': ['MagpieData minimum Number',
                  'MagpieData maximum Number',
                  'MagpieData range Number',
                  'MagpieData mean Number',
                  'MagpieData avg_dev Number',
                  'MagpieData mode Number',
                  'MagpieData minimum MendeleevNumber',
                  'MagpieData maximum MendeleevNumber',
                  'MagpieData range MendeleevNumber',
                  'MagpieData mean MendeleevNumber',
                  'MagpieData avg_dev MendeleevNumber',
                  'MagpieData mode MendeleevNumber',
                  'MagpieData minimum AtomicWeight',
                  'MagpieData maximum AtomicWeight',
                  'MagpieData range AtomicWeight',
                  'MagpieData mean AtomicWeight',
                  'MagpieData avg_dev AtomicWeight',
                  'MagpieData mode AtomicWeight',
                  'MagpieData minimum MeltingT',
                  'MagpieData maximum MeltingT',
                  'MagpieData range MeltingT',
                  'MagpieData mean MeltingT',
                  'MagpieData avg_dev MeltingT',
                  'MagpieData mode MeltingT',
                  'MagpieData minimum Column',
                  'MagpieData maximum Column',
                  'MagpieData range Column',
                  'MagpieData mean Column',
                  'MagpieData avg_dev Column',
                  'MagpieData mode Column',
                  'MagpieData minimum Row',
                  'MagpieData maximum Row',
                  'MagpieData range Row',
                  'MagpieData mean Row',
                  'MagpieData avg_dev Row',
                  'MagpieData mode Row',
                  'MagpieData minimum CovalentRadius',
                  'MagpieData maximum CovalentRadius',
                  'MagpieData range CovalentRadius',
                  'MagpieData mean CovalentRadius',
                  'MagpieData avg_dev CovalentRadius',
                  'MagpieData mode CovalentRadius',
                  'MagpieData minimum Electronegativity',
                  'MagpieData maximum Electronegativity',
                  'MagpieData range Electronegativity',
                  'MagpieData mean Electronegativity',
                  'MagpieData avg_dev Electronegativity',
                  'MagpieData mode Electronegativity',
                  'MagpieData minimum NsValence',
                  'MagpieData maximum NsValence',
                  'MagpieData range NsValence',
                  'MagpieData mean NsValence',
                  'MagpieData avg_dev NsValence',
                  'MagpieData mode NsValence',
                  'MagpieData minimum NpValence',
                  'MagpieData maximum NpValence',
                  'MagpieData range NpValence',
                  'MagpieData mean NpValence',
                  'MagpieData avg_dev NpValence',
                  'MagpieData mode NpValence',
                  'MagpieData minimum NdValence',
                  'MagpieData maximum NdValence',
                  'MagpieData range NdValence',
                  'MagpieData mean NdValence',
                  'MagpieData avg_dev NdValence',
                  'MagpieData mode NdValence',
                  'MagpieData minimum NfValence',
                  'MagpieData maximum NfValence',
                  'MagpieData range NfValence',
                  'MagpieData mean NfValence',
                  'MagpieData avg_dev NfValence',
                  'MagpieData mode NfValence',
                  'MagpieData minimum NValence',
                  'MagpieData maximum NValence',
                  'MagpieData range NValence',
                  'MagpieData mean NValence',
                  'MagpieData avg_dev NValence',
                  'MagpieData mode NValence',
                  'MagpieData minimum NsUnfilled',
                  'MagpieData maximum NsUnfilled',
                  'MagpieData range NsUnfilled',
                  'MagpieData mean NsUnfilled',
                  'MagpieData avg_dev NsUnfilled',
                  'MagpieData mode NsUnfilled',
                  'MagpieData minimum NpUnfilled',
                  'MagpieData maximum NpUnfilled',
                  'MagpieData range NpUnfilled',
                  'MagpieData mean NpUnfilled',
                  'MagpieData avg_dev NpUnfilled',
                  'MagpieData mode NpUnfilled',
                  'MagpieData minimum NdUnfilled',
                  'MagpieData maximum NdUnfilled',
                  'MagpieData range NdUnfilled',
                  'MagpieData mean NdUnfilled',
                  'MagpieData avg_dev NdUnfilled',
                  'MagpieData mode NdUnfilled',
                  'MagpieData minimum NfUnfilled',
                  'MagpieData maximum NfUnfilled',
                  'MagpieData range NfUnfilled',
                  'MagpieData mean NfUnfilled',
                  'MagpieData avg_dev NfUnfilled',
                  'MagpieData mode NfUnfilled',
                  'MagpieData minimum NUnfilled',
                  'MagpieData maximum NUnfilled',
                  'MagpieData range NUnfilled',
                  'MagpieData mean NUnfilled',
                  'MagpieData avg_dev NUnfilled',
                  'MagpieData mode NUnfilled',
                  'MagpieData minimum GSvolume_pa',
                  'MagpieData maximum GSvolume_pa',
                  'MagpieData range GSvolume_pa',
                  'MagpieData mean GSvolume_pa',
                  'MagpieData avg_dev GSvolume_pa',
                  'MagpieData mode GSvolume_pa',
                  'MagpieData minimum GSbandgap',
                  'MagpieData maximum GSbandgap',
                  'MagpieData range GSbandgap',
                  'MagpieData mean GSbandgap',
                  'MagpieData avg_dev GSbandgap',
                  'MagpieData mode GSbandgap',
                  'MagpieData minimum GSmagmom',
                  'MagpieData maximum GSmagmom',
                  'MagpieData range GSmagmom',
                  'MagpieData mean GSmagmom',
                  'MagpieData avg_dev GSmagmom',
                  'MagpieData mode GSmagmom',
                  'MagpieData minimum SpaceGroupNumber',
                  'MagpieData maximum SpaceGroupNumber',
                  'MagpieData range SpaceGroupNumber',
                  'MagpieData mean SpaceGroupNumber',
                  'MagpieData avg_dev SpaceGroupNumber',
                  'MagpieData mode SpaceGroupNumber',
                  'sine coulomb matrix eig 0',
                  'sine coulomb matrix eig 1',
                  'sine coulomb matrix eig 2',
                  'sine coulomb matrix eig 3',
                  'sine coulomb matrix eig 4',
                  'sine coulomb matrix eig 5',
                  'sine coulomb matrix eig 6',
                  'sine coulomb matrix eig 7',
                  'sine coulomb matrix eig 8',
                  'sine coulomb matrix eig 9',
                  'sine coulomb matrix eig 10',
                  'sine coulomb matrix eig 11',
                  'sine coulomb matrix eig 12',
                  'sine coulomb matrix eig 13',
                  'sine coulomb matrix eig 14',
                  'sine coulomb matrix eig 15',
                  'sine coulomb matrix eig 16',
                  'sine coulomb matrix eig 17',
                  'sine coulomb matrix eig 18',
                  'sine coulomb matrix eig 19',
                  'sine coulomb matrix eig 20',
                  'sine coulomb matrix eig 21',
                  'sine coulomb matrix eig 22',
                  'sine coulomb matrix eig 23',
                  'sine coulomb matrix eig 24',
                  'sine coulomb matrix eig 25',
                  'sine coulomb matrix eig 26',
                  'sine coulomb matrix eig 27',
                  'sine coulomb matrix eig 28',
                  'sine coulomb matrix eig 29',
                  'sine coulomb matrix eig 30',
                  'sine coulomb matrix eig 31',
                  'sine coulomb matrix eig 32',
                  'sine coulomb matrix eig 33',
                  'sine coulomb matrix eig 34',
                  'sine coulomb matrix eig 35',
                  'sine coulomb matrix eig 36',
                  'sine coulomb matrix eig 37',
                  'sine coulomb matrix eig 38',
                  'sine coulomb matrix eig 39',
                  'sine coulomb matrix eig 40',
                  'sine coulomb matrix eig 41',
                  'sine coulomb matrix eig 42',
                  'sine coulomb matrix eig 43',
                  'sine coulomb matrix eig 44',
                  'sine coulomb matrix eig 45',
                  'sine coulomb matrix eig 46',
                  'sine coulomb matrix eig 47',
                  'sine coulomb matrix eig 48',
                  'sine coulomb matrix eig 49',
                  'sine coulomb matrix eig 50',
                  'sine coulomb matrix eig 51',
                  'sine coulomb matrix eig 52',
                  'sine coulomb matrix eig 53',
                  'sine coulomb matrix eig 54',
                  'sine coulomb matrix eig 55',
                  'sine coulomb matrix eig 56',
                  'sine coulomb matrix eig 57',
                  'sine coulomb matrix eig 58',
                  'sine coulomb matrix eig 59',
                  'sine coulomb matrix eig 60',
                  'sine coulomb matrix eig 61',
                  'sine coulomb matrix eig 62',
                  'sine coulomb matrix eig 63',
                  'sine coulomb matrix eig 64',
                  'sine coulomb matrix eig 65',
                  'sine coulomb matrix eig 66',
                  'sine coulomb matrix eig 67',
                  'sine coulomb matrix eig 68',
                  'sine coulomb matrix eig 69',
                  'sine coulomb matrix eig 70',
                  'sine coulomb matrix eig 71',
                  'sine coulomb matrix eig 72',
                  'sine coulomb matrix eig 73',
                  'sine coulomb matrix eig 74',
                  'sine coulomb matrix eig 75',
                  'sine coulomb matrix eig 76',
                  'sine coulomb matrix eig 77',
                  'sine coulomb matrix eig 78',
                  'sine coulomb matrix eig 79',
                  'sine coulomb matrix eig 80',
                  'sine coulomb matrix eig 81',
                  'sine coulomb matrix eig 82',
                  'sine coulomb matrix eig 83',
                  'sine coulomb matrix eig 84',
                  'sine coulomb matrix eig 85',
                  'sine coulomb matrix eig 86',
                  'sine coulomb matrix eig 87',
                  'sine coulomb matrix eig 88',
                  'sine coulomb matrix eig 89',
                  'sine coulomb matrix eig 90',
                  'sine coulomb matrix eig 91',
                  'sine coulomb matrix eig 92',
                  'sine coulomb matrix eig 93',
                  'sine coulomb matrix eig 94',
                  'sine coulomb matrix eig 95',
                  'sine coulomb matrix eig 96',
                  'sine coulomb matrix eig 97',
                  'sine coulomb matrix eig 98',
                  'sine coulomb matrix eig 99',
                  'sine coulomb matrix eig 100',
                  'sine coulomb matrix eig 101',
                  'sine coulomb matrix eig 102',
                  'sine coulomb matrix eig 103',
                  'sine coulomb matrix eig 104',
                  'sine coulomb matrix eig 105',
                  'sine coulomb matrix eig 106',
                  'sine coulomb matrix eig 107',
                  'sine coulomb matrix eig 108',
                  'sine coulomb matrix eig 109',
                  'sine coulomb matrix eig 110',
                  'sine coulomb matrix eig 111',
                  'sine coulomb matrix eig 112',
                  'sine coulomb matrix eig 113',
                  'sine coulomb matrix eig 114',
                  'sine coulomb matrix eig 115',
                  'sine coulomb matrix eig 116',
                  'sine coulomb matrix eig 117',
                  'sine coulomb matrix eig 118',
                  'sine coulomb matrix eig 119',
                  'sine coulomb matrix eig 120',
                  'sine coulomb matrix eig 121',
                  'sine coulomb matrix eig 122',
                  'sine coulomb matrix eig 123',
                  'sine coulomb matrix eig 124',
                  'sine coulomb matrix eig 125',
                  'sine coulomb matrix eig 126',
                  'sine coulomb matrix eig 127',
                  'sine coulomb matrix eig 128',
                  'sine coulomb matrix eig 129',
                  'sine coulomb matrix eig 130',
                  'sine coulomb matrix eig 131',
                  'sine coulomb matrix eig 132',
                  'sine coulomb matrix eig 133',
                  'sine coulomb matrix eig 134',
                  'sine coulomb matrix eig 135',
                  'sine coulomb matrix eig 136',
                  'sine coulomb matrix eig 137',
                  'sine coulomb matrix eig 138',
                  'sine coulomb matrix eig 139',
                  'sine coulomb matrix eig 140',
                  'sine coulomb matrix eig 141',
                  'sine coulomb matrix eig 142',
                  'sine coulomb matrix eig 143',
                  'sine coulomb matrix eig 144',
                  'sine coulomb matrix eig 145',
                  'sine coulomb matrix eig 146',
                  'sine coulomb matrix eig 147',
                  'sine coulomb matrix eig 148',
                  'sine coulomb matrix eig 149',
                  'sine coulomb matrix eig 150',
                  'sine coulomb matrix eig 151',
                  'sine coulomb matrix eig 152',
                  'sine coulomb matrix eig 153',
                  'sine coulomb matrix eig 154',
                  'sine coulomb matrix eig 155',
                  'sine coulomb matrix eig 156',
                  'sine coulomb matrix eig 157',
                  'sine coulomb matrix eig 158',
                  'sine coulomb matrix eig 159',
                  'sine coulomb matrix eig 160',
                  'sine coulomb matrix eig 161',
                  'sine coulomb matrix eig 162',
                  'sine coulomb matrix eig 163',
                  'sine coulomb matrix eig 164',
                  'sine coulomb matrix eig 165',
                  'sine coulomb matrix eig 166',
                  'sine coulomb matrix eig 167',
                  'sine coulomb matrix eig 168',
                  'sine coulomb matrix eig 169',
                  'sine coulomb matrix eig 170',
                  'sine coulomb matrix eig 171',
                  'sine coulomb matrix eig 172',
                  'sine coulomb matrix eig 173',
                  'sine coulomb matrix eig 174',
                  'sine coulomb matrix eig 175',
                  'sine coulomb matrix eig 176',
                  'sine coulomb matrix eig 177',
                  'sine coulomb matrix eig 178',
                  'sine coulomb matrix eig 179',
                  'sine coulomb matrix eig 180',
                  'sine coulomb matrix eig 181',
                  'sine coulomb matrix eig 182',
                  'sine coulomb matrix eig 183',
                  'sine coulomb matrix eig 184',
                  'sine coulomb matrix eig 185',
                  'sine coulomb matrix eig 186',
                  'sine coulomb matrix eig 187',
                  'sine coulomb matrix eig 188',
                  'sine coulomb matrix eig 189',
                  'sine coulomb matrix eig 190',
                  'sine coulomb matrix eig 191',
                  'sine coulomb matrix eig 192',
                  'sine coulomb matrix eig 193',
                  'sine coulomb matrix eig 194',
                  'sine coulomb matrix eig 195',
                  'sine coulomb matrix eig 196',
                  'sine coulomb matrix eig 197',
                  'sine coulomb matrix eig 198',
                  'sine coulomb matrix eig 199',
                  'sine coulomb matrix eig 200',
                  'sine coulomb matrix eig 201',
                  'sine coulomb matrix eig 202',
                  'sine coulomb matrix eig 203',
                  'sine coulomb matrix eig 204',
                  'sine coulomb matrix eig 205',
                  'sine coulomb matrix eig 206',
                  'sine coulomb matrix eig 207',
                  'sine coulomb matrix eig 208',
                  'sine coulomb matrix eig 209',
                  'sine coulomb matrix eig 210',
                  'sine coulomb matrix eig 211',
                  'sine coulomb matrix eig 212',
                  'sine coulomb matrix eig 213',
                  'sine coulomb matrix eig 214',
                  'sine coulomb matrix eig 215',
                  'sine coulomb matrix eig 216',
                  'sine coulomb matrix eig 217',
                  'sine coulomb matrix eig 218',
                  'sine coulomb matrix eig 219',
                  'sine coulomb matrix eig 220',
                  'sine coulomb matrix eig 221',
                  'sine coulomb matrix eig 222',
                  'sine coulomb matrix eig 223',
                  'sine coulomb matrix eig 224',
                  'sine coulomb matrix eig 225',
                  'sine coulomb matrix eig 226',
                  'sine coulomb matrix eig 227',
                  'sine coulomb matrix eig 228',
                  'sine coulomb matrix eig 229',
                  'sine coulomb matrix eig 230',
                  'sine coulomb matrix eig 231',
                  'sine coulomb matrix eig 232',
                  'sine coulomb matrix eig 233',
                  'sine coulomb matrix eig 234',
                  'sine coulomb matrix eig 235',
                  'sine coulomb matrix eig 236',
                  'sine coulomb matrix eig 237',
                  'sine coulomb matrix eig 238',
                  'sine coulomb matrix eig 239',
                  'sine coulomb matrix eig 240',
                  'sine coulomb matrix eig 241',
                  'sine coulomb matrix eig 242',
                  'sine coulomb matrix eig 243',
                  'sine coulomb matrix eig 244',
                  'sine coulomb matrix eig 245',
                  'sine coulomb matrix eig 246',
                  'sine coulomb matrix eig 247',
                  'sine coulomb matrix eig 248',
                  'sine coulomb matrix eig 249',
                  'sine coulomb matrix eig 250',
                  'sine coulomb matrix eig 251',
                  'sine coulomb matrix eig 252',
                  'sine coulomb matrix eig 253',
                  'sine coulomb matrix eig 254',
                  'sine coulomb matrix eig 255',
                  'sine coulomb matrix eig 256',
                  'sine coulomb matrix eig 257',
                  'sine coulomb matrix eig 258',
                  'sine coulomb matrix eig 259',
                  'sine coulomb matrix eig 260',
                  'sine coulomb matrix eig 261',
                  'sine coulomb matrix eig 262',
                  'sine coulomb matrix eig 263',
                  'sine coulomb matrix eig 264',
                  'sine coulomb matrix eig 265',
                  'sine coulomb matrix eig 266',
                  'sine coulomb matrix eig 267',
                  'sine coulomb matrix eig 268',
                  'sine coulomb matrix eig 269',
                  'sine coulomb matrix eig 270',
                  'sine coulomb matrix eig 271',
                  'sine coulomb matrix eig 272',
                  'sine coulomb matrix eig 273',
                  'sine coulomb matrix eig 274',
                  'sine coulomb matrix eig 275',
                  'sine coulomb matrix eig 276',
                  'sine coulomb matrix eig 277',
                  'sine coulomb matrix eig 278',
                  'sine coulomb matrix eig 279',
                  'sine coulomb matrix eig 280',
                  'sine coulomb matrix eig 281',
                  'sine coulomb matrix eig 282',
                  'sine coulomb matrix eig 283',
                  'sine coulomb matrix eig 284',
                  'sine coulomb matrix eig 285',
                  'sine coulomb matrix eig 286',
                  'sine coulomb matrix eig 287'],
 'learner_kwargs': {'n_estimators': 500},
 'learner_name': 'rf',
 'reducer_kwargs': {'reducers': []}}
```

### Metadata:

Tasks recorded: 13 of 13 total

Benchmark is complete? True

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3042| 0.7850| 0.1176| 14.5979 |
 | fold_1 | 0.4079| 1.2316| 0.1509| 20.1279 |
 | fold_2 | 0.5220| 2.9832| 0.1370| 59.1201 |
 | fold_3 | 0.3879| 2.1680| 0.1057| 49.4924 |
 | fold_4 | 0.4760| 2.1012| 0.1886| 31.0645 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.4196 | 0.5220 | 0.3042 | 0.0750 |
| rmse | 1.8538 | 2.9832 | 0.7850 | 0.7700 |
| mape* | 0.1400 | 0.1886 | 0.1057 | 0.0289 |
| max_error | 34.8806 | 59.1201 | 14.5979 | 16.9980 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.4360| 0.7985| 0.3380| 5.1654 |
 | fold_1 | 0.4387| 0.7819| 0.3044| 4.7122 |
 | fold_2 | 0.4812| 0.9435| 0.4019| 9.5428 |
 | fold_3 | 0.4345| 0.8059| 0.3647| 5.2288 |
 | fold_4 | 0.4400| 0.7918| 0.4385| 5.5833 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.4461 | 0.4812 | 0.4345 | 0.0177 |
| rmse | 0.8243 | 0.9435 | 0.7819 | 0.0601 |
| mape* | 0.3695 | 0.4385 | 0.3044 | 0.0470 |
| max_error | 6.0465 | 9.5428 | 4.7122 | 1.7700 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_expt_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9249| 0.9248| 0.9236| 0.9248 |
 | fold_1 | 0.9167| 0.9166| 0.9156| 0.9166 |
 | fold_2 | 0.9096| 0.9095| 0.9076| 0.9095 |
 | fold_3 | 0.9228| 0.9227| 0.9221| 0.9227 |
 | fold_4 | 0.9096| 0.9096| 0.9104| 0.9096 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9167 | 0.9249 | 0.9096 | 0.0064 |
| balanced_accuracy | 0.9167 | 0.9248 | 0.9095 | 0.0064 |
| f1 | 0.9159 | 0.9236 | 0.9076 | 0.0063 |
| rocauc | 0.9167 | 0.9248 | 0.9095 | 0.0064 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_glass`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9199| 0.8860| 0.9449| 0.8860 |
 | fold_1 | 0.8856| 0.8402| 0.9217| 0.8402 |
 | fold_2 | 0.8847| 0.8495| 0.9200| 0.8495 |
 | fold_3 | 0.8891| 0.8526| 0.9233| 0.8526 |
 | fold_4 | 0.8979| 0.8651| 0.9292| 0.8651 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8954 | 0.9199 | 0.8847 | 0.0131 |
| balanced_accuracy | 0.8587 | 0.8860 | 0.8402 | 0.0158 |
| f1 | 0.9278 | 0.9449 | 0.9200 | 0.0091 |
| rocauc | 0.8587 | 0.8860 | 0.8402 | 0.0158 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 42.7473| 72.7391| 23.7625| 295.7437 |
 | fold_1 | 45.7510| 94.3771| 0.4382| 581.4859 |
 | fold_2 | 66.2421| 153.0635| 0.8747| 836.6225 |
 | fold_3 | 44.0340| 81.5112| 0.4818| 337.7693 |
 | fold_4 | 51.4457| 159.6390| 0.6384| 1538.6073 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 50.0440 | 66.2421 | 42.7473 | 8.6271 |
| rmse | 112.2660 | 159.6390 | 72.7391 | 36.7066 |
| mape* | 5.2391 | 23.7625 | 0.4382 | 9.2629 |
| max_error | 718.0457 | 1538.6073 | 295.7437 | 453.6473 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1046| 0.1515| 0.0817| 1.1754 |
 | fold_1 | 0.1024| 0.1557| 0.0815| 1.6942 |
 | fold_2 | 0.1025| 0.1533| 0.0798| 1.0668 |
 | fold_3 | 0.1037| 0.1495| 0.0777| 0.9041 |
 | fold_4 | 0.1067| 0.1601| 0.0832| 0.9480 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1040 | 0.1067 | 0.1024 | 0.0016 |
| rmse | 0.1540 | 0.1601 | 0.1495 | 0.0037 |
| mape* | 0.0808 | 0.0832 | 0.0777 | 0.0019 |
| max_error | 1.1577 | 1.6942 | 0.9041 | 0.2845 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0809| 0.1415| 0.0522| 1.4432 |
 | fold_1 | 0.0808| 0.1503| 0.0532| 1.7642 |
 | fold_2 | 0.0783| 0.1383| 0.0509| 1.1189 |
 | fold_3 | 0.0863| 0.1478| 0.0608| 1.1620 |
 | fold_4 | 0.0836| 0.1489| 0.0558| 1.3775 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0820 | 0.0863 | 0.0783 | 0.0027 |
| rmse | 0.1454 | 0.1503 | 0.1383 | 0.0046 |
| mape* | 0.0546 | 0.0608 | 0.0509 | 0.0035 |
| max_error | 1.3732 | 1.7642 | 1.1189 | 0.2311 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1158| 0.2386| 0.9331| 4.2469 |
 | fold_1 | 0.1160| 0.2459| 0.5068| 5.4382 |
 | fold_2 | 0.1179| 0.2443| 0.5549| 4.0782 |
 | fold_3 | 0.1159| 0.2373| 0.7206| 2.9374 |
 | fold_4 | 0.1166| 0.2435| 0.6836| 3.8910 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1165 | 0.1179 | 0.1158 | 0.0008 |
| rmse | 0.2419 | 0.2459 | 0.2373 | 0.0034 |
| mape* | 0.6798 | 0.9331 | 0.5068 | 0.1492 |
| max_error | 4.1183 | 5.4382 | 2.9374 | 0.8008 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3456| 0.6097| 5.6881| 6.3322 |
 | fold_1 | 0.3417| 0.6104| 4.3547| 7.0601 |
 | fold_2 | 0.3445| 0.6047| 6.9303| 5.9201 |
 | fold_3 | 0.3427| 0.6101| 11.9090| 6.6456 |
 | fold_4 | 0.3512| 0.6276| 9.2752| 6.0212 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3452 | 0.3512 | 0.3417 | 0.0033 |
| rmse | 0.6125 | 0.6276 | 0.6047 | 0.0079 |
| mape* | 7.6315 | 11.9090 | 4.3547 | 2.6835 |
| max_error | 6.3958 | 7.0601 | 5.9201 | 0.4182 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9080| 0.9025| 0.8905| 0.9025 |
 | fold_1 | 0.9027| 0.8968| 0.8839| 0.8968 |
 | fold_2 | 0.9049| 0.8987| 0.8862| 0.8987 |
 | fold_3 | 0.9051| 0.8994| 0.8869| 0.8994 |
 | fold_4 | 0.9047| 0.8984| 0.8858| 0.8984 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9051 | 0.9080 | 0.9027 | 0.0017 |
| balanced_accuracy | 0.8992 | 0.9025 | 0.8968 | 0.0019 |
| f1 | 0.8866 | 0.8905 | 0.8839 | 0.0022 |
| rocauc | 0.8992 | 0.9025 | 0.8968 | 0.0019 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2357| 0.3292| 0.2634| 2.8870 |
 | fold_1 | 0.2367| 0.3394| 0.2888| 2.2083 |
 | fold_2 | 0.2365| 0.3382| 0.2631| 2.5900 |
 | fold_3 | 0.2395| 0.3369| 0.2827| 2.6112 |
 | fold_4 | 0.2291| 0.3293| 0.2411| 2.4921 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2355 | 0.2395 | 0.2291 | 0.0034 |
| rmse | 0.3346 | 0.3394 | 0.3292 | 0.0044 |
| mape* | 0.2678 | 0.2888 | 0.2411 | 0.0168 |
| max_error | 2.5577 | 2.8870 | 2.2083 | 0.2185 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 82.3863| 171.1524| 0.1348| 1004.2770 |
 | fold_1 | 72.8871| 172.8015| 0.1172| 2024.7301 |
 | fold_2 | 59.2712| 128.7871| 0.1040| 1206.8703 |
 | fold_3 | 58.6036| 122.1566| 0.1167| 861.9005 |
 | fold_4 | 64.9149| 136.4846| 0.1197| 1255.6664 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 67.6126 | 82.3863 | 58.6036 | 8.9900 |
| rmse | 146.2764 | 172.8015 | 122.1566 | 21.4752 |
| mape* | 0.1185 | 0.1348 | 0.1040 | 0.0098 |
| max_error | 1270.6889 | 2024.7301 | 861.9005 | 402.7307 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 114.6331| 196.3586| 0.0731| 1121.1276 |
 | fold_1 | 85.6694| 113.1549| 0.0654| 362.6630 |
 | fold_2 | 110.0055| 150.1283| 0.0807| 448.9038 |
 | fold_3 | 111.5273| 153.4522| 0.0801| 633.6092 |
 | fold_4 | 95.7271| 133.8257| 0.0730| 408.6042 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 103.5125 | 114.6331 | 85.6694 | 11.0368 |
| rmse | 149.3839 | 196.3586 | 113.1549 | 27.4893 |
| mape* | 0.0745 | 0.0807 | 0.0654 | 0.0056 |
| max_error | 594.9816 | 1121.1276 | 362.6630 | 278.7002 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_1 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_2 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_3 | `{'note': 'single config; see benchmark user metadata'}` |
| fold_4 | `{'note': 'single config; see benchmark user metadata'}` |




