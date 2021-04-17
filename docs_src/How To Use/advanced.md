# Advanced usage

Once you have recorded some data, you can examine it with the `MatbenchBenchmark` object. If you are looking to record data, see the [Recording data page](2run.md).

**Pretty much everything in Matbench - including scoring, saving, loading, recording, inspecting, and more - can be done thru `MatbenchBenchmark` directly.**


## Loading and saving

### Load a completed, valid benchmark from disk:

```python
mb = MatbenchBechmark.from_file("path/to/my_results.json.gz")

>>> mb
<MatbenchBenchmark>

```


### Save a completed, valid benchmark to disk

```python
mb.to_file("path/to/my_results.json.gz")
```


## Task data

Tasks (`MatbenchTask`) are accessible as `MatbenchBenchmark` attributes through their names.


Let's say we are interested in `matbench_dielectric`. 
```python

# Access task thru attribute
task = mb.matbench_dielectric


# This task is a MatbenchTask object
>>> print(task)

<MatbenchTask>
```


### See task metadata

See metadata for an individual task.

```python
metadata = mb.matbench_dielectric.metadata

>>> metadata
{
'input_type': 'structure',
 'mad': 0.808534704217072,
 'n_samples': 4764,
 'target': 'n',
 'task_type': 'regression',
 'unit': 'unitless',
 'bibtex_refs': ["@Article{Dunn2020,\nauthor={Dunn, Alexander\nand Wang, Qi\nand Ganose, Alex\nand Dopp, Daniel\nand Jain, Anubhav},\ntitle={Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm},\njournal={npj Computational Materials},\nyear={2020},\nmonth={Sep},\nday={15},\nvolume={6},\nnumber={1},\npages={138},\nabstract={We present a benchmark test suite and an automated machine learning procedure for evaluating supervised machine learning (ML) models for predicting properties of inorganic bulk materials. The test suite, Matbench, is a set of 13{\\thinspace}ML tasks that range in size from 312 to 132k samples and contain data from 10 density functional theory-derived and experimental sources. Tasks include predicting optical, thermal, electronic, thermodynamic, tensile, and elastic properties given a material's composition and/or crystal structure. The reference algorithm, Automatminer, is a highly-extensible, fully automated ML pipeline for predicting materials properties from materials primitives (such as composition and crystal structure) without user intervention or hyperparameter tuning. We test Automatminer on the Matbench test suite and compare its predictive power with state-of-the-art crystal graph neural networks and a traditional descriptor-based Random Forest model. We find Automatminer achieves the best performance on 8 of 13 tasks in the benchmark. We also show our test suite is capable of exposing predictive advantages of each algorithm---namely, that crystal graph methods appear to outperform traditional machine learning methods given {\\textasciitilde}104 or greater data points. We encourage evaluating materials ML algorithms on the Matbench benchmark and comparing them against the latest version of Automatminer.},\nissn={2057-3960},\ndoi={10.1038/s41524-020-00406-3},\nurl={https://doi.org/10.1038/s41524-020-00406-3}\n}\n",
  '@article{Jain2013,\nauthor = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},\ndoi = {10.1063/1.4812323},\nissn = {2166532X},\njournal = {APL Materials},\nnumber = {1},\npages = {011002},\ntitle = {{The Materials Project: A materials genome approach to accelerating materials innovation}},\nurl = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\\&Agg=doi},\nvolume = {1},\nyear = {2013}\n}',
  '@article{Petousis2017,\nauthor={Petousis, Ioannis and Mrdjenovich, David and Ballouz, Eric\nand Liu, Miao and Winston, Donald and Chen, Wei and Graf, Tanja\nand Schladt, Thomas D. and Persson, Kristin A. and Prinz, Fritz B.},\ntitle={High-throughput screening of inorganic compounds for the\ndiscovery of novel dielectric and optical materials},\njournal={Scientific Data},\nyear={2017},\nmonth={Jan},\nday={31},\npublisher={The Author(s)},\nvolume={4},\npages={160134},\nnote={Data Descriptor},\nurl={http://dx.doi.org/10.1038/sdata.2016.134}\n}'],
 'columns': {'n': 'Target variable. Refractive index (unitless).',
  'structure': 'Pymatgen Structure of the material.'},
 'description': 'Matbench v0.1 test dataset for predicting refractive index from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having refractive indices less than 1 and those containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.',
 'file_type': 'json.gz',
 'num_entries': 4764,
 'url': 'https://ml.materialsproject.org/projects/matbench_dielectric.json.gz',
 'hash': '83befa09bc2ec2f4b6143afc413157827a90e5e2e42c1eb507ccfa01bf26a1d6',
 'reference': 'Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,\nChen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.\nHigh-throughput screening of inorganic compounds for the discovery\nof novel dielectric and optical materials. Sci. Data 4, 160134 (2017).',
}



# Metadata is also accessible as attributes
>>> metadata.unit

"unitless"
```

### See which folds of this task are recorded

```python

recorded_folds = mb.matbench_dielectric.is_recorded

# In this example, we only have folds 0 and 1 recorded.
>>> recorded_folds

{0: True, 1: True, 2: False, 3: False, 4: False}

```


### See task score stats among folds

All folds must be recorded to see score stats.

```python

scores = mb.matbench_dielectric.scores


# Show score stats taken over all folds
>>> scores

{'mae': {'mean': 0.31502894856879793,
  'max': 0.42569840085084304,
  'min': 0.21883030230732342,
  'std': 0.0672172232063864},
 'rmse': {'mean': 1.7202043807691947,
  'max': 2.9472145483123082,
  'min': 0.6855155532720747,
  'std': 0.8140297551209411},
 'mape': {'mean': 0.08510552426501797,
  'max': 0.09872854141937873,
  'min': 0.07201546203802894,
  'std': 0.009760258167856002},
 'max_error': {'mean': 34.996903717427166,
  'max': 59.01119325894446,
  'min': 14.665353016975205,
  'std': 17.978224948280573}
 }


# scores are also accessible as attrs

>>> scores.mae.max

0.42569840085084304

```


### See outputs, parameters, and scores for individual task folds

```python

# Get all of our recorded results
results = mb.matbench_dielectric.results

>>> results
{
    'fold_0': 
         {'data': 
              {
                'mb-dielectric-0008': 2.1816278769942685,
                'mb-dielectric-0010': 2.1449892069940995,
                'mb-dielectric-0019': 3.9022885489716175,
                'mb-dielectric-0025': 4.105947591302149,
                ...
               },
          'parameters': {
                'best_pipeline': '["(selectfwe, SelectFwe(alpha=0.006, score_func=<function f_regression at 0x2aaaef1a0840>))..."'
                ...
               },
          'scores': {
                'mae': 0.21883030230732342,
                'mape': 0.07602888421332273,
                'max_error': 14.665353016975205,
                'rmse': 0.6855155532720747
            }
          },
    'fold_1': {...},
    ...
}

      
# Individual fold data are available thru attrs
>>> results.fold_4.data['mb-dielectric-4751']
2.5696947646331614

# Including ML parameters for a specific fold, if made available
>>> results.fold_4.parameters
{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.034, score_func=<function f_regression at 0x2aaaf35a08c8>))',
    '(zerocount, ZeroCount())',
    '(gradientboostingregressor, GradientBoostingRegressor(alpha=0.85, criterion=friedman_mse, init=null,\n                          learning_rate=0.1, loss=huber, max_depth=9,\n                          max_features=0.7500000000000001, max_leaf_nodes=null,\n                          min_impurity_decrease=0.0, min_impurity_split=null,\n                          min_samples_leaf=13, min_samples_split=17,\n                          min_weight_fraction_leaf=0.0, n_estimators=100,\n                          n_iter_no_change=null, presort=auto,\n                          random_state=null, subsample=0.7500000000000001,\n                          tol=0.0001, validation_fraction=0.1, verbose=0,\n                          warm_start=false))'],
   'features_reduced': ['MagpieData maximum Number',
    'MagpieData maximum MendeleevNumber',
    'MagpieData mean MendeleevNumber',
    'MagpieData avg_dev MendeleevNumber',
    'MagpieData range AtomicWeight',
    ...
 }

# Get score metrics on fold 4
>>> results.fold_4.scores
{'mae': 0.3264316502622554,
 'mape': 0.09872854141937873,
 'max_error': 28.160118784575193,
 'rmse': 1.6137009708660595
 }
```

### Validate an individual task's results


```python

>>> mb.matbench_dielectric.validate()

# If does not throw an error, it's valid!

```



## Benchmark data

A `MatbenchBenchmark` is a collection of tasks. Once your benchmark is recorded, you can inspect it.



### Get information about the state of a benchmark

```python

>>> mb.get_info()

"""
Matbench package 0.1.0 running benchmark 'matbench_v0.1'
	is complete: True
	is recorded: True
	is valid: True

Results:

	- 'matbench_dielectric' MAE mean: 29.09435441521901
	- 'matbench_expt_gap' MAE mean: 5.097990146029299
	- 'matbench_expt_is_metal' ROCAUC mean: 0.490515739562644
	- 'matbench_glass' ROCAUC mean: 0.4915206231191361
	- 'matbench_mp_e_form' MAE mean: 1.9798749618345852
	- 'matbench_jdft2d' MAE mean: 624.8594821594436
	- 'matbench_log_gvrh' MAE mean: 0.7503117195807093
	- 'matbench_log_kvrh' MAE mean: 0.8337265925158915
	- 'matbench_mp_gap' MAE mean: 3.9947345263133185
	- 'matbench_mp_is_metal' ROCAUC mean: 0.4995330363104962
	- 'matbench_perovskites' MAE mean: 1.6494389339807394
	- 'matbench_phonons' MAE mean: 1442.1910745917485
	- 'matbench_steels' MAE mean: 514.6879431114869
"""
```


### Access a summary of score data, across all tasks

Access score data for multiple metrics, including fold statistics, programmatically
```python
>>> mb.scores

{'matbench_dielectric': {'mae': {'mean': 29.09435441521901, 'max': 29.790913986352297, 'min': 26.50764023789047, 'std': 1.2938287761791334}, 'rmse': {'mean': 33.654269974352744, 'max': 34.44945162692406, 'min': 30.707221665034698, 'std': 1.4740060199828717}, 'mape': {'mean': 14.169387576348942, 'max': 14.56764274096521, 'min': 12.928095832225917, 'std': 0.6228030143476618}, 'max_error': {'mean': 58.85621300050616, 'max': 60.1966146990726, 'min': 53.98208657241693, 'std': 2.4395502402545453}}, 'matbench_expt_gap': {'mae': {'mean': 5.097990146029299, 'max': 5.290261095781455, 'min': 4.6298670001648965, 'std': 0.2397514292575463}, 'rmse': {'mean': 6.006638705150991, 'max': 6.226508032402611, 'min': 5.47028276176484, 'std': 0.27274122238814}, 'mape': {'mean': 1.38641021305497e+16, 'max': 1.5276180519639252e+16, 'min': 1.2259552001352658e+16, 'std': 986247659935790.8}, 'max_error': {'mean': 11.407347551284193, 'max': 11.688512264782567, 'min': 10.489690494035637, 'std': 0.45961704429199657}}, 'matbench_expt_is_metal': {'accuracy': {'mean': 0.4903474887540754, 'max': 0.5050813008130082, 'min': 0.47459349593495936, 'std': 0.013195738662206162}, 'balanced_accuracy': {'mean': 0.490515739562644, 'max': 0.5052590266875981, 'min': 0.4747707180038007, 'std': 0.013195964150335589}, 'f1': {'mean': 0.5107296153663292, 'max': 0.5248780487804879, 'min': 0.49560975609756097, 'std': 0.012667909247509207}, 'rocauc': {'mean': 0.490515739562644, 'max': 0.5052590266875981, 'min': 0.4747707180038007, 'std': 0.013195964150335589}}, 'matbench_glass': {'accuracy': {'mean': 0.5059859154929578, 'max': 0.528169014084507, 'min': 0.477112676056338, 'std': 0.018718357549298598}, 'balanced_accuracy': {'mean': 0.4915206231191361, 'max': 0.518476250739163, 'min': 0.4564355205025932, 'std': 0.022745473256365906}, 'f1': {'mean': 0.6019858156028368, 'max': 0.6198581560283688, 'min': 0.5787234042553191, 'std': 0.015080889486527119}, 'rocauc': {'mean': 0.4915206231191361, 'max': 0.518476250739163, 'min': 0.4564355205025932, 'std': 0.022745473256365906}}, 'matbench_mp_e_form': {'mae': {'mean': 1.9798749618345852, 'max': 1.9820103943808465, 'min': 1.9764313221160588, 'std': 0.0018588951040352502}, 'rmse': {'mean': 2.376419875235826, 'max': 2.3794812432136196, 'min': 2.3722602233100063, 'std': 0.0023430849418330816}, 'mape': {'mean': 6989111302031.963, 'max': 7492035787402.213, 'min': 6236081301418.79, 'std': 476980899991.28485}, 'max_error': {'mean': 6.9650087167699155, 'max': 7.057955130739103, 'min': 6.878168095265195, 'std': 0.06657839974500762}}, 'matbench_jdft2d': {'mae': {'mean': 624.8594821594436, 'max': 662.8351790033564, 'min': 484.0870035426516, 'std': 70.41763851884579}, 'rmse': {'mean': 754.6594168930902, 'max': 802.851398577492, 'min': 575.4212296101125, 'std': 89.65203353138263}, 'mape': {'mean': 12.691214729498025, 'max': 22.18652735053058, 'min': 9.642403294653164, 'std': 4.833743597331997}, 'max_error': {'mean': 1455.537803743586, 'max': 1532.911339763068, 'min': 1229.7021907932801, 'std': 113.62938957056699}}, 'matbench_log_gvrh': {'mae': {'mean': 0.7503117195807093, 'max': 0.7567499426463542, 'min': 0.7458321525860483, 'std': 0.004177000349054263}, 'rmse': {'mean': 0.8922201073043177, 'max': 0.8965161788869266, 'min': 0.8860255812982848, 'std': 0.0034322474625259137}, 'mape': {'mean': 15059325260426.266, 'max': 26158506009539.293, 'min': 4541885118479.488, 'std': 6978350942510.934}, 'max_error': {'mean': 2.4294014472589063, 'max': 2.7078341735946374, 'min': 2.2460171713812693, 'std': 0.17276767393879686}}, 'matbench_log_kvrh': {'mae': {'mean': 0.8337265925158915, 'max': 0.84093059152486, 'min': 0.8252194857104939, 'std': 0.005960109281798535}, 'rmse': {'mean': 1.0122909056359641, 'max': 1.0190702248693488, 'min': 1.0038726599443502, 'std': 0.005051164726815858}, 'mape': {'mean': 5205086458416.939, 'max': 10062434398435.773, 'min': 1547244178802.9026, 'std': 3081512230166.448}, 'max_error': {'mean': 2.5243586576102204, 'max': 2.7538971602513187, 'min': 2.4510034654636557, 'std': 0.11592723389441413}}, 'matbench_mp_gap': {'mae': {'mean': 3.9947345263133185, 'max': 4.040261917311839, 'min': 3.8419572019120563, 'std': 0.07657166015944829}, 'rmse': {'mean': 4.802562096614456, 'max': 4.852934760261583, 'min': 4.621350867969822, 'std': 0.09070340779972745}, 'mape': {'mean': 9376100521414580.0, 'max': 9530024251770120.0, 'min': 9017068673849420.0, 'std': 191246529837308.34}, 'max_error': {'mean': 9.641818242181197, 'max': 9.721159283071396, 'min': 9.326360936678295, 'std': 0.15772886643823172}}, 'matbench_mp_is_metal': {'accuracy': {'mean': 0.49927909555806177, 'max': 0.5032513429459994, 'min': 0.49498185930358574, 'std': 0.002961735738880825}, 'balanced_accuracy': {'mean': 0.4995330363104962, 'max': 0.5035756141508568, 'min': 0.4951605437227332, 'std': 0.003013293507682773}, 'f1': {'mean': 0.465575654865608, 'max': 0.46982498491249247, 'min': 0.46097364715349026, 'std': 0.0031704147980028555}, 'rocauc': {'mean': 0.4995330363104962, 'max': 0.5035756141508567, 'min': 0.4951605437227331, 'std': 0.0030132935076827893}}, 'matbench_perovskites': {'mae': {'mean': 1.6494389339807394, 'max': 1.6643604327414814, 'min': 1.6083671212370563, 'std': 0.02130042981456539}, 'rmse': {'mean': 1.9895605050492304, 'max': 2.0097384860674103, 'min': 1.9348806762983708, 'std': 0.028069501175258544}, 'mape': {'mean': 8474366075980.172, 'max': 17109693350693.695, 'min': 170695202621.18396, 'std': 5913986606286.262}, 'max_error': {'mean': 5.122830203832267, 'max': 5.401364835279832, 'min': 4.933113748862263, 'std': 0.15432057817183506}}, 'matbench_phonons': {'mae': {'mean': 1442.1910745917485, 'max': 1460.5342302638428, 'min': 1404.6727173726108, 'std': 19.87835062913105}, 'rmse': {'mean': 1739.1638204522908, 'max': 1748.4453111626615, 'min': 1714.4001958506544, 'std': 12.506318415067186}, 'mape': {'mean': 4.535426963268569, 'max': 4.692503460859966, 'min': 4.356729242935622, 'std': 0.11577855407899913}, 'max_error': {'mean': 3387.1756802926197, 'max': 3490.7322416780676, 'min': 3312.8239446861567, 'std': 60.586867518772216}}, 'matbench_steels': {'mae': {'mean': 514.6879431114869, 'max': 548.5353510044772, 'min': 488.97286237333986, 'std': 24.98451122832146}, 'rmse': {'mean': 619.9832706475461, 'max': 651.1520235084482, 'min': 591.9607445092288, 'std': 24.183510586935057}, 'mape': {'mean': 0.39220921441643364, 'max': 0.4201053232886023, 'min': 0.368378839458224, 'std': 0.02076964611162295}, 'max_error': {'mean': 1331.6729147023618, 'max': 1389.1259692340998, 'min': 1272.982373277621, 'std': 40.71026078669035}}}
```


### Validate an entire benchmark

You can validate an entire benchmark with the `validate` method of `MatbenchBenchmark`. 

```python

>>> mb.is_valid

True
```

If your results are valid, it ensures the automated leaderboard can understand your data and that all folds for all tasks are recorded.


### See if a benchmark is complete

A benchmark is complete if it contains all the tasks specified in the benchmark specification. In the case of the benchmark Matbench v0.1, this means all 13 tasks are present in your benchmark (though they may not be recorded yet!).


```python
>>> mb.is_complete

True


```