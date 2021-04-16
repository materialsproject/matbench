

## Step 1: Install the python package

See the [installation page](installation.md) for more details.


## Step 2: Record your data.

You can use the matbench python package to retrieve the training and testing splits as well as
record new predictions. Recording and saving your data with matbench should take no more than 
10 lines of matbench code.

```python
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    for fold in task.folds:

        # training inputs are either chemical compositions as strings
        # or crystal structures as pymatgen.Structure objects
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        
        # train and validate your model
        my_model.train_and_Validate(train_inputs, train_outputs)
        
        
        test_inputs = task.get_test_data(fold, include_target=False)
        predictions = my_model.predict(test_inputs)
        task.record(fold, predictions)

mb.to_file("my_models_benchmark.json.gz")

```

The output file, in this case `my_models_benchmark.json.gz` contains everything predicted by your 
benchmark. **Keep this file, as it is the core result that will be submitted to the leaderboard.**


### Note: Benchmark subsets
If you want to benchmark on a subset of Matbench tasks, set the `subset` argument when creating `MatbenchBenchmark`
and use the same code as above. The repo accepts subsets of matbench tasks as well which will appear on a separate "task-specific" leaderboard.


## Step 2.5: Explore data (optional)

### Accessing benchmark data

Benchmarking data, including scores and dataset info, can be accessed directly through `MatbenchBenchmark`:


Get info about the benchmark
```python

# continuing from the example above ...
mb.get_info()
```

```
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
```


Access score data for multiple metrics, including fold statistics, programmatically
```python
mb.scores
```

```
{'matbench_dielectric': {'mae': {'mean': 29.09435441521901, 'max': 29.790913986352297, 'min': 26.50764023789047, 'std': 1.2938287761791334}, 'rmse': {'mean': 33.654269974352744, 'max': 34.44945162692406, 'min': 30.707221665034698, 'std': 1.4740060199828717}, 'mape': {'mean': 14.169387576348942, 'max': 14.56764274096521, 'min': 12.928095832225917, 'std': 0.6228030143476618}, 'max_error': {'mean': 58.85621300050616, 'max': 60.1966146990726, 'min': 53.98208657241693, 'std': 2.4395502402545453}}, 'matbench_expt_gap': {'mae': {'mean': 5.097990146029299, 'max': 5.290261095781455, 'min': 4.6298670001648965, 'std': 0.2397514292575463}, 'rmse': {'mean': 6.006638705150991, 'max': 6.226508032402611, 'min': 5.47028276176484, 'std': 0.27274122238814}, 'mape': {'mean': 1.38641021305497e+16, 'max': 1.5276180519639252e+16, 'min': 1.2259552001352658e+16, 'std': 986247659935790.8}, 'max_error': {'mean': 11.407347551284193, 'max': 11.688512264782567, 'min': 10.489690494035637, 'std': 0.45961704429199657}}, 'matbench_expt_is_metal': {'accuracy': {'mean': 0.4903474887540754, 'max': 0.5050813008130082, 'min': 0.47459349593495936, 'std': 0.013195738662206162}, 'balanced_accuracy': {'mean': 0.490515739562644, 'max': 0.5052590266875981, 'min': 0.4747707180038007, 'std': 0.013195964150335589}, 'f1': {'mean': 0.5107296153663292, 'max': 0.5248780487804879, 'min': 0.49560975609756097, 'std': 0.012667909247509207}, 'rocauc': {'mean': 0.490515739562644, 'max': 0.5052590266875981, 'min': 0.4747707180038007, 'std': 0.013195964150335589}}, 'matbench_glass': {'accuracy': {'mean': 0.5059859154929578, 'max': 0.528169014084507, 'min': 0.477112676056338, 'std': 0.018718357549298598}, 'balanced_accuracy': {'mean': 0.4915206231191361, 'max': 0.518476250739163, 'min': 0.4564355205025932, 'std': 0.022745473256365906}, 'f1': {'mean': 0.6019858156028368, 'max': 0.6198581560283688, 'min': 0.5787234042553191, 'std': 0.015080889486527119}, 'rocauc': {'mean': 0.4915206231191361, 'max': 0.518476250739163, 'min': 0.4564355205025932, 'std': 0.022745473256365906}}, 'matbench_mp_e_form': {'mae': {'mean': 1.9798749618345852, 'max': 1.9820103943808465, 'min': 1.9764313221160588, 'std': 0.0018588951040352502}, 'rmse': {'mean': 2.376419875235826, 'max': 2.3794812432136196, 'min': 2.3722602233100063, 'std': 0.0023430849418330816}, 'mape': {'mean': 6989111302031.963, 'max': 7492035787402.213, 'min': 6236081301418.79, 'std': 476980899991.28485}, 'max_error': {'mean': 6.9650087167699155, 'max': 7.057955130739103, 'min': 6.878168095265195, 'std': 0.06657839974500762}}, 'matbench_jdft2d': {'mae': {'mean': 624.8594821594436, 'max': 662.8351790033564, 'min': 484.0870035426516, 'std': 70.41763851884579}, 'rmse': {'mean': 754.6594168930902, 'max': 802.851398577492, 'min': 575.4212296101125, 'std': 89.65203353138263}, 'mape': {'mean': 12.691214729498025, 'max': 22.18652735053058, 'min': 9.642403294653164, 'std': 4.833743597331997}, 'max_error': {'mean': 1455.537803743586, 'max': 1532.911339763068, 'min': 1229.7021907932801, 'std': 113.62938957056699}}, 'matbench_log_gvrh': {'mae': {'mean': 0.7503117195807093, 'max': 0.7567499426463542, 'min': 0.7458321525860483, 'std': 0.004177000349054263}, 'rmse': {'mean': 0.8922201073043177, 'max': 0.8965161788869266, 'min': 0.8860255812982848, 'std': 0.0034322474625259137}, 'mape': {'mean': 15059325260426.266, 'max': 26158506009539.293, 'min': 4541885118479.488, 'std': 6978350942510.934}, 'max_error': {'mean': 2.4294014472589063, 'max': 2.7078341735946374, 'min': 2.2460171713812693, 'std': 0.17276767393879686}}, 'matbench_log_kvrh': {'mae': {'mean': 0.8337265925158915, 'max': 0.84093059152486, 'min': 0.8252194857104939, 'std': 0.005960109281798535}, 'rmse': {'mean': 1.0122909056359641, 'max': 1.0190702248693488, 'min': 1.0038726599443502, 'std': 0.005051164726815858}, 'mape': {'mean': 5205086458416.939, 'max': 10062434398435.773, 'min': 1547244178802.9026, 'std': 3081512230166.448}, 'max_error': {'mean': 2.5243586576102204, 'max': 2.7538971602513187, 'min': 2.4510034654636557, 'std': 0.11592723389441413}}, 'matbench_mp_gap': {'mae': {'mean': 3.9947345263133185, 'max': 4.040261917311839, 'min': 3.8419572019120563, 'std': 0.07657166015944829}, 'rmse': {'mean': 4.802562096614456, 'max': 4.852934760261583, 'min': 4.621350867969822, 'std': 0.09070340779972745}, 'mape': {'mean': 9376100521414580.0, 'max': 9530024251770120.0, 'min': 9017068673849420.0, 'std': 191246529837308.34}, 'max_error': {'mean': 9.641818242181197, 'max': 9.721159283071396, 'min': 9.326360936678295, 'std': 0.15772886643823172}}, 'matbench_mp_is_metal': {'accuracy': {'mean': 0.49927909555806177, 'max': 0.5032513429459994, 'min': 0.49498185930358574, 'std': 0.002961735738880825}, 'balanced_accuracy': {'mean': 0.4995330363104962, 'max': 0.5035756141508568, 'min': 0.4951605437227332, 'std': 0.003013293507682773}, 'f1': {'mean': 0.465575654865608, 'max': 0.46982498491249247, 'min': 0.46097364715349026, 'std': 0.0031704147980028555}, 'rocauc': {'mean': 0.4995330363104962, 'max': 0.5035756141508567, 'min': 0.4951605437227331, 'std': 0.0030132935076827893}}, 'matbench_perovskites': {'mae': {'mean': 1.6494389339807394, 'max': 1.6643604327414814, 'min': 1.6083671212370563, 'std': 0.02130042981456539}, 'rmse': {'mean': 1.9895605050492304, 'max': 2.0097384860674103, 'min': 1.9348806762983708, 'std': 0.028069501175258544}, 'mape': {'mean': 8474366075980.172, 'max': 17109693350693.695, 'min': 170695202621.18396, 'std': 5913986606286.262}, 'max_error': {'mean': 5.122830203832267, 'max': 5.401364835279832, 'min': 4.933113748862263, 'std': 0.15432057817183506}}, 'matbench_phonons': {'mae': {'mean': 1442.1910745917485, 'max': 1460.5342302638428, 'min': 1404.6727173726108, 'std': 19.87835062913105}, 'rmse': {'mean': 1739.1638204522908, 'max': 1748.4453111626615, 'min': 1714.4001958506544, 'std': 12.506318415067186}, 'mape': {'mean': 4.535426963268569, 'max': 4.692503460859966, 'min': 4.356729242935622, 'std': 0.11577855407899913}, 'max_error': {'mean': 3387.1756802926197, 'max': 3490.7322416780676, 'min': 3312.8239446861567, 'std': 60.586867518772216}}, 'matbench_steels': {'mae': {'mean': 514.6879431114869, 'max': 548.5353510044772, 'min': 488.97286237333986, 'std': 24.98451122832146}, 'rmse': {'mean': 619.9832706475461, 'max': 651.1520235084482, 'min': 591.9607445092288, 'std': 24.183510586935057}, 'mape': {'mean': 0.39220921441643364, 'max': 0.4201053232886023, 'min': 0.368378839458224, 'std': 0.02076964611162295}, 'max_error': {'mean': 1331.6729147023618, 'max': 1389.1259692340998, 'min': 1272.982373277621, 'std': 40.71026078669035}}}
```

Access task metadata

```python
# number of samples
mb.matbench_dielectric.metadata.n_samples
>>> 4764

# bibtex references for source data, intemediate repos, and this package
mb.matbench_dielectric.metadata.bibtex_refs
>>> '["@Article{Dunn2020,\nauthor={Dunn, Alexander\nand Wang, Qi\nand Ganose, Alex\nand Dopp, ...'

# human-readable description of the data
mb.matbench_dielectric.metadata.description
>>> 'Matbench v0.1 test dataset for predicting refractive index from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having refractive indices less than 1 and those containing noble gases. Retrieved April 2, 2019. For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data; refer to the Automatminer/Matbench publication for more details.'

```



Easily access data for each task and fold
```python
# Look at the raw data for the second fold test set for an example benchmark on matbench_dielectric
mb.matbench_dielectric.results.fold_2.data
>>> {'mb-dielectric-0006': 39.07812613512991, 'mb-dielectric-0007': 41.33770582195621, 'mb-dielectric-0012': 45.125030619739405, ...}


# Look at the scores on the second fold test set
mb.matbench_dielectric.results.fold_2.scores
>>> {'mae': 26.50764023789047, 'rmse': 30.707221665034698, 'mape': 12.928095832225917, 'max_error': 53.98208657241693}


# Get statistics of each metric over all the folds for this problem
mb.matbench_dielectric.scores
>>> {'mae': {'mean': 29.09435441521901, 'max': 29.790913986352297, 'min': 26.50764023789047, 'std': 1.2938287761791334}, 'rmse': {'mean': 33.654269974352744, 'max': 34.44945162692406, 'min': 30.707221665034698, 'std': 1.4740060199828717}, 'mape': {'mean': 14.169387576348942, 'max': 14.56764274096521, 'min': 12.928095832225917, 'std': 0.6228030143476618}, 'max_error': {'mean': 58.85621300050616, 'max': 60.1966146990726, 'min': 53.98208657241693, 'std': 2.4395502402545453}}


# Access individual statistics easily
mb.matbench_dielectric.scores.rmse.max
>>>  34.44945162692406
```


## Step 3: Make a [PR](https://guides.github.com/activities/hello-world/#:~:text=Pull%20Requests%20are%20the%20heart,merge%20them%20into%20their%20branch.&text=You%20can%20even%20open%20pull,repository%20and%20merge%20them%20yourself.)

Follow each of the steps below to create a pull request to the [Matbench repo](https://github.com/hackingmaterials/matbench).

**Note: the files must have these names exactly for your PR to go through without problems, automatically.**


### Step 3a: Create the 3 required files
#### 1. `results.json.gz`
**This output file**, the .json from Step 2 which is automatically formatted.


#### 2. `info.json`
**A metadata file** about your algorithm, the authors, and any relevant citations. Please ensure the following keys are included, as they are required by our automated leaderboard:

  - `"authors"`: The author names for this PR
  - `"algorithm"`: The short or abbreviated name for your algorithm, e.g., `"MegNET v1.0"`. Should be 5-15 characters.
  - `"algorithm_long"`: A longer description of your algorithm, to be shown as details for your results.
  - `"bibtex_refs"`: A comprehensive list of references for your algorithm, including manuscripts and preprints for the algorithm itself, formatted as bibtex.
  - `"notes"`: Any other freeform notes you'd like to include as details for your algorithm/submission.


#### 3. `notebook.ipynb`
**A jupyter notebook** with some code for running your algorithm on Matbench. 


![example_notebook](static/notebook_example.png)

The notebook should generally follow the format of the example notebook `/benchmarks/matbench_v0.1_dummy/notebook.ipynb`. Try to include a long form, human readable description of how your algorithm works, any package versions needed to have it run correctly, and **most importantly, a link to a publication for your algorithm**. 

Aside from that, what goes in your notebook is pretty freeform; **put whatever is needed to
allow someone else to train and run your algorithm on the benchmark**. Also feel free to include extra code and small supporting files (<1MB) if they are crucial to the code working.

You can find an example for all of these files in the repo under `/benchmarks/matbench_v0.1_dummy.`



### Step 3b: Put these files in a folder in the `/benchmarks` directory.

You can put the files from [Step 3a](#step-3a-create-the-3-required-files) into a folder in `/benchmarks`. Name the file `<benchmark name>_<algorithm name>` according to your algorithm and the benchmark you ran (e.g., `matbench_v0.1`).


The files should look like:

```
├── benchmarks
│   └── matbench_v0.1_<your algorithm name>
│       ├── info.json
│       ├── notebook.ipynb
│       └── results.json.gz
```

Please make sure the names of these files are exactly as shown here. You can include any other small files (no naming scheme required) for running your code in this directory. 



### Step 3c: Make a PR to the [matbench repo](htttps://github.com/hackingmaterials/matbench)

You can find info on how to make a pull request [on GitHub's official documentation](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

![pr example](static/pr_example.png)

Your submission (output file, metadata, and notebook) will be automatically validated by the workflows on the matbench repo; once it's validated, it will be merged and will
automatically appear in the full benchmark data!