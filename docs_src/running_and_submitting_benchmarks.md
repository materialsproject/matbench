

*This documentation si a work in progress. Please check back later for more detailed tutorials.*


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

mb.to_file("my_models_benchmark.json")

```

The output file, in this case `my_models_benchmark.json` contains everything predicted by your 
benchmark. **Keep this file, as it is the core result that will be submitted to the leaderboard.**


#### Note: Benchmark subsets
If you want to benchmark on a subset of Matbench tasks, set the `subset` argument when creating `MatbenchBenchmark`
and use the same code as above. The repo accepts subsets of matbench tasks as well which will appear on a separate "task-specific" leaderboard.


## Step 3: Make a [PR](https://guides.github.com/activities/hello-world/#:~:text=Pull%20Requests%20are%20the%20heart,merge%20them%20into%20their%20branch.&text=You%20can%20even%20open%20pull,repository%20and%20merge%20them%20yourself.)

Make a pull request to the [Matbench repo](https://github.com/hackingmaterials/matbench) with the following 3 items:

- **the output file**, the .json from Step 2 which is automatically formatted.
- **a reference** to a peer-reviewed publication or preprint describing your algorithm.
- **a jupyter notebook** with some code for running your algorithm on Matbench


Your output file will be automatically validated by the workflows on the matbench repo; once it's validated, it will be merged and will
automatically appear in the [Full Benchmark Data](full_benchmark_data.md)!