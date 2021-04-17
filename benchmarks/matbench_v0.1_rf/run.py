# Example py file


from automatminer import MatPipe
from automatminer.automl.adaptors import SinglePipelineAdaptor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matbench.bench import MatbenchBenchmark



learner = SinglePipelineAdaptor(
                regressor=RandomForestRegressor(**learner_kwargs),
                classifier=RandomForestClassifier(**learner_kwargs),
            )
pipe_config = {
            "learner": learner,
            "reducer": FeatureReducer(**reducer_kwargs),
            "cleaner": DataCleaner(**cleaner_kwargs),
            "autofeaturizer": AutoFeaturizer(**autofeaturizer_kwargs),
        }

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    for fold in task.folds:

        # Inputs are either chemical compositions as strings
        # or crystal structures as pymatgen.Structure objects.
        # Outputs are either floats (regression tasks) or bools (classification tasks)
        train_inputs, train_outputs = task.get_train_and_val_data(fold)

        # train and validate your model
        my_model.train_and_validate(train_inputs, train_outputs)

        # Get testing data
        test_inputs = task.get_test_data(fold, include_target=False)

        # Predict on the testing data
        # Your output should be a pandas series, numpy array, or python iterable
        # where the array elements are floats or bools
        predictions = my_model.predict(test_inputs)

        # Record your data!
        task.record(fold, predictions)

# Save your results
mb.to_file("my_models_benchmark.json.gz")


