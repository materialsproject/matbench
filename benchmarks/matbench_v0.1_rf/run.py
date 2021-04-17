"""
Code for training and recording the matbench_v0.1 random forest benchmark.

The ML pipeline is placed within the Automatminer pipeline code infrastructure for convenience.

All training and inference was done on a single 128-core HPC node.

Reduce the number of jobs n_jobs for less memory usage on consumer machines.
"""


from automatminer import MatPipe
from automatminer.featurization import AutoFeaturizer
from automatminer.automl.adaptors import SinglePipelineAdaptor
from automatminer.preprocessing import DataCleaner, FeatureReducer
from automatminer.automl.adaptors import TPOTAdaptor, SinglePipelineAdaptor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from matbench.bench import MatbenchBenchmark



# The learner is a single 500-estimator Random Forest model
learner = SinglePipelineAdaptor(
                regressor=RandomForestRegressor(n_estimators=500),
                classifier=RandomForestClassifier(n_estimators=500),
            )
pipe_config = {
            "learner": learner,
            "reducer": FeatureReducer(reducers=[]),
            "cleaner": DataCleaner(feature_na_method="mean", max_na_frac=0.01, na_method_fit="drop", na_method_transform="mean"),
            "autofeaturizer": AutoFeaturizer(n_jobs=10, preset="debug"),
        }

pipe = MatPipe(**pipe_config)

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    for fold in task.folds:

        df_train = task.get_train_and_val_data(fold, as_type="df")

        # Fit the RF with matpipe
        pipe.fit(df_train, task.metadata.target)

        df_test = task.get_test_data(fold, include_target=False, as_type="df")
        predictions = pipe.predict(df_test)[f"{task.metadata.target} predicted"]

        # A single configuration is used
        params = {'note': 'single config; see benchmark user metadata'}

        task.record(fold, predictions, params=params)

# Save your results
mb.to_file("results.json.gz")


