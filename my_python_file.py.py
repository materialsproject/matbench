"""
Code for training and recording the matbench_v0.1 random forest benchmark.

The ML pipeline is placed within the Automatminer pipeline code infrastructure for convenience.

All training and inference was done on a single 128-core HPC node.

Reduce the number of jobs n_jobs for less memory usage on consumer machines.
"""

if __name__ == '__main__':
    from automatminer import MatPipe
    from automatminer.automl.adaptors import SinglePipelineAdaptor, TPOTAdaptor
    from automatminer.featurization import AutoFeaturizer
    from automatminer.preprocessing import DataCleaner, FeatureReducer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from matbench.bench import MatbenchBenchmark
    from multiprocessing import set_start_method
    
    set_start_method("spawn", force=True)

    # The learner is a single 500-estimator Random Forest model
    learner = SinglePipelineAdaptor(
                    regressor=RandomForestRegressor(n_estimators=500),
                    classifier=RandomForestClassifier(n_estimators=500),
                )
    pipe_config = {
                "learner": learner,
                "reducer": FeatureReducer(reducers=[]),
                "cleaner": DataCleaner(feature_na_method="mean", max_na_frac=0.01, na_method_fit="drop", na_method_transform="mean"),
                "autofeaturizer": AutoFeaturizer(n_jobs=8, preset="debug"),
            }

    pipe = MatPipe(**pipe_config)

    mb = MatbenchBenchmark(autoload=False)

    i = 0

    #for task in mb.tasks:
    task = mb.matbench_jdft2d
    print(task)
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

    mb.to_file("results_" + str(i) + ".json.gz")
    i += 1

    # Save your results
    mb.to_file("results.json.gz")


