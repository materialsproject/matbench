from matbench.metadata import metadata


'''
Core functions for benchmarking.
---


Final, bedrock truth results JSON format:

{
    "raw": {
        "matbench_steels": {fold0: {data: { index1: prediction1, index2: prediction2...}, params}, fold1: ...}
        "matbench_dielectric": {fold0: {data: { index1: prediction1, index2: prediction2...}, params}, fold1: ...}
        ...
    },
    "processed": {
        fold0: {rmse: rmse1, mae: mae1, mape: mape1, ...}
        fold1: {rmse: rmse2, mae: mae2, mape: mape2, ...}
        ...
        mean: {rmse: mean_rmse, mae: mean_mae, ...}
        std: {rmse: std_rmse, mae: std_mae, ...}
        ...
    },
    "metadata": {
        "is_full" : True if it was a full benchmark
    }
    
    
    
Usage should be something like:


mb = MatbenchBenchmark()

for task in mb.tasks:
    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        my_model.train_and_validate(train_inputs, train_outputs)
        
        test_inputs = task.get_test_data(fold)
        predictions = my_model.predict(test_inputs)
        
        task.record(predictions, params=my_model.hyperparams)  
mb.score()


# holds all the scoring info
mb.scores

# access the raw results of an individual test
mb.tasks.matbench_dielectric.results.fold0.data
mb.tasks.matbench_dielectric.results.fold0.params


# access the individual scores of a fold
mb.tasks.matbench_steels.scores.fold0

'''


class MatbenchBenchmark:

    def __init__(self):
        self.metadata = metadata


        #todo
        self.folds = "something"


    def


