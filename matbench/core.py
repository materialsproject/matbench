'''
Core functions for benchmarking.



User:
{
    "matbench_steels: {fold0: {data: fold0_df, params: {}},  {fold1: fold1_df, params: {}}, fold3: ...},
    ...
    [[any number of these]]
}

params is a freeform dictionary representing the hyperparameters of the model, which may have been determined via automated processes.

Each df should be of the form

    index    target prop   structure obj/composition  predicted target prop

    10       3.751         Al2O3                      3.013
    1022     4.013         Ce3Fe4                     4.918
    857      1.001         TiO2                       0.888
    1747     2.567         BaTiO4                     2.765


---


Final, bedrock truth results format:

{
    "raw": {
        "matbench_steels": {fold0: {data: { index1: prediction1, index2: prediction2...}, params, fold1: ...}
        "matbench_dielectric": {fold0: { index1: prediction1, index2: prediction2...}, fold1: ...}
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

'''




def get_splits(problem_name):
    """

    Args:
        problem_name:

    Returns:

    data.train/val/test
    """
    pass


def score(predictions):
    pass


def mb_results_to_json(predictions)


