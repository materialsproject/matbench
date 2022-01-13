# Notes on Benchmarking


## General-purpose vs Task-specific algorithms

"General purpose" algorithms are treated differently from task-specific algorithms in Matbench for the purposes of ranking.

We make this distinction because some algorithms can be trained and used - in theory - for predicting **any property** of a material as long as they are trained on sufficient data.
Others are specialized for particular domains and need a separate comparison for fair analysis.


### General purpose algorithms 

General purpose algorithms are valid for all the tasks in a benchmark **using the same human-chosen configuration**.
Beyond defining a single configuration before beginning a benchmark, a human should not be hand-tuning or informing the algorithm about architecture, parameters, or hyperparameters.
However, general purpose algorithms can automatically determine hyperparameters and parameters as part of their fitting processes in each fold.


**We include algorithms as "general purpose" to include on the general purpose leaderboard if any one of the following criteria is met for Matbench v0.1:**

 - All 13 tasks are recorded, OR...
 - All 10 regression tasks are recorded, OR...
 - All 9 structure tasks are recorded. If only the 9 structure tasks are recorded, the algorithm is marked with "requires structure".


General purpose algorithms' results will appear on both the [General Purpose Leaderboard](../index.md) as well as the Task-specific leaderboards.


### Task specific algorithms

Task-specific algorithms can fit on any subset of tasks; for example, a single task. Task-specific algorithms may be valid or specialized only for a subset of the tasks in the benchmark.  

For example, if you have a model which was specifically created for predicting bulk metallic glasses, you may submit a benchmark containing only results for the `matbench_glass` dataset. 

Task-specific results **will only appear on the Task-specific leaderboards, not on the General Purpose Leaderboard**. 


---

## Why MAE?

Mean absolute error was chosen as the ranking metric for regression because:

1. The meaning of MAE is the most easily inferred
2. Dataset targets which should be analyzed according to relative error (such as bulk moduli) have their target transformed to order-of-magnitude form (e.g., log10).
3. MAE are valid for all target values, unlike mean absolute percentage errors, which are invalid for 0-valued targets.

That being said, other error metrics are also informative beyond what MAE can offer. Therefore, Matbench offers multiple error metrics to help assess generalization error.

---

## Mean absolute percentage error (`mape*`) scores
Mean absolute percentage error is only valid on sets of data without any true values of zero.
Also, small true values can result in very large MAPE for samples with even very small predicted error.
A threshold of 1e-5 is applied to mask samples with true absolute values smaller than this from the
MAPE calculation. The reported MAPE is the decimal (not percentage) among these masked samples; i.e., 
a MAPE of 11% corresponds to `mape*=0.11`, and a MAPE of 11,000% corresponds to `mape*=110`.

**Please use the given MAPE scores with a grain of salt, as they are not complete for the reason given above**.





