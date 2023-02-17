# matbench_v0.1: AMMExpress v2020

### Algorithm description: 

Automatminer express v1.03.20200727. Based on automatic featurization, tree-based feature reduction, and genetic-algorithm based AutoML with the TPOT package.

#### Notes:
All data was generated using the same config (express, default). The automatminer version requirement specifies the versions of many dependent packages, such as matminer, which are required for the algorithm to work in your virtualenv.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_automatminer_expressv2020).

### References (in bibtex format): 

```
('@article{Dunn2020,\n'
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
 '}')
```

### User metadata:

```
{'autofeaturizer_kwargs': {'n_jobs': 10, 'preset': 'express'},
 'cleaner_kwargs': {'feature_na_method': 'drop',
                    'max_na_frac': 0.1,
                    'na_method_fit': 'mean',
                    'na_method_transform': 'mean'},
 'learner_kwargs': {'max_eval_time_mins': 20,
                    'max_time_mins': 1440,
                    'memory': 'auto',
                    'n_jobs': 10,
                    'population_size': 200},
 'learner_name': 'TPOTAdaptor',
 'reducer_kwargs': {'reducers': ['corr', 'tree'],
                    'tree_importance_percentile': 0.99}}
```

### Metadata:

| tasks recorded | 13/13 |
|----------------|-------------------------------------|
| complete? | ✓ | 
| composition complete? | ✓ | 
| structure complete? | ✓ | 
| regression complete? | ✓ | 
| classification complete? | ✓ | 

### Software Requirements

```
{'python': ['automatminer==1.0.3.20200727', 'matbench==0.1.0']}
```

### Task data:

#### `matbench_dielectric`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2188| 0.6855| 0.0760| 14.6654 |
 | fold_1 | 0.2844| 1.0764| 0.0899| 19.6283 |
 | fold_2 | 0.4257| 2.9472| 0.0889| 59.0112 |
 | fold_3 | 0.3198| 2.2782| 0.0720| 53.5196 |
 | fold_4 | 0.3264| 1.6137| 0.0987| 28.1601 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.3150 | 0.4257 | 0.2188 | 0.0672 |
| rmse | 1.7202 | 2.9472 | 0.6855 | 0.8140 |
| mape* | 0.0851 | 0.0987 | 0.0720 | 0.0098 |
| max_error | 34.9969 | 59.0112 | 14.6654 | 17.9782 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.006, score_func=<function f_regression at 0x2aaaef1a0840>))', '(robustscaler, RobustScaler(copy=true, quantile_range=(25.0, 75.0), with_centering=true...` |
| fold_1 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(zerocount, ZeroCount())', '(gradientboostingregressor, GradientBoostingRegressor(alpha=0.75, criterion=friedman_mse, in...` |
| fold_2 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.001))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(gradientboostingregressor, GradientBoostingRegressor(al...` |
| fold_3 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.023, score_func=<function f_regression at 0x2aaaef19f950>))', '(standardscaler, StandardScaler(copy=true, with_mean=true, with_std=true))', '(gradient...` |
| fold_4 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.034, score_func=<function f_regression at 0x2aaaf35a08c8>))', '(zerocount, ZeroCount())', '(gradientboostingregressor, GradientBoostingRegressor(alpha...` |




#### `matbench_expt_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.3998| 0.9435| 0.3372| 8.0111 |
 | fold_1 | 0.4061| 0.9354| 0.3085| 8.6887 |
 | fold_2 | 0.4538| 1.0955| 0.3916| 12.7533 |
 | fold_3 | 0.4061| 1.0273| 0.3019| 12.6296 |
 | fold_4 | 0.4150| 0.9573| 0.4503| 6.0779 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.4161 | 0.4538 | 0.3998 | 0.0194 |
| rmse | 0.9918 | 1.0955 | 0.9354 | 0.0612 |
| mape* | 0.3579 | 0.4503 | 0.3019 | 0.0560 |
| max_error | 9.6321 | 12.7533 | 6.0779 | 2.6411 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.035, score_func=<function f_regression at 0x2aaaf35a18c8>))', '(standardscaler, StandardScaler(copy=true, with_mean=true, with_std=true))', '(gradient...` |
| fold_1 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.046, score_func=<function f_regression at 0x2aaaef19f8c8>))', '(onehotencoder, OneHotEncoder(categorical_features=[false, false, false, false, false, ...` |
| fold_2 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0005))', '(robustscaler, RobustScaler(copy=true, quantile_range=(25.0, 75.0), with_centering=true,\n             with_scaling=true...` |
| fold_3 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=85,\n                 score_func=<function f_regression at 0x2aaaf39a38c8>))', '(onehotencoder, OneHotEncoder(categorical_features=[f...` |
| fold_4 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0005))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(randomforestregressor, RandomForestRegressor(bootstrap=false, criterion=mse,...` |




#### `matbench_expt_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9218| 0.9218| 0.9205| 0.9218 |
 | fold_1 | 0.9157| 0.9156| 0.9145| 0.9156 |
 | fold_2 | 0.9207| 0.9207| 0.9193| 0.9207 |
 | fold_3 | 0.9228| 0.9228| 0.9223| 0.9228 |
 | fold_4 | 0.9238| 0.9238| 0.9235| 0.9238 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9210 | 0.9238 | 0.9157 | 0.0028 |
| balanced_accuracy | 0.9209 | 0.9238 | 0.9156 | 0.0028 |
| f1 | 0.9200 | 0.9235 | 0.9145 | 0.0031 |
| rocauc | 0.9209 | 0.9238 | 0.9156 | 0.0028 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.009000000000000001,\n          score_func=<function f_classif at 0x2aaaf35a16a8>))', '(onehotencoder, OneHotEncoder(categorical_features=[false, false...` |
| fold_1 | `{'best_pipeline': ['(rfe, RFE(estimator=ExtraTreesClassifier(bootstrap=false, class_weight=null,\n                                   criterion=gini, max_depth=null,\n                                  ...` |
| fold_2 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.001))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(gradientboostingclassifier, GradientBoostingClassifier(criterion=friedman_mse...` |
| fold_3 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.03, score_func=<function f_classif at 0x2aaaf35a0730>))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(gradientboostingclassifier, GradientBoostingCla...` |
| fold_4 | `{'best_pipeline': ['(rfe, RFE(estimator=ExtraTreesClassifier(bootstrap=false, class_weight=null,\n                                   criterion=entropy, max_depth=null,\n                               ...` |




#### `matbench_glass`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.8283| 0.8441| 0.8697| 0.8441 |
 | fold_1 | 0.8125| 0.8383| 0.8548| 0.8383 |
 | fold_2 | 0.8574| 0.8546| 0.8956| 0.8546 |
 | fold_3 | 0.9173| 0.8742| 0.9437| 0.8742 |
 | fold_4 | 0.9375| 0.8921| 0.9579| 0.8921 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.8706 | 0.9375 | 0.8125 | 0.0490 |
| balanced_accuracy | 0.8607 | 0.8921 | 0.8383 | 0.0199 |
| f1 | 0.9043 | 0.9579 | 0.8548 | 0.0404 |
| rocauc | 0.8607 | 0.8921 | 0.8383 | 0.0199 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(selectfrommodel, SelectFromModel(estimator=ExtraTreesClassifier(bootstrap=false,\n                                               class_weight=null,\n                              ...` |
| fold_1 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(standardscaler, StandardScaler(copy=true, with_mean=true, with_std=true))', '(extratreesclassifier, ExtraTreesClassifie...` |
| fold_2 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=74,\n                 score_func=<function f_classif at 0x2aaaf35a0730>))', '(onehotencoder, OneHotEncoder(categorical_features=[fals...` |
| fold_3 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(standardscaler, StandardScaler(copy=true, with_mean=true, with_std=true))', '(gradientboostingclassifier, GradientBoost...` |
| fold_4 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(standardscaler, StandardScaler(copy=true, with_mean=true, with_std=true))', '(gradientboostingclassifier, GradientBoost...` |




#### `matbench_jdft2d`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 29.5070| 57.7719| 18.9726| 362.2752 |
 | fold_1 | 44.3036| 98.1137| 0.3191| 551.7742 |
 | fold_2 | 54.4690| 164.0162| 0.5117| 847.0618 |
 | fold_3 | 28.0759| 55.8345| 0.2371| 316.2185 |
 | fold_4 | 42.8931| 156.9938| 0.5429| 1552.9102 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 39.8497 | 54.4690 | 28.0759 | 9.8835 |
| rmse | 106.5460 | 164.0162 | 55.8345 | 46.6251 |
| mape* | 4.1167 | 18.9726 | 0.2371 | 7.4289 |
| max_error | 726.0480 | 1552.9102 | 316.2185 | 453.6535 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(gradientboostingregressor, GradientBoostingRegressor(alpha...` |
| fold_1 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=40,\n                 score_func=<function f_regression at 0x2aaaf35a08c8>))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(gradientb...` |
| fold_2 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=62,\n                 score_func=<function f_regression at 0x2aaaf35a08c8>))', '(onehotencoder, OneHotEncoder(categorical_features=[f...` |
| fold_3 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=82,\n                 score_func=<function f_regression at 0x2aab561f6620>))', '(robustscaler, RobustScaler(copy=true, quantile_range...` |
| fold_4 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=62,\n                 score_func=<function f_regression at 0x2aaaf35a08c8>))', '(zerocount, ZeroCount())', '(gradientboostingregresso...` |




#### `matbench_log_gvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0891| 0.1270| 0.0692| 1.1580 |
 | fold_1 | 0.0852| 0.1261| 0.0666| 1.0887 |
 | fold_2 | 0.0849| 0.1261| 0.0668| 0.9631 |
 | fold_3 | 0.0884| 0.1279| 0.0670| 0.8959 |
 | fold_4 | 0.0894| 0.1313| 0.0690| 0.9810 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0874 | 0.0894 | 0.0849 | 0.0020 |
| rmse | 0.1277 | 0.1313 | 0.1261 | 0.0019 |
| mape* | 0.0677 | 0.0692 | 0.0666 | 0.0012 |
| max_error | 1.0173 | 1.1580 | 0.8959 | 0.0937 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.2))', '(zerocount, ZeroCount())', '(gradientboostingregressor, GradientBoostingRegressor(alpha=0.99, criterion=friedman_mse, init=...` |
| fold_1 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.01))', '(robustscaler, RobustScaler(copy=true, quantile_range=(25.0, 75.0), with_centering=true,\n             with_scaling=true))...` |
| fold_2 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.01, score_func=<function f_regression at 0x2aaaef19e8c8>))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(randomforestregressor,...` |
| fold_3 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(standardscaler, StandardScaler(copy=true, with_mean=true, with_std=true))', '(gradientboostingregressor, GradientBoosti...` |
| fold_4 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=96,\n                 score_func=<function f_regression at 0x2aaaf35a08c8>))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(extratree...` |




#### `matbench_log_kvrh`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.0639| 0.1179| 0.0417| 1.4823 |
 | fold_1 | 0.0659| 0.1231| 0.0432| 1.2686 |
 | fold_2 | 0.0627| 0.1115| 0.0411| 1.1316 |
 | fold_3 | 0.0668| 0.1217| 0.0464| 1.1890 |
 | fold_4 | 0.0640| 0.1172| 0.0417| 1.4335 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.0647 | 0.0668 | 0.0627 | 0.0015 |
| rmse | 0.1183 | 0.1231 | 0.1115 | 0.0041 |
| mape* | 0.0428 | 0.0464 | 0.0411 | 0.0019 |
| max_error | 1.3010 | 1.4823 | 1.1316 | 0.1362 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.032, score_func=<function f_regression at 0x2aaaf35a2840>))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(extratreesregressor, ...` |
| fold_1 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.029, score_func=<function f_regression at 0x2aaaf35a08c8>))', '(zerocount, ZeroCount())', '(gradientboostingregressor, GradientBoostingRegressor(alpha...` |
| fold_2 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.2))', '(onehotencoder, OneHotEncoder(categorical_features=[false, false, false, false, false, false,\n                            ...` |
| fold_3 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.016, score_func=<function f_regression at 0x2aaaf79a28c8>))', '(onehotencoder, OneHotEncoder(categorical_features=[false, false, false, false, false, ...` |
| fold_4 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(extratreesregressor, ExtraTreesRegressor(bootstrap=fal...` |




#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.1586| 0.2508| 1.0829| 4.0713 |
 | fold_1 | 0.2026| 0.2955| 0.9253| 5.8108 |
 | fold_2 | 0.1473| 0.2256| 0.7722| 2.7696 |
 | fold_3 | 0.2080| 0.3062| 1.3958| 5.5190 |
 | fold_4 | 0.1467| 0.2226| 0.8028| 3.3888 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.1726 | 0.2080 | 0.1467 | 0.0270 |
| rmse | 0.2602 | 0.3062 | 0.2226 | 0.0348 |
| mape* | 0.9958 | 1.3958 | 0.7722 | 0.2280 |
| max_error | 4.3119 | 5.8108 | 2.7696 | 1.1826 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(gradientboostingregressor, GradientBoostingRegressor(alpha=0.75, criterion=friedman_mse, init=null,\n             learning_rate=0.5, loss=huber, max_depth=5,\n             max_fea...` |
| fold_1 | `{'best_pipeline': ['(polynomialfeatures, PolynomialFeatures(degree=2, include_bias=false, interaction_only=false))', '(pca, PCA(copy=true, iterated_power=3, n_components=null, random_state=null,\n  sv...` |
| fold_2 | `{'best_pipeline': ['(stackingestimator, StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, criterion=friedman_mse, init=null,\n             learning_rate=0.5, loss=huber, max_depth=4,\n ...` |
| fold_3 | `{'best_pipeline': ['(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(selectfwe, SelectFwe(alpha=0.027, score_func=<function f_regression at 0x2b2eb18422f0>))', '(stackingestimator, St...` |
| fold_4 | `{'best_pipeline': ['(xgbregressor, XGBRegressor(base_score=0.5, booster=gbtree, colsample_bylevel=1,\n       colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=0,\n       max_depth=5, min_...` |




#### `matbench_mp_gap`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2799| 0.5481| 3.5712| 5.4792 |
 | fold_1 | 0.2850| 0.5671| 3.1533| 6.9105 |
 | fold_2 | 0.2724| 0.5477| 4.6097| 6.2045 |
 | fold_3 | 0.2909| 0.5710| 10.0191| 6.4590 |
 | fold_4 | 0.2837| 0.5714| 6.8322| 5.5333 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2824 | 0.2909 | 0.2724 | 0.0061 |
| rmse | 0.5611 | 0.5714 | 0.5477 | 0.0109 |
| mape* | 5.6371 | 10.0191 | 3.1533 | 2.5347 |
| max_error | 6.1173 | 6.9105 | 5.4792 | 0.5480 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(stackingestimator-1, StackingEstimator(estimator=RandomForestRegressor(bootstrap=false, criterion=mse, max_depth=null,\n           max_features=0.4, max_leaf_nodes=null,\n        ...` |
| fold_1 | `{'best_pipeline': ['(stackingestimator-1, StackingEstimator(estimator=RandomForestRegressor(bootstrap=true, criterion=mse, max_depth=null,\n           max_features=0.35000000000000003, max_leaf_nodes=...` |
| fold_2 | `{'best_pipeline': ['(stackingestimator-1, StackingEstimator(estimator=RandomForestRegressor(bootstrap=false, criterion=mse, max_depth=null,\n           max_features=0.45, max_leaf_nodes=null,\n       ...` |
| fold_3 | `{'best_pipeline': ['(stackingestimator-1, StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=false, criterion=mse, max_depth=null,\n          max_features=0.45, max_leaf_nodes=null,\n          ...` |
| fold_4 | `{'best_pipeline': ['(stackingestimator-1, StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.85, criterion=friedman_mse, init=null,\n             learning_rate=0.01, loss=lad, max_depth=1,\...` |




#### `matbench_mp_is_metal`

###### Fold scores

| fold | accuracy | balanced_accuracy | f1 | rocauc |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.9133| 0.9094| 0.8982| 0.9094 |
 | fold_1 | 0.9123| 0.9086| 0.8972| 0.9086 |
 | fold_2 | 0.9129| 0.9089| 0.8976| 0.9089 |
 | fold_3 | 0.9146| 0.9108| 0.8998| 0.9108 |
 | fold_4 | 0.9129| 0.9086| 0.8974| 0.9086 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| accuracy | 0.9132 | 0.9146 | 0.9123 | 0.0008 |
| balanced_accuracy | 0.9093 | 0.9108 | 0.9086 | 0.0008 |
| f1 | 0.8981 | 0.8998 | 0.8972 | 0.0009 |
| rocauc | 0.9093 | 0.9108 | 0.9086 | 0.0008 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(randomforestclassifier, RandomForestClassifier(bootstrap=false, class_weight=null,\n            criterion=entropy,...` |
| fold_1 | `{'best_pipeline': ['(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(randomforestclassifier, RandomForestClassifier(bootstrap=false, class_weight=null,\n            criterion=entropy,...` |
| fold_2 | `{'best_pipeline': ['(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(randomforestclassifier, RandomForestClassifier(bootstrap=false, class_weight=null,\n            criterion=entropy,...` |
| fold_3 | `{'best_pipeline': ['(stackingestimator, StackingEstimator(estimator=RandomForestClassifier(bootstrap=false, class_weight=null,\n            criterion=entropy, max_depth=null, max_features=0.5,\n      ...` |
| fold_4 | `{'best_pipeline': ['(featureunion, FeatureUnion(n_jobs=null,\n       transformer_list=[(functiontransformer, FunctionTransformer(accept_sparse=false, check_inverse=true,\n          func=<function copy...` |




#### `matbench_perovskites`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.2159| 0.3114| 0.2077| 2.7651 |
 | fold_1 | 0.1904| 0.2857| 0.1944| 2.6783 |
 | fold_2 | 0.1962| 0.2869| 0.1933| 2.4466 |
 | fold_3 | 0.1992| 0.2907| 0.2209| 3.3116 |
 | fold_4 | 0.2006| 0.3023| 0.1886| 2.4386 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.2005 | 0.2159 | 0.1904 | 0.0085 |
| rmse | 0.2954 | 0.3114 | 0.2857 | 0.0099 |
| mape* | 0.2010 | 0.2209 | 0.1886 | 0.0118 |
| max_error | 2.7280 | 3.3116 | 2.4386 | 0.3186 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(robustscaler, RobustScaler(copy=true, quantile_range=(25.0, 75.0), with_centering=true,\n             with_scaling=true))'...` |
| fold_1 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(zerocount, ZeroCount())', '(randomforestregressor, RandomForestRegressor(bootstrap=false, criterion=mse, max_depth=null,\n...` |
| fold_2 | `{'best_pipeline': ['(selectfwe, SelectFwe(alpha=0.03, score_func=<function f_regression at 0x2aaaf35a08c8>))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(gradientboostingregres...` |
| fold_3 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(robustscaler, RobustScaler(copy=true, quantile_range=(25.0, 75.0), with_centering=true,\n             with_scaling=true))'...` |
| fold_4 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.05))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(randomforestregressor, RandomForestRegressor(bootstrap=false, criterion=mse, m...` |




#### `matbench_phonons`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 67.5727| 146.7970| 0.1079| 1151.5570 |
 | fold_1 | 54.0755| 100.2097| 0.1048| 890.4159 |
 | fold_2 | 50.9853| 96.5991| 0.0931| 680.9361 |
 | fold_3 | 59.6458| 127.8555| 0.1142| 926.0969 |
 | fold_4 | 48.5738| 77.0626| 0.0958| 383.1912 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 56.1706 | 67.5727 | 48.5738 | 6.7981 |
| rmse | 109.7048 | 146.7970 | 77.0626 | 24.6280 |
| mape* | 0.1032 | 0.1142 | 0.0931 | 0.0078 |
| max_error | 806.4394 | 1151.5570 | 383.1912 | 258.9850 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.01))', '(robustscaler, RobustScaler(copy=true, quantile_range=(25.0, 75.0), with_centering=true,\n             with_scaling=true))...` |
| fold_1 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.005))', '(maxabsscaler, MaxAbsScaler(copy=true))', '(gradientboostingregressor, GradientBoostingRegressor(alpha=0.8, criterion=fri...` |
| fold_2 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(extratreesregressor, ExtraTreesRegressor(bootstrap=false,...` |
| fold_3 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.0001))', '(onehotencoder, OneHotEncoder(categorical_features=[false, false, false, false, false, false,\n                         ...` |
| fold_4 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.2))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(extratreesregressor, ExtraTreesRegressor(bootstrap=false,...` |




#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 109.3058| 188.8049| 0.0693| 1082.7703 |
 | fold_1 | 80.4188| 109.2771| 0.0569| 416.3620 |
 | fold_2 | 83.5360| 120.2935| 0.0607| 424.5913 |
 | fold_3 | 98.7186| 136.5898| 0.0722| 473.4563 |
 | fold_4 | 115.4851| 215.1149| 0.0891| 1142.9223 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 97.4929 | 115.4851 | 80.4188 | 13.7919 |
| rmse | 154.0161 | 215.1149 | 109.2771 | 40.9531 |
| mape* | 0.0696 | 0.0891 | 0.0569 | 0.0112 |
| max_error | 708.0205 | 1142.9223 | 416.3620 | 331.6607 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{'best_pipeline': ['(selectfrommodel, SelectFromModel(estimator=ExtraTreesRegressor(bootstrap=false, criterion=mse,\n                                              max_depth=null,\n                    ...` |
| fold_1 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(fastica, FastICA(algorithm=parallel, fun=logcosh, fun_args=null, max_iter=200,\n        n_components=null, random_state=nu...` |
| fold_2 | `{'best_pipeline': ['(selectpercentile, SelectPercentile(percentile=53,\n                 score_func=<function f_regression at 0x2aaaf79a38c8>))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=...` |
| fold_3 | `{'best_pipeline': ['(variancethreshold, VarianceThreshold(threshold=0.1))', '(minmaxscaler, MinMaxScaler(copy=true, feature_range=(0, 1)))', '(kneighborsregressor, KNeighborsRegressor(algorithm=auto, ...` |
| fold_4 | `{'best_pipeline': ['(selectfrommodel, SelectFromModel(estimator=ExtraTreesRegressor(bootstrap=false, criterion=mse,\n                                              max_depth=null,\n                    ...` |




