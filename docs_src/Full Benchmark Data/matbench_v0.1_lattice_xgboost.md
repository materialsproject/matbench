# matbench_v0.1: Lattice-XGBoost

### Algorithm description: 

eXtreme Gradient Boosting trees (XGBoost) is applied on basic tabular data that describes the crystal lattice of each compound: lattice parameter lengths and angles, space group number, and unit cell volume. Fixed XGBoost hyperparameters were used. This serves as part of a baseline to answer the question: how much predictive performance is present in the basic details of a crystal lattice (i.e. no composition, no site information)?

#### Notes:
Designed for use on the `matbench_mp_e_form` task as an alternative perspective in a more established domain (i.e. model accuracy) to that of generative model benchmarking. This is specifically part of a series of baselines and tests related to the xtal2png (https://xtal2png.readthedocs.io/en/latest/) representation.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_lattice_xgboost).

### References (in bibtex format): 

```
('@inproceedings{Chen:2016:XST:2939672.2939785,\n'
 ' author = {Chen, Tianqi and Guestrin, Carlos},\n'
 ' title = {{XGBoost}: A Scalable Tree Boosting System},\n'
 ' booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on '
 'Knowledge Discovery and Data Mining},\n'
 " series = {KDD '16},\n"
 ' year = {2016},\n'
 ' isbn = {978-1-4503-4232-2},\n'
 ' location = {San Francisco, California, USA},\n'
 ' pages = {785--794},\n'
 ' numpages = {10},\n'
 ' url = {http://doi.acm.org/10.1145/2939672.2939785},\n'
 ' doi = {10.1145/2939672.2939785},\n'
 ' acmid = {2939785},\n'
 ' publisher = {ACM},\n'
 ' address = {New York, NY, USA},\n'
 ' keywords = {large-scale machine learning}\n'
 '}, @article{article,\n'
 ' author = {Ong, Shyue and Richards, William and Jain, Anubhav and Hautier, '
 'Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and '
 'Chevrier, Vincent and Persson, Kristin and Ceder, Gerbrand},\n'
 ' year = {2013},\n'
 ' month = {02},\n'
 ' pages = {314-319},\n'
 ' title = {Python Materials Genomics (pymatgen): A robust, open-source python '
 'library for materials analysis},\n'
 ' volume = {68},\n'
 ' journal = {Computational Materials Science},\n'
 ' doi = {10.1016/j.commatsci.2012.10.028}}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 1/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'python': ['matbench==0.5.0',
            'scikit-learn==1.1.1',
            'xgboost==1.6.1',
            'pandas==1.4.2',
            'numpy==1.22.1 ',
            'typing==3.10.5']}
```

### Task data:

#### `matbench_mp_e_form`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------ |------ |------ |------ |------ |
 | fold_0 | 0.7549| 0.9441| 8.1981| 4.0717 |
 | fold_1 | 0.7464| 0.9374| 4.8884| 4.2425 |
 | fold_2 | 0.7560| 0.9454| 5.8547| 3.9905 |
 | fold_3 | 0.7465| 0.9363| 7.3725| 4.0495 |
 | fold_4 | 0.7536| 0.9441| 8.2081| 3.9335 |


###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 0.7515 | 0.7560 | 0.7464 | 0.0042 |
| rmse | 0.9415 | 0.9454 | 0.9363 | 0.0038 |
| mape* | 6.9044 | 8.2081 | 4.8884 | 1.3235 |
| max_error | 4.0575 | 4.2425 | 3.9335 | 0.1043 |


###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




