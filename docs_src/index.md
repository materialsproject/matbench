# MatBench v0.1 - Machine Learning Benchmark for Materials Science

## Overview

[MatBench](https://doi.org/10.1038/s41524-020-00406-3) is an [ImageNet](http://www.image-net.org) for **materials science**; a
**curated set of 13 supervised, pre-cleaned, ready-to-use ML tasks** for benchmarking and fair comparison. The tasks span a wide domain of
inorganic materials science applications including electronic, thermodynamic, mechanical, and thermal properties among crystals, 2D materials,
disordered metals, and more.  

**The [MatBench python package](https://github.com/hackingmaterials/matbench) provides everything needed to use MatBench with your ML algorithm in ~10 lines of code or less.**

![infographic](static/infographic_matbench.png)



## What can MatBench offer?

This website offers a 

The diversity of MatBench's tasks allows materials properties prediction algorithms to 



## Summary of MatBench's Tasks

MatBench's 13 tasks can be broken down into various categories; it includes both the small - less than 10,000 samples - datasets that characterize
experimental materials data as well as larger datasets from computer modelling methods like density functional theory (DFT).


![breakdown](static/datasets_breakdown_inverted.png)


Each task in MatBench consists of a three things:

1. **A set of inputs:** crystal structures or compositions.
2. **A set of outputs:** target properties, such as formation energy.
3. **A test procedure:** a way to get a score for your algorithm


The MatBench Python package provides functions for getting the first two (packaged together for each task as a _dataset_) as well as running 
the test procedure. See the "How to use MatBench" documentation page to get started.


## Leaderboard

Last updated: 

| task name                | verified top score (MAE or ROCAUC) | algorithm name, config,             | is algorithm general purpose? (same config on all problems) |
|--------------------------|------------------------------------|-------------------------------------|-------------------------------------------------------------|
| `matbench_dielectric`    | 0.299 (unitless)                   | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_expt_gap`      | 0.416 eV                           | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_expt_is_metal` | 0.92                               | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_glass`         | 0.861                              | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_jdft2d`        | 38.6 meV/atom                      | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_log_gvrh`      | 0.0849 log(GPa)                    | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_log_kvrh`      | 0.0679 log(GPa)                    | Automatminer express v1.0.3.2019111 | yes                                                         |
| `matbench_mp_e_form`     | 0.0327 eV/atom                     | MEGNet v0.2.2                       | yes, structure only                                         |
| `matbench_mp_gap`        | 0.228 eV                           | CGCNN (2019)                        | yes, structure only                                         |
| `matbench_mp_is_metal`   | 0.977                              | MEGNet v0.2.2                       | yes, structure only                                         |
| `matbench_perovskites`   | 0.0417                             | MEGNet v0.2.2                       | yes, structure only                                         |
| `matbench_phonons`       | 36.9 cm^-1                         | MEGNet v0.2.2                       | yes, structure only                                         |
| `matbench_steels`        | 95.2 MPa                           | Automatminer express v1.0.3.2019111 | yes                                                         |




## Citing MatBench

You can find details and results on the benchmark in our paper [Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference ](https://doi.org/10.1038/s41524-020-00406-3). 
Please consider citing this paper if you use Matbench v0.1 for benchmarking, comparison, or prototyping.


You can cite MatBench using this reference:

```
Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. 
Benchmarking Materials Property Prediction Methods: 
The Matbench Test Set and Automatminer Reference Algorithm. 
npj Computational Materials 6, 138 (2020). 
https://doi.org/10.1038/s41524-020-00406-3
```



















