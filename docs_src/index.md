# MatBench v0.1 - Machine Learning Benchmark for Materials Science

## Overview

[MatBench](https://doi.org/10.1038/s41524-020-00406-3) is an [ImageNet](http://www.image-net.org) for **materials science**; a
**curated set of 13 supervised, pre-cleaned, ready-to-use ML tasks** for benchmarking and fair comparison. The tasks span a wide domain of
inorganic materials science applications including electronic, thermodynamic, mechanical, and thermal properties among crystals, 2D materials,
disordered metals, and more.  

**The [MatBench python package](https://github.com/hackingmaterials/matbench) provides everything needed to use MatBench with your ML algorithm in ~10 lines of code or less.**

![infographic](static/infographic_matbench.png)



## What can MatBench offer?

### This website


- **Leaderboard of results for state-of-the-art materials ML algorithms on standardized test problems**
- Interactively explore and download the tasks on [MPContribs-ML](https://ml.materialsproject.org/browse), a platform hosted by [The Materials Project](https://materialsproject.org). See [All MatBench Datasets](#all-matbench-datasets) for links to each dataset.
- **Each and every result is backed by a peer-reviewed publication and a jupyter notebook** (similar to Papers With Code) - i.e., how were these results were obtained?
- Glossary of all algorithms' results on the MatBench problems


### The MatBench Python package

- Probe ML algorithms strengths and weaknesses across a wide range of materials property prediction tasks
- Run a full benchmark in ~10 lines of code
- Submit results as a PR to the [MatBench repo](https://github.com/hackingmaterials/matbench) to compare with other algorithms and appear on the leaderboard
- Benchmark both **general purpose** ML models as well as algorithms specialized for particular domains


## Summary of MatBench's Tasks

MatBench's 13 tasks can be broken down into various categories; it includes both the small - less than 10,000 samples - datasets that characterize
experimental materials data as well as larger datasets from computer modelling methods like density functional theory (DFT).


![breakdown](static/datasets_breakdown_inverted.png)


Each task in MatBench consists of a three things:

1. **A set of inputs:** crystal structures or chemical compositions.
2. **A set of outputs:** target properties, such as formation energy.
3. **A test procedure:** a way to get a score for your algorithm


The MatBench Python package provides functions for getting the first two (packaged together for each task as a _dataset_) as well as running 
the test procedure. See the "How to use MatBench" documentation page to get started.


## All MatBench Datasets

| task name                | target column (unit)         | number of samples | task type      | links                                                                                                                                                                |
| ------------------------ | ---------------------------- | ----------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `matbench_dielectric`    | `n` (unitless)               | 4,764              | regression     | [download](https://ml.materialsproject.org/projects/matbench_dielectric.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_dielectric/)       |
| `matbench_expt_gap`      | `gap expt` (eV)              | 4,604              | regression     | [download](https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_expt_gap/)           |
| `matbench_expt_is_metal` | `is_metal` (unitless)        | 4,921              | classification | [download](https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_expt_is_metal/) |
| `matbench_glass`         | `gfa` (unitless)             | 5,680              | classification | [download](https://ml.materialsproject.org/projects/matbench_glass.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_glass/)                 |
| `matbench_jdft2d`        | `exfoliation_en` (meV/atom)  | 636               | regression     | [download](https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_jdft2d/)               |
| `matbench_log_gvrh`      | `log10(G_VRH)` (log(GPa))    | 10,987             | regression     | [download](https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_log_gvrh/)           |
| `matbench_log_kvrh`      | `log10(K_VRH)` (log(GPa))    | 10,987             | regression     | [download](https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_log_kvrh/)           |
| `matbench_mp_e_form`     | `e_form` (eV/atom)           | 132,752            | regression     | [download](https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_e_form/)         |
| `matbench_mp_gap`        | `gap pbe` (eV)               | 106,113            | regression     | [download](https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_gap/)               |
| `matbench_mp_is_metal`   | `is_metal` (unitless)        | 106,113            | classification | [download](https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_is_metal/)     |
| `matbench_perovskites`   | `e_form` (eV, per unit cell) | 18,928             | regression     | [download](https://ml.materialsproject.org/projects/matbench_perovskites.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_perovskites/)     |
| `matbench_phonons`       | `last phdos peak` (cm^-1)    | 1,265              | regression     | [download](https://ml.materialsproject.org/projects/matbench_phonons.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_phonons/)             |
| `matbench_steels`        | `yield strength` (MPa)       | 312               | regression     | [download](https://ml.materialsproject.org/projects/matbench_steels.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_steels/)               |





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



















