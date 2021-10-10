# Leaderboard

## Leaderboard: General Purpose Algorithms on `matbench_v0.1`

Find more information about this benchmark on [the benchmark info page](Benchmark%20Info/matbench_v0.1.md)

| Task name | Samples | Algorithm | Verified MAE (unit) or ROCAUC | Notes |
|------------------|---------|-----------|----------------------|-------|
| matbench_steels | 312 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **97.4929 (MPa)** |  |
| matbench_jdft2d | 636 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **39.8497 (meV/atom)** |  |
| matbench_phonons | 1,265 | [CrabNet](Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **55.1114 (cm^-1)** |  |
| matbench_expt_gap | 4,604 | [CrabNet](Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **0.3463 (eV)** |  |
| matbench_dielectric | 4,764 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.3150 (unitless)** |  |
| matbench_expt_is_metal | 4,921 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.9209** |  |
| matbench_glass | 5,680 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.8607** |  |
| matbench_log_gvrh | 10,987 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.0874 (log10(GPa))** |  |
| matbench_log_kvrh | 10,987 | [AMMExpress v2020](Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020.md) | **0.0647 (log10(GPa))** |  |
| matbench_perovskites | 18,928 | [CGCNN v2019](Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **0.0452 (eV/unit cell)** | structure required |
| matbench_mp_gap | 106,113 | [CrabNet](Full%20Benchmark%20Data/matbench_v0.1_CrabNet.md) | **0.2655 (eV)** |  |
| matbench_mp_is_metal | 106,113 | [CGCNN v2019](Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **0.9520** | structure required |
| matbench_mp_e_form | 132,752 | [CGCNN v2019](Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019.md) | **0.0337 (eV/atom)** | structure required |



<iframe src="static/scaled_errors.html" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>

## Overview

[Matbench](https://doi.org/10.1038/s41524-020-00406-3) is an [ImageNet](http://www.image-net.org) for **materials science**; a
**curated set of 13 supervised, pre-cleaned, ready-to-use ML tasks** for benchmarking and fair comparison. The tasks span a wide domain of
inorganic materials science applications including electronic, thermodynamic, mechanical, and thermal properties among crystals, 2D materials,
disordered metals, and more.  

**The [Matbench python package](https://github.com/hackingmaterials/matbench) provides everything needed to use Matbench with your ML algorithm in ~10 lines of code or less.**

![infographic](static/infographic_matbench.png)


## What can Matbench offer?

### This website


- **Leaderboard of results for state-of-the-art materials ML algorithms on standardized test problems**
- Interactively explore and download the tasks on [MPContribs-ML](https://ml.materialsproject.org/browse), a platform hosted by [The Materials Project](https://materialsproject.org). See [Benchmark Info](Benchmark%20Info/matbench_v0.1/) for links to each dataset.
- **Each and every result is backed by a peer-reviewed publication and a jupyter notebook** (similar to Papers With Code) - i.e., how were these results were obtained?
- Glossary of all algorithms' results on the Matbench problems


### The Matbench Python package

- Probe ML algorithms strengths and weaknesses across a wide range of materials property prediction tasks
- Run a full benchmark in ~10 lines of code
- Submit results as a PR to the [Matbench repo](https://github.com/hackingmaterials/matbench) to compare with other algorithms and appear on the leaderboard
- Benchmark both **general purpose** ML models as well as algorithms specialized for particular domains


## Summary of Matbench's Tasks

Matbench's 13 tasks can be broken down into various categories; it includes both the small - less than 10,000 samples - datasets that characterize
experimental materials data as well as larger datasets from computer modelling methods like density functional theory (DFT).


![breakdown](static/datasets_breakdown_inverted.png)


Each task in Matbench consists of a three things:

1. **A set of inputs:** crystal structures or chemical compositions.
2. **A set of outputs:** target properties, such as formation energy.
3. **A test procedure:** a way to get a score for your algorithm


The Matbench Python package provides functions for getting the first two (packaged together for each task as a _dataset_) as well as running 
the test procedure. See the [How to use](How%20To%20Use/) documentation page to get started.



## Citing Matbench

You can find details and results on the benchmark in our paper [Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm](https://doi.org/10.1038/s41524-020-00406-3). 
Please consider citing this paper if you use Matbench v0.1 for benchmarking, comparison, or prototyping.


You can cite Matbench using this reference:

```
Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. 
Benchmarking Materials Property Prediction Methods: 
The Matbench Test Set and Automatminer Reference Algorithm. 
npj Computational Materials 6, 138 (2020). 
https://doi.org/10.1038/s41524-020-00406-3
```