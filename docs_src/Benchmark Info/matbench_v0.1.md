# Benchmark info for `matbench_v0.1`

The `matbench_v0.1` benchmark contains 13 tasks:

| Task name | Task type | Target column (unit) | Input type | Samples | MAD (regression) or Fraction True (classification) | Links | Submissions|
|-------|-------|-------|-------|-------|-------|-------|-------|
| `matbench_steels` | regression | `yield strength` (MPa) | composition | 312 | 229.3743 | [download](https://ml.materialsproject.org/projects/matbench_steels.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_steels) | 5 |
| `matbench_jdft2d` | regression | `exfoliation_en` (meV/atom) | structure | 636 | 67.2020 | [download](https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_jdft2d) | 7 |
| `matbench_phonons` | regression | `last phdos peak` (cm^-1) | structure | 1,265 | 323.7870 | [download](https://ml.materialsproject.org/projects/matbench_phonons.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_phonons) | 7 |
| `matbench_expt_gap` | regression | `gap expt` (eV) | composition | 4,604 | 1.1432 | [download](https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_expt_gap) | 8 |
| `matbench_dielectric` | regression | `n` (unitless) | structure | 4,764 | 0.8085 | [download](https://ml.materialsproject.org/projects/matbench_dielectric.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_dielectric) | 7 |
| `matbench_expt_is_metal` | classification | `is_metal` | composition | 4,921 | 0.4981 | [download](https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_expt_is_metal) | 4 |
| `matbench_glass` | classification | `gfa` | composition | 5,680 | 0.7104 | [download](https://ml.materialsproject.org/projects/matbench_glass.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_glass) | 4 |
| `matbench_log_gvrh` | regression | `log10(G_VRH)` (log10(GPa)) | structure | 10,987 | 0.2931 | [download](https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_log_gvrh) | 7 |
| `matbench_log_kvrh` | regression | `log10(K_VRH)` (log10(GPa)) | structure | 10,987 | 0.2897 | [download](https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_log_kvrh) | 7 |
| `matbench_perovskites` | regression | `e_form` (eV/unit cell) | structure | 18,928 | 0.5660 | [download](https://ml.materialsproject.org/projects/matbench_perovskites.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_perovskites) | 7 |
| `matbench_mp_gap` | regression | `gap pbe` (eV) | structure | 106,113 | 1.3271 | [download](https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_gap) | 7 |
| `matbench_mp_is_metal` | classification | `is_metal` | structure | 106,113 | 0.4349 | [download](https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_is_metal) | 6 |
| `matbench_mp_e_form` | regression | `e_form` (eV/atom) | structure | 132,752 | 1.0059 | [download](https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_e_form) | 7 |
