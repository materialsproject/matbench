# Benchmark info for `matbench_v0.1`

The `matbench_v0.1` benchmark contains 13 tasks:

| Task name | Task type/input | Target column (unit) | Samples | MAD (regression) or Fraction True (classification) | Links | Submissions|
|-------|-------|-------|-------|-------|-------|-------|
| `matbench_steels` | regression/composition | `yield strength` (MPa) | 312 | 229.3743 | [download](https://ml.materialsproject.org/projects/matbench_steels.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_steels) | 9 |
| `matbench_jdft2d` | regression/structure | `exfoliation_en` (meV/atom) | 636 | 67.2020 | [download](https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_jdft2d) | 14 |
| `matbench_phonons` | regression/structure | `last phdos peak` (cm^-1) | 1,265 | 323.7870 | [download](https://ml.materialsproject.org/projects/matbench_phonons.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_phonons) | 14 |
| `matbench_expt_gap` | regression/composition | `gap expt` (eV) | 4,604 | 1.1432 | [download](https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_expt_gap) | 11 |
| `matbench_dielectric` | regression/structure | `n` (unitless) | 4,764 | 0.8085 | [download](https://ml.materialsproject.org/projects/matbench_dielectric.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_dielectric) | 14 |
| `matbench_expt_is_metal` | classification/composition | `is_metal` | 4,921 | 0.4981 | [download](https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_expt_is_metal) | 6 |
| `matbench_glass` | classification/composition | `gfa` | 5,680 | 0.7104 | [download](https://ml.materialsproject.org/projects/matbench_glass.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_glass) | 6 |
| `matbench_log_gvrh` | regression/structure | `log10(G_VRH)` (log10(GPa)) | 10,987 | 0.2931 | [download](https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_log_gvrh) | 14 |
| `matbench_log_kvrh` | regression/structure | `log10(K_VRH)` (log10(GPa)) | 10,987 | 0.2897 | [download](https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_log_kvrh) | 14 |
| `matbench_perovskites` | regression/structure | `e_form` (eV/unit cell) | 18,928 | 0.5660 | [download](https://ml.materialsproject.org/projects/matbench_perovskites.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_perovskites) | 14 |
| `matbench_mp_gap` | regression/structure | `gap pbe` (eV) | 106,113 | 1.3271 | [download](https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_gap) | 14 |
| `matbench_mp_is_metal` | classification/structure | `is_metal` | 106,113 | 0.4349 | [download](https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_is_metal) | 11 |
| `matbench_mp_e_form` | regression/structure | `e_form` (eV/atom) | 132,752 | 1.0059 | [download](https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz), [interactive](https://ml.materialsproject.org/projects/matbench_mp_e_form) | 15 |
