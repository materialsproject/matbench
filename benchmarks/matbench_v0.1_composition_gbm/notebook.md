# CompGBM Reproduction Note

## Scope

This folder submits one Matbench v0.1 task:

- `matbench_glass`, using composition features plus HistGradientBoostingClassifier.

## Data And Folds

- Dataset: `https://ml.materialsproject.org/projects/matbench_glass.json.gz`
- Official folds: `https://raw.githubusercontent.com/materialsproject/matbench/main/matbench/matbench_v0.1_validation.json`
- Local source artifact: `outputs/matbench_glass/results.json.gz`
- Rows: 5,680
- Official test count per fold: 1,136

No custom folds were generated.

## Features

- `python/discovery/composition_features_v2.py`
- 156 numeric composition features.
- Inputs use the `composition` column only.

## Model

- `HistGradientBoostingClassifier`
- Seed: 42
- Inner model selection was performed inside each official training fold.
- The selected model was refit on the full official training fold before recording test predictions.

## Result

- Mean official `rocauc`: 0.8658
- Mean official `balanced_accuracy`: 0.8658
- Mean official `f1`: 0.9307

The classification artifact stores probability-like float predictions in `[0, 1]`.

## Reproduction

Run standalone from this folder (no dependency on any path outside it):

```bash
cd src
pip install pymatgen scikit-learn pandas numpy
python3 run_matbench_glass.py
```

This downloads the dataset and official validation folds, refits the model,
and reproduces the headline number above; verified byte-for-byte identical
per-fold predictions and scores against `results.json.gz` before submission.

The local Matbench package import path failed before import while building `scikit-learn==1.0.1`, so the runner used the raw dataset plus official validation JSON fallback and wrote a MatbenchBenchmark-shaped artifact.
