# CompPHGBM Reproduction Note

## Scope

This folder submits two Matbench v0.1 tasks under one shared algorithm:

- `matbench_log_gvrh`, using composition plus persistent-homology structure features.
- `matbench_log_kvrh`, using the same feature and model family.

## Data And Folds

- GVRH dataset: `https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz`
- KVRH dataset: `https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz`
- Official folds: `https://raw.githubusercontent.com/materialsproject/matbench/main/matbench/matbench_v0.1_validation.json`
- Local source artifacts:
  - `outputs/matbench_gvrh/results.json.gz`
  - `outputs/matbench_kvrh/results.json.gz`

No custom folds were generated.

## Features

- Composition branch: `python/discovery/composition_features_v2.py`, 156 features.
- Persistent-homology branch: 51 structure-derived features.
- Point cloud: central-cell atoms plus deterministic periodic images within 6.0 angstrom of central-cell atoms.
- Filtration: Vietoris-Rips with threshold 6.0 angstrom.
- Diagram dimensions: H0 and H1.
- Library path: `ripser==0.6.15` plus `persim==0.3.8`.
- Declared companions: unit-cell atom count, volume, atom density, point-cloud count, expansion-fallback flag.

## Model

- `HistGradientBoostingRegressor`
- Seed: 42
- Inner grid: `max_iter={300, 600}` x `max_leaf_nodes={31, 63}`, `learning_rate=0.1`.
- Inner model selection used only a 90/10 split inside each official training fold.
- The selected model was refit on the full official training fold before recording test predictions.

## Results

- `matbench_log_gvrh` mean official MAE: 0.0813
- `matbench_log_kvrh` mean official MAE: 0.0636
- Both tasks improved over the composition-only control in all five official folds.

The result supports the complete structure-derived feature track as complementary to composition. The five geometry and density companion features are explicitly declared, so a separate ablation is still needed before assigning the full lift uniquely to topological statistics.

## Reproduction

Run standalone from this folder (no dependency on any path outside it;
`run_matbench_kvrh.py` imports the sibling `ph_features.py` in this same
`src/` directory):

```bash
cd src
pip install pymatgen scikit-learn pandas numpy ripser persim
python3 run_matbench_gvrh.py
python3 run_matbench_kvrh.py
```

This downloads both datasets and the official validation folds, recomputes
the persistent-homology feature cache, refits both models, and reproduces
the headline numbers above; verified byte-for-byte identical per-fold
predictions and scores against `results.json.gz` before submission. That
merged artifact combines the two independently validated single-task runs
into one MatbenchBenchmark-shaped record.
