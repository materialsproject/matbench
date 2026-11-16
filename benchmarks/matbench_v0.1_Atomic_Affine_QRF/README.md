# Atomic-AffQRF (matbench_v0.1, matbench_steels)

Composition-only regression on `matbench_steels`. Reference seed 417:
mean fold MAE 75.5426 MPa on the five official Matbench v0.1 folds.

## Method

- Features: atomic-fraction composition vectors under four fixed transforms.
- 16 fixed ExtraTrees configurations; each gives a leaf-distribution conditional
  mean and lower-median prediction (32 predictors).
- Within each official outer training fold, five inner folds produce
  out-of-fold (OOF) predictions for all 32 predictors.
- 128 Bayesian-bootstrap weighted-MAE linear programs fit nonnegative
  coefficients (sum <= 2) plus a free intercept on the OOF predictions;
  coefficients and intercepts are averaged. Outer-test labels are never used.
- Final forests (900 trees) are refit on the full outer training fold.

## Files

- `results.json.gz`, `info.json`, `notebook.ipynb`: required submission files.
- `run.py`, `inputs/base_qrf.py`: fitting source (base forests, OOF libraries,
  affine LP aggregation). `inputs/protocol.json`, `protocol.json`: frozen settings.
- `inputs/matbench_steels.json.gz`, `inputs/steels_splits.json`: the official
  dataset and fold indices used (snapshots of the Matbench release).
- `fresh.py`, `export.py`, `reproduce_entry.py`: end-to-end rebuild from raw
  inputs and export to a Matbench results file.
- `preflight.py`, `environment_lock.json`, `manifest.json`,
  `upstream_source_hashes.json`: refuse to run with mismatched package
  versions, changed inputs, or pre-existing fit caches.
- `all_seed_scores.csv`: mean fold MAE for every run seed (0-6 and 417) and the
  two ablation aggregators.

## Reproduce

```bash
uv venv --python 3.13.16 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -B reproduce_entry.py --output fresh_script_review --workers 3
```

This rebuilds 400 inner forests and all active final forests from an empty fit
directory and asserts that the complete official fold dictionaries equal
`results.json.gz` exactly. The notebook runs the same pipeline.

## Limitations

- The official folds were used repeatedly during method development, so
  model-family selection across runs may be optimistic even though each run
  fits only on outer-training labels. This is not an untouched holdout.
- Seed repeats 0-6: 75.77 +/- 0.33 MPa (sample SD); the primary is the
  pre-declared seed 417, not the best repeat.
- On a separate composition-grouped five-fold stress test (201 groups; not the
  official task) seed 417 scores 99.56 MPa versus 99.09 MPa for the released
  TPOT pipeline refit with the same seed, so no chemistry-transfer advantage
  is claimed.
