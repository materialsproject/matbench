# AlchemBERT results

All 13 tasks / 65 folds were rescored on 2026-11-13 against the current,
checksum-verified official target tables. Predictions, official sample IDs,
fold parameters and uncertainties are unchanged. No model was retrained.

The mean is the unweighted mean of the five official fold scores; standard
deviation is the population value (`ddof=0`). Regression MAE retains each task's
original target units. Classification ROC-AUC uses the original unrounded
positive-class probabilities.

| Task | Metric | Historical cached mean | Current mean | Current fold std. |
| --- | --- | ---: | ---: | ---: |
| `matbench_dielectric` | MAE | 0.334517302 | 0.334517302 | 0.068362560 |
| `matbench_expt_gap` | MAE | 0.374580207 | 0.374580207 | 0.020017598 |
| `matbench_expt_is_metal` | ROC-AUC | 0.891674127 | 0.959589686 | 0.003370929 |
| `matbench_glass` | ROC-AUC | 0.770211636 | 0.883284181 | 0.014559718 |
| `matbench_jdft2d` | MAE | 45.491824712 | 45.491824712 | 11.216087118 |
| `matbench_log_gvrh` | MAE | 0.099820535 | 0.099820535 | 0.002060095 |
| `matbench_log_kvrh` | MAE | 0.076519978 | 0.076519978 | 0.001693070 |
| `matbench_mp_e_form` | MAE | 0.061889702 | 0.061889702 | 0.001614849 |
| `matbench_mp_gap` | MAE | 0.254341043 | 0.254341043 | 0.002548657 |
| `matbench_mp_is_metal` | ROC-AUC | 0.886022805 | 0.955124203 | 0.001665531 |
| `matbench_perovskites` | MAE | 0.043188111 | 0.043188111 | 0.001914046 |
| `matbench_phonons` | MAE | 44.987414162 | 44.987414162 | 2.484883230 |
| `matbench_steels` | MAE | 132.912130507 | 132.912130507 | 7.176195707 |

The historical classification `rocauc` values equal balanced accuracy because
of the old thresholded-scoring behavior. They are retained above for comparison,
not described as probability ROC-AUC. All three classification tasks now have
corrected probability-based AUC. See [rescoring.json](rescoring.json) for all
metrics and fold differences, and [RESCORING.md](RESCORING.md) for the method.

## Artifact identities

- Current result: `results.json.gz`, SHA256 `c96b467e07f521e3e3d9f8f9f2c32f881dd58e2a523f87a630e2b3b67c9ed13a`.
- Unchanged original: `historical/results.json.gz`, SHA256
  `a0bb054a3a32b09041940aa90a7adc4e63a538811af430194057736ea5bd3daf`.
- Original Git blob: `ecc70fbbbd7776c443d769a29ec6b02692136385`, from
  [PR #358's original artifact](https://github.com/liuxiaotong15/matbench/blob/b44dca2f98b1b7112bdb1900183475876fb01551/benchmarks/matbench_v0.1_AlchemBERT/results.json.gz).
- Original serialized Matbench version: `0.6`; datestamp:
  `2024.12.09 08:47.31`. Original fold parameter dictionaries and user metadata
  are empty. These fields do not establish the training environment.

The current artifact includes `user_metadata.score_migration` with the original
identity, scorer/source hashes, label provenance and software versions. Matbench
regenerates its serialization content hash. The migrated file's datestamp
records serialization, not model training.

## Validate

```bash
python benchmarks/matbench_v0.1_AlchemBERT/validate_results.py
python -m scripts.test_submission --submission matbench_v0.1_AlchemBERT
```

The first command checks the current SHA256 from `provenance.json`, structure,
completeness and recorded status. The second is upstream's current strict
classification gate, including official target recomputation and cached-score
agreement. Neither command establishes historical model provenance or reproduces
training; the remaining limitations are in [README.md](README.md).
