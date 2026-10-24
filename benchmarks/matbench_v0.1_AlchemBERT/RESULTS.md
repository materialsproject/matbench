# Archived AlchemBERT results

These are summaries of the submitted artifact, not new experiments or independently recomputed scores. `results.json.gz` is preserved byte for byte.

- Source: [PR #358 head `b44dca2f98b1b7112bdb1900183475876fb01551`](https://github.com/liuxiaotong15/matbench/blob/b44dca2f98b1b7112bdb1900183475876fb01551/benchmarks/matbench_v0.1_AlchemBERT/results.json.gz).
- Git blob: `ecc70fbbbd7776c443d769a29ec6b02692136385`.
- SHA256: `a0bb054a3a32b09041940aa90a7adc4e63a538811af430194057736ea5bd3daf`.
- Serialized Matbench version: `0.6`; serialized datestamp: `2024.12.09 08:47.31` (timezone unspecified). These fields do not establish the exact training environment or run dates.
- 13 tasks, five folds per task. All 65 `parameters` dictionaries and the top-level `user_metadata` are empty.

## Stored primary scores

Mean and standard deviation below are calculated across the five stored fold scores, with population standard deviation (`ddof=0`). MAE retains each task's target units; ROC-AUC is dimensionless.

| Task | Metric | Stored mean | Fold standard deviation |
| --- | --- | ---: | ---: |
| `matbench_dielectric` | MAE | 0.334517302 | 0.0683625597 |
| `matbench_expt_gap` | MAE | 0.374580207 | 0.0200175981 |
| `matbench_expt_is_metal` | ROCAUC | 0.891674127 | 0.0146521198 |
| `matbench_glass` | ROCAUC | 0.770211636 | 0.0103112965 |
| `matbench_jdft2d` | MAE | 45.4918247 | 11.2160871 |
| `matbench_log_gvrh` | MAE | 0.0998205348 | 0.00206009541 |
| `matbench_log_kvrh` | MAE | 0.0765199783 | 0.00169306961 |
| `matbench_mp_e_form` | MAE | 0.061889702 | 0.00161484946 |
| `matbench_mp_gap` | MAE | 0.254341043 | 0.00254865661 |
| `matbench_mp_is_metal` | ROCAUC | 0.886022805 | 0.0039596367 |
| `matbench_perovskites` | MAE | 0.0431881109 | 0.00191404572 |
| `matbench_phonons` | MAE | 44.9874142 | 2.48488323 |
| `matbench_steels` | MAE | 132.912131 | 7.17619571 |

## Classification predictions and historical scoring

All stored predictions, including the classification predictions, are floating-point numbers. The classification values are non-binary scores inside `[0, 1]`:

| Task | Minimum across folds | Maximum across folds |
| --- | ---: | ---: |
| `matbench_expt_is_metal` | 0.00040915323 | 0.9991026 |
| `matbench_glass` | 0.0036448804 | 0.9989999 |
| `matbench_mp_is_metal` | 0.00019700835 | 0.99985945 |

For all 15 classification folds, the stored ROC-AUC equals stored balanced accuracy to `1e-12`. This is consistent with the historical thresholded-prediction scoring issue fixed in main; it is not evidence of data leakage. The table reports those historical values transparently and does not relabel them as probability-based ROC-AUC. Independent rescoring with verified labels and the stored sample-ID-to-prediction maps is now reported separately in [RESCORING.md](RESCORING.md). That audit identifies corrected probability-based scores and its remaining data-version limit without silently overwriting this artifact.

## Read-only validation

After installing the intended Matbench checkout, run from the repository root:

```bash
python benchmarks/matbench_v0.1_AlchemBERT/validate_results.py
```

The script checks the original SHA256, calls `MatbenchBenchmark.from_file`, and requires `is_valid`, `is_complete`, and `is_recorded`. It emits every stored metric's mean, population variance and standard deviation, plus every fold's prediction count, type, range and recorded parameters. It also reports the serialized version and the currently imported Matbench version. No training, model inference, target-data download or rescoring occurs.

On 2026-10-24, validation against the core code from main commit `4b7d0714849a917814115be380a64e7ace94be55` returned `is_valid=True`, `is_complete=True`, and `is_recorded=True`. The archive checksum matched, and an intentionally incorrect expected checksum was rejected. This is a local check; CI on the final PR commit must be checked separately.

Passing these checks establishes artifact structure, official fold membership and coverage. It does not establish model provenance, absence of leakage, agreement between predictions and checkpoints, or reproducibility of the training run.
