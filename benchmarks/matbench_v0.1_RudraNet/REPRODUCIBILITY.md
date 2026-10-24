# RudraNet / TRIADS V13A: reproduction and provenance

This document describes the submitted `matbench_steels` benchmark. It separates
**verified observations** from **unverified historical claims**. The submission
predictions have NOT been retrained or changed during the October 2026 repairs.

## Sources

- PR: https://github.com/materialsproject/matbench/pull/360
- Submitted training script: `trm13.py` (original source at PR head 89be878).
- Public research archive: https://github.com/Rtx09x/TRIADS
- TRIADS code: `matbench_steels/model.py` (Git blob 5282d83405b8db2ecf0aade121be7d180c5498e8).
- Archived results: `matbench_steels/results.json` and
  `archive/research_notes/steels/summary_v13.json`
  (Git blob 6684ffe674e9dfadce0398372cd955e6e791c45f).
- Model/paper DOI: https://doi.org/10.5281/zenodo.19200579
- Mat2Vec publication: https://doi.org/10.1038/s41586-019-1335-8

The submitted `trm13.py` and archived `TRIADS/matbench_steels/model.py`
were identical after normalizing Windows/Unix line endings at audit time
(prior to adding the raw-prediction export path here).

## Submitted result and fold membership

V13A (`V13A-2xSA-StdDS`) uses five seeds: 42, 123, 7, 0, 99.
The 5-seed ensemble fold MAEs in MPa are:

| Fold | Test samples | MAE (MPa) |
| --- | ---: | ---: |
| 0 | 63 | 114.324803 |
| 1 | 63 | 81.463865 |
| 2 | 62 | 80.551309 |
| 3 | 62 | 90.491534 |
| 4 | 62 | 89.182703 |

Mean fold MAE: 91.202843 MPa (91.20 rounded); across-fold standard
deviation approximately 12.23 MPa. This standard deviation is **not**
a per-seed uncertainty estimate.

The original submitted JSON includes all 312 unique `mb-steels-NNN` IDs.
Comparison against scikit-learn `KFold(n_splits=5, shuffle=True,
random_state=18012019)` showed **zero fold membership mismatches**.
Matbench's `data_ops.load` wraps the same Matminer dataset loader and adds
MBIDs in row order. The upstream submission validator also passed on the
gzip-corrected, otherwise unmodified, predictions file.

Run `python verify_submission.py` to repeat the lightweight audit.
Run `python verify_submission.py --official` in an environment with
Matbench and the dataset available to additionally check the official task
splits and built-in validation.

## Official result generation for new training runs

The amended `trm13.py` now writes:
`trm_models_v13/v13a_fold_predictions_by_mbid.json`.
This output is keyed by MBID and captures the V13A ensemble predictions
per sample, independent of Matbench availability during Kaggle training.

In an environment where Matbench can load the dataset:

```shell
python export_predictions.py --input trm_models_v13/v13a_fold_predictions_by_mbid.json
```

The exporter uses `MatbenchBenchmark`, `get_train_and_val_data`,
`get_test_data`, `record`, and `to_file`, checks exact official sample
membership, and writes `results.rebuilt.json.gz` **without overwriting**
the historical PR file. No claim is made that this new exporter was used
for the original March 2026 results; it provides a reproducible export
path for subsequent runs.

## Dependencies and compute

`requirements-reproduction.txt` contains an exact-version **candidate
reproduction target** for Python 3.11. The original Kaggle `pip freeze`
is unavailable. These versions have not been installed and run together
as a verified historic environment; they must not be described as
versions actually used for the submitted training run.

Training environment reported by the author: Kaggle, a single NVIDIA T4 GPU.
Author-reported training runtime: **under 20 minutes** on one NVIDIA T4 GPU. The exact timing was not recorded or independently verified.

## Mat2Vec checksum capture (pending upstream bytes)

The author's training code used the official Mat2Vec pretrained model set:
`pretrained_embeddings`, `pretrained_embeddings.wv.vectors.npy`, and
`pretrained_embeddings.trainables.syn1neg.npy` from
`https://storage.googleapis.com/mat2vec/`.

Run in a networked, trusted environment (for example Kaggle):

```sh
python mat2vec_integrity.py capture --cache ./mat2vec_cache
python mat2vec_integrity.py verify --cache ./mat2vec_cache
```

This writes `mat2vec_sha256.json` with observed SHA-256 hashes, lengths,
source metadata, and UTC capture time. Only after checking the real bytes
and the source should that manifest be committed and checksum verification
be enforced in `trm13.py` before `Word2Vec.load`. **No placeholder hashes
are included.** A first download from HTTPS records current content but
cannot, alone, establish which exact bytes were used historically.

## Open reproducibility issues

- Original individual checkpoint-to-prediction correspondence has not
  been independently regenerated from archived weights.
- Comparison of V10-V13 and V13A/B on test-fold scores, and its impact
  on model selection, remains under scientific review.
- The original training script loads Mat2Vec Word2Vec pickle-compatible
  files from `https://storage.googleapis.com/mat2vec/` without
  independently pinned checksums. The three official source filenames have
  been confirmed in the original Mat2Vec project's documentation, but a
  trusted SHA-256 digest has **not** been retrieved or pinned yet.
  `mat2vec_integrity.py` provides a read-only-to-model SHA-256 capture and
  validation workflow. It does not unpickle the model; a completed manifest
  must be independently reviewed and committed before treating downloads
  as authenticated. The historic Kaggle file identity cannot be established
  merely by hashing the current upstream objects.
- Per-seed averages are preserved in the TRIADS archive; individual
  seed-by-fold metrics have not been restored from an original run log.

Neither the fixes nor the included scripts reselect or modify the
historic submitted predictions.
