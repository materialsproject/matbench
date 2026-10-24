# RudraNet (TRIADS V13A) • Matbench steels reproducibility

## Submission and research sources

- Benchmark PR: https://github.com/materialsproject/matbench/pull/360
- Original model/training implementation: [trm13.py](trm13.py)
- Research archive and V13 experiment summary:
  https://github.com/Rtx09x/TRIADS/tree/main/matbench_steels
- Paper DOI: https://doi.org/10.5281/zenodo.19200579
- Mat2Vec source paper: https://doi.org/10.1038/s41586-019-1335-8

The originally submitted training code matches `TRIADS/matbench_steels/model.py`
(after normalization of line endings). The submission specifies the
`V13A-2xSA-StdDS` model with self-attention, cross-attention to Mat2Vec,
20-step recursive refinement, deep supervision, and a five-seed ensemble
(`42, 123, 7, 0, 99`).

## Data protocol and evaluated scores

The run used the 312-composition `matbench_steels` dataset through Matminer
because the author's Kaggle training environment could not access the
Matbench package/resources. Outer splits were generated with
`KFold(n_splits=5, shuffle=True, random_state=18012019)`. Within each
outer training fold, a separate 15% validation holdout was used for model
training/early stopping. This is **five-fold benchmark evaluation with an
inner holdout**, not a claim of inner multi-fold hyperparameter-search CV.

All 312 submitted Matbench sample IDs were checked against the seeded
outer fold memberships: there were **no missing, repeated, or mismatched
test IDs**. The official Matbench submission check passes.

| Outer fold | Samples | MAE (MPa) |
| --- | ---: | ---: |
| 0 | 63 | 114.324803 |
| 1 | 63 | 81.463865 |
| 2 | 62 | 80.551309 |
| 3 | 62 | 90.491534 |
| 4 | 62 | 89.182703 |

The five-fold mean MAE is **91.2028 MPa**. Each value agrees with the
archived V13A experiment summary in the main TRIADS repository to its
recorded precision. The standard deviation across *folds* is about
12.23 MPa, not a per-seed uncertainty estimate.

The archived individual-seed mean MAEs (MPa) are **98.7218** (42),
**96.7742** (123), **98.7624** (7), **102.5524** (0), and
**102.1098** (99). Across these five seed-level means, the arithmetic
mean is **99.7841 MPa** and the population standard deviation is
**2.2047 MPa**. The separately averaged five-seed predictions yield
**91.2028 MPa**. These numbers are recovered from the archived V13
summary, not from new retraining.

## Verified software and export

- **Inference environment:** [requirements-reproduction.txt](requirements-reproduction.txt)
  is based on the TRIADS repository's dependencies, narrowed to the
  steels task. GitHub Actions installed Python 3.11 and CPU PyTorch 2.3.1,
  including the required Matminer, Pymatgen, Gensim, SciPy and NumPy
  versions, then successfully imported `DeepHybridTRM` and produced
  finite predictions on a synthetic input. GPU-specific training is not
  part of this CPU smoke test.
- **Official benchmark export:** in a separate Python 3.12 environment,
  GitHub Actions installed the current upstream Matbench source, used
  [export_predictions.py](export_predictions.py) to record all five folds,
  and reloaded the output with the official `MatbenchBenchmark` API.
  The exported 312 per-sample values matched the historical submission
  with **zero numerical mismatches** and the rebuilt benchmark was
  `is_valid=True`, `is_recorded=True`.
- **Repeatable checks:** [verify_submission.py](verify_submission.py)
  checks split membership, finite predictions and fold scores.
  `--official` additionally checks official Matbench task splits.
  [export_predictions.py](export_predictions.py) accepts sample-ID-keyed
  raw predictions for any subsequent V13A run.

Use an inference environment for model execution and an independent
Matbench environment for output serialization, since the Matbench package
and model pipeline have different release dependencies.

## Mat2Vec data provenance

The model downloads three original upstream objects from
`https://storage.googleapis.com/mat2vec/`. GitHub Actions captured
the **actual upstream SHA-256 hashes, byte lengths, and GCS generation
metadata** in [mat2vec_sha256.json](mat2vec_sha256.json).
[mat2vec_integrity.py](mat2vec_integrity.py) verifies those files without
executing or deserializing model content. `trm13.py` also checks all three
files before passing them to `Word2Vec.load`. The official upstream
objects carry June 2019 modification dates.

## Compute and archived checkpoint scope

Training was reported by the author as performed on Kaggle with **one NVIDIA
T4 GPU**, in **under 20 minutes**. The submitted per-sample predictions
have not been changed during these reproducibility repairs.

The published Hugging Face repository
(https://huggingface.co/Rtx09x/TRIADS) contains
`weights/steels/weights.pt`. Its archive has **five fold-level state
dictionaries** labelled `avg_42_123_7_0_99`, each containing weights for
the 224,685-parameter V13A model. These are archived fold-level
weights, not an identified set of 25 separately saved seed/fold
checkpoints. The package, protocol, prediction-to-score mapping,
and model weights should not be conflated with a verified replay of all
five historic seed predictions.

Historical cross-version model selection is reviewed separately from
these data-integrity, software and serialization checks.
