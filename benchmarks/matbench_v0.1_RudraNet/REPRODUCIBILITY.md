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
outer training fold, a separate approximately 15% validation holdout is used.
The scaler is fitted only to the inner training rows, separately for each
seed and fold. The default code runs all 300 epochs, averages weights over
zero-based epochs 200–299 with SWA, and returns that SWA model. It does
**not** stop early. Validation tracks the best pre-SWA snapshot, which is
returned only if training ends before SWA starts. This is **five-fold benchmark evaluation with an
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

## Exact feature behavior in the verified environment

The source featurizer builds Magpie (132 dimensions), the pretrained
Mat2Vec context (200 dimensions), and an optional Matminer
extra-descriptor block. With the pinned `matminer==0.9.3` stack, the
original `MultipleFeaturizer` combination raises a charge-balance
exception on the 312 fractional steel compositions. The original
training code catches this exception and supplies a **200-dimensional
zero-filled block** for those extra descriptors.

An independent audit of all 312 compositions confirmed this behavior
for every row, yielding uniform descriptor width and a model input of
132 + 200 + 200 = 532 values. The archived five fold-level model
states have a pooling-layer input dimension of 264 (64 attended
features + 200 extra channels), matching the 200-channel fallback.
Thus, the supplementary Matminer block is *inactive* for this tested
steels pipeline, while Magpie and Mat2Vec remain the informative
input components. This clarification does not alter the historical
predictions or recorded MAEs.

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
  checks split membership, finite predictions and stored fold MAEs.
  `--official` additionally checks official Matbench task splits and
  independently recomputes all four regression metrics from the official
  held-out targets, aligning predictions by sample ID.
  [export_predictions.py](export_predictions.py) accepts sample-ID-keyed
  raw predictions for any subsequent V13A run.

Use an inference environment for model execution and an independent
Matbench environment for output serialization, since the Matbench package
and model pipeline have different release dependencies. The inference file pins
11 direct dependencies; it is not a complete transitive lock or a recovered
historical Kaggle environment. The local integration audit on 2026-11-15 uses
upstream Matbench revision `6374ebf52d636d4f059daec170dc6f8ac7ca3a3a`
with Python 3.12 for official export and score recomputation. To select that
same Matbench source in a separate export environment:

```bash
python -m pip install 'git+https://github.com/materialsproject/matbench.git@6374ebf52d636d4f059daec170dc6f8ac7ca3a3a'
```

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
T4 GPU**, in **under 20 minutes**. The scope of that duration still needs
clarification: the [archived V13 summary](https://github.com/Rtx09x/TRIADS/blob/47ebe2202cfa1498fce8e8b3d6a04e321fa5589e/matbench_steels/results.json)
records `total_min=76.6` for a run containing both V13A and V13B; this does
not establish V13A-only time. The submitted per-sample predictions have
not been changed during these reproducibility repairs.

The published Hugging Face repository
(https://huggingface.co/Rtx09x/TRIADS) contains
`weights/steels/weights.pt`. Its archive has **five fold-level state
dictionaries** labelled `avg_42_123_7_0_99`, each containing weights for
the 224,685-parameter V13A model. These five model states have been
strictly loaded into the original architecture with PyTorch 2.3.1 CPU,
and each produced finite outputs. For loading this archive, the input
width is derived from the saved tensor shapes (200 extra channels)
rather than relying on the archive's descriptive `n_extra=212` value,
which disagrees with the saved tensor dimensions. The archive contains no
scalers. The [author's archive documentation](https://github.com/Rtx09x/TRIADS/blob/47ebe2202cfa1498fce8e8b3d6a04e321fa5589e/README.md#L327-L332)
identifies these as **averaged-weight states**, not the original 25
separately saved seed/fold checkpoints. Averaging weights of this nonlinear
model is not equivalent to averaging predictions from its independently
trained seed models; each seed also uses its own inner-training scaler.
The submitted training code averages predictions. The archive alone
therefore cannot replay that ensemble, and the original seed states,
matching preprocessing parameters and packaging procedure remain needed. The package, protocol, prediction-to-score mapping,
and model weights should not be conflated with a verified replay of all
five historic seed predictions.

## Historical model-selection evidence

The [author's PR statement](https://github.com/materialsproject/matbench/pull/360#issuecomment-6386373087)
says architecture choices used training/validation behavior rather than
outer-test MAEs. Public historical research notes do exist. For example,
the [V13 log](https://github.com/Rtx09x/TRIADS/blob/47ebe2202cfa1498fce8e8b3d6a04e321fa5589e/archive/research_notes/steels/experiment_log_v13.md)
compares V13A/V13B outer-fold outcomes, and the V14/V15 logs discuss
subsequent experiments and earlier performance peaks. These records show
that outer-test results were observed and compared, but do not establish
the timing and criteria used to select this submission.

A dated selection point, inner-validation results, experiment identifiers
and an explanation of how the research narratives relate to the author's
statement are still needed to assess historical selection. Removing V13B
from the current code does not settle that question. Conversely, the
available evidence does not establish test-set leakage. This unresolved
protocol question is separate from verified fold membership, numerical
score consistency and whether the current training code can execute.


## Complete new CPU run (2026-11-15)

The local integration audit also executed the submitted configuration from
scratch on CPU: Python 3.11.12, PyTorch 2.3.1+cpu, the pinned direct
dependencies, all five seeds and five folds, 300 epochs per model and SWA
starting at zero-based epoch 200. The run took about 17.7 minutes with two
CPU intra-op threads and one inter-op thread. No architecture or training
hyperparameters were changed. All 25 new checkpoints and their individual
inner-training scalers were saved and reloaded; their 312 ensemble
predictions matched the fresh training outputs exactly.

The replayed predictions were then recorded with the official Matbench API
at the revision listed above, reloaded, and independently rescored. All
IDs and values survived that round trip, with `is_valid=True` and
`is_recorded=True`. The new fold MAEs were **113.1600, 78.4441, 81.5675,
88.1359, 94.4825 MPa**; their mean was **91.157996 MPa**.

This is a complete **new run**, not a numerical replay of the historical
**91.202843 MPa** submission: all 312 predictions differ, with a mean
absolute prediction difference of 18.817552 MPa and a maximum of
134.466919 MPa. The original result file remains unchanged.

Source comparison found no change to V13A's training computation or
seed/fold loop. CPU dropout and the DataLoader share the default CPU RNG,
whereas CUDA dropout uses a CUDA RNG, so the same seed does not define
identical cross-device random trajectories. That is a concrete possible
source of differences, not proof of their sole cause; the historical
Kaggle environment, cache and 25 original checkpoints remain unavailable.
The new run therefore verifies execution, checkpoint replay and official
serialization without resolving historical provenance or model selection.
