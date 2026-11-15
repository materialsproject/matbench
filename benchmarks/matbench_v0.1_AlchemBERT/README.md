# AlchemBERT

AlchemBERT fine-tunes BERT-base-cased on natural-language descriptions of
materials for regression and classification. This submission covers all **13
Matbench v0.1 tasks and 65 official test folds** using the natural-language
representation described in the published paper:

Yuhang Wang, Xiaotong Liu, Haotian Chen, Tao Yang, Xingchen Liu and Xiaodong Wen,
[Exploring lightweight language models for materials informatics with AlchemBERT](https://doi.org/10.1016/j.xcrp.2025.102724),
*Cell Reports Physical Science* **6**(8), 102724 (2025).

## Results and compatibility

`results.json.gz` contains the original sample-ID-keyed predictions with all
fold metrics recomputed using current Matbench and checksum-verified official
labels. Classification uses unrounded positive-class probabilities for ROC-AUC
and the fixed `p > 0.5` rule for decision-based metrics. This corrects the old
thresholded ROC-AUC values; it is a scoring migration, not a new training run.
The original result is retained byte for byte in `historical/results.json.gz`.

- [RESULTS.md](RESULTS.md): current primary scores and historical comparison.
- [RESCORING.md](RESCORING.md): reference identities and migration commands.
- [rescoring.json](rescoring.json): every original/recomputed fold metric,
  differences, dataset checksums and software versions.
- [provenance.json](provenance.json): original and current artifact identities,
  plus archived source and external input hashes.

The scoring migration was validated against upstream main
`2f7512ef40935b5d642127887fc4daf42236a2b7`. Core scoring and website generation
remain unchanged. A subsequent archive-preservation repair adds explicit source
verification and benchmark-local tests to CI without changing the predictions,
scores or active runtime dependency pins.

## Model and input representation

The published model uses the first token representation of BERT-base-cased
with a 768-to-1 linear head and a sigmoid for classification. The
[base model card](https://huggingface.co/google-bert/bert-base-cased/blob/main/README.md)
describes BookCorpus and English Wikipedia pretraining. The historical model
and tokenizer revisions are not recorded.

| Tasks | Material input |
| --- | --- |
| `expt_gap`, `expt_is_metal`, `glass` | Formula-derived composition descriptions |
| `steels` | Alloy composition fractions |
| `jdft2d`, `phonons`, `dielectric`, `log_gvrh`, `log_kvrh`, `perovskites`, `mp_e_form`, `mp_gap`, `mp_is_metal` | Structure-derived formula, lattice, symmetry, sites and local environments |

The archived generator and `upstream/constants.py` show the descriptions and
fixed task/unit instructions. They do not query a property-text database or
Robocrystallographer. The original JSON inputs store targets separately as
`output`; `QADataset` constructs encoder input only from `instruction + input`.
The target field is parsed separately as a label.

The paper reports an RTX 4090, a 90/10 inner training/validation split, learning
rate 1e-5, batch size 32 and early-stopping patience 200. The archived scheduler
selects `bert-base-cased`, encoder fine-tuning, every task and all five folds.
Source defaults additionally specify PyTorch/CUDA seed 42, maximum sequence
length 512, AdamW, at most 5000 epochs, and lowest validation loss (BCE for
classification, MAE for regression). These are paper/source observations;
[observed_config.json](observed_config.json) and
[PAPER_EVIDENCE.md](PAPER_EVIDENCE.md) distinguish them from historical run logs.

## Source and reproduction

The original property-task source is
[liuxiaotong15/alchemBERT on Gitee](https://gitee.com/liuxiaotong15/alchemBERT),
pinned for inspection to commit `3ff872f6510cc88e96061844a2ca41d0c6041c8f`
(2025-01-26). `upstream/` contains ten source files with their original bytes,
including training, prediction, recording, templates and the GPL-3.0 license.
This snapshot postdates the original submission and is not established as the
exact historical run revision. Matbench-Discovery assets are a separate
benchmark and are not used here.

The original Gitee `requirements.txt` is stored locally as
[`upstream/requirements.txt.archive`](upstream/requirements.txt.archive).
`provenance.json` retains its original filename and SHA256 and records the
local archive-path mapping. The rename distinguishes this preserved evidence
from an installation requirements file; its bytes must remain unchanged.
Do not install from it. New experiments use `requirements-rerun-cpu.txt`, where
dependency compatibility and security updates are maintained separately.

```bash
python benchmarks/matbench_v0.1_AlchemBERT/verify_source.py
# Optional: also verify all 130 v1 input archives in the pinned Gitee checkout.
python benchmarks/matbench_v0.1_AlchemBERT/verify_source.py --source alchembert-source
```

[RUNNING.md](RUNNING.md) gives a complete path for **new experiments**:
`prepare_inputs.py` creates identified, target-free inputs; `train_fold.py`
trains a fresh model for each fold, saves checkpoints and run records, and
exports probabilities with sample IDs; `record_predictions.py` enforces exact
fold membership and records through the current Matbench API. The environment
in `requirements-rerun-cpu.txt` is for these prospective tools, not a recovered
historical training lockfile.

## Historical experiment limitations

The archived code and inputs make the implementation inspectable, while these
remaining distinctions matter when interpreting the published results:

- **Input identity:** the visible generator produces v2 descriptions while the
  trainer reads v1 archives; the original v1 generator was not recovered.
  [INPUT_AUDIT.md](INPUT_AUDIT.md) documents composition identity checks for all
  40 relevant archives, including the older official reference typo. The
  keyword scan and formula checks do not verify every structure description
  or establish which files were used in the original run.
- **Inner validation preprocessing:** the archived code fits the target
  `PowerTransformer` before the 90/10 inner split for `jdft2d`, `phonons`,
  `dielectric` and `steels`. Inner validation labels therefore influence that
  preprocessing; this does not establish outer-test leakage. The new runner
  fits only on inner-training labels. That repair does not change past runs.
- **Run provenance:** the 65 original trained checkpoints, original prediction
  CSVs and run logs are not included. Exact model revisions, selected epochs,
  tuning history, runtime and full historical environment are not recovered.
  The historical dependency list also has incompatible pins. Original CSVs
  lack sample IDs, whereas the submitted result already contains official IDs.

Rescoring verifies the saved predictions against official targets. It neither
reproduces historical model training nor proves absence of data leakage or
pretraining-corpus overlap. [REVIEW_STATUS.md](REVIEW_STATUS.md) maps the review
to the included evidence and these limits.

## Validation

Install this checkout into a Python 3.12+ environment. For migration and
benchmark-local non-training tests, install `rescore_requirements.txt` and
`inflect`; optional tiny-model integration tests also require the prospective
PyTorch/Transformers dependencies.

```bash
python benchmarks/matbench_v0.1_AlchemBERT/verify_source.py
python benchmarks/matbench_v0.1_AlchemBERT/validate_results.py
python -m unittest discover -s benchmarks/matbench_v0.1_AlchemBERT -p 'test_*.py'
python -m unittest scripts.test_submission
python -m scripts.test_submission --submission matbench_v0.1_AlchemBERT
```

The last command is upstream's current probability-submission gate. It checks
canonical classification fold IDs, probability form and all cached metrics
against official targets. Structural validation additionally requires the
full artifact to be valid, complete and recorded.

The `benchmark-submission` CI job explicitly runs `verify_source.py`, its
regression tests and the benchmark-local test suite. Default repository-wide
test discovery alone does not include this directory. CI does not install the
training stack, so optional tiny-model training tests skip there; follow
[RUNNING.md](RUNNING.md) to run them in the separate CPU environment. The source
check verifies the archived source files; it does not install archived dependencies
or validate historical training. The original submission's remote checks are
listed on [PR #358](https://github.com/materialsproject/matbench/pull/358/checks).
