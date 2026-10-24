# AlchemBERT benchmark: source and reproducibility status

This submission preserves the original `results.json.gz` and adds inspectable
source, provenance, and read-only checks. It does **not** claim that historical
training has been reproduced or that all review questions have been resolved.
See [RESULTS.md](RESULTS.md) for every task's stored score and the historical
classification-scoring limitation.

## Scope and preserved evidence

The PR has been synchronized with Matbench main
`4b7d0714849a917814115be380a64e7ace94be55`. Its final diff against that commit is
confined to `benchmarks/matbench_v0.1_AlchemBERT/`. Core scoring, NumPy support,
tests, dependency files and workflows use main unchanged. The original PR head
is `b44dca2f98b1b7112bdb1900183475876fb01551`; its result bytes and citation are
preserved. `provenance.json` records the result's SHA256 and Git blob.

The official source linked by the original submission is
[liuxiaotong15/alchemBERT on Gitee](https://gitee.com/liuxiaotong15/alchemBERT).
It was accessible for this review. The fixed snapshot is
[`3ff872f6510cc88e96061844a2ca41d0c6041c8f`](https://gitee.com/liuxiaotong15/alchemBERT/tree/3ff872f6510cc88e96061844a2ca41d0c6041c8f),
dated 2025-01-26. It postdates the original submission and is **not established
as the commit that produced the submitted results**.

`upstream/` contains byte-for-byte copies of ten files from that snapshot,
including its GPL-3.0 [LICENSE](upstream/LICENSE). That source retains its
original license. The copies are archival evidence and retain the defects and
machine-specific assumptions discussed below; they are not a certified,
portable reproduction recipe.

| File | Evidence now included |
| --- | --- |
| `upstream/train.py` | Model architecture, training, validation, checkpoint loading, prediction |
| `upstream/matbench_submit.py` | Original CSV-to-Matbench recording; API is `task.record(...)` |
| `upstream/str_matbench_data_generation.py` | Visible structure and composition text generation |
| `upstream/constants.py`, `upstream/converter.py` | Per-task instructions, units, transforms and text helpers |
| `upstream/scheduling.sh` | Explicit model choice, 13 tasks and five folds |
| `upstream/requirements.txt` | Original dependency declarations, including the conflicts below |
| `upstream/print_log.py`, `upstream/README.md` | Original log-reading utility and usage instructions |
| `observed_config.json` | Machine-readable settings found in source; unknown historical fields are null |
| `provenance.json` | Source/result hashes and hashes of all 130 external `data_nl_v1` archives |

The external repository contains the v1 input archives. They are not duplicated
here (about 286 MB compressed). Check their exact bytes before using them:

```bash
git clone https://gitee.com/liuxiaotong15/alchemBERT.git alchembert-source
git -C alchembert-source checkout --detach 3ff872f6510cc88e96061844a2ca41d0c6041c8f
python benchmarks/matbench_v0.1_AlchemBERT/verify_source.py --source alchembert-source
```

This command checks the ten local source copies, their external counterparts,
and all 130 external v1 archives. Without `--source`, only local copies are checked.
Hash agreement establishes file identity, not their use in the historical run.

## What the recovered code establishes

The scheduler explicitly selects `bert-base-cased`, encoder fine-tuning, every
Matbench task and folds 0–4. The training script's default is instead
`bert-base-uncased`; reproducing the scheduler requires its explicit arguments.
The public [bert-base-cased model card](https://huggingface.co/google-bert/bert-base-cased/blob/main/README.md)
describes BookCorpus and English Wikipedia pretraining. The historical model
and tokenizer revisions/hashes are unknown; a model name cannot establish their
identity or exclude overlap between pretraining material and benchmark data.

The visible model uses the first token representation with a linear scalar head
and a sigmoid for classification. The source sets PyTorch/CUDA seed 42, token
length 512, train/validation batch size 32, test batch size 1, AdamW learning rate
1e-5 with fine-tuning (1e-4 with a frozen encoder), an unstratified random 90/10
inner split, at most 5000 epochs and early stopping patience 200. It chooses the
lowest validation loss: BCE for classification and MAE for regression. A fresh
model is constructed for each fold/job. These are observed source settings,
not recovered run logs. Python/NumPy seeds, deterministic settings, the actual
best epochs and the hyperparameter selection process are not recorded.

## Inputs, folds and leakage checks

The visible generator obtains official outer train/validation data. It derives
composition descriptions from formulas (alloy fractions for steels), and
structure descriptions from formula, lattice, symmetry, sites and neighbors.
It does not call Robocrystallographer or a property-text database.
`upstream/constants.py` specifies each task's instruction and units.

| Tasks | Visible input source | Requested property in the fixed instruction |
| --- | --- | --- |
| `expt_gap`, `expt_is_metal`, `glass` | Formula/composition description | Experimental gap, metal classification, glass classification |
| `steels` | Alloy composition fractions | Yield strength |
| `jdft2d`, `phonons`, `dielectric` | Crystal structure description | Exfoliation energy, phonon peak, refractive index |
| `log_gvrh`, `log_kvrh` | Crystal structure description | Log shear and bulk moduli |
| `perovskites`, `mp_e_form`, `mp_gap`, `mp_is_metal` | Crystal structure description | Formation energy, gap or metal classification |

The generator requests test targets and stores them in a separate `output`
field. `QADataset` constructs model text solely from `instruction + input`;
`output` is parsed separately as the label. The test labels are used for
post-prediction metrics. Their presence in a JSON file is not, by itself,
evidence that they entered the encoder input or training.

The review read all 130 v1 archives: all have `instruction/input/output`, one
fixed instruction per task, and ten files per task. A case-insensitive
whole-word scan of `input` for `band[- ]gap`, `yield strength`, `metallic`,
`refractive`, `glass`, `target`, `True`, `False`, `MPa`, and `eV` found no matches.
This is a limited check, not proof of the absence of semantic proxies or leakage.

For the four composition tasks above, all 40 archive lengths match official
fold sizes. Assigning official ordered fold IDs to rows gives the same complete
row for every ID across all five folds, with zero conflicts; train/test complete
rows do not intersect within a fold. This establishes internal partition
consistency, but cannot rule out a consistent global permutation. The official
dataset endpoint returned HTTP 403 in this environment, preventing an independent
material-to-ID comparison. The archives lack explicit sample IDs. Actual input
identity and original prediction order therefore still require verification.

## Known reproduction problems, not hidden by this update

1. **v1/v2 mismatch.** `train.py` reads `data_nl_v1`, but the visible generator
   writes `data_nl_v2` and spells numbers as words. Existing v1 text differs.
   The available 15-commit history did not provide the original v1 generator.
   Structure generation also joins a set without a fixed order, which can
   affect truncation. Do not regenerate with v2 and call the result identical
   to the historical inputs.
2. **Inner validation isolation.** For `jdft2d`, `phonons`, `dielectric`, and
   `steels`, `PowerTransformer` fits the entire outer training/validation label
   set before the 90/10 inner split. Thus inner validation labels influence
   preprocessing. This does not establish outer-test leakage. A corrected run
   must split first, fit only inner-training labels, transform validation with
   that fitted transformer and save the transformer with its checkpoint.
   Such a change needs a new experiment; it cannot validate the old scores.
3. **Classification scoring.** Original test logging rounds probabilities for
   ROC-AUC, while prediction CSVs retain unrounded values. All 15 stored
   classification fold ROC-AUC values equal balanced accuracy to 1e-12,
   consistent with the historical Matbench scoring issue. Updating main or a
   logging expression does not repair historical stored scores. Independently
   re-score aligned official labels/probabilities into a separate report before
   deciding with the maintainer how to present these results.
4. **Prediction identity.** Original CSVs contain predictions and ratings but no
   sample IDs; the original recorder assumes row order. It also reads
   `nl_results/bert-base-cased`, while training writes a differently named log
   directory. The new `record_predictions.py` refuses to guess sample order.
5. **Environment conflicts.** Upstream pins NumPy 1.23.5, pandas 1.5.3 and
   matplotlib 3.7.0 with pymatgen 2024.10.22. The latter's
   [versioned dependency declarations](https://github.com/materialsproject/pymatgen/blob/v2024.10.22/pyproject.toml)
   require NumPy >=1.25, pandas >=2 and matplotlib >=3.8. The supplied list is
   not a consistent installable lockfile. It is retained as evidence, without
   substituting invented historical versions. `/data/cache`, `~/lxt/venv`, GPU
   IDs 0–7 and a forced `.cuda()` call are additional portability assumptions.

## Recording new, identified predictions safely

This update adds an independent recorder using the current Matbench API. It
requires CSV columns `task,fold,sample_id,prediction`; sample IDs must be the
official Matbench test IDs. It checks exact ID sets for each fold, reorders
predictions to official test order, rejects duplicates/non-finite values and
invalid probability ranges, and passes classification probabilities directly to
`task.record`. It fetches test inputs with `include_target=False`.

```bash
python benchmarks/matbench_v0.1_AlchemBERT/record_predictions.py predictions.csv \
  --output fresh_results.json.gz
```

The default requires all tasks/folds. `--tasks matbench_steels` permits a subset
but still requires all folds of each selected task. `--params-json` accepts
explicitly supplied task/fold parameters; unknown parameters are left empty.
Input hashes are recorded. Existing output files and `results.json.gz` cannot
be overwritten. The original unlabelled CSV format is intentionally rejected:
the author must supply a trustworthy ID mapping. This utility is not evidence
that the historical predictions came from the archived model code.

## Evidence still needed from the original experiment

- Exact source revision and commands that produced the submitted artifact;
  original v1 generator, or confirmation of the hashed input files used.
- Per-task/fold mapping of input rows and prediction CSVs to official sample IDs.
- Immutable base-model/tokenizer revisions and hashes; all 65 trained checkpoints,
  their hashes/download locations, and their association with prediction CSVs.
  The source tree contains no checkpoints or original CSVs; the original PR
  offers about 25 GB of models on request.
- A working original Python/CUDA/dependency lockfile resolving the conflicting
  pins; actual random seeds, deterministic settings, hardware and training time.
- Training/validation logs, selected epochs, hyperparameter trials and selection
  policy, and evidence that outer test data were not used for model selection.
- Details of any continued pretraining and evidence relevant to corpus overlap.
  No definitive claim of either leakage or its absence is made here.

## Validation

Use an environment for the current Matbench checkout separately from any
historical training environment. Main now requires Python >=3.12. From the root:

```bash
python benchmarks/matbench_v0.1_AlchemBERT/verify_source.py
python benchmarks/matbench_v0.1_AlchemBERT/validate_results.py
python -m unittest discover -s benchmarks/matbench_v0.1_AlchemBERT -p 'test_*.py'
python -m unittest scripts/test_submission.py
```

Local validation against main `4b7d071` confirmed the original hash and
`is_valid`, `is_complete`, `is_recorded` for 13 tasks/65 folds. New recorder
unit tests cover identity checks and overwrite protection, without training a
model. The sparse local submission check covers this benchmark only. The full
upstream CI matrix and documentation build must be checked on the updated PR;
July's successful checks do not establish success for this revision. A local
full data-operations test encountered a failed official dataset download/hash
check, so it is not reported as passing. No new training run has been performed.
