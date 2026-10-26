# Review response and remaining author evidence

This document maps the review to concrete changes and remaining limitations.
It is not a claim that every historical experiment has been reproduced or that
the submission has been approved for merge.

| Review point | Work completed | Remaining limitation |
| --- | --- | --- |
| Benchmark-only scope / merge conflicts | Synchronized with upstream `4b7d071`; final changes stay within this benchmark directory. Core code, dependencies and workflows are unchanged from main. | Maintainer review and merge remain pending. |
| Training, prediction, recording code | Include the checksummed original source and license. Add `prepare_inputs.py`, `train_fold.py`, and ID-validating `record_predictions.py`; commands in `RUNNING.md`. | The new corrected implementation does not establish the exact historical source revision. |
| Checkpoint / pretraining corpus | Source scheduler names BERT-base-cased; original paper and public model card are documented. | Historical immutable model/tokenizer identities, fine-tuned checkpoints and any additional pretraining still require original evidence. |
| Text templates / target information | Original task templates and generator are inspectable. New preparation uses official inputs and never serializes targets. Audit all 40 composition archives against identified official references. | Original v1 generator is missing; structure-neighborhood text and historical usage are not independently established. No definitive absence-of-leakage claim is made. |
| Outer folds / fresh model / preprocessing | Official IDs and all five folds are enforced. New runner creates a fresh model per fold and fits target transforms only after the inner split, on inner-training labels. | Archived source fits preprocessing before the inner split for four tasks. A new implementation cannot correct the already-completed historical experiments. |
| Hyperparameters / seeds / compute | Observed settings in `observed_config.json`; paper-reported RTX 4090 in `PAPER_EVIDENCE.md`. New runs record configs, seeds, best epoch, timing and runtime. | Historical logs, actual environment, tuning trials and outer-test-use policy are not recovered. |
| Completeness / prediction form | Original artifact validates as complete and recorded for 13 tasks/65 folds; classification predictions are floating probabilities. Original bytes and citation remain intact. | Artifact validity does not establish its model provenance. |
| Stored scores | Per-task historical means in `RESULTS.md`; independent ID-aligned rescoring in `RESCORING.md` identifies probability-based AUC corrections. | Current `expt_is_metal` label file is still required. Canonical historical results have not been silently overwritten. |
| Installable dependencies | Tested prospective CPU dependency versions and `pip check`; archival requirements retained separately. | A new working environment is not the historical environment. |
| Tests / CI | Benchmark-local unit and tiny-BERT integration tests; upstream submission/data-operations checks. Upstream workflows remain unchanged. | Tiny random-model tests are software validation, not full BERT training. Latest remote checks are on the PR Checks tab. |

## Smallest remaining data request

To complete current-version scoring for the final task, provide a cached copy
or accessible download of `matbench_expt_is_metal.json.gz` with SHA256:

```text
8f2a4f9bacdcbc5c2c73615629ee7986f09d39bed40ba7db52b61b2889730887
```

The public older Figshare version has a different checksum. It is documented
as a historical input-audit reference and is not accepted by the scoring script
as the current label table. The official download endpoint returned HTTP 403
in the audit environment. No target labels have been inferred from model
predictions or fabricated.

## Original-experiment evidence request

A read-only download link to the original run directory is sufficient to begin;
one task/fold can establish the chain first, followed by the remaining folds:

1. Training source revision and actual launch commands; the v1 generator or
   confirmation of the exact input archive hashes used.
2. Base-model/tokenizer identities, trained checkpoints, original prediction
   CSVs, and their task/fold/sample-ID correspondence.
3. Training/validation logs, selected epochs, actual Python/CUDA/package export,
   seeds, hardware/time and the model/hyperparameter selection procedure.
4. Whether additional pretraining occurred and whether outer-test scores were
   used to choose settings, epochs or models.

These are requests for existing experimental evidence, not requests to invent
parameters or rerun an experiment and call it the historical one. Broader
GitHub permissions cannot recover missing experimental records.
