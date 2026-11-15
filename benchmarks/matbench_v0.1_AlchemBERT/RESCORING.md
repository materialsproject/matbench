# Migration to current Matbench scoring

All **13 tasks / 65 folds** are independently rescored using the exact current
official Matminer datasets and upstream main
`2f7512ef40935b5d642127887fc4daf42236a2b7`. All downloaded files match the
SHA256 identities in `rescore_results.py`. The formerly inaccessible experimental
metallicity and gap datasets are now available and verified; no historical
Figshare or derivative-table fallback is used in this migration.

The [per-fold report](rescoring.json) contains every stored metric, recomputed
metric, difference, label source, official split/metadata hash and software
version. [RESULTS.md](RESULTS.md) shows the primary score means and standard
deviations. All three classification ROC-AUC values now use probabilities.

## What changes

`migrate_results.py` starts from the hash-locked
`historical/results.json.gz`. It aligns targets and predictions by official
sample ID, rejects incomplete tasks/folds, and replaces each fold's metric
cache with the current `matbench.data_ops.score_array` output. Classification
uses unrounded probabilities for ROC-AUC and the fixed `p > 0.5` rule for the
other metrics. The 13 tasks, 65 folds, prediction values and types, parameters
and uncertainties are checked to remain identical through serialization.

Matbench regenerates its internal content hash, version and serialization
datestamp; `user_metadata.score_migration` records the source and migration.
The original result remains byte-identical. The top-level `results.json.gz` is
the migrated submission consumed by the website and submission checks.
This is not a new model experiment or evidence of historical training replay.

## Reproduce the migration

Install this checkout in a Python 3.12+ environment:

```bash
python -m pip install -e .
python -m pip install -r benchmarks/matbench_v0.1_AlchemBERT/rescore_requirements.txt
```

Download each `TASK.json.gz` from
`https://ml.materialsproject.org/projects/TASK.json.gz` into a dataset directory,
where `TASK` is each of the 13 names in [RESULTS.md](RESULTS.md). Use these same
files as the Matminer cache when running the strict submission gate:

```bash
export MATMINER_DATA=/absolute/path/to/verified-datasets
python benchmarks/matbench_v0.1_AlchemBERT/migrate_results.py \
  --dataset-dir "$MATMINER_DATA" \
  --output fresh_results.json.gz --report-output fresh_rescoring.json
python benchmarks/matbench_v0.1_AlchemBERT/validate_results.py \
  fresh_results.json.gz --expected-sha256 "$(sha256sum fresh_results.json.gz | cut -d' ' -f1)"
```

Both output paths must be new. The migration requires all tasks and performs
structural round-trip and prediction-preservation checks before publishing its
output. Review the output before promoting it to `results.json.gz` and update
`provenance.json.current_artifact` with its identity. Independent reruns can have
different compressed bytes and serialization timestamps while retaining the
same predictions and scores; the committed output's identity is locked in
`provenance.json`.

For a read-only score report, without generating a migrated result:

```bash
python benchmarks/matbench_v0.1_AlchemBERT/rescore_results.py \
  --dataset-dir "$MATMINER_DATA" --output independent_rescoring.json
```

Without `--dataset-dir`, that report command instead checks only `glass`,
`dielectric` and `steels` against hash-locked ground-truth fixtures in Matbench's
tests. This three-task offline mode is distinct from the full migration above.
The script retains an explicitly identified `expt_gap_kingsbury` fallback for
older environments, but it was not needed for this 13-task report.

## Validation and provenance

```bash
python -m unittest discover -s benchmarks/matbench_v0.1_AlchemBERT \
  -p 'test_rescore_results.py'
python -m unittest discover -s benchmarks/matbench_v0.1_AlchemBERT \
  -p 'test_migrate_results.py'
python -m scripts.test_submission --submission matbench_v0.1_AlchemBERT
```

The focused tests cover probability scoring, ID alignment, dataset/scorer
identity, incomplete reports, unchanged predictions and protection of existing
outputs. Upstream's strict gate independently recomputes all classification
metrics from canonical official targets.

The reference files and stored predictions are sufficient to validate scoring.
They do not identify the training run that produced those predictions. Original
checkpoint, input-generation and training-protocol limitations remain documented
in [README.md](README.md) and [REVIEW_STATUS.md](REVIEW_STATUS.md).
