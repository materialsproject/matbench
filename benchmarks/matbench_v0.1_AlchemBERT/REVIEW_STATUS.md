# Review response

| Review point | Included evidence / resolution |
| --- | --- |
| Scope and conflicts | The scoring migration was validated against main `2f7512e`. The archive-preservation repair adds targeted CI checks without changing core scoring, website generation, predictions or active runtime dependency pins. |
| Training, prediction and recording code | Checksummed original code in `upstream/`; executable prospective preparation, training, prediction and current-API recording in `RUNNING.md`. |
| Archived dependency list | Original Gitee `requirements.txt` bytes are preserved as `upstream/requirements.txt.archive`; the manifest retains the original filename/hash and maps the local archive path. This is evidence, not an installation entry point. Active dependency updates belong in `requirements-rerun-cpu.txt`. |
| Publication and checkpoint | Formal 2025 journal citation replaces the preprint; BERT-base-cased and its public pretraining description are identified. |
| Templates and composition-only inputs | Per-task source templates are included. Composition tasks use formulas/alloy fractions; structure tasks use structure descriptions. Input audits and limitations are documented. |
| Complete folds and prediction form | All 13 tasks / 65 folds are recorded; classification predictions are the original floating-point probabilities in `[0, 1]`. |
| Current scoring | All task/fold metrics recomputed from verified official labels, with sample-ID alignment. Original predictions and fold parameters remain unchanged. The original artifact and per-fold comparison are retained. |
| Hyperparameters and compute | Paper/source observations, including the RTX 4090 and training defaults, are documented without inventing missing run records. |
| Validation | Current Matbench structural validation, strict probability gate and documentation build. CI explicitly verifies the source archive and runs source-verifier regression tests plus benchmark-local non-training tests; optional tiny-model training tests require the separate CPU environment and skip in CI. See PR Checks for remote results. |

The remaining historical evidence limits are listed in [README.md](README.md):
original run/checkpoint identities, v1 generator, hyperparameter-selection
history, and the source's pre-split target preprocessing for four tasks.
The new training pipeline corrects that preprocessing for future experiments;
a scoring migration cannot retroactively correct it. Maintainer scientific
review remains distinct from passing software and artifact checks.
