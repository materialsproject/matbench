# Review response

| Review point | Included evidence / resolution |
| --- | --- |
| Scope and conflicts | Synchronized with main `2f7512e`; only `benchmarks/matbench_v0.1_AlchemBERT/` differs. Core code, workflows and dependencies use main unchanged. |
| Training, prediction and recording code | Checksummed original code in `upstream/`; executable prospective preparation, training, prediction and current-API recording in `RUNNING.md`. |
| Publication and checkpoint | Formal 2025 journal citation replaces the preprint; BERT-base-cased and its public pretraining description are identified. |
| Templates and composition-only inputs | Per-task source templates are included. Composition tasks use formulas/alloy fractions; structure tasks use structure descriptions. Input audits and limitations are documented. |
| Complete folds and prediction form | All 13 tasks / 65 folds are recorded; classification predictions are the original floating-point probabilities in `[0, 1]`. |
| Current scoring | All task/fold metrics recomputed from verified official labels, with sample-ID alignment. Original predictions and fold parameters remain unchanged. The original artifact and per-fold comparison are retained. |
| Hyperparameters and compute | Paper/source observations, including the RTX 4090 and training defaults, are documented without inventing missing run records. |
| Validation | Current Matbench structural validation, strict probability gate, benchmark-local tests and documentation build; see PR Checks for remote results. |

The remaining historical evidence limits are listed in [README.md](README.md):
original run/checkpoint identities, v1 generator, hyperparameter-selection
history, and the source's pre-split target preprocessing for four tasks.
The new training pipeline corrects that preprocessing for future experiments;
a scoring migration cannot retroactively correct it. Maintainer scientific
review remains distinct from passing software and artifact checks.
