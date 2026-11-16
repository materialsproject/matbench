# TabPFN-HYD-FT: trained JDFT2D exfoliation-energy regression

**Built with PriorLabs-TabPFN.** This entry covers only `matbench_jdft2d`, not the Matbench 13-task general-purpose leaderboard.

The recorded native official five-fold mean MAE is **28.733328891325755 meV/atom** (fold std **9.38087794130482**), retaining all **636** held-out predictions. This file reports a measured benchmark result; leaderboard acceptance is determined by the maintainers.

## Method and training

The public non-gated `Prior-Labs/TabPFN-v2-reg` checkpoint is pinned to revision `02a44705ff7790b50cded404bc59adb94da456a1`, file `tabpfn-v2-regressor-v2_default.ckpt`, SHA256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Its original license, including the additional attribution requirements, is in LICENSE-TabPFN.txt. This derivative starts its public name with TabPFN and does not claim original foundation pretraining.

118 elemental fractions, Matminer stoichiometry/Magpie/valence features, and analytic cell/packing/layer-gap/periodic-neighbor descriptors yield 370 features. Feature construction uses structures only, with no target-dependent global fitting. TabPFN preprocessing is fitted within each relevant TRAIN context.

Each official outer TRAIN fold independently uses reduced-composition GroupKFold(3) to choose the lowest mean-fold MAE from a fixed four-configuration grid: full-network lr 1e-6 for 5/10 epochs, full-network lr 5e-6 for 5 epochs, or readout-only lr 5e-5 for 10 epochs. The minimum-index tie break is fixed. All candidates genuinely update weights; the original no-gradient model is not an eligible fallback. Fine-tuning uses 128 context-plus-query samples, query fraction .25, two estimators, activation checkpointing, MAE weight 1 and CRPS weight .25, no external validation or early stopping; final predictions use eight estimators and the median, seed3042. Complete settings and the actual executed training workers are included.

The selected configuration is genuinely refitted on all 508/509 rows of its corresponding TRAIN fold. Final refits have 15/30/30/30/30 optimizer steps. Four final models update 129 parameter tensors; the remaining readout-only model updates four tensors while retaining its backbone. The five models and configurations are frozen before a separate process replays 16 TRAIN witnesses per model and generates all 636 official held-out predictions. Official scoring follows only then. Per-fold choices and checksums are in selection-freezes.json.

Exploratory development used official fold0 TRAIN only. Its final inner results were reused after exact feature/label/ID/CV/config comparison; its final full-TRAIN model was newly refitted. Other outer folds select within their own TRAIN. Different official TRAIN folds naturally share labels, as required by five-fold cross validation; this is not a globally disjoint-label claim. Official TEST scores, labels and others' prediction files never feed back into configuration or checkpoint selection. The original development failures and inferior configurations are retained in the experiment archive.

## Reproduce

Use Python3.12 and a CUDA GPU (actual runs: PyTorch2.13.0+cu126, A100-SXM4-80GB). Retaining all intermediate checkpoints needs roughly 12GiB disk. Install requirements.txt in a new environment; PyTorch must match the local CUDA installation. Matbench is pinned to commit `35f88110a51adf817a53515780d99fc34f614bf8`.

Download the official Matminer `matbench_jdft2d.json.gz` dataset to a data directory. Its required SHA256 is `26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf`. Do not change its filename or row order.

```bash
python reproduce.py --data /path/to/matbench_jdft2d.json.gz --output /new/output-directory
```

The optional `--checkpoint /path/to/tabpfn-v2-regressor-v2_default.ckpt` uses an already downloaded exact checkpoint. Otherwise Hugging Face downloads the pinned public revision. No private course account or API is required. `--prepare-only` constructs features and per-fold TRAIN bundles on CPU without training, prediction or scoring. Existing output directories are refused; execution failures stop rather than automatically retrying or changing models.

The public reproduction recomputes all inner folds, including fold0 rather than relying on the development cache. It freezes all five trained models, launches a separate process for replay/inference, then calls native `MatbenchTask.record` for all five official folds and writes a new results.json.gz. It does not import the submitted predictions to reproduce them.

## Verification scope

All downloaded training/prediction files have been independently SHA256-rechecked, genuine updates and optimizer checkpoints are retained, and all jobs released their GPU. Separate-process TRAIN witness replay passed for all five models with maximum absolute prediction difference 0.0 under the predeclared atol1e-3/rtol1e-4. All 636 held-out IDs are unique and complete. Native results were validated after writing and reloading. The original actual training workers are byte-identical to the submitted worker files. The public predictor changes only input-root plumbing. Public preparation equivalence and contribution-format validation are reported in validation.json. The complete public command has not been separately retrained end-to-end; no such claim or fabricated notebook output is made.
