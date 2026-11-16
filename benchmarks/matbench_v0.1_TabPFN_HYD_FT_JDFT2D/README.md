# TabPFN-HYD-FT: JDFT2D review correction in progress

**Built with PriorLabs-TabPFN.** Only matbench_jdft2d is covered, not the 13-task general-purpose leaderboard.

**Draft, not ready for acceptance.** The current results.json.gz is the preserved initial result (MAE28.733328891325755 meV/atom), which has a design-selection bias risk: initial development used fold0 TRAIN, overlapping test labels in other folds, and narrowed the candidate grid. It is not asserted to be a fully unbiased benchmark result or official first. See REVIEW_RESPONSE.md for all four answers, development-history.json for every trial, and the immutable original PR commit340be4d210abbc5367fea241f7e496818f98c171.

The reviewer-requested correction is frozen before execution and reselects all historical alternatives independently inside each official TRAIN. Corrected results will replace the initial result regardless of score direction. Historical exposure cannot be erased by rerunning. Nothing from old/new outer-test scores or predictions is used for fitting, calibration, blending selection or checkpoint selection.

## Reproduce the corrected protocol

Use Python3.12, a compatible CUDA PyTorch installation and requirements.txt. The non-gated Prior-Labs/TabPFN-v2-reg checkpoint revision02a44705ff7790b50cded404bc59adb94da456a1 has required SHA256 2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736. The original license and required attribution are included; foundation pretraining is not claimed.

Download the official Matminer matbench_jdft2d.json.gz (SHA25626057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf). Use matbench/matbench_v0.1_validation.json from the pinned Matbench revision35f88110a51adf817a53515780d99fc34f614bf8 (SHA25694c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d). Both SHA values are checked before preparation.

```bash
python reproduce.py --data /path/to/matbench_jdft2d.json.gz --splits /path/to/matbench_v0.1_validation.json --protocol corrected-protocol-freeze.json --output /new/output-directory
```

Optional --checkpoint supplies the exact pinned checkpoint. Without it, the pinned public revision is downloaded. No course account or private API is needed for default local-CUDA execution. Existing outputs/intents are refused; failures stop without automatic GPU retries. Allow roughly12GiB free storage for retained checkpoints and evidence. An optional --executor/--executor-python accepts explicit bounded-job transport; it cannot change scientific inputs or recipe.

The exact public master and six scientific files are currently being executed with bounded stage transport. CPU preparation/contracts have passed; completed end-to-end corrected execution and a separate full default-backend rerun are not claimed while this remains pending. Historical reproduce_original.py and old worker/selection/replay artifacts remain for provenance, not as the recommended corrected command. selection-protocol.json source hashes describe historical provenance, not missing current imports. The current source dependencies are all included and pinned in corrected-protocol-freeze.json.

## Method

Pure composition/periodic-geometry features; per-TRAIN composition-group three-fold selection over1206 alternatives; actual full/readout TabPFN gradient fine-tuning or random MLP training; positive blends selected only with inner predictions. All five final fits freeze before a new process replays TRAIN witnesses then predicts all636 held-out rows. The native Matbench scorer runs once afterward. The process does not import submitted predictions to reproduce them. Read the full schedules, checkpoint-retention limits and residual retrospective-exposure limitation in REVIEW_RESPONSE.md.
