# TabPFN-HYD-FT: corrected JDFT2D exfoliation-energy regression

**Built with PriorLabs-TabPFN.** Only matbench_jdft2d is covered, not the 13-task general-purpose leaderboard.

The corrected native five-fold MAE is **29.99157698166102 meV/atom**, std **9.44184383860221**, using 636 real predictions. This result replaces the initial 28.733328891325755 regardless of metric direction. It is a measured result awaiting scientific review, not a claim of official first.

The initial result had design-selection bias risk: exploratory fold0 TRAIN overlaps other folds' TEST labels and influenced a narrowed grid. The corrected run reselects the full historical candidate closure in each own TRAIN. Historical labels/results cannot be made unseen; no pristine blind-design claim is made. See development-history.json and REVIEW_RESPONSE.md. Original metadata are explicitly prefixed original- and the initial commit/results SHA remain disclosed.

All 21 stages completed with exact SHA-pinned public scientific source. The initial public master was interrupted before one GPU submission by private course catalog resolution and subsequently host-resumed using the same public prepare/select/train/predict/scoring functions, data and protocol. Prior completed stages were SHA-adopted, not recomputed. No GPU failure was automatically retried. The recovery also corrected the monitor's expected response path while observing the already-created job. Those were transport/observation repairs, not scientific changes. The original master is not claimed to have completed uninterrupted; the default local-CUDA backend has not separately been fully retrained.

validation.json records full output integrity, actual per-stage compute, genuine training and fresh-process replay. All five final models froze before held-out inference; all 636 IDs are present. Historical exposure and scientific acceptance remain explicit limitations.

## Reproduce the corrected protocol

Use Python3.12, a compatible CUDA PyTorch installation and requirements.txt. The non-gated Prior-Labs/TabPFN-v2-reg checkpoint revision02a44705ff7790b50cded404bc59adb94da456a1 has required SHA256 2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736. The original license and required attribution are included; foundation pretraining is not claimed.

Download the official Matminer matbench_jdft2d.json.gz (SHA25626057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf). Use matbench/matbench_v0.1_validation.json from the pinned Matbench revision35f88110a51adf817a53515780d99fc34f614bf8 (SHA25694c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d). Both SHA values are checked before preparation.

```bash
python reproduce.py --data /path/to/matbench_jdft2d.json.gz --splits /path/to/matbench_v0.1_validation.json --protocol corrected-protocol-freeze.json --output /new/output-directory
```

Optional --checkpoint supplies the exact pinned checkpoint. Without it, the pinned public revision is downloaded. No course account or private API is needed for default local-CUDA execution. Existing outputs/intents are refused; failures stop without automatic GPU retries. Allow roughly12GiB free storage for retained checkpoints and evidence. An optional --executor/--executor-python accepts explicit bounded-job transport; it cannot change scientific inputs or recipe.


## Method and actual execution

The per-TRAIN 1206-candidate protocol, all schedules, true final-fit updates and complete inner selection metrics are included in corrected-protocol-freeze.json and selection-freezes.json. Every public scientific dependency is included. requirements.txt, LICENSE-TabPFN.txt and source-lock.json retain the pinned original non-gated checkpoint and attribution. The command reconstructs and trains, rather than loading submitted predictions. Failures stop. Default reproduction requires no course account; an optional explicit stage executor is only transport.
