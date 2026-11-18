# TabPFN-HYD-FT: corrected JDFT2D exfoliation-energy regression

**Built with PriorLabs-TabPFN.** Only matbench_jdft2d is covered, not the 13-task general-purpose leaderboard.

The corrected native five-fold MAE is **29.99157698166102 meV/atom**, std **9.44184383860221**, using 636 real predictions. This result replaces the initial 28.733328891325755 regardless of metric direction. It is a measured result awaiting scientific review, not a claim of official first.

The initial result had design-selection bias risk: exploratory fold0 TRAIN overlaps other folds' TEST labels and influenced a narrowed grid. The corrected run reselects the full historical candidate closure in each own TRAIN. Historical labels/results cannot be made unseen; no pristine blind-design claim is made. See development-history.json and REVIEW_RESPONSE.md. Original metadata are explicitly prefixed original- and the initial commit/results SHA remain disclosed.

All 21 stages completed with exact SHA-pinned public scientific source. The initial public master was interrupted before one GPU submission by private course catalog resolution and subsequently host-resumed using the same public prepare/select/train/predict/scoring functions, data and protocol. Prior completed stages were SHA-adopted, not recomputed. No GPU failure was automatically retried. The recovery also corrected the monitor's expected response path while observing the already-created job. Those were transport/observation repairs, not scientific changes. The original master is not claimed to have completed uninterrupted; the default local-CUDA backend has not separately been fully retrained.

validation.json records full output integrity, actual per-stage compute, genuine training and fresh-process replay. All five final models froze before held-out inference; all 636 IDs are present. Historical exposure and scientific acceptance remain explicit limitations.

## Reproduce the corrected protocol

Use Python3.12, an **NVIDIA A100 GPU (the actual runs used an A100-SXM4-80GB)**, a compatible CUDA PyTorch installation and requirements.txt. The candidate worker explicitly rejects non-A100 GPUs; CUDA availability alone is insufficient. Other GPU architectures are not claimed supported or numerically verified. The non-gated Prior-Labs/TabPFN-v2-reg checkpoint revision02a44705ff7790b50cded404bc59adb94da456a1 has required SHA256 2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736. The original license and required attribution are included; foundation pretraining is not claimed.

Download the official Matminer matbench_jdft2d.json.gz (SHA25626057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf). Use matbench/matbench_v0.1_validation.json from the pinned Matbench revision35f88110a51adf817a53515780d99fc34f614bf8 (SHA25694c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d). Both SHA values are checked before preparation.

```bash
python reproduce.py --data /path/to/matbench_jdft2d.json.gz --splits /path/to/matbench_v0.1_validation.json --protocol corrected-protocol-freeze.json --output /new/output-directory
```

Optional --checkpoint supplies the exact pinned checkpoint. Without it, the pinned public revision is downloaded. No course account or private API is needed for default local-CUDA execution. Existing outputs/intents are refused; failures stop without automatic GPU retries. Allow roughly12GiB free storage for retained checkpoints and evidence. An optional --executor/--executor-python accepts explicit bounded-job transport; it cannot change scientific inputs or recipe.


## Method and actual execution

The per-TRAIN 1206-candidate protocol, all schedules, true final-fit updates and complete inner selection metrics are included in corrected-protocol-freeze.json and selection-freezes.json. Every public scientific dependency is included. requirements.txt, LICENSE-TabPFN.txt and source-lock.json retain the pinned original non-gated checkpoint and attribution. The command reconstructs and trains, rather than loading submitted predictions. Failures stop. Default reproduction requires no course account; an optional explicit stage executor is only transport.

## Maintainer-requested frozen-model replay

The five exact final checkpoint files, SHA256 hashes, original byte-identical pre-inference freeze and TRAIN/test-geometry replay contexts are listed in `frozen-model-assets.json`. They are available from [the frozen-model release](https://github.com/donkeey11/matbench/releases/tag/jdft2d-corrected-models-v1). Checkpoints include the saved optimizer state exactly as frozen; they are not retrained or rewritten. **Built with PriorLabs-TabPFN**, under the included Prior Labs license. Load only these SHA-pinned files.

On an A100 with the pinned environment, replay all 636 submitted predictions without training, selecting a recipe, loading TEST targets or computing a benchmark score:

```bash
python verify_full_predictions.py --assets /new/frozen-assets --download --output /new/full-replay-report.json
python -m unittest test_checkpoint_path.py
```

The replay uses the original TRAIN contexts needed by TabPFN and the five frozen recipes; test NPZs contain only IDs and analytic features. It checks every file's SHA and size, the original freeze, exact own-TRAIN/TEST ID separation, all five model hashes, all 636 predictions, and the pre-existing atol=1e-3/rtol=1e-4 tolerances. It never uses discrepancies to choose a model. `--check-only` checks assets without executing GPU inference. The default foundation checkpoint is downloaded from the pinned public revision, or `--checkpoint` accepts its exact hash-verified local file.

The cache path fix stages the exact checkpoint bytes under the required filename rather than resolving a snapshot symlink to its hash-named blob directory. Its CPU regression test reproduces the standard Hugging Face snapshots-to-blobs layout and a custom filename, and rejects a wrong hash. `runtime-amendment.json` preserves the original executed launcher hash and declares the amended launcher; the original protocol, five scientific worker files, weights, selected recipes and submitted result bytes remain unchanged. No claim is made that the amended standalone training launcher has been fully retrained end to end. Historical fold-0 exposure remains disclosed above.

The extra full636 GPU rerun of this new verification script has not completed: input-storage lease creation failed before a GPU Pod was created, and manual operational recovery was blocked by unverifiable lease release. Local checks verified every asset hash, original freeze/model/recipe links and test-geometry-only inputs, and the cache-path CPU regression tests passed. The original C3 inference already generated all636 submitted predictions; its80 TRAIN-witness replay scope is unchanged. No additional complete-GPU-replay success is claimed. See followup-verification.json.
