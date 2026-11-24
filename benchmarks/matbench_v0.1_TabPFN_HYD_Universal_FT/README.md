# TabPFN-HYD-Universal-FT: three materials regression tasks

**Built with PriorLabs-TabPFN.** This submission applies one task-parameterized regression method to three complete Matbench tasks. It is neither an all-13-task benchmark nor a claim of official first place. Acceptance and public ranking remain the maintainers' decision.

The native five-fold results are: phonons MAE **24.924304469958248 cm^-1** (fold std **2.5179944964811543**, 1265 held-out rows); refractive index, `matbench_dielectric`, MAE **0.2564567309263447** (std **0.07292975450284772**, 4764 rows); experimental band gap MAE **0.31707419178786395 eV** (std **0.021642045162478318**, 4604 rows). Full RMSE, MAPE and maximum-error statistics are in each task's native-score.json. No claim of superiority on every metric or statistical significance is made.

## One method, prospectively fixed selection

The method transfers the reviewed steel/JDFT2D machinery into input-only features, own-TRAIN selection, genuine gradient training, final refitting, model freezing and inference. Composition260 contains 118 elemental fractions and Matminer stoichiometry/Magpie/fractional-valence descriptors. All370 adds 110 analytic cell/packing/periodic-neighbor descriptors when a structure is available. Experimental band gap provides composition only: its structural slots are missing, not measured crystal geometry. Learned imputation/scaling uses only the corresponding fit TRAIN.

The same predeclared 1206-recipe closure is independently considered in each official outer TRAIN: two feature views, six TabPFN full-network/readout schedules, four/eight inference estimators, mean/median summaries, eight frozen comparison variants, positive trained/frozen mixtures with trained weights .25/.5/.75/1, and six genuinely trained MLPs. This is **1206 recipes, not 1206 independent fits**. Composition-group GroupKFold3 mean validation MAE and a fixed-index tie-break select one recipe per outer TRAIN. Pure frozen/tree recipes cannot independently win because positive-gradient training was required prospectively. Each selected model is refit on that entire outer TRAIN. Its own official TEST is excluded from fitting, preprocessing, calibration and selection.

Candidate axes and losses were fixed before new-task target-driven development. No fold-0 observation changed another fold's grid; no official TEST labels, predictions, errors, scores or other contributors' predictions selected a model, recipe, feature, blend or checkpoint. Every fit starts from the exact original TabPFN-v2 foundation, not a checkpoint selected using another completed task's TEST score. All five final models for a task froze before any of its held-out inference/scoring. All results are retained regardless of direction.

The exact six executed scientific files and original protocols are retained under `tasks/phonons`, `tasks/dielectric` and `tasks/expt_gap`. Four scientific files match across all tasks. Larger tasks change only the training time allowance 120 to 300 seconds per fit and schedule partitioning from three to one per job; their prescribed complete epochs, candidates, loss, learning rates and selection remain the same. A metadata limitation sentence also differs. These are preserved historical snapshots, not post-score source replacements. Incomplete epochs fail instead of shortening training to fit a limit.

## Actual training and verification scope

Each task completed 180 TabPFN and 90 MLP inner fits, followed by five final trained refits. Across these three tasks this is **810 inner gradient fits and 15 final refits**, plus separate-process inference. Phonons completed 31 bounded stages; each larger task completed 71. Final gradient steps are phonons 35/35/140/140/140; refractive index 290/290/290/290/580; band gap 280/140/140/280/280. Phonon and band-gap finals each changed 129 parameter tensors; the first four refractive-index finals trained the output head (four tensors), and its fifth changed 129 tensors. Training evidence, every inner recipe score and all selected recipes are included.

Every exact output passed size/SHA verification and owned GPUs were released before subsequent jobs. A fresh inference process replayed 16 TRAIN witnesses per final model (80 per task), then generated that task's entire held-out predictions. All TEST replay NPZs contain IDs and features only; TRAIN labels are retained solely as the required own-TRAIN TabPFN context.

An **additional independent A100 execution of the public phonon checker reproduced all 1265 predictions with maximum absolute difference 0**, under original atol1e-3/rtol1e-4. Additional full 4764/4604 GPU replays have **not** been performed for refractive index/band gap; their supplied full checkers and contexts passed CPU hash, ID, selection-link and submitted-prediction equality checks. These scopes are distinct. No extra training, scoring, checkpoint selection or tolerance change is part of replay.

Measured A100 occupation was 2.3544444444, 6.9775 and 7.4280555556 card-hours respectively. Container execution was 7801.4165, 23142.1426 and 24584.5540 seconds, including installation/inference/non-gradient work. These are not pure gradient times or end-to-end elapsed days.

## Provenance, limitations and operational interruptions

Steel and JDFT2D experiments informed this transferred method. The first JDFT2D design adaptively narrowed a grid using fold-0 development whose TRAIN contained other outer folds' TEST labels. That design-exposure risk and its old result 28.7333 were disclosed; the corrected per-TRAIN selection result 29.9916 replaced it irrespective of direction and was accepted in PR435. This history is **not erased** by the current runs and is not described as pristine blind. Third-party foundation pretraining may include related materials and is not claimed entirely unseen.

The public master started, but private host transport needed continuation. Phonons had pre-GPU inherited-launcher metadata and dataset-name length fixes. The larger runs encountered pre-GPU upload/create/prewarm and storage-lease failures, unknown submission responses, interrupted readonly catalog/download calls, a local CLI-access error and local credential-expiry guards. Already successful stages were SHA-adopted without retraining. Unknown submissions were first reconciled to the original job; only explicitly authorized, fully verified recovery used the exact original command/idempotency key at most once per intent. Diagnosed pre-GPU storage failures used the same job after exact input/prewarm verification, not a failed-compute search retry. Original failures, sources and immutable evidence are retained locally.

All scientific prepare/select/train/predict/native functions were the frozen public ones. The original public command is **not claimed uninterrupted**, and default standalone local-CUDA end-to-end full retraining has not been separately performed. Private course transport, account, runtime and machine logs are not public dependencies. The publication timing changed at the user's explicit request to submit these three completed tasks now; declared shear/bulk-modulus tasks remain in the research plan and will be retained/reported regardless of score. A shear-modulus candidate stopped before TEST because its prescribed 20 epochs were incomplete; its failure is retained and no automatic GPU retry was performed. Bulk-modulus training has not started. No task was removed from that declared batch because of its TEST score.

## Reproduce training

Use **Python 3.12 and NVIDIA A100** (actual A100-SXM4-80GB). Other GPU architectures have not been verified. Install requirements.txt with compatible CUDA PyTorch; allow at least roughly 12GiB for a new full run. Download each task's official Matminer JSON without reordering rows and the official Matbench validation splits. Task protocols pin dataset hashes, split SHA `94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d` and native revision `35f88110a51adf817a53515780d99fc34f614bf8`.

```bash
python reproduce.py --task phonons --data /data/matbench_phonons.json.gz --splits /data/matbench_v0.1_validation.json --protocol tasks/phonons/protocol-freeze.json --output /new/phonons-run
python reproduce.py --task dielectric --data /data/matbench_dielectric.json.gz --splits /data/matbench_v0.1_validation.json --protocol tasks/dielectric/protocol-freeze.json --output /new/dielectric-run
python reproduce.py --task expt_gap --data /data/matbench_expt_gap.json.gz --splits /data/matbench_v0.1_validation.json --protocol tasks/expt_gap/protocol-freeze.json --output /new/gap-run
python -m unittest discover -s tasks/expt_gap -p test_checkpoint_path.py
```

An optional `--checkpoint` accepts the exact original foundation; otherwise the pinned public revision is downloaded. Repository `Prior-Labs/TabPFN-v2-reg`, revision `02a44705ff7790b50cded404bc59adb94da456a1`, checkpoint SHA `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Standard Hugging Face snapshot-to-blob symlinks are staged under the correct filename; a CPU regression test checks this and wrong-SHA rejection. The default backend uses local CUDA Python subprocesses without a course account. An optional explicit stage executor replaces only job transport. Existing outputs/intents are refused; failures stop and there is no hidden untrained fallback.

## Exact public models and prediction replay

Release: https://github.com/donkeey11/matbench/releases/tag/unified-regression-three-tasks-v1

All **15 full final checkpoints with original optimizers**, three original pre-inference freezes, and TRAIN/target-free TEST contexts are provided. Per-task frozen-model-assets.json pins actual URLs, lengths, SHA and inference source hashes. All 26 public release assets passed complete anonymous HTTPS GET with exact original byte lengths and SHA256; see public-assets-verification.json. The original per-task asset manifests retain their pre-publication publication_status field for provenance; this independent verification report records the completed actual checks. No check is inferred only from an API digest. Freeze byte hashes and original task result bytes are unchanged. The root results.json.gz is a native metadata/container merge whose three task objects are exactly identical to the original results, not a new model evaluation.

```bash
python verify_full_predictions.py --task phonons --assets /new/phonons-assets --download --check-only
python verify_full_predictions.py --task phonons --assets /new/phonons-assets --download --output /new/phonons-replay.json
python verify_full_predictions.py --task dielectric --assets /new/dielectric-assets --download --output /new/dielectric-replay.json
python verify_full_predictions.py --task expt_gap --assets /new/gap-assets --download --output /new/gap-replay.json
```

Replay verifies only saved predictions with atol1e-3/rtol1e-4. It does not train, select, load TEST targets or calculate a benchmark score. Comparison failures must be reported, not used to change a model or recipe. This work retains the TabPFN name prefix, **Built with PriorLabs-TabPFN**, the supplied Apache2/extra attribution license and foundation citations.
