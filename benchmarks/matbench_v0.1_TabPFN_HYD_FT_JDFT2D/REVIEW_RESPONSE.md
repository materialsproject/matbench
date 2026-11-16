# Response to the four protocol/reproducibility questions

This draft preserves the initial result while a frozen correction runs. It is not ready for scientific acceptance. A correction cannot make previously observed labels or scores unseen; no pristine blind-design claim is made.

## 1. Development trials and fixed/adapted choices

V1 had 18 configurations: four ExtraTrees diagnostics, eight frozen TabPFN feature/inference alternatives, and six random-initialized MLPs genuinely trained for 600 steps each in each of three composition-group folds. The seed 3042, analytic 370 descriptors, 260-composition subset and these alternatives were frozen before the first pilot result. V2 introduced two full-network fine-tuning schedules (lr5e-6/2e-5,20 epochs). MAE1+CRPS.25 loss and context128/query.25 were fixed before the first fine-tuning result, not swept. Positive baseline blends .25/.5/.75/1 were inspected on development OOF predictions. V3 introduced four shorter/full/readout schedules after V2 worsened development error. The narrowed V3 grid and particular feature/inference choice were consequently influenced by fold0 development. All configurations, freeze times and development metrics are in development-history.json.

Fold0 TRAIN508 is disjoint from fold0 TEST128 but contains 127 TEST rows of each of folds1–4. Separate own-fold fitting does not eliminate this design-level selection risk. We accept the concern.

## 2. Previous official results

The first complete five-fold native score found in the local indexes, frozen input/output manifests and source entrypoints was 28.733328891325755 meV/atom at 2026-11-16T14:17:07.753744Z, the submitted initial result. No earlier complete official JDFT2D evaluation was found in that evidence scope. V1/V2/V3 scores were development group-CV scores, not complete official scores. This is a scoped audit, not a claim about inaccessible external activity. The initial result SHA and original public commit are retained.

## 3. Corrected per-TRAIN selection and residual limitation

Before any corrected GPU stage, corrected-protocol-freeze.json fixed the deterministic closure of the historical alternatives: both feature sets; all six V2+V3 schedules; four/eight estimators and mean/median; all eight original frozen baselines for positive .25/.5/.75 blends; pure trained models; and all six original trained MLPs. This gives 1206 alternatives per outer TRAIN. Reduced-composition GroupKFold(3) selects minimum mean validation MAE, with a fixed-index tie break. Pure untrained baselines and tree diagnostics are not eligible standalone because genuine gradient training was a user requirement before V1. Every outer fit sees only its own TRAIN labels, with fitted preprocessing restricted to that context. No old or new official score, test prediction or test label is supplied to candidate workers or selection.

All five selected final models must be frozen before corrected outer-test inference, then a different process replays TRAIN witnesses and generates 636 actual predictions for one native scoring pass. The corrected result will replace the old result regardless of metric direction. Historical exposure and retrospective protocol construction remain disclosed; rerunning does not erase them. Scientific acceptance is the maintainers' decision.

## 4. Missing historical scripts and actual public reproduction

The historical selection-protocol.json lists original source hashes for provenance, not a list of portable execution dependencies. matbench_jdft2d_probe.py was exploratory development; prepare_jdft2d_nested.py was course packaging; score_frozen_jdft2d.py wrapped native scoring; run_jdft2d_nested_queue.py was private course transport. The original public prepare/fit/predict equivalence checks did not constitute an end-to-end public rerun.

The new complete public implementation includes jdft2d_geometry_features.py, matbench_jdft2d_training.py, jdft2d_review_candidates.py, jdft2d_review_workflow.py, jdft2d_review_stage.py and jdft2d_review_reproduce.py. It reconstructs features, executes all inner fits, selects/refits/freezes all models, launches fresh inference and records native results. These exact SHA-pinned public scientific files and master are now executing across explicit bounded single-A100 stage transport. The default local-CUDA transport is also included; no separate complete run of that backend is claimed. Private transport contains no scientific alternative and is not required for local reproduction. Failures stop, without automatic GPU retry or hidden model fallback.

Candidate checkpoints retain learned weights and every parameter hash. To bound storage, optimizer tensors are omitted only after full-checkpoint replay, with every optimizer tensor hash and both original/retained checkpoint SHA preserved; this is explicitly not a resumable optimizer checkpoint. Final refits retain their full optimizer state. A further prediction check follows stripping.

## Compute accounting

The original nine development/final/inference jobs total 0.5805555556 A100 card-hours in the platform occupation ledger (queue/archive excluded by the ledger), including installation and other container work. This is not pure gradient time. Corrected per-fold stage compute and proof will be recorded from actual completed execution; no zero or completed total is invented while the run is pending.
