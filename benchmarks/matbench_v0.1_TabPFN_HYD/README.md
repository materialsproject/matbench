# TabPFN-HYD — composition-only steel yield-strength regression

**Built with PriorLabs-TabPFN.** The unmodified TabPFN-v2 regression checkpoint
is combined with a training-only-selected conventional regression baseline.
This submission covers **only `matbench_steels`**, not the 13-task general-purpose
Matbench leaderboard.

The native Matbench five-fold result is **MAE 76.59472316987244 MPa**, with
fold standard deviation **12.406993875998648 MPa**. All 312 official test-fold
predictions are present in `results.json.gz`. This is the submitted experiment's
score; public acceptance and ranking are determined by the maintainers.

## Model and selection protocol

1. Compute per-composition features without targets: 118 elemental atomic
   fractions, 142 descriptors from `Stoichiometry`, Magpie `ElementProperty`, and
   fractional `ValenceOrbital`, and their 260-feature concatenation. Missing
   descriptors remain NaN. No rows are discarded.
2. Within each official outer training fold, compare the same 42 conventional
   configurations by three-fold inner MAE (`KFold(shuffle=True, random_state=1701)`).
   The grid in `model_candidates.py` includes ExtraTrees, RandomForest, CatBoost,
   histogram gradient boosting and kernel ridge. Imputation and scaling, where
   needed, are fitted separately on the inner training split. Refit that fold's
   winner on its complete outer training split.
3. Independently within **each** official outer training fold, compare that
   baseline with all 48 preregistered TabPFN/mixture configurations: feature set
   `{fractions, magpie, combined}`, `n_estimators={4,8}`, prediction summary
   `{mean,median}`, and TabPFN mixture weight `{0.25,0.5,0.75,1}`. Baseline-only
   is an additional option. Recompute baseline inner out-of-fold predictions
   using **fresh clones**, never the model fitted on all outer training rows.
   Choose by mean three-fold inner MAE, with deterministic JSON-order tie breaks.
4. Fit the selected TabPFN training context on the outer training split, retain
   the corresponding already-fitted baseline component, and save the selection,
   component hashes and fitted TabPFN context. Foundation weights receive
   **zero gradient updates**. TabPFN uses training examples as context.
5. Freeze **all five** final selections before final test inference. Record the
   predictions once with the native `MatbenchTask.record` scoring code. No outer
   test scores, labels, public prediction files or cross-fold pooled scores are
   used to select settings. Each outer fold reads only its own training-label
   file; reuse of samples across different folds follows the official CV design.

The initial development pilot used only official fold-0 training rows: 199 for
development CV and 50 for a one-time baseline holdout. The TabPFN family probe
used the original 199 development rows; that 50-row holdout was not reused.
Final settings above were selected independently inside every outer training
fold, rather than transferring the pilot's best setting globally. First-stage
baseline test predictions were generated but were never inspected or scored
to select the final system. The first official outer-test score was computed
after every final model and mixture had been frozen.

## Reproduction

The actual experiments used Python 3.12.3, PyTorch 2.13.0+cu126, one NVIDIA
A100-SXM4-80GB, TabPFN 9.2.0, scikit-learn 1.9.1 and NumPy 2.5.4. The public
runners retain the original A100 hardware check. The conventional first stage
uses CPU estimators; TabPFN inference uses CUDA. The complete TabPFN nested
stage took about 107 seconds including environment installation, excluding
queueing and platform export. This is not an end-to-end timing claim.

Install the dependencies in `requirements.txt`, then install the pinned
Matbench source into that environment:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps git+https://github.com/materialsproject/matbench.git@35f88110a51adf817a53515780d99fc34f614bf8
python reproduce.py --workdir ./replay
```

Alternatively pass the original checkpoint explicitly:

```bash
python reproduce.py --workdir ./replay --weights /path/to/tabpfn-v2-regressor-v2_default.ckpt
```

`reproduce.py` downloads the original small Matbench dataset, verifies its hash,
prepares per-fold training-label files, runs both frozen search stages and then
records the final predictions with the native API. A fresh work directory is
required. `--prepare-only` prepares and verifies features without fitting or
scoring. `notebook.ipynb` provides a thin entry point to the same source.

Foundation model: `Prior-Labs/TabPFN-v2-reg`, revision
`02a44705ff7790b50cded404bc59adb94da456a1`, file
`tabpfn-v2-regressor-v2_default.ckpt`, SHA256
`2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.
The default public v2 checkpoint is used; no gated/newer model or API service
is required. The authors describe TabPFN-v2 pretraining on synthetic data; we
performed no Matbench-specific foundation pretraining or gradient fine-tuning.

Raw dataset SHA256:
`473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252`.
The native scorer and official fold metadata are pinned to Matbench commit
`35f88110a51adf817a53515780d99fc34f614bf8`; source hashes are recorded in
`source-lock.json`. `selection-freezes.json` contains the actual experiment's
five selections and component checksums. Foundation weights and fitted binary
models are not added to this repository; the source, public checkpoint and
official data are sufficient to refit them.

## Validation evidence

The actual saved fitted contexts were reloaded in a fresh A100 process. All
312 predictions matched within a maximum absolute difference of
`4.547473508864641e-13` MPa; see `frozen-model-replay.json`. No true targets
were used in replay. The public preparation entry point also recreated every
feature array exactly (including NaNs) and all five training-label files
byte-for-byte. The public conventional and TabPFN training runners are byte
identical to the ones actually executed on the course worker. The notebook
is an unexecuted entry point; no notebook outputs are fabricated.

## Attribution

The TabPFN-v2 weights use the Prior Labs License (Apache 2.0 plus additional
attribution). See `LICENSE-TabPFN.txt`. The public derivative name begins with
TabPFN as required. Citations for Matbench, TabPFN, ExtraTrees and the feature
libraries are supplied in `info.json`.
