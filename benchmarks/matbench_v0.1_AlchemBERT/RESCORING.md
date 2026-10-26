# Independent check of the archived predictions

The submitted `results.json.gz` contains predictions keyed by official Matbench
sample IDs. Consequently, their metrics can be recomputed without obtaining the
original model checkpoints or CSV exports. Those materials are still needed to
establish how the predictions were generated.

`rescore_results.py` has now independently checked **12 tasks / 60 folds** using
verified, public reference labels. It keeps the original artifact unchanged.
The detailed report is [rescoring.json](rescoring.json), including every fold's
original score, recomputed score, difference, reference-data hashes and software
versions. This is a scoring check, not a new experiment or a claim that model
training has been reproduced.

## Results

The table shows the unweighted mean of the five official fold scores. MAE is
reported in each task's original target units.

| Task | Metric | Stored mean | Recomputed mean |
| --- | --- | ---: | ---: |
| matbench_dielectric | MAE | 0.334517302 | 0.334517302 |
| matbench_expt_gap | MAE | 0.374580207 | 0.374580207 |
| matbench_glass | ROC-AUC | 0.770211636 | **0.883284181** |
| matbench_jdft2d | MAE | 45.491824712 | 45.491824712 |
| matbench_log_gvrh | MAE | 0.099820535 | 0.099820535 |
| matbench_log_kvrh | MAE | 0.076519978 | 0.076519978 |
| matbench_mp_e_form | MAE | 0.061889702 | 0.061889702 |
| matbench_mp_gap | MAE | 0.254341043 | 0.254341043 |
| matbench_mp_is_metal | ROC-AUC | 0.886022805 | **0.955124203** |
| matbench_perovskites | MAE | 0.043188111 | 0.043188111 |
| matbench_phonons | MAE | 44.987414162 | 44.987414162 |
| matbench_steels | MAE | 132.912130507 | 132.912130507 |

For `glass` and `mp_is_metal`, ROC-AUC is recomputed from the **unrounded stored
probabilities**. The remaining classification metrics and all checked regression
metrics agree with the stored values to floating-point precision. The original
classification ROC-AUC values are not the appropriate probability-based scores.
The canonical benchmark artifact has not been replaced while the remaining task
and the presentation of historical corrections are unresolved.

## Reference-label provenance

Eleven original public Figshare downloads have SHA256 hashes exactly matching
the current Matminer dataset metadata. The report records both the standard
Materials Project download URL and the verified public Figshare mirror.

For `expt_gap`, the old Figshare benchmark archive predates a composition typo
correction and does not have the current benchmark hash. Instead, this check
uses `expt_gap_kingsbury.json.gz`, SHA256
`b36bf5bfe3782c2a2372422c64b8a86fd1d30709559a6dc0c30e4c8ac5ab3e96`.
The [pinned official Matminer metadata](https://github.com/hackingmaterials/matminer/blob/be0f096b6b4f59374a9ae00b8fd84321fc3dccbc/src/matminer/datasets/dataset_metadata.json)
describes it as the identical dataset with likely Materials Project IDs added
and columns renamed. We use its `expt_gap` column and unchanged integer indices.
This is reported explicitly as a verified official derivative, **not** as a
byte-identical copy of the benchmark archive. Independent comparison finds the
same 4,604 index/target pairs as the older official archive; the only formula
change is the documented `G1128Ga1As0.1P0.9` correction at zero-based index 1393.

The following downloads reproduce the reference files used here. Save them under
the filenames in the first column; their SHA256 hashes are checked by the script.

| Filename | Public download |
| --- | --- |
| matbench_dielectric.json.gz | [Figshare 17476067](https://ndownloader.figshare.com/files/17476067) |
| expt_gap_kingsbury.json.gz | [Figshare 28232598](https://ndownloader.figshare.com/files/28232598) |
| matbench_glass.json.gz | [Figshare 17494637](https://ndownloader.figshare.com/files/17494637) |
| matbench_jdft2d.json.gz | [Figshare 17476046](https://ndownloader.figshare.com/files/17476046) |
| matbench_log_gvrh.json.gz | [Figshare 17476064](https://ndownloader.figshare.com/files/17476064) |
| matbench_log_kvrh.json.gz | [Figshare 17476061](https://ndownloader.figshare.com/files/17476061) |
| matbench_mp_e_form.json.gz | [Figshare 17476028](https://ndownloader.figshare.com/files/17476028) |
| matbench_mp_gap.json.gz | [Figshare 17084741](https://ndownloader.figshare.com/files/17084741) |
| matbench_mp_is_metal.json.gz | [Figshare 17476034](https://ndownloader.figshare.com/files/17476034) |
| matbench_perovskites.json.gz | [Figshare 17494805](https://ndownloader.figshare.com/files/17494805) |
| matbench_phonons.json.gz | [Figshare 17476037](https://ndownloader.figshare.com/files/17476037) |
| matbench_steels.json.gz | [Figshare 17476040](https://ndownloader.figshare.com/files/17476040) |

## Reproduce the check

Install this Matbench checkout and the additional streaming parser dependency
separately from a training environment. Run from the repository root:

```bash
python -m pip install -e .
python -m pip install -r benchmarks/matbench_v0.1_AlchemBERT/rescore_requirements.txt
python benchmarks/matbench_v0.1_AlchemBERT/rescore_results.py \
  --dataset-dir /path/to/verified-downloads \
  --tasks matbench_dielectric matbench_expt_gap matbench_glass \
    matbench_jdft2d matbench_log_gvrh matbench_log_kvrh matbench_mp_e_form \
    matbench_mp_gap matbench_mp_is_metal matbench_perovskites \
    matbench_phonons matbench_steels \
  --output fresh_rescoring.json
python -m unittest discover -s benchmarks/matbench_v0.1_AlchemBERT \
  -p 'test_rescore_results.py'
```

The script locks the original result hash, the official fold/metadata hashes,
and the corrected scoring implementation from upstream commit
`4b7d0714849a917814115be380a64e7ace94be55`. It checks exact fold ID sets and
looks up predictions and targets by ID. It rejects incorrect dataset hashes,
invalid labels/probabilities and existing output files. Large structure datasets
are streamed one row at a time. Seven focused tests cover probability scoring,
ID alignment, untrusted data, scorer identity and overwrite protection.

Without `--dataset-dir`, the script offers an entirely offline cross-check of
`glass`, `dielectric` and `steels` using hash-locked ground-truth fixtures already
in `matbench/tests/`. The upstream `test_msonability` code constructs these
fixtures by recording `test_outputs` as predictions. Their results agree with
the corresponding raw-dataset checks above. This mode requires no dataset
downloads and no `ijson` dependency.

## The remaining task

`matbench_expt_is_metal` has **not** been independently rescored against the
current reference dataset. The standard download returned HTTP 403 in this
environment. The original public Figshare file matches the old official hash,
but predates the same composition typo correction described in
[Matminer commit edfcff3](https://github.com/hackingmaterials/matminer/commit/edfcff311fea550c1d361300d40d5c5bdefba2e1).
The old file is not silently accepted as the current reference.

The specific remaining public-data input is:

- Filename: `matbench_expt_is_metal.json.gz`
- [Standard download](https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz)
- Required SHA256: `8f2a4f9bacdcbc5c2c73615629ee7986f09d39bed40ba7db52b61b2889730887`

Once that file is accessible, omitting `--tasks` in dataset mode checks all
13 tasks. A genuine existing Matminer cache with this exact hash is sufficient;
no original AlchemBERT model checkpoint is needed for the scoring calculation.

For model reproducibility, the original checkpoints, run records, and evidence
linking each input and prediction to the actual historical experiment remain
necessary. These score checks alone neither prove nor disprove data leakage.
