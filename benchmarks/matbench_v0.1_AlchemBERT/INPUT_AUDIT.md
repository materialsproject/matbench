# Independent composition-input identity audit

`verify_inputs.py` compares all 40 v1 input archives for `expt_gap`,
`expt_is_metal`, `glass`, and `steels` with independent official datasets and
the official ordered train/test fold IDs. Every archive and reference dataset
is checked by SHA256 before parsing. Full machine-readable results, source
identities and individual mismatches are in [input_audit.json](input_audit.json).

This extends the earlier internal partition-consistency check: the archived
formula or alloy description is now compared with the identified reference
material, and each separate `output` field is compared with its reference label.
No model is trained and no benchmark result is modified.

| Task | Unique materials | Row appearances across five folds | Input comparison | Target / instruction mismatches |
| --- | ---: | ---: | --- | --- |
| `expt_gap` | 4,604 | 23,020 | One formula typo difference, present once in each fold | 0 / 0 |
| `expt_is_metal` | 4,921 | 24,605 | The same formula typo difference, once in each fold | 0 / 0 |
| `glass` | 5,680 | 28,400 | Exact formula-prefix agreement | 0 / 0 |
| `steels` | 312 | 1,560 | All element sets agree; percentages agree within displayed rounding precision | 0 / 0 |

All 77,585 row appearances had the expected three fields, reference-aligned
targets and task-fixed instructions. No input matched the limited property-word
scan recorded in the JSON report.

## Explicit treatment of the experimental-dataset typo

The public official Figshare originals for
[expt_gap](https://api.figshare.com/v2/articles/9765779) and
[expt_is_metal](https://api.figshare.com/v2/articles/9765803) have their historical
Matminer checksums. Their bytes differ from the current corrected dataset
checksums. The script accepts these originals only with the explicit
`--allow-historical-expt` option and labels them accordingly.

At `mb-expt-gap-1394` and `mb-expt-is-metal-1501`, the historical reference
formula is `G1128Ga1As0.1P0.9`; the archived input says `Ga1As0.1P0.9`.
The reference labels are respectively 1.22 and false, and all corresponding
archive rows agree. These are the only discrepancies found; they are reported,
not silently normalized.

Matminer's [2020 correction commit](https://github.com/hackingmaterials/matminer/commit/edfcff311fea550c1d361300d40d5c5bdefba2e1)
specifically attributes the two changed checksums to the `1128G` typo. For
`expt_gap`, the separately published official `expt_gap_kingsbury` table supplies
additional evidence: after verifying its checksum, all 4,604 indices and labels
match the original, and the sole formula change is exactly the correction above.
Its added MP-ID column is not used. This does not assert that the current
compressed `matbench_expt_gap` bytes were recovered, or establish the current
`expt_is_metal` target-table identity.

## Repeating the check

With the pinned Gitee checkout and verified official dataset downloads:

```bash
python benchmarks/matbench_v0.1_AlchemBERT/verify_inputs.py \
  --source alchembert-source --dataset-dir official-matbench-data \
  --allow-historical-expt --output new_input_audit.json
```

The output must be a new file. Supplying the checksum-verified
`expt_gap_kingsbury.json.gz` in the same dataset directory enables the additional
derivative-table comparison. Without `--allow-historical-expt`, historical
experimental-dataset bytes are rejected.

This audit checks formula prefixes and rounded alloy quantities, not every
natural-language clause or any structure-task neighborhood description.
Agreement narrows the sample-identity uncertainty for these archived files;
it does not establish that the files were used in the original experiments,
exclude semantic proxies, or prove absence of pretraining-corpus overlap.
