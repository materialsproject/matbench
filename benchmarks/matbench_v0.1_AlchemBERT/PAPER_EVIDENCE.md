# Publication evidence and remaining provenance gaps

The original citation in `info.json` is retained. Published descriptions supplement
the recovered code; they do not identify the exact run that produced
`results.json.gz`.

## Original preprint

The [original ChemRxiv PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/67540a28085116a133a62b85/original/AlchemBERT__Exploring_Lightweight_Language_Models_for_Materials_Informatics-6.pdf)
(DOI: `10.26434/chemrxiv-2024-r4dnl`, 13 pages) reports:

| Evidence | Location | Paper-reported detail |
| --- | --- | --- |
| Hardware | Methods, page 8 | Nvidia RTX 4090 |
| Model | Methods, page 8 | BERT-base-cased; first-token representation and a 768-to-1 linear head |
| Training | Methods, page 8 | 9:1 train/validation split; learning rate 1e-5; batch size 32; early-stopping patience 200 epochs |
| Text examples | Pages 9–11 | Alloy fractions for steels; formula-based descriptions for other composition tasks; lattice, symmetry and neighbors for structure tasks; fixed task/unit instruction |
| Pretraining control | Figure 1, page 5 | Predictions without fine-tuning compared with a dummy model |

**Interpretation:** the control does not establish that the pretraining corpus has
no overlap with benchmark data. Hardware is paper-reported, not verified from
historical run logs. Runtime, checkpoint hashes, exact input identity and the
hyperparameter-selection history still require original evidence. The paper's
template examples cannot recover the missing v1 generator or establish that
particular input archives were used.

## Later publication and public model assets

The [2025 journal article](https://doi.org/10.1016/j.xcrp.2025.102724) links the
Gitee repository for Matbench property tasks, and separate resources for
Matbench-Discovery. The following public metadata were checked:

| Resource | Verified scope |
| --- | --- |
| [Figshare 28690583](https://api.figshare.com/v2/articles/28690583), version 1 when checked | One checkpoint explicitly described as Matbench-Discovery |
| [Figshare 28387640](https://api.figshare.com/v2/articles/28387640), version 1 when checked | WBM IS2RE predictions for Matbench-Discovery |
| [Zenodo 15985169](https://zenodo.org/records/15985169) | Archive of `WangYuHang-WYH/AlchemBERT` v1.0.0; description specifies Discovery IS2RE |

These assets are evidence for a different benchmark. They do not supply the
65 task/fold checkpoints required to establish this submission's historical
training provenance. No Discovery checkpoint or prediction file has been
substituted for a Matbench property artifact.
