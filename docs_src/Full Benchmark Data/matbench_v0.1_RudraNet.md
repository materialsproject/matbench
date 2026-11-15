# matbench_v0.1: RudraNet

### Algorithm description: 

RudraNet is a novel hybrid recursive architecture (Hybrid-Attention-MLP-Recursive-Model) designed specifically for low-data materials property prediction. It features an Attention-based Feature Extractor (AFE) that utilizes multi-head self-attention on 22 Magpie property tokens, each containing six statistics, followed by a cross-attention gating mechanism with Mat2Vec embeddings. The core reasoning engine is a recursive MLP-TRM loop with shared weights that performs iterative state refinement over 20 discrete steps. The model is trained using a linear-weighted Deep Supervision loss and Stochastic Weight Averaging (SWA) to ensure generalizability on small tabular datasets.

#### Notes:
TRIADS/RudraNet V13A five-seed ensemble on matbench_steels: mean five-fold MAE 91.2028 MPa. Training was performed on Kaggle using one NVIDIA T4, reported under 20 minutes. Matminer loaded the dataset when Matbench was unavailable in Kaggle. The script uses five outer KFold partitions (shuffle=True, random_state=18012019), with inner validation holdouts; all 312 submitted sample IDs match the official partitioning. In the pinned inference environment, the additional Matminer descriptor calculation falls back to a 200-dimensional zero block for all 312 samples; the informative features are Magpie and Mat2Vec. Mat2Vec pretrained inputs are SHA-256 verified against the captured upstream assets, not a recovered historical Kaggle cache. See REPRODUCIBILITY.md for the archived source, numerical results, Matbench export validation and software details.

Raw data download and example notebook available [on the matbench repo](https://github.com/hackingmaterials/matbench/tree/main/benchmarks/matbench_v0.1_RudraNet).

### References (in bibtex format): 

```
('@misc{tiwari2026triads,\n'
 ' author = {Rudra Tiwari},\n'
 ' title = {TRIADS: Tiny Recursive Information-Attention with Deep '
 'Supervision},\n'
 ' year = {2026},\n'
 ' doi = {10.5281/zenodo.19200579},\n'
 ' url = {https://doi.org/10.5281/zenodo.19200579},\n'
 ' note = {Research archive: https://github.com/Rtx09x/TRIADS}\n'
 '}\n'
 '@article{tshitoyan2019unsupervised,\n'
 ' title = {Unsupervised word embeddings capture latent knowledge from '
 'materials science literature},\n'
 ' author = {Tshitoyan, Vahe and others},\n'
 ' journal = {Nature},\n'
 ' year = {2019},\n'
 ' doi = {10.1038/s41586-019-1335-8}\n'
 '}')
```

### User metadata:

```
{}
```

### Metadata:

| tasks recorded | 1/13 |
|----------------|-------------------------------------|
| complete? | ✗ | 
| composition complete? | ✗ | 
| structure complete? | ✗ | 
| regression complete? | ✗ | 
| classification complete? | ✗ | 

### Software Requirements

```
{'benchmark_export': 'Matbench source 6374ebf52d636d4f059daec170dc6f8ac7ca3a3a '
                     'on Python 3.12 (separate environment; 2026-11-15 '
                     'integration audit)',
 'inference': {'packages': ['torch==2.3.1',
                            'numpy==1.26.4',
                            'pandas==2.2.2',
                            'scipy==1.13.1',
                            'scikit-learn==1.5.1',
                            'pymatgen==2024.7.18',
                            'matminer==0.9.3',
                            'gensim==4.3.3',
                            'matplotlib==3.9.4',
                            'tqdm==4.66.4',
                            'huggingface_hub==0.24.7'],
               'python': '3.11'}}
```

### Task data:

#### `matbench_steels`

###### Fold scores

| fold | mae | rmse | mape* | max_error |
|------|------|------|------|------|
| fold_0 | 114.3248 | 197.3315 | 0.0727 | 1127.7518 |
| fold_1 | 81.4639 | 113.4375 | 0.0576 | 354.2149 |
| fold_2 | 80.5513 | 130.8272 | 0.0558 | 482.6379 |
| fold_3 | 90.4915 | 127.5304 | 0.0627 | 453.0268 |
| fold_4 | 89.1827 | 135.3460 | 0.0673 | 527.2985 |

###### Fold score stats

| metric | mean | max | min | std |
|--------|------|-----|-----|-----|
| mae | 91.2028 | 114.3248 | 80.5513 | 12.2271 |
| rmse | 140.8945 | 197.3315 | 113.4375 | 29.1539 |
| mape* | 0.0632 | 0.0727 | 0.0558 | 0.0062 |
| max_error | 588.9860 | 1127.7518 | 354.2149 | 275.3130 |

###### Fold parameters

| fold | params dict|
|------|------------|
| fold_0 | `{}` |
| fold_1 | `{}` |
| fold_2 | `{}` |
| fold_3 | `{}` |
| fold_4 | `{}` |




