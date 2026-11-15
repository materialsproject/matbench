# Prospective CPU runtime maintenance

`requirements-rerun-cpu.txt` is the installation entry point for **new**
experiments with `train_fold.py`. Its dependencies are maintained separately
from the archived training source. Updating this environment does not rerun
the published experiments or alter either saved results file.

## November 2026 dependency update

The previous CPU pins included PyTorch 2.9.1 and Transformers 4.57.6. This
update responds to the following published advisories; it is not a claim that
the environment is free of every vulnerability.

| Advisory | Published affected and patched versions | Relevance to this runner |
| --- | --- | --- |
| [CVE-2026-24747 / GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) | PyTorch through 2.9.1; patched in 2.10.0 | A malicious checkpoint can cause memory corruption even with `torch.load(weights_only=True)`. The runner uses this API for checkpoint loading and replay. |
| [CVE-2026-9856 / GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | Transformers before 5.10.0; patched in 5.10.0 | Named chat templates from a tokenizer configuration can traverse outside the destination when `save_pretrained` writes them. The runner loads a chosen tokenizer and saves it locally. |
| [CVE-2025-3000 / GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) | PyTorch through 2.12.1; patched in 2.13.0 | This concerns `torch.jit.script`, which this runner does not call. Choosing 2.13.0 also moves beyond this affected package range. |

The Transformers [upstream fix](https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5)
validates chat-template paths before writing them. Pinning an immutable model
revision and disabling remote Python code do not themselves fix a data-file
path traversal vulnerability.

The updated direct pins are:

| Package | Previous pin | Updated pin |
| --- | --- | --- |
| `torch` | `2.9.1+cpu` | `2.13.0+cpu` |
| `transformers` | `4.57.6` | `5.10.1` |
| `huggingface-hub` | `0.36.2` | `1.5.0` |

The hub change meets the `>=1.5.0,<2.0` requirement declared in
[Transformers 5.10.1 package metadata](https://pypi.org/pypi/transformers/5.10.1/json).
The existing `tokenizers==0.22.2` pin satisfies its `>=0.22.0,<=0.23.0`
requirement and is retained. Other direct pins are unchanged. This environment
does not install or use Lightning.

## Verification scope

On 2026-11-13, a fresh Python 3.12.7 environment installed this requirements
file together with editable Matbench 0.6 (core source from upstream commit
`ded3a3f261e880705ebcdcb66dc3385202a39b9a`). `pip check` reported no broken
requirements. All three tests in `test_train_smoke.py` passed with zero skips,
using PyTorch 2.13.0+cpu, Transformers 5.10.1, huggingface-hub 1.5.0 and
tokenizers 0.22.2.

To repeat the CPU checks after installing the environment in [RUNNING.md](RUNNING.md):

```bash
python -m pip check
python -c "import torch, transformers; from transformers import BertModel, BertTokenizerFast"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover \
  -s benchmarks/matbench_v0.1_AlchemBERT -p 'test_train_smoke.py' -v
```

The dedicated `AlchemBERT CPU runtime` workflow installs this environment with
Python 3.12 on changes to the benchmark, Matbench package, setup configuration
or workflow. It checks imports before running the three tests so missing or
broken training dependencies fail the job rather than silently skipping them.
The Hugging Face offline flags are enabled for the test step.

The CPU smoke suite uses synthetic data and a tiny randomly initialized BERT.
It exercises classification and transformed regression training, verifies
changes to encoder weights, and compares saved-checkpoint replay predictions
with the initial prediction file. It also verifies that a checkpoint remains
usable when prediction fails after training. The tokenizer fixture explicitly
checks that `Fe O` produces meaningful vocabulary tokens rather than only
unknown tokens under Transformers 5.

These checks do not cover remote pretrained-model downloads, full-size BERT
training, CUDA, the full 13-task experiment, or reproduction of historical
scores. Passing functional tests is not a security audit of every dependency.

Continue to select trusted model revisions and run artifacts. The runner keeps
`trust_remote_code=False`, immutable revision requirements, local artifact
hashes, and `weights_only=True` checkpoint loads. A hash recorded inside an
imported run establishes consistency with that run's manifest, not independent
trust in its author or content.
