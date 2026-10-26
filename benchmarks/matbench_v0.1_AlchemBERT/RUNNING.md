# Running a new corrected experiment

These commands implement a **new experiment**. They do not reproduce or replace
the historical `results.json.gz`. The original source is preserved in
`upstream/`; the historical checkpoints, original run settings and v1 generator
are not supplied by these tools.

## Environment

From the Matbench repository root, use a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv-alchembert
source .venv-alchembert/bin/activate
pip install -e . -r benchmarks/matbench_v0.1_AlchemBERT/requirements-rerun-cpu.txt
pip check
```

The listed versions were tested together with upstream Matbench commit
`4b7d0714849a917814115be380a64e7ace94be55`, Python 3.12.14 and CPU PyTorch.
They are prospective tool dependencies, not a recovered historical lockfile.
The runtime manifest records the installed versions. For CUDA, select an
appropriate official PyTorch build and record that different environment;
CUDA execution was not tested in this review.

## Prepare identified, target-free inputs

For each task and outer fold, generate a new directory:

```bash
python benchmarks/matbench_v0.1_AlchemBERT/prepare_inputs.py \
  --task matbench_steels --fold 0 --output-dir new-inputs/steels-0
```

This obtains official Matbench inputs and retains their sample IDs. Test data
are requested with `include_target=False`. The generated JSONL records contain
only `sample_id`, `instruction` and `input`, never target labels. The manifest
records exact file hashes, task/fold and template provenance.

The template version `archived-v2-deterministic-1` derives from the recovered
v2 source. It sorts descriptions, uses actual periodic-image neighbor distances,
handles zero explicitly and propagates neighbor-generation failures. These
repairs can change text and truncation. **Neither v1 identity nor equivalence
to historical predictions is claimed.** The original formula-description
heuristics remain visible in the source; their descriptions are not independently
established chemical interpretations.

## Supply a real immutable model revision

Choose the model and tokenizer revisions for the new experiment. The observed
original scheduler names `bert-base-cased`, but its historical commit hashes
are unknown. A branch name such as `main` is rejected. Do not fill in a guessed
hash or describe a newly selected revision as the historical one.

Create `run.json` next to `new-inputs/steels-0/manifest.json` with this Python
snippet after setting `BERT_REVISION` to the actual 40-character commit of the
chosen Hugging Face model. A separate tokenizer revision may be used if needed.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

directory = Path('new-inputs/steels-0')
manifest = json.loads((directory / 'manifest.json').read_text())
keys = ('task', 'fold', 'train_file', 'test_file',
        'train_sha256', 'test_sha256', 'input_provenance')
config = {key: manifest[key] for key in keys}
config.update(
    model_name='google-bert/bert-base-cased',
    model_revision=os.environ['BERT_REVISION'],
    tokenizer_revision=os.environ['BERT_REVISION'],
    output_dir='../../new-runs/steels-0',
    device='cpu',
    seed=42,
    validation_fraction=0.1,
    max_length=512,
    batch_size=32,
    test_batch_size=1,
    learning_rate=1e-5,
    max_epochs=5000,
    patience=200,
    fine_tune_encoder=True,
)
with (directory / 'run.json').open('x') as stream:
    json.dump(config, stream, indent=2)
PY
python benchmarks/matbench_v0.1_AlchemBERT/train_fold.py train \
  --config new-inputs/steels-0/run.json
```

The numeric settings above are observable in the archived scheduler/trainer.
The repaired runner adds Python/NumPy seeding and deterministic operations and
uses a NumPy inner split, so its split/order is a new implementation choice.
It splits before fitting target transformations and fits only on inner-training
labels. Regression training uses MSE and validation uses MAE (in transformed
space when applicable); classification uses BCE. Each outer fold initializes
a fresh model and selects its best epoch using only inner validation loss.

Training saves the config, split IDs, best model, tokenizer, transform parameters,
logs, runtime details and checksums. It then writes unthresholded probabilities
or regression values with official IDs to `predictions.csv`. Existing run
directories are rejected. Full BERT training can be expensive; no full real-data
training run was performed as part of this update.

## Replay and record

Reload the locally saved best checkpoint and tokenizer without fetching model
weights again:

```bash
python benchmarks/matbench_v0.1_AlchemBERT/train_fold.py predict \
  --run-dir new-runs/steels-0 --output replay-steels-0.csv --device cpu
```

Repeat preparation/training for folds 0–4. Combine their CSV rows under one
header, and combine each run's `config.json` and `run.json` into a parameters
object keyed by task then fold string. For example:

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

params = {'matbench_steels': {}}
with Path('steels-predictions.csv').open('x', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=['task', 'fold', 'sample_id', 'prediction'])
    writer.writeheader()
    for fold in range(5):
        run = Path(f'new-runs/steels-{fold}')
        with (run / 'predictions.csv').open(newline='') as source:
            writer.writerows(csv.DictReader(source))
        params['matbench_steels'][str(fold)] = {
            'config': json.loads((run / 'config.json').read_text()),
            'run': json.loads((run / 'run.json').read_text()),
            'artifacts': json.loads((run / 'artifacts.json').read_text()),
        }
with Path('steels-parameters.json').open('x') as out:
    json.dump(params, out, indent=2)
PY
python benchmarks/matbench_v0.1_AlchemBERT/record_predictions.py \
  steels-predictions.csv --tasks matbench_steels \
  --params-json steels-parameters.json --output new-steels-results.json.gz
```

All selected folds and exact official sample-ID sets are required. Recording
passes probabilities directly to `task.record(...)` and leaves the historical
result untouched. A full submission requires all 13 tasks, each with five folds.

## Tests and their limits

```bash
python -m unittest discover -s benchmarks/matbench_v0.1_AlchemBERT -p 'test_*.py'
```

The optional CPU integration tests use a tiny randomly initialized BERT and
synthetic data. They execute training, verify actual weight changes, reload the
checkpoint and compare prediction files. They also test recovery after an
inference failure. They require no network or pretrained weights and skip when
PyTorch/Transformers are absent. Passing them verifies software behavior, not
historical scores or full-size training performance. The upstream CI workflows
are unchanged and do not automatically discover these benchmark-local tests;
their results are reported separately.
