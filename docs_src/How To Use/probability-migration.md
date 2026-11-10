# Migrating classification submissions to probability ROC AUC

Current classification leaderboards rank the mean ROC AUC computed from the
saved positive-class probabilities in the five official test folds. The same
recomputed scores appear in task tables, homepage winners, scaled-error plots,
and submission detail pages. Original submission files remain unchanged.

## Which historical submissions can be rescored?

| Saved predictions | Current ranking | Required action for a new submission |
|---|---|---|
| Floating-point probabilities in `[0, 1]`, finite in every fold | Eligible; the website recomputes scores from official test targets | Recompute cached classification metrics before submitting the result file |
| Boolean or integer class labels | Excluded | Recover probabilities from the original fold models or rerun the benchmark |
| Mixed label/probability types, missing predictions, or invalid probabilities | Excluded | Correct the predictions and validate every fold |

The old threshold-based results are retained in the **Historical balanced
accuracy leaderboard** section on each classification task page. A model can
still appear on regression leaderboards when its classification result is
ineligible.

Converting `True`/`False` to `1.0`/`0.0` cannot recover probabilities or their
ranking information. Do not use a type cast as a substitute for obtaining model
probabilities. Genuine endpoint probabilities are allowed; the file validator
cannot determine whether a floating-point number was previously a hard label.

## Record the probability of the correct class

Use the probability of the target being `True`: `is_metal=True` for metallicity
and `gfa=True` for glass-forming ability. Establish class polarity from the
model's class mapping, not by comparing either orientation against test scores.
For a scikit-learn compatible classifier trained on the Boolean targets:

```python
import numpy as np

positive_columns = np.flatnonzero(np.asarray(model.classes_) == True)
if len(positive_columns) != 1:
    raise ValueError("The model must expose exactly one True target class")
probabilities = np.asarray(model.predict_proba(test_inputs))[:, positive_columns[0]]
if probabilities.dtype.kind != "f":
    raise TypeError("Expected model probabilities, not predicted class labels")
if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
    raise ValueError("Probabilities must be finite and in [0, 1]")
task.record(fold, probabilities)
```

Preserve the order returned by `task.get_test_data(fold)`. Do not threshold,
round, apply `argmax`, or call `.astype(bool)` before recording. For a model
returning logits, use the model's documented sigmoid or softmax conversion and
select the positive-class probability. An arbitrary decision score is not a
probability and must not be clipped into `[0, 1]`.

## Inspect corrected scores without changing an artifact

```python
from matbench.bench import MatbenchBenchmark
from matbench.leaderboard import prepare_classification_scores

benchmark = MatbenchBenchmark.from_file("results.json.gz")
packet = {"results": benchmark}
prepare_classification_scores({"submission": packet})
for task_name, result in packet["classification_scores"].items():
    if result["eligible"]:
        print(task_name, result["scores"]["rocauc"])
    else:
        print(task_name, "excluded:", result["reason"])
```

This operation loads official targets, aligns saved predictions by sample ID,
and evaluates each fold separately. It neither changes the saved predictions
nor overwrites their stored scores. The reported mean is the unweighted mean
of the five fold AUCs; the standard deviation uses `ddof=0`. It is not the AUC
of all out-of-fold predictions pooled into one array.

When preparing a replacement submission, generate a separate result file with
the current `task.record` scoring implementation, retain the original artifact
for provenance, and run:

```sh
python -m scripts.test_submission --submission matbench_v0.1_your_algorithm
```

The check requires all classification folds to contain eligible probabilities
and verifies that their stored metrics match recomputation. A legacy file may
pass the structural/deserialization checks while failing this stricter
submission check.

## Archived reproduction scripts

Scripts and notebooks within `benchmarks/` describe historical submissions and
may require the original software versions. They are not all suitable as
current submission templates. In particular, the illustrative MODNet v0.1.10
and v0.1.12 `run.py` scripts request probabilities but then cast predictions to
Boolean labels. The archived ALIGNN, GPTChem, and Dummy workflows also include label-based
classification paths. The historical Matformer result artifact stores Boolean
labels, even though its checked-in script mentions sigmoid predictions. Their original code and
result artifacts are retained so published work remains traceable.

For new work, follow the probability-recording example above and the
[Running a benchmark](2run.md) and [Submitting results](3submit.md) instructions.
Do not assume that a printed historical field named `rocauc` is the corrected
probability ROC AUC shown on the current website.
