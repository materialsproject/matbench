# Migrating classification submissions to probability ROC AUC

The homepage best-model table displays separate **balanced accuracy (BA)** and
**probability ROC AUC** champions for each classification task. Each champion is
selected by its own metric within the existing general-purpose model scope.
All exact top-score ties are shown in algorithm-name order.
The two scaled-error plots continue to use BA for classification, with error
scaled as `(1 - balanced_accuracy) / 0.5`; their best-model markers therefore
identify BA winners.

The task leaderboards show both BA and probability ROC AUC:
[experimental metallicity](../Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_is_metal.md),
[glass formation](../Leaderboards%20Per-Task/matbench_v0.1_matbench_glass.md), and
[MP metallicity](../Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_is_metal.md).
The **ROC AUC available** group is sorted by mean probability ROC AUC, followed
by **Balanced accuracy only**, sorted by mean BA. Exact ties use algorithm-name
ordering. These are groups with different available metrics, not a continuous
ranking across metrics. BA and probability ROC AUC can select different
champions, including on the homepage. The homepage and task pages may also have
different model inclusion scopes.

Task plots show gray BA circles for every valid result and additional blue
probability ROC AUC diamonds where available. Each series uses its own fold
standard deviation. The original result files and their cached scores remain
unchanged; collapsed archives preserve the original scores and provenance.
The [score audit](../static/classification_scores.json) records availability,
exclusion reasons, fold results, summary statistics, and source-file hashes.

## Which historical submissions can be rescored?

| Saved predictions | Current display | Required action for a new submission |
|---|---|---|
| Floating-point probabilities in `[0, 1]`, finite in every official fold | Recomputed BA and probability ROC AUC | Recompute cached classification metrics before submitting the result file |
| Historical Boolean class decisions conforming to the original result-file protocol, valid in every official fold | Recomputed BA in the main table; probability ROC AUC is unavailable (`—`) | Recover probabilities from the original fold models or rerun the benchmark |
| Mixed label/probability types, missing or extra sample IDs, incomplete folds, non-finite or out-of-range values | Excluded from both formal metric displays, with a reason | Correct the predictions and validate every fold |

All displayed classification results are aligned by official test sample ID and
checked against the official labels. BA from probabilities uses `p > 0.5`; BA
from historical hard labels uses the saved decisions. The original decision
threshold for those historical labels may not be recoverable, so this is not a
claim that every historical model was reevaluated at a common threshold.

A single probability ROC AUC score cannot be converted to BA: it does not
contain the confusion matrix at a particular threshold. If the original
probabilities and official labels are available, BA can instead be recomputed
from decisions at `p > 0.5`. Do not select or tune that threshold on the test
set. The benchmark uses the fixed rule for probability submissions.

Standard binary AUC computed from hard labels equals BA. It cannot recover ROC
AUC from the model's original continuous predictions. BA remains a useful metric,
and probability ROC AUC is not necessarily larger than BA. When usable
probabilities were not saved, the probability ROC AUC cell is missing, not zero;
a genuine AUC of `0.0` is retained as a valid score.

Converting `True`/`False` to `1.0`/`0.0` cannot recover probabilities or their
ranking information and does not satisfy the submission policy. Do not use a
type cast as a substitute for obtaining model probabilities. Genuine endpoint
probabilities are allowed; the numerical validator cannot determine whether a
floating-point value was previously a hard label. Preserve the original
probability-generation code and provenance. New and updated classification
results still require probabilities and matching cached metrics; restoring
historical BA visibility does not relax that requirement. Regression results
and benchmark-completeness definitions are unchanged.

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
    if result["ba_available"]:
        print(task_name, "balanced accuracy:", result["scores"]["balanced_accuracy"])
    else:
        print(task_name, "BA excluded:", result["ba_reason"])
    if result["auc_available"]:
        print(task_name, "probability ROC AUC:", result["scores"]["rocauc"])
    else:
        print(task_name, "probability ROC AUC unavailable:", result["auc_reason"])
```

This operation loads official targets, aligns saved predictions by sample ID,
and evaluates each fold separately. It neither changes the saved predictions
nor overwrites their stored scores. The reported mean is the unweighted mean
of the five official fold scores for each metric; the standard deviation uses `ddof=0`. It is not the AUC
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
