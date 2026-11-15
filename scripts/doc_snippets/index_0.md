Scaled errors for regressions on this leaderboard plot are assessed as the ratio of mean absolute error to mean absolute deviation:

$$
\text{Scaled Error} = \frac{\text{MAE}}{\text{MAD}} = \frac{\sum_i^N | y_i - y_i^{pred} |}{\sum_i^N | y_i - \bar{y} | }
$$

Scaled classification errors in both homepage plots use balanced accuracy (BA):

$$
\text{Scaled classification error} = \frac{1 - \text{balanced accuracy}}{0.5}
$$

The homepage best-model table lists separate BA and probability ROC AUC leaders
for every classification task, including all exact ties. Both metrics use the
same general-purpose model scope and their own verified scores; each model's
input requirement is shown beside its name. Both scaled-error plots continue to
use BA verified against official five-fold test labels. Saved probabilities use the existing `p > 0.5` decision
rule; historical hard-label submissions use their saved decisions. Their original
decision thresholds may be unknown. The general-purpose table and plot retain
their existing model inclusion rules; the all-algorithm plot also includes
specialized models. Best-model markers use the same BA values as the corresponding
plot. Regression points and inclusion rules are unchanged.

For full model lists and the two-metric plots, see the
[experimental metallicity](Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_is_metal.md),
[metallic glass](Leaderboards%20Per-Task/matbench_v0.1_matbench_glass.md), and
[DFT metallicity](Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_is_metal.md)
task leaderboards. BA and probability ROC AUC measure different aspects of the
predictions, so their leading models can differ. A model with only saved hard
labels has no recoverable probability ROC AUC; this does not invalidate its BA.
