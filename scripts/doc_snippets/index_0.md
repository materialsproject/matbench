Scaled errors for regressions on this leaderboard plot are assessed as the ratio of mean absolute error to mean absolute deviation:

$$
\text{Scaled Error} = \frac{\text{MAE}}{\text{MAD}} = \frac{\sum_i^N | y_i - y_i^{pred} |}{\sum_i^N | y_i - \bar{y} | }
$$

While, scaled errors for classifications are assessed as:

$$
\text{Scaled Error} = \frac{1 - \text{ROCAUC}}{0.5}
$$
Classification ROC-AUC is recomputed from saved floating-point probabilities for
`target=True` on each official test fold, then averaged across folds. The plot
uses only eligible probability submissions for classification; hard-label results
are omitted from those points and retained in the historical balanced-accuracy
tables on the task pages. Regression points are unchanged.
