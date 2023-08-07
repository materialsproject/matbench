Scaled errors for regressions on this leaderboard plot are assessed as the ratio of mean absolute error to mean absolute deviation:

$$
\text{Scaled Error} = \frac{\text{MAE}}{\text{MAD}} = \frac{\sum_i^N | y_i - y_i^{pred} |}{\sum_i^N | y_i - \bar{y} | }
$$

While, scaled errors for classifications are assessed as:

$$
\text{Scaled Error} = \frac{1 - \text{ROCAUC}}{0.5}
$$
