#!/usr/bin/env python3
# modified from source: ax\metrics\hartmann6.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import numpy as np


def extraordinary_probability(
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    mx: float = None,
    mn: float = None,
    thresh: float = 0.10,
    verbose: bool = True,
    minimize: bool = True,
    use_quantile: bool = False,
):
    """Determine the probability of finding an extraordinary candidate.

    Note that the predicted values are used for thresholding, while the true values are
    for determining whether or not an extraordinary candidate was found. In other words,
    this assumes you are on a fixed budget and using this as a screening tool.
    Candidates which had a low predicted value and a high true value wouldn't be
    considered in the pool of extraordinary candidates. This is similar to doing repeats
    to verify extraordinary observations subject to noise.

    Parameters
    ----------
    y_true : Union[list, np.ndarray]
        True property values.
    y_pred : Union[list, np.ndarray]
        Predicted property values.
    mx : float, optional
        Estimate of true maximum. If None, then max(y_true) is used. By default None.
    mn : float, optional
        Estimate of true minimum. If None, then min(y_true) is used. By default None.
    thresh : float, optional
        Threshold to use for defining an extraordinary candidate, by default 0.10
    verbose : bool, optional
        Whether or not to print information about the number of candidates required to
        find and the probability of finding an extraordinary compound, by default True
    minimize : bool, optional
        Whether lower values are considered more optimal, by default True
    use_quantile : bool, optional
        Whether to use quantile methods of thresholding rather than a percentage
        relative to the true optimum, by default False

    Returns
    -------
    p
        probability of finding an extraordinary candidate
    ids
        ids of candidates considered to be extraordinary
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if mx is None:
        mx = max(y_true)
    if mn is None:
        mn = min(y_true)
    if minimize:
        if use_quantile:
            cutoff = np.quantile(y_true, 0.02)
        else:
            cutoff = mn + (mx - mn) * thresh
        ids = y_pred < cutoff
        n_ext = sum(y < cutoff for y in y_pred)
        p = n_ext / len(y_true)
    else:
        if use_quantile:
            cutoff = np.quantile(y_true, 0.02)
        else:
            cutoff = mx - (mx - mn) * thresh
        ids = y_pred > cutoff
        n_ext = sum(y > cutoff for y in y_pred)
        p = n_ext / len(y_true)
    if verbose:
        if n_ext > 0:
            print(
                f"probability of finding candidate within {100*thresh:.1f}% of best estimated optimum (f<{cutoff:.4f}): {100*p:.4f}%. In other words, on average {int(np.round(1/p)) if p != 0 else np.inf} candidates required to find one extraordinary candidate."
            )
        else:
            print(
                f"no candidates found within {100*thresh:.1f}% of best estimated optimum (f<{cutoff:.4f}): using {len(y_true)} observations. "
            )
    return p, ids
