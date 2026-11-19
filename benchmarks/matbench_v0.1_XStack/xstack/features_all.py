"""Assemble Amendment-2 feature matrices (all label-free, plus the disclosed DFT transfer block)."""
from __future__ import annotations
import pandas as pd
from .data import featurise, featurise_extra, load_task_frames
from .features_more import block, BLOCKS


def features(task, df=None, blocks=("base", "extra") + tuple(BLOCKS) + ("dft",)):
    if df is None:
        df, _ = load_task_frames(task)
    parts = []
    for b in blocks:
        if b == "base":
            parts.append(featurise(task, df))
        elif b == "extra":
            parts.append(featurise_extra(task, df))
        elif b == "dft":
            from .external_dft import dft_features
            parts.append(dft_features(task, df))
        else:
            try:
                parts.append(block(task, b, df))
            except FileNotFoundError:
                raise
    X = pd.concat(parts, axis=1)
    import re
    X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    return X
