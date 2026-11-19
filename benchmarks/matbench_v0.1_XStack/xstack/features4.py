"""Amendment-4 feature matrices (ANALYSIS_PLAN.md Section 14): composition-only, NO cross-row statistic.

Two changes against src/features_all.py (Amendments 2/3), nothing else:
  1. the DFT transfer block ("dft") is not available at all (asserted);
  2. the two blocks whose builders filled missing cells with column medians over ALL rows of a task
     (the `extra` block of src/data.py and the CBFV Oliynyk block) are rebuilt with the missing cells left as NaN
     (`*_raw.pkl` caches). The medians are then computed inside every fit from that fit's training rows only
     (src/experts4.py), i.e. from T_k for the final refit and from the inner-training rows for every inner-CV fit.

Every value in these matrices is a deterministic function of the row's own formula. `verify()` checks that
re-applying the old all-row median fill to the raw blocks reproduces the Amendment-2/3 caches exactly, so the raw
blocks differ from the frozen ones only in the previously imputed cells. No target is read here.

Usage: python -m src.features4 [task ...]      # build + verify the raw caches
"""
from __future__ import annotations
import re, sys, warnings
import numpy as np, pandas as pd

from .data import CACHE, TASKS, load_task_frames, featurise, featurise_extra
from .features_more import block

IMPUTED_PREFIX = ("extra_", "cbfv_oliynyk_")       # column prefixes (after sanitisation) of the two raw blocks


def extra_raw(task, df=None):
    """src.data.featurise_extra without its `X.fillna(X.median())` line (same featurisers, same order)."""
    out = CACHE / f"{task}_features_extra_raw.pkl"
    if out.exists():
        return pd.read_pickle(out)
    if df is None:
        df, _ = load_task_frames(task)
    from pymatgen.core import Composition
    from matminer.featurizers.composition import (Stoichiometry, ValenceOrbital, IonProperty,
                                                  BandCenter, AtomicOrbitals)
    warnings.filterwarnings("ignore")
    comps = [Composition(c) for c in df["composition"]]
    fz = [Stoichiometry(), ValenceOrbital(impute_nan=True), IonProperty(fast=True, impute_nan=True),
          BandCenter(impute_nan=True), AtomicOrbitals()]
    blocks, names = [], []
    for f in fz:
        vals = []
        for c in comps:
            try:
                vals.append(f.featurize(c))
            except Exception:
                vals.append([np.nan] * len(f.feature_labels()))
        blocks.append(np.array([[v if isinstance(v, (int, float, np.floating)) else np.nan for v in r] for r in vals], dtype=float))
        names += [f"extra_{type(f).__name__}_{n}" for n in f.feature_labels()]
    X = pd.DataFrame(np.hstack(blocks), index=df.index, columns=names)
    X = X.loc[:, X.notna().any()]          # columns that are NaN for every row of the task carry no value at all
    X.to_pickle(out)
    return X


def cbfv_oliynyk_raw(task, df=None):
    """The CBFV Oliynyk block exactly as src.features_more.block builds it, with CBFV's own median fill
    (`X[cols] = X[cols].fillna(X[cols].median())`, the only cross-row step of generate_features) switched off."""
    out = CACHE / f"{task}_fb_cbfv_oliynyk_raw.pkl"
    if out.exists():
        return pd.read_pickle(out)
    if df is None:
        df, _ = load_task_frames(task)
    warnings.filterwarnings("ignore")
    from CBFV.composition import generate_features
    orig = pd.DataFrame.fillna

    def no_series_fill(self, value=None, *a, **k):
        if isinstance(value, pd.Series):       # the median fill: leave the missing cells missing
            return self
        return orig(self, value, *a, **k)

    pd.DataFrame.fillna = no_series_fill
    try:
        tmp = pd.DataFrame({"formula": df["composition"].values, "target": 0.0})   # dummy target, as in features_more
        X, _, _, skipped = generate_features(tmp, elem_prop="oliynyk", drop_duplicates=False, extend_features=False,
                                             sum_feat=True)
    finally:
        pd.DataFrame.fillna = orig
    if len(X) != len(df):
        raise RuntimeError(f"CBFV skipped rows: {skipped}")
    X.index = df.index
    X.columns = [f"cbfv_oliynyk_{c}" for c in X.columns]
    X = X.loc[:, X.notna().any()]
    X.to_pickle(out)
    return X


def verify(task, df=None):
    """Raw block + the old all-row median fill == the frozen Amendment-2/3 cache (same columns, same values)."""
    if df is None:
        df, _ = load_task_frames(task)
    rep = {}
    for name, raw, old in (("extra", extra_raw(task, df), featurise_extra(task, df)),
                           ("cbfv_oliynyk", cbfv_oliynyk_raw(task, df), block(task, "cbfv_oliynyk", df))):
        assert list(raw.columns) == list(old.columns) and list(raw.index) == list(old.index), (task, name)
        refilled = raw.fillna(raw.median())
        assert np.array_equal(refilled.to_numpy(), old.to_numpy(), equal_nan=True), (task, name)
        rep[name] = {"nan_cells": int(raw.isna().sum().sum()), "rows_with_nan": int(raw.isna().any(axis=1).sum()),
                     "columns_with_nan": int(raw.isna().any(axis=0).sum()), "shape": list(raw.shape)}
    return rep


def features4(task, df=None, blocks=("base", "extra")):
    """Same assembly as features_all.features (order, name sanitisation, de-duplication), raw blocks swapped in."""
    if df is None:
        df, _ = load_task_frames(task)
    assert "dft" not in blocks, "Amendment 4 uses no DFT block"
    parts = []
    for b in blocks:
        if b == "base":
            parts.append(featurise(task, df))
        elif b == "extra":
            parts.append(extra_raw(task, df))
        elif b == "cbfv_oliynyk":
            parts.append(cbfv_oliynyk_raw(task, df))
        else:
            parts.append(block(task, b, df))
    X = pd.concat(parts, axis=1)
    X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    assert not any(c.startswith("dft_") for c in X.columns)
    return X


if __name__ == "__main__":
    import json
    out = {}
    for t in (sys.argv[1:] or TASKS):
        d, _ = load_task_frames(t)
        out[t] = {"n_rows": len(d), **verify(t, d)}
        print(t, out[t], flush=True)
    print(json.dumps(out))
