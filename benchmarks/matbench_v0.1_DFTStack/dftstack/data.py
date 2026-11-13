"""Load the four Matbench v0.1 composition tasks with their OFFICIAL folds and
cache compositions, targets, fold membership and label-free composition features.

Featurisation is label-free (a deterministic function of each formula), so it is
computed once for all rows; no target or fold information is used by it.
"""
from __future__ import annotations
import json, pathlib, warnings
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
TASKS = ["matbench_steels", "matbench_expt_gap", "matbench_expt_is_metal", "matbench_glass"]


def load_task_frames(task_name: str):
    """Return (df, folds) where df has columns composition(str), target, and folds is a
    list of 5 dicts {train: index list, test: index list} using the official matbench folds."""
    out = CACHE / f"{task_name}_frame.pkl"
    fold_file = CACHE / f"{task_name}_folds.json"
    if out.exists() and fold_file.exists():
        return pd.read_pickle(out), json.loads(fold_file.read_text())
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = getattr(mb, task_name)
    task.load()
    df = task.df.copy()
    target_col = task.metadata.target
    input_col = task.metadata.input_type  # 'composition'
    df = df.rename(columns={target_col: "target"})
    df["composition"] = df[input_col].astype(str) if input_col in df else df.iloc[:, 0].astype(str)
    folds = []
    for fold in task.folds:
        tr_in, _ = task.get_train_and_val_data(fold)
        te_in = task.get_test_data(fold, include_target=False)
        folds.append({"train": [str(i) for i in tr_in.index], "test": [str(i) for i in te_in.index]})
    df.index = df.index.astype(str)
    df = df[["composition", "target"]]
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    fold_file.write_text(json.dumps(folds))
    return df, folds


def featurise(task_name: str, df: pd.DataFrame):
    """Magpie ElementProperty descriptors + element-fraction vector (label-free)."""
    out = CACHE / f"{task_name}_features.pkl"
    if out.exists():
        return pd.read_pickle(out)
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty, ElementFraction
    warnings.filterwarnings("ignore")
    comps = [Composition(c) for c in df["composition"]]
    ep = ElementProperty.from_preset("magpie", impute_nan=True)
    ef = ElementFraction()
    mag = np.array([ep.featurize(c) for c in comps], dtype=float)
    frac = np.array([ef.featurize(c) for c in comps], dtype=float)
    mag_cols = ["magpie_" + n for n in ep.feature_labels()]
    frac_cols = ["frac_" + n for n in ef.feature_labels()]
    feats = pd.DataFrame(np.hstack([mag, frac]), index=df.index, columns=mag_cols + frac_cols)
    feats.to_pickle(out)
    return feats


def featurise_extra(task_name: str, df: pd.DataFrame):
    """Additional label-free matminer composition descriptors (MODNet-style breadth):
    Stoichiometry, ValenceOrbital, IonProperty(fast), BandCenter, AtomicOrbitals (HOMO/LUMO)."""
    out = CACHE / f"{task_name}_features_extra.pkl"
    if out.exists():
        return pd.read_pickle(out)
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
    X = X.loc[:, X.notna().any()]
    X = X.fillna(X.median())  # label-free imputation of rare failures
    X.to_pickle(out)
    return X


if __name__ == "__main__":
    import sys
    for t in TASKS:
        df, folds = load_task_frames(t)
        f = featurise(t, df)
        if "--extra" in sys.argv:
            fe = featurise_extra(t, df)
            print(t, "extra", fe.shape, flush=True)
        print(t, df.shape, [len(x["test"]) for x in folds], f.shape, int(f.isna().sum().sum()), df["target"].dtype, flush=True)

