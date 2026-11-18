"""Amendment-2 label-free composition feature blocks (cached per task and block).

Blocks: matminer ElementProperty presets (deml, matminer, matscholar_el, megnet_el), Meredig,
Miedema, YangSolidSolution, AtomicPackingEfficiency, WenAlloys, TMetalFraction, and CBFV
element-property statistics (oliynyk, jarvis, mat2vec). Each block is a deterministic function of the
formula (no labels). Featuriser failures -> NaN (handled downstream: trees natively, others by
train-fold median imputation inside the expert pipeline).
"""
from __future__ import annotations
import sys, time, warnings
import numpy as np, pandas as pd
from .data import CACHE, TASKS, load_task_frames

BLOCKS = ["ep_deml", "ep_matminer", "ep_matscholar", "ep_megnet", "meredig", "miedema", "yang", "ape",
          "wen", "tmetal", "cbfv_oliynyk", "cbfv_jarvis", "cbfv_mat2vec"]


def _matminer_block(name):
    from matminer.featurizers import composition as C
    return {"ep_deml": lambda: C.ElementProperty.from_preset("deml", impute_nan=True),
            "ep_matminer": lambda: C.ElementProperty.from_preset("matminer", impute_nan=True),
            "ep_matscholar": lambda: C.ElementProperty.from_preset("matscholar_el", impute_nan=True),
            "ep_megnet": lambda: C.ElementProperty.from_preset("megnet_el", impute_nan=True),
            "meredig": lambda: C.Meredig(),
            "miedema": lambda: C.Miedema(),
            "yang": lambda: C.YangSolidSolution(),
            "ape": lambda: C.AtomicPackingEfficiency(),
            "wen": lambda: C.WenAlloys(),
            "tmetal": lambda: C.TMetalFraction()}[name]()


def block(task, name, df=None):
    out = CACHE / f"{task}_fb_{name}.pkl"
    if out.exists():
        return pd.read_pickle(out)
    warnings.filterwarnings("ignore")
    if df is None:
        df, _ = load_task_frames(task)
    t0 = time.time()
    if name.startswith("cbfv_"):
        from CBFV.composition import generate_features
        prop = name.split("_", 1)[1]
        tmp = pd.DataFrame({"formula": df["composition"].values, "target": 0.0})
        X, _, formulae, skipped = generate_features(tmp, elem_prop=prop, drop_duplicates=False, extend_features=False,
                                                    sum_feat=True)
        X.index = df.index[:len(X)] if len(X) == len(df) else X.index
        if len(X) != len(df):
            raise RuntimeError(f"CBFV skipped rows: {skipped}")
        X.columns = [f"{name}_{c}" for c in X.columns]
    else:
        from pymatgen.core import Composition
        fz = _matminer_block(name)
        comps = [Composition(c) for c in df["composition"]]
        vals = []
        for c in comps:
            try:
                v = fz.featurize(c)
                vals.append([x if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) else
                             (float(x) if isinstance(x, bool) else np.nan) for x in v])
            except Exception:
                vals.append([np.nan] * len(fz.feature_labels()))
        X = pd.DataFrame(np.array(vals, dtype=float), index=df.index, columns=[f"{name}_{c}" for c in fz.feature_labels()])
    X = X.loc[:, X.notna().any()]
    X.to_pickle(out)
    print(f"  {task} block {name}: {X.shape[1]} cols, {int(X.isna().any(axis=1).sum())} rows with NaN, {time.time()-t0:.0f}s", flush=True)
    return X


if __name__ == "__main__":
    tasks = sys.argv[1:] or TASKS
    for t in tasks:
        df, _ = load_task_frames(t)
        for b in BLOCKS:
            try:
                block(t, b, df)
            except Exception as e:
                print(f"  {t} block {b}: FAILED {type(e).__name__}: {e}", flush=True)
