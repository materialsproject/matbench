"""Disclosed transfer-learning features from OFFLINE DFT (PBE) data (ANALYSIS_PLAN Amendment 2).

Source: Materials Project DFT data as redistributed in matminer's `matbench_mp_gap` (PBE band gap,
106,113 structures, e_hull < 150 meV/atom) and `matbench_mp_e_form` (PBE formation energy).
No experimental labels are used. Entries are matched to task compositions by reduced formula. For each
formula the "most stable" polymorph (lowest formation energy per atom, matched by formula + nsites +
volume) supplies the primary features; polymorph aggregates (min/median/max gap, count) are added. No
match -> NaN plus indicator column `dft_matched` = 0.
(The figshare-hosted `mp_nostruct_20181018` table, which has e_hull directly, was unreachable: HTTP 500.)
"""
from __future__ import annotations
import pathlib
import numpy as np, pandas as pd
from .data import ROOT

EXT = ROOT / "data" / "cache" / "external"


def build_reference():
    out = EXT / "mp_dft_by_formula.pkl"
    if out.exists():
        return pd.read_pickle(out)
    from matminer.datasets import load_dataset
    EXT.mkdir(parents=True, exist_ok=True)
    g = load_dataset("matbench_mp_gap", data_home=str(EXT))
    e = load_dataset("matbench_mp_e_form", data_home=str(EXT))
    def keys(df):
        return pd.DataFrame({"formula": [s.composition.reduced_formula for s in df["structure"]],
                             "nsites": [len(s) for s in df["structure"]],
                             "vol": [round(s.volume, 1) for s in df["structure"]]}, index=df.index)
    kg, ke = keys(g), keys(e)
    kg["gap"] = g["gap pbe"].values
    ke["e_form"] = e["e_form"].values
    m = kg.merge(ke.drop_duplicates(["formula", "nsites", "vol"]), on=["formula", "nsites", "vol"], how="left")
    rows = []
    for f, sub in m.groupby("formula"):
        withE = sub.dropna(subset=["e_form"])
        best = withE.loc[withE["e_form"].idxmin()] if len(withE) else sub.iloc[0]
        rows.append({"formula": f, "dft_gap_stable": best["gap"], "dft_metal_stable": float(best["gap"] <= 0.0),
                     "dft_eform_stable": best.get("e_form", np.nan), "dft_gap_min": sub["gap"].min(),
                     "dft_gap_median": sub["gap"].median(), "dft_gap_max": sub["gap"].max(),
                     "dft_metal_frac": float((sub["gap"] <= 0).mean()), "dft_n_polymorphs": len(sub),
                     "dft_eform_known": float(len(withE) > 0)})
    ref = pd.DataFrame(rows).set_index("formula")
    ref.to_pickle(out)
    return ref


def dft_features(task_name: str, df: pd.DataFrame):
    out = ROOT / "data" / "cache" / f"{task_name}_features_dft.pkl"
    if out.exists():
        return pd.read_pickle(out)
    from pymatgen.core import Composition
    ref = build_reference()
    f = [Composition(c).reduced_formula for c in df["composition"]]
    X = ref.reindex(f)
    X.index = df.index
    X["dft_matched"] = X["dft_gap_stable"].notna().astype(float)
    X.to_pickle(out)
    return X


if __name__ == "__main__":
    from .data import TASKS, load_task_frames
    ref = build_reference()
    print("reference formulas:", len(ref))
    for t in TASKS:
        df, _ = load_task_frames(t)
        X = dft_features(t, df)
        print(t, "matched", int(X["dft_matched"].sum()), "of", len(X))
