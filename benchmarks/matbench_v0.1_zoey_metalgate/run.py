"""
Zoey-MetalGate: a metal-classifier-gated gradient-boosting blend for
matbench_expt_gap (experimental band gap from composition alone).

WHAT THIS DOES
--------------
matbench_expt_gap asks for the experimentally measured band gap of a compound
given only its chemical formula. 53% of the labels in this dataset are exactly
0.0 eV, because those compounds are metals. A single regressor trained on the
whole distribution spends a lot of its error budget smearing small positive
predictions across that zero mass.

This submission therefore uses a two-stage ("metal-gated") model:

    1. A LightGBM classifier predicts P(metal), trained on the binary label
       (gap == 0) of the training fold only.
    2. A gradient-boosted regressor is fit on the NON-METAL rows of the
       training fold only, so it models the shape of real gaps without being
       pulled toward zero.
    3. The two are combined per compound, either softly
       (pred = (1 - P_metal) * max(reg, 0)) or hard
       (pred = 0 where P_metal > 0.5, else max(reg, 0)).

The final submission averages three such models with fixed weights
(0.5 / 0.25 / 0.25), chosen by greedy forward selection on out-of-fold
predictions computed inside the official TRAINING folds only.

Features are 493 stateless composition descriptors from matminer. "Stateless"
means every column is a lookup of tabulated element properties combined by the
formula's stoichiometry; nothing is fit to the dataset during featurization, so
featurizing all rows at once cannot leak label information.

EVALUATION HYGIENE
------------------
Model selection (feature sets, model families, hyperparameters, blend weights)
was done entirely by internal 5-fold cross-validation INSIDE the official
training folds. The official test data is loaded only through
`task.get_test_data(fold, include_target=False)` and scored only by
`task.record(...)`, i.e. by matbench's own machinery. The full campaign was
preregistered before results existed, including the success criteria and a
requirement to investigate for leakage before reporting any result that beat
the published leaderboard.

Independent verification performed before submission:
  * Recomputing per-fold MAE directly from the raw prediction arrays, bypassing
    matbench's scorer, reproduces the recorded score to five decimals.
  * The five official folds are index-disjoint, cover the dataset exactly, and
    share zero normalized reduced formulas between any fold's train and test
    splits.

RUNTIME
-------
About 45 minutes total on 8 CPU cores (AMD Ryzen Z1 Extreme), no GPU.
Featurization is the slow part (~30 min, dominated by AtomicPackingEfficiency)
and is cached to disk on first run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition
from xgboost import XGBRegressor

SEED = 42
N_JOBS = os.cpu_count() or 8
FEATURE_CACHE = "zoey_features.parquet"


# --------------------------------------------------------------------------
# Featurization: 493 stateless composition descriptors
# --------------------------------------------------------------------------
def build_features(compositions: pd.Series) -> pd.DataFrame:
    """Compute the feature matrix. Cached, since this is the slow step."""
    if os.path.exists(FEATURE_CACHE):
        return pd.read_parquet(FEATURE_CACHE)

    from matminer.featurizers.composition import (
        AtomicOrbitals,
        AtomicPackingEfficiency,
        BandCenter,
        ElementFraction,
        ElementProperty,
        IonProperty,
        Meredig,
        Miedema,
        Stoichiometry,
        TMetalFraction,
        ValenceOrbital,
        YangSolidSolution,
    )

    featurizers = {
        "magpie": ElementProperty.from_preset("magpie"),
        "valence": ValenceOrbital(props=["frac"]),
        "atomic_orbitals": AtomicOrbitals(),
        "band_center": BandCenter(),
        "stoichiometry": Stoichiometry(),
        "element_fraction": ElementFraction(),
        "deml": ElementProperty.from_preset("deml"),
        "meredig": Meredig(),
        "miedema": Miedema(),
        "yang": YangSolidSolution(),
        "ape": AtomicPackingEfficiency(),
        "tmetal": TMetalFraction(),
        "ionprop": IonProperty(fast=True),
    }

    comps = [Composition(s) for s in compositions]
    parts = []
    for name, fz in featurizers.items():
        labels = fz.feature_labels()
        rows = []
        for c in comps:
            try:
                rows.append(fz.featurize(c))
            except Exception:
                # A handful of compositions fail AtomicOrbitals; NaN is fine,
                # gradient-boosted trees handle missing values natively.
                rows.append([np.nan] * len(labels))
        df = pd.DataFrame(rows, columns=[f"{name}::{l}" for l in labels])
        # Drop non-numeric columns (e.g. AtomicOrbitals element/character names)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.drop(columns=[c for c in df.columns if df[c].isna().all()])
        parts.append(df)
        print(f"  featurized {name}: {df.shape[1]} columns")

    feats = pd.concat(parts, axis=1)
    feats.index = compositions.index
    feats.to_parquet(FEATURE_CACHE)
    return feats


# --------------------------------------------------------------------------
# The metal-gated two-stage model
# --------------------------------------------------------------------------
class MetalGate:
    """Metal classifier gating a gap regressor.

    Both stages see only the data passed to fit(), i.e. one official training
    fold. The classifier's target is derived from the training labels, never
    from any external metal/nonmetal annotation.
    """

    def __init__(self, regressor, gate: str = "soft", seed: int = SEED):
        self.regressor = regressor
        self.gate = gate
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        # Stage 1: is this compound a metal (gap exactly zero)?
        self.clf_ = LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=127,
            colsample_bytree=0.7,
            subsample=0.8,
            bagging_freq=1,
            random_state=self.seed,
            n_jobs=N_JOBS,
            verbosity=-1,
        )
        self.clf_.fit(X, (y <= 0.0).astype(int))
        # Stage 2: regress the gap using non-metals only
        nonmetal = y > 0.0
        self.regressor.fit(X[nonmetal], y[nonmetal])
        return self

    def predict(self, X):
        p_metal = self.clf_.predict_proba(X)[:, 1]
        gap = np.maximum(self.regressor.predict(X), 0.0)
        if self.gate == "hard":
            return np.where(p_metal > 0.5, 0.0, gap)
        return (1.0 - p_metal) * gap


def xgb_regressor():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        random_state=SEED,
        n_jobs=N_JOBS,
        verbosity=0,
    )


def lgbm_regressor():
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=2500,
        learning_rate=0.02,
        num_leaves=255,
        subsample=0.8,
        bagging_freq=1,
        colsample_bytree=0.7,
        random_state=SEED,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


# The three blend members and their weights, fixed by out-of-fold greedy
# selection on the training folds before any test evaluation.
BLEND = [
    (0.50, lambda: MetalGate(xgb_regressor(), gate="soft")),
    (0.25, lambda: MetalGate(xgb_regressor(), gate="hard")),
    (0.25, lambda: MetalGate(lgbm_regressor(), gate="hard")),
]


# --------------------------------------------------------------------------
# Benchmark run
# --------------------------------------------------------------------------
def main():
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])
    task = mb.tasks_map["matbench_expt_gap"]
    task.load()

    print(f"featurizing {len(task.df)} compositions (cached after first run)")
    features = build_features(task.df["composition"])

    for fold in task.folds:
        X_train, y_train = task.get_train_and_val_data(fold)
        X_test = task.get_test_data(fold, include_target=False)

        train_feats = features.loc[X_train.index].values
        test_feats = features.loc[X_test.index].values
        y = np.asarray(y_train.values, dtype=float)

        prediction = np.zeros(len(X_test))
        for weight, make_model in BLEND:
            model = make_model().fit(train_feats, y)
            prediction += weight * model.predict(test_feats)

        task.record(fold, prediction)
        print(f"fold {fold}: MAE {task.results[f'fold_{fold}']['scores']['mae']:.4f}")

    print(f"\nmean MAE over 5 official folds: {task.scores['mae']['mean']:.4f}")
    print(f"mean RMSE: {task.scores['rmse']['mean']:.4f}")
    mb.to_file("results.json.gz")


if __name__ == "__main__":
    main()
