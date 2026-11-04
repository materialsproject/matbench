"""Expert (level-0) models. Every expert exposes fit(X_df, y) and predict(X_df) -> 1-D array
(regression value, or probability of the positive class for classification).

Hyperparameters live in CONFIG and are fixed during the design phase (inner CV on fold 0's
training data only), then frozen.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import joblib


def cols(X: pd.DataFrame, which: str):
    if which == "magpie":
        return [c for c in X.columns if c.startswith("magpie_")]
    if which == "frac":
        return [c for c in X.columns if c.startswith("frac_")]
    if which == "all":
        return list(X.columns)
    raise ValueError(which)


class Expert:
    def __init__(self, name, features, model, task_type, target_transform=None):
        self.name, self.features, self.model, self.task_type = name, features, model, task_type
        self.tt = target_transform  # None or "log"

    def _X(self, X):
        return X[cols(X, self.features)].to_numpy(dtype=np.float64)

    def fit(self, X, y):
        y = np.asarray(y)
        if self.task_type == "regression":
            yt = np.log(y) if self.tt == "log" else y
            self.y_lo, self.y_hi = float(np.min(y)), float(np.max(y))
            # threading backend for the internal KRR grid search: identical results, avoids loky
            # worker-process hangs when many folds run concurrently
            with joblib.parallel_config(backend="threading"):
                self.model.fit(self._X(X), yt)
        else:
            self.model.fit(self._X(X), y.astype(int))
        return self

    def predict(self, X):
        Xa = self._X(X)
        if self.task_type == "regression":
            p = self.model.predict(Xa)
            p = np.exp(np.clip(p, -50, 50)) if self.tt == "log" else p
            # predictions are clipped to the training-target range (guards against MLP/KRR blow-ups)
            return np.clip(p, self.y_lo, self.y_hi)
        return self.model.predict_proba(Xa)[:, 1]


def _lgb(task_type, n, seed, objective=None):
    common = dict(n_estimators=1500 if n > 1000 else 800, learning_rate=0.02, num_leaves=31,
                  min_child_samples=5 if n < 1000 else 10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.4, reg_lambda=1.0, random_state=seed, n_jobs=4, verbose=-1,
                  deterministic=True, force_col_wise=True)  # reproducible with a fixed thread count
    return lgb.LGBMRegressor(objective=objective or ("l1" if n < 1000 else "l2"), **common) if task_type == "regression" \
        else lgb.LGBMClassifier(**common)


def make_experts(task_type: str, n_train: int, seed: int = 0, cfg: dict | None = None):
    """Return the list of experts used for one task. cfg can override per-task choices
    (e.g. {"target_transform": "log"}) fixed during design."""
    cfg = cfg or {}
    tt = cfg.get("target_transform")
    reg = task_type == "regression"
    E = []
    E.append(Expert("lgbm_all", "all", _lgb(task_type, n_train, seed), task_type, tt))
    if reg and cfg.get("lgbm_l1_expert"):
        E.append(Expert("lgbm_l1_all", "all", _lgb(task_type, n_train, seed, objective="l1"), task_type, tt))
    rf_kw = dict(n_estimators=500, max_features=0.33, min_samples_leaf=1, n_jobs=4, random_state=seed)
    E.append(Expert("rf_all", "all", RandomForestRegressor(**rf_kw) if reg else RandomForestClassifier(**rf_kw), task_type, tt))
    et_kw = dict(n_estimators=500, max_features=0.5, min_samples_leaf=1, n_jobs=4, random_state=seed)
    E.append(Expert("et_all", "all", ExtraTreesRegressor(**et_kw) if reg else ExtraTreesClassifier(**et_kw), task_type, tt))
    if reg:
        # kernel ridge has no intercept: standardise the target; (alpha, gamma) tuned by an internal
        # 3-fold grid search on the rows the expert is fitted on (nested inside every training set)
        def krr(scale_X):
            g = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                             "gamma": list(np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]) * (1.0 / 132 if scale_X else 1.0))},
                             cv=3, scoring="neg_mean_absolute_error", n_jobs=4)
            m = make_pipeline(StandardScaler(), g) if scale_X else g
            return TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
        E.append(Expert("krr_frac", "frac", krr(False), task_type, tt))
        E.append(Expert("krr_magpie", "magpie", krr(True), task_type, tt))
        E.append(Expert("mlp_all", "all", TransformedTargetRegressor(regressor=make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal"), MLPRegressor(
            hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3, max_iter=600,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=seed)),
            transformer=StandardScaler()), task_type, tt))
    else:
        E.append(Expert("svc_frac", "frac", SVC(kernel="rbf", C=cfg.get("svc_frac_C", 10.0), gamma=cfg.get("svc_frac_gamma", 1.0),
                                               probability=True, random_state=seed), task_type))
        E.append(Expert("svc_magpie", "magpie", make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=cfg.get("svc_mag_C", 10.0), gamma=cfg.get("svc_mag_gamma", 1.0 / 132),
            probability=True, random_state=seed)), task_type))
        E.append(Expert("mlp_all", "all", make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal"), MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3, max_iter=600,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=seed)), task_type))
    return E
