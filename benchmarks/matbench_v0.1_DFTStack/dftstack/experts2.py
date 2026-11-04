"""Amendment-2 expert pool. Each expert = (model factory, feature group). Hyperparameters of the
gradient-boosting experts are tuned with Optuna by inner CV inside the official training fold T_k
(see pipeline2.tune_expert); everything else is fixed.

Feature groups (column prefixes after name sanitisation, see features_all.py):
  F1     : magpie_, frac_, extra_, dft_            (Amendment-1 features + DFT transfer block)
  F2     : F1 + ep_* + meredig_ + miedema_ + yang_ + ape_ + wen_ + tmetal_ + cbfv_oliynyk_
  magpie : magpie_ only;  frac : frac_ only;  comp : the composition string (CrabNet)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

F1_PREFIX = ("magpie_", "frac_", "extra_", "dft_")
F2_PREFIX = F1_PREFIX + ("ep_", "meredig_", "miedema_", "yang_", "ape_", "wen_", "tmetal_", "cbfv_oliynyk_")


def group_cols(X: pd.DataFrame, group: str):
    if group == "F1":
        return [c for c in X.columns if c.startswith(F1_PREFIX)]
    if group == "F2":
        return [c for c in X.columns if c.startswith(F2_PREFIX)]
    if group == "magpie":
        return [c for c in X.columns if c.startswith("magpie_")]
    if group == "frac":
        return [c for c in X.columns if c.startswith("frac_")]
    raise ValueError(group)


class Expert2:
    def __init__(self, name, group, factory, task_type, tunable=None):
        self.name, self.group, self.factory, self.task_type, self.tunable = name, group, factory, task_type, tunable
        self.params = {}

    def clone(self):
        e = Expert2(self.name, self.group, self.factory, self.task_type, self.tunable)
        e.params = dict(self.params)
        return e

    def _X(self, X):
        if self.group == "comp":
            return X[["composition"]]
        return X[group_cols(X, self.group)]

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.model = self.factory(self.params)
        Xs = self._X(X)
        if self.task_type == "regression":
            self.lo, self.hi = float(y.min()), float(y.max())
            with joblib.parallel_config(backend="threading"):
                self.model.fit(Xs if self.group == "comp" else Xs.to_numpy(np.float64), y)
        else:
            with joblib.parallel_config(backend="threading"):
                self.model.fit(Xs if self.group == "comp" else Xs.to_numpy(np.float64), y.astype(int))
        return self

    def predict(self, X):
        Xs = self._X(X)
        A = Xs if self.group == "comp" else Xs.to_numpy(np.float64)
        if self.task_type == "regression":
            return np.clip(self.model.predict(A), self.lo, self.hi)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(A)[:, 1]
        return self.model.predict(A)


class Hurdle:
    """Zero-inflated target (e.g. experimental band gap, ~half exactly 0): a classifier for y > 0 and a
    regressor fitted on the y > 0 rows. rule="exp": p * r (expectation); rule="median": r if p >= 0.5 else 0
    (the MAE-optimal choice when the conditional distribution is a point mass at 0 plus a positive part)."""
    def __init__(self, clf, reg, rule):
        self.clf, self.reg, self.rule = clf, reg, rule

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.clf.fit(X, (y > 0).astype(int))
        nz = y > 0
        self.reg.fit(X[nz], y[nz])
        return self

    def predict(self, X):
        p = self.clf.predict_proba(X)[:, 1]
        r = np.clip(self.reg.predict(X), 0, None)
        return p * r if self.rule == "exp" else np.where(p >= 0.5, r, 0.0)


def _imp(m):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), m)


def make_experts2(task_type, n_train, seed=0, cfg=None):
    cfg = cfg or {}
    reg = task_type == "regression"
    small = n_train < 1000
    E = []

    # --- tuned gradient boosting (params filled by Optuna inside T_k) ---
    def lgbm(p):
        q = dict(n_estimators=1000, learning_rate=0.03, num_leaves=31, min_child_samples=5 if small else 10, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, random_state=seed, n_jobs=4, verbose=-1,
                 deterministic=True, force_col_wise=True)
        q.update(p)
        return lgb.LGBMRegressor(**q) if reg else lgb.LGBMClassifier(**q)
    E.append(Expert2("lgbm_F2_tuned", "F2", lgbm, task_type, tunable="lgbm"))

    def xgbm(p):
        q = dict(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.3, min_child_weight=1,
                 reg_lambda=1.0, tree_method="hist", random_state=seed, n_jobs=4)
        q.update(p)
        return xgb.XGBRegressor(objective="reg:absoluteerror" if reg and q.pop("l1", False) else "reg:squarederror", **q) \
            if reg else xgb.XGBClassifier(eval_metric="logloss", **{k: v for k, v in q.items() if k != "l1"})
    E.append(Expert2("xgb_F2", "F2", xgbm, task_type))  # fixed defaults (tuning too slow on CPU, see plan)

    def cat(p):
        q = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, random_seed=seed, thread_count=4, verbose=0,
                 allow_writing_files=False)
        q.update(p)
        return CatBoostRegressor(loss_function="MAE" if q.pop("mae", False) else "RMSE", **q) if reg \
            else CatBoostClassifier(**{k: v for k, v in q.items() if k != "mae"})
    E.append(Expert2("cat_F1", "F1", cat, task_type))  # F1: CatBoost on ~1k features was too slow on CPU

    # --- fixed tree ensembles ---
    for g in ("F1", "F2"):
        E.append(Expert2(f"et_{g}", g, (lambda p: _imp(ExtraTreesRegressor(n_estimators=500, max_features=0.5, n_jobs=4, random_state=seed))) if reg
                         else (lambda p: _imp(ExtraTreesClassifier(n_estimators=500, max_features=0.5, n_jobs=4, random_state=seed))), task_type))
    # extremely randomised trees on element fractions only, max_features chosen by an internal 3-fold grid
    E.append(Expert2("et_frac", "frac", (lambda p: GridSearchCV(ExtraTreesRegressor(n_estimators=1000, n_jobs=4, random_state=seed),
                                                               {"max_features": [0.3, 0.5, 0.8, 1.0]}, cv=3,
                                                               scoring="neg_mean_absolute_error")) if reg else
                     (lambda p: GridSearchCV(ExtraTreesClassifier(n_estimators=1000, n_jobs=4, random_state=seed),
                                             {"max_features": [0.1, 0.3, 0.5, 0.8]}, cv=3, scoring="roc_auc")), task_type))
    # (random forest on F2 removed after design: slow on CPU and near-zero stack weight)

    # --- hurdle experts for zero-inflated regression targets (enabled per task in the config) ---
    if reg and cfg.get("hurdle"):
        hp = dict(n_estimators=1500, learning_rate=0.02, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.3, random_state=seed, n_jobs=4, verbose=-1, deterministic=True, force_col_wise=True)
        for rule in ("exp", "median"):
            E.append(Expert2(f"hurdle_lgbm_F2_{rule}", "F2", (lambda p, rule=rule: Hurdle(lgb.LGBMClassifier(**hp),
                     lgb.LGBMRegressor(objective="l1", **hp), rule)), task_type))
        E.append(Expert2("hurdle_et_F1_exp", "F1", lambda p: Hurdle(_imp(ExtraTreesClassifier(n_estimators=500, max_features=0.5, n_jobs=4, random_state=seed)),
                 _imp(ExtraTreesRegressor(n_estimators=500, max_features=0.5, n_jobs=4, random_state=seed)), "exp"), task_type))

    # --- kernel / neighbour / neural experts (Amendment-1 style) ---
    if reg:
        def krr(scaled):
            def f(p):
                g = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                                 "gamma": list(np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]) * (1.0 / 132 if scaled else 1.0))},
                                 cv=3, scoring="neg_mean_absolute_error", n_jobs=4)
                return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), g) if scaled else g, transformer=StandardScaler())
            return f
        E.append(Expert2("krr_frac", "frac", krr(False), task_type))
        E.append(Expert2("krr_magpie", "magpie", krr(True), task_type))
        E.append(Expert2("knn_frac", "frac", lambda p: GridSearchCV(KNeighborsRegressor(weights="distance", metric="manhattan"),
                                                                   {"n_neighbors": [1, 2, 3, 5, 8, 13, 21]}, cv=3,
                                                                   scoring="neg_mean_absolute_error"), task_type))
        E.append(Expert2("mlp_F1", "F1", lambda p: TransformedTargetRegressor(regressor=make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), QuantileTransformer(n_quantiles=200, output_distribution="normal"),
            MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=600, early_stopping=True, n_iter_no_change=25,
                         random_state=seed)), transformer=StandardScaler()), task_type))
    else:
        E.append(Expert2("svc_frac", "frac", lambda p: SVC(kernel="rbf", C=10.0, gamma=1.0, probability=True, random_state=seed), task_type))
        E.append(Expert2("svc_magpie", "magpie", lambda p: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma=1.0 / 132,
                                                                                             probability=True, random_state=seed)), task_type))
        E.append(Expert2("knn_frac", "frac", lambda p: GridSearchCV(KNeighborsClassifier(weights="distance", metric="manhattan"),
                                                                   {"n_neighbors": [3, 5, 8, 13, 21, 34]}, cv=3, scoring="roc_auc"), task_type))
        E.append(Expert2("mlp_F1", "F1", lambda p: make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                                                 QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                                                                 MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=600,
                                                                               early_stopping=True, n_iter_no_change=25, random_state=seed)), task_type))
    if cfg.get("crabnet"):
        from .crabnet_expert import CrabNetExpert
        E.append(Expert2("crabnet", "comp", lambda p: CrabNetExpert(task_type, seed=seed, **cfg.get("crabnet_kw", {})), task_type))
    return E


# ---------------------------------------------------------------- Optuna search spaces
def suggest(kind, trial, task_type, small):
    reg = task_type == "regression"
    if kind == "lgbm":
        p = dict(num_leaves=trial.suggest_int("num_leaves", 4, 96, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 n_estimators=trial.suggest_int("n_estimators", 300, 1500, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 2, 40, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.05, 0.6),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        if reg:
            p["objective"] = trial.suggest_categorical("objective", ["l1", "l2", "huber"])
        return p
    if kind == "xgb":
        p = dict(max_depth=trial.suggest_int("max_depth", 2, 10),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 n_estimators=trial.suggest_int("n_estimators", 300, 2500, log=True),
                 min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.05, 0.6),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        if reg:
            p["l1"] = trial.suggest_categorical("l1", [True, False])
        return p
    raise ValueError(kind)


def _lgbm_default(task_type, small):
    d = dict(num_leaves=31, learning_rate=0.03, n_estimators=1000, min_child_samples=5 if small else 10,
             colsample_bytree=0.3, subsample=0.8, reg_lambda=1.0)
    if task_type == "regression":
        d["objective"] = "l1" if small else "l2"
    return d


def _xgb_default(task_type, small):
    d = dict(max_depth=6, learning_rate=0.03, n_estimators=800, min_child_weight=1.0, colsample_bytree=0.3,
             subsample=0.8, reg_lambda=1.0)
    if task_type == "regression":
        d["l1"] = bool(small)
    return d


DEFAULTS = {"lgbm": _lgbm_default, "xgb": _xgb_default}
