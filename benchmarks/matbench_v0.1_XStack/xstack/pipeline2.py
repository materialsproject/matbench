"""Amendment-2 pipeline for one (train, test) split. Uses only training rows/targets.

1. inner splits (J=5, seed 0) of T_k;
2. Optuna tuning of the tunable experts (LightGBM, XGBoost) on T_k with those inner splits (objective:
   mean inner-validation MAE / ROC-AUC). The tuned parameters are then used for every inner fit and the
   full refit. (Disclosed: the OOF predictions of tuned experts are mildly optimistic, which can only
   affect the level-1 stack weights, never the test fold.)
3. cross-fitting of all experts -> OOF and test predictions (cached per expert, resumable);
4. simplex stack (L1 for regression / Brier for classification) + cross-fitted OOF stack;
5. optional naive local correction selected by inner CV (alpha = 0 allowed) and kept only if a NESTED
   inner-CV check confirms it (gated_correction) -> FINAL arm `stack_local`;
6. balanced-accuracy thresholds from training OOF predictions (classification).
"""
from __future__ import annotations
import json, pathlib, time
import numpy as np

from .pipeline import (inner_splits, simplex_weights, crossfitted_stack, select_correction, predict_correction,
                       corrected_oof, balanced_threshold, score, better)
from .experts2 import make_experts2, suggest, DEFAULTS


def gated_correction(task_type, X_tr, y_tr, base_oof, splits):
    """Select the local correction on all training rows, then check it by NESTED inner CV: for each inner
    fold f the correction is re-selected on the other folds only and applied to f. The correction is kept
    only if this honest corrected OOF score beats the uncorrected OOF score; otherwise alpha = 0."""
    sel = select_correction(task_type, X_tr, y_tr, base_oof)
    info = {"selected_alpha": sel["alpha"], "uncorrected_oof": float(score(task_type, y_tr, base_oof))}
    if sel["alpha"] == 0:
        info.update(honest_oof=info["uncorrected_oof"], kept=False)
        return sel, info
    honest = base_oof.copy()
    for tr_i, va_i in splits:
        sf = select_correction(task_type, X_tr.iloc[tr_i], y_tr[tr_i], base_oof[tr_i])
        honest[va_i] = predict_correction(task_type, X_tr.iloc[tr_i], y_tr[tr_i], base_oof[tr_i], X_tr.iloc[va_i],
                                          base_oof[va_i], sf)
    h = float(score(task_type, y_tr, honest))
    keep = better(task_type, h, info["uncorrected_oof"])
    info.update(honest_oof=h, kept=bool(keep))
    if not keep:
        sel = dict(sel, alpha=0.0, space=None)
    return sel, info


def tune_expert(ex, task_type, X, y, splits, n_trials, cache: pathlib.Path, seed=0, log=print):
    f = cache / f"{ex.name}_params.json"
    if f.exists():
        return json.loads(f.read_text())
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    small = len(y) < 1000

    def objective(trial):
        p = suggest(ex.tunable, trial, task_type, small)
        vals = []
        for tr_i, va_i in splits:
            e = ex.clone(); e.params = dict(p)
            e.fit(X.iloc[tr_i], y[tr_i])
            vals.append(score(task_type, y[va_i], e.predict(X.iloc[va_i])))
        return float(np.mean(vals))
    t0 = time.time()
    st = optuna.create_study(direction="minimize" if task_type == "regression" else "maximize",
                             sampler=optuna.samplers.TPESampler(seed=seed))
    # the expert's default parameters are always evaluated first, so tuning can only improve on them
    st.enqueue_trial(DEFAULTS[ex.tunable](task_type, small))
    st.optimize(objective, n_trials=n_trials)
    best = st.best_params
    f.write_text(json.dumps({"params": best, "value": st.best_value, "n_trials": n_trials,
                             "seconds": time.time() - t0}, indent=1))
    log(f"    tuned {ex.name}: {st.best_value:.4f} in {time.time()-t0:.0f}s ({n_trials} trials)")
    return json.loads(f.read_text())


def crossfit2(task_type, experts, X_tr, y_tr, X_te, splits, cache: pathlib.Path, log=print):
    names, oofs, tests, secs = [], [], [], {}
    for ex in experts:
        f = cache / f"{ex.name}.npz"
        if f.exists():
            z = np.load(f); oof, te = z["oof"], z["test"]; secs[ex.name] = float(z["seconds"])
        else:
            t0 = time.time()
            oof = np.full(len(y_tr), np.nan)
            for tr_i, va_i in splits:
                e = ex.clone()
                e.fit(X_tr.iloc[tr_i], y_tr[tr_i])
                oof[va_i] = e.predict(X_tr.iloc[va_i])
            e = ex.clone(); e.fit(X_tr, y_tr); te = e.predict(X_te)
            secs[ex.name] = time.time() - t0
            np.savez(f, oof=oof, test=te, seconds=secs[ex.name])
            log(f"    expert {ex.name}: {secs[ex.name]:.0f}s, OOF {score(task_type, y_tr, oof):.4f}")
        names.append(ex.name); oofs.append(oof); tests.append(te)
    return np.column_stack(oofs), np.column_stack(tests), names, secs


def run_split2(task_type, X_tr, y_tr, X_te, cache_dir, cfg=None, n_inner=5, seed=0, log=print):
    cfg = cfg or {}
    cache = pathlib.Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    y_tr = np.asarray(y_tr, dtype=float)
    splits = inner_splits(task_type, y_tr, n_inner, seed)
    experts = make_experts2(task_type, len(y_tr), seed=seed, cfg=cfg)
    drop = set(cfg.get("drop_experts", []))
    experts = [e for e in experts if e.name not in drop]
    tuned = {}
    for ex in experts:
        if ex.tunable:
            r = tune_expert(ex, task_type, X_tr, y_tr, splits, cfg.get("n_trials", 30), cache, seed, log)
            ex.params = r["params"]; tuned[ex.name] = r
    P_oof, P_te, names, secs = crossfit2(task_type, experts, X_tr, y_tr, X_te, splits, cache, log)
    out, meta = {}, {"experts": names, "expert_seconds": secs, "tuned": tuned}
    oof_scores = {n: float(score(task_type, y_tr, P_oof[:, j])) for j, n in enumerate(names)}
    meta["expert_oof_scores"] = oof_scores
    for j, n in enumerate(names):
        out[f"single:{n}"] = P_te[:, j]
    pick = min if task_type == "regression" else max
    best_j = names.index(pick(names, key=lambda n: oof_scores[n]))
    meta["best_single"] = names[best_j]
    out["best_single"] = P_te[:, best_j]
    out["mean_ensemble"] = P_te.mean(1)
    loss = cfg.get("stack_loss", "l1" if task_type == "regression" else "l2")
    w = simplex_weights(P_oof, y_tr, task_type, loss)
    meta["stack_loss"] = loss
    meta["stack_weights"] = dict(zip(names, map(float, w)))
    s_te = P_te @ w
    s_oof = crossfitted_stack(P_oof, y_tr, splits, task_type, loss)
    meta["stack_oof_score"] = float(score(task_type, y_tr, s_oof))
    out["stack"] = s_te
    sel, gate = gated_correction(task_type, X_tr, y_tr, s_oof, splits)
    meta["sel_stack_local"] = sel
    meta["gate_stack_local"] = gate
    out["stack_local"] = predict_correction(task_type, X_tr, y_tr, s_oof, X_te, s_te, sel)
    b_oof = P_oof[:, best_j]
    sel_b, gate_b = gated_correction(task_type, X_tr, y_tr, b_oof, splits)
    meta["sel_single_local"] = sel_b
    meta["gate_single_local"] = gate_b
    out["single_local"] = predict_correction(task_type, X_tr, y_tr, b_oof, X_te, P_te[:, best_j], sel_b)
    if task_type != "regression":
        for k in out:
            out[k] = np.clip(out[k], 0.0, 1.0)
        meta["thresholds"] = {
            "stack": balanced_threshold(y_tr, s_oof),
            "stack_local": balanced_threshold(y_tr, corrected_oof(task_type, X_tr, y_tr, s_oof, sel)),
            "best_single": balanced_threshold(y_tr, b_oof),
            "single_local": balanced_threshold(y_tr, corrected_oof(task_type, X_tr, y_tr, b_oof, sel_b)),
        }
    return out, meta
