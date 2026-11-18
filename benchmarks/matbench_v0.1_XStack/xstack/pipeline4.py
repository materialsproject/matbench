"""Amendment-4 pipeline for one (train, test) split (ANALYSIS_PLAN.md Section 14).

`run_split4` is `pipeline2.run_split2` line for line, with ONE change: the expert pool comes from
`experts4.make_experts4` (medians of the `extra` / CBFV Oliynyk cells computed inside each fit from that fit's
training rows) instead of `experts2.make_experts2`. Tuning, cross-fitting, the simplex stack, the nested-CV-gated
local correction and the thresholds are the unchanged Amendment-2 functions. Uses only training rows/targets.
"""
from __future__ import annotations
import pathlib
import numpy as np

from .pipeline import simplex_weights, crossfitted_stack, predict_correction, corrected_oof, balanced_threshold, score, inner_splits
from .pipeline2 import gated_correction, tune_expert, crossfit2
from .experts4 import make_experts4


def run_split4(task_type, X_tr, y_tr, X_te, cache_dir, cfg=None, n_inner=5, seed=0, log=print):
    cfg = cfg or {}
    cache = pathlib.Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    y_tr = np.asarray(y_tr, dtype=float)
    splits = inner_splits(task_type, y_tr, n_inner, seed)
    experts = make_experts4(task_type, len(y_tr), seed=seed, cfg=cfg)
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
