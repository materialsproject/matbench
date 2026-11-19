"""Cross-fitted stacking + novelty-damped local OOF residual correction.

One call of `run_split(...)` handles one (train, test) split:
  1. cross-fit every expert with K inner folds on the TRAINING rows -> OOF predictions;
     refit every expert on all training rows -> test predictions;
  2. simplex-constrained global stack (non-negative weights summing to 1), fitted on OOF
     predictions; a cross-fitted copy (weights fitted without the row's inner fold) gives
     OOF stack predictions for the training rows;
  3. local correction: kernel-weighted average of neighbouring training rows' OOF stack
     residuals, damped by a pseudo-count kappa (shrinks to 0 for novel queries), scaled by alpha.
     (k, bandwidth, kappa, alpha, kernel space) are chosen on the TRAINING rows only
     (leave-self-out neighbour sets over OOF residuals).
Nothing in this module ever sees test targets; `run_split` receives only X_test.
"""
from __future__ import annotations
import itertools, json, pathlib, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error

from .experts import make_experts, cols

EPS = 1e-6


# ---------------------------------------------------------------- metrics
def score(task_type, y, p):
    if task_type == "regression":
        return mean_absolute_error(y, p)
    return roc_auc_score(np.asarray(y).astype(int), p)


def better(task_type, a, b):
    """True if score a is strictly better than b."""
    return a < b if task_type == "regression" else a > b


# ---------------------------------------------------------------- stage 1: cross-fitting
def inner_splits(task_type, y, n_inner, seed):
    if task_type == "regression":
        kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
        return list(kf.split(np.zeros(len(y))))
    kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(len(y)), np.asarray(y).astype(int)))


def crossfit_experts(task_type, X_tr, y_tr, X_te, splits, cache_dir: pathlib.Path, cfg=None, seed=0, log=print,
                     fold_matched=False):
    """Returns (oof [n_tr, E], test [n_te, E], names, fm) and caches one npz per expert (resumable).
    With fold_matched=True also returns fm = {"test_inner": [E, J, n_te] predictions of each inner-fold
    model on the test rows, "pair": [E, J, J, n_tr] predictions of the pair-excluded refits (trained on
    T minus (P_m u P_m')) on the rows of P_m u P_m'}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    experts = make_experts(task_type, len(y_tr), seed=seed, cfg=cfg)
    ex_names = [e.name for e in experts]
    J = len(splits)
    fold_of = np.empty(len(y_tr), dtype=int)
    for m, (_, va_i) in enumerate(splits):
        fold_of[va_i] = m
    names, oofs, tests, tinner, pairs = [], [], [], [], []
    for ex in experts:
        f = cache_dir / f"{ex.name}.npz"
        if f.exists():
            z = np.load(f)
            oof, te = z["oof"], z["test"]
            ti = z["test_inner"] if "test_inner" in z.files else None
        else:
            t0 = time.time()
            oof = np.full(len(y_tr), np.nan)
            ti = np.full((J, len(X_te)), np.nan)
            for m, (tr_i, va_i) in enumerate(splits):
                mdl = make_experts(task_type, len(tr_i), seed=seed, cfg=cfg)[ex_names.index(ex.name)]
                mdl.fit(X_tr.iloc[tr_i], y_tr[tr_i])
                oof[va_i] = mdl.predict(X_tr.iloc[va_i])
                ti[m] = mdl.predict(X_te)
            ex.fit(X_tr, y_tr)
            te = ex.predict(X_te)
            np.savez(f, oof=oof, test=te, test_inner=ti, seconds=time.time() - t0)
            log(f"    expert {ex.name}: {time.time() - t0:.1f}s, OOF score {score(task_type, y_tr, oof):.4f}")
        names.append(ex.name); oofs.append(oof); tests.append(te); tinner.append(ti)
        if fold_matched:
            fp = cache_dir / f"{ex.name}_pairs.npz"
            if fp.exists():
                pr = np.load(fp)["pair"]
            else:
                t0 = time.time()
                pr = np.full((J, J, len(y_tr)), np.nan)
                for m in range(J):
                    for mp in range(m + 1, J):
                        held = (fold_of == m) | (fold_of == mp)
                        tr_i, ho_i = np.where(~held)[0], np.where(held)[0]
                        mdl = make_experts(task_type, len(tr_i), seed=seed, cfg=cfg)[ex_names.index(ex.name)]
                        mdl.fit(X_tr.iloc[tr_i], y_tr[tr_i])
                        pv = mdl.predict(X_tr.iloc[ho_i])
                        pr[m, mp, ho_i] = pv
                        pr[mp, m, ho_i] = pv
                np.savez(fp, pair=pr, seconds=time.time() - t0)
                log(f"    expert {ex.name}: pair-excluded refits {time.time() - t0:.1f}s")
            pairs.append(pr)
    fm = None
    if fold_matched:
        if any(t is None for t in tinner):
            raise RuntimeError("cached expert lacks test_inner; delete the cache to recompute")
        fm = {"test_inner": np.stack(tinner), "pair": np.stack(pairs), "fold_of": fold_of}
    return np.column_stack(oofs), np.column_stack(tests), names, fm


# ---------------------------------------------------------------- stage 2: simplex stack
def simplex_weights(P, y, task_type, loss="l2"):
    """Non-negative weights summing to 1 minimising MSE (regression) or Brier score (classification;
    which is also an MSE on 0/1 targets). loss="l1" (regression only): minimise mean absolute error
    by linear programming."""
    y = np.asarray(y, dtype=float)
    E = P.shape[1]
    if loss == "l1" and task_type == "regression":
        from scipy.optimize import linprog
        from scipy.sparse import hstack, vstack, identity, csr_matrix
        n = len(y)
        c = np.concatenate([np.zeros(E), np.ones(n) / n])
        Ps, I = csr_matrix(P), identity(n, format="csr")
        A_ub = vstack([hstack([Ps, -I]), hstack([-Ps, -I])])
        b_ub = np.concatenate([y, -y])
        A_eq = csr_matrix(np.concatenate([np.ones(E), np.zeros(n)])[None, :])
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                    bounds=[(0, 1)] * E + [(0, None)] * n, method="highs")
        w = np.clip(r.x[:E], 0, None)
        return w / w.sum()
    w0 = np.full(E, 1.0 / E)
    fun = lambda w: np.mean((P @ w - y) ** 2)
    jac = lambda w: 2.0 * P.T @ (P @ w - y) / len(y)
    res = minimize(fun, w0, jac=jac, method="SLSQP", bounds=[(0.0, 1.0)] * E,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(E)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def crossfitted_stack(P_oof, y, splits, task_type, loss="l2"):
    """OOF stack predictions for training rows: weights for inner fold f fitted on rows not in f."""
    s = np.full(len(y), np.nan)
    for tr_i, va_i in splits:
        w = simplex_weights(P_oof[tr_i], y[tr_i], task_type, loss)
        s[va_i] = P_oof[va_i] @ w
    return s


# ---------------------------------------------------------------- stage 3: local correction
class Space:
    """A composition space for neighbour search: standardised Magpie (Euclidean) or element fractions (L1)."""
    def __init__(self, name):
        self.name = name

    def fit(self, X):
        if self.name == "magpie":
            self.sc = StandardScaler().fit(X[cols(X, "magpie")].to_numpy())
        return self

    def transform(self, X):
        if self.name == "magpie":
            return self.sc.transform(X[cols(X, "magpie")].to_numpy())
        return X[cols(X, "frac")].to_numpy()

    @property
    def metric(self):
        return "euclidean" if self.name == "magpie" else "manhattan"


def neighbour_table(space: Space, X_tr, X_q, k_max, exclude_self):
    Z_tr = space.transform(X_tr)
    nn = NearestNeighbors(n_neighbors=min(k_max + int(exclude_self), len(Z_tr)), metric=space.metric).fit(Z_tr)
    d, idx = nn.kneighbors(space.transform(X_q))
    if exclude_self:
        # drop the query itself (the first column is the row itself, distance 0); keep exact-duplicate
        # compositions of OTHER rows, since a random test split sees those too.
        own = np.arange(len(X_q))[:, None]
        keep = idx != own
        d = np.array([r[k][:k_max] for r, k in zip(d, keep)])
        idx = np.array([r[k][:k_max] for r, k in zip(idx, keep)])
    return d, idx


def correction(d, T, k, h, kappa):
    """c(q) = sum_j w_j T_qj / (sum_j w_j + kappa), w_j = exp(-(d_j/h)^2) over the k nearest.
    T_qj is the transferred quantity for neighbour j of query q (naive: j's OOF residual r_j;
    fold-matched: r_j minus the absorption term, see fm_transfer_*). kappa > 0 damps the
    correction toward 0 when all neighbours are far (novel query)."""
    dk, Tk = d[:, :k], T[:, :k]
    w = np.exp(-(dk / h) ** 2)
    den = w.sum(1) + kappa
    num = (w * Tk).sum(1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-300)


def naive_T(resid):
    return lambda sp_name, idx, query_rows=None: resid[idx]


def fm_train_T(y, oof_stack, pair_stack, fold_of):
    """Deployment-matched transfer for training row i (as a manufactured query) and neighbour j:
    same inner fold -> r_j (that model saw neither i nor j); other fold m' -> y_j - F^(-m,-m')(x_j)
    + F^(-m,-m')(x_i) - F^(-m)(x_i), using the pair-excluded refit that saw neither i nor j."""
    r = y - oof_stack
    def T(sp_name, idx, query_rows=None):
        n = idx.shape[0]
        qi = np.arange(n)[:, None] if query_rows is None else np.asarray(query_rows)[:, None]
        mi, mj = fold_of[qi], fold_of[idx]
        same = mi == mj
        pj = pair_stack[mi, mj, idx]
        pi = pair_stack[mi, mj, np.broadcast_to(qi, idx.shape)]
        other = y[idx] - pj + pi - oof_stack[qi]
        return np.where(same, r[idx], np.nan_to_num(other))
    return T


def fm_test_T(y, oof_stack, test_inner_stack, base_te, fold_of):
    """Deployment transfer for test query q and neighbour j: r_j + F^(-m(j))(x_q) - F^full(x_q)."""
    r = y - oof_stack
    def T(sp_name, idx, query_rows=None):
        q = np.arange(idx.shape[0])[:, None]
        return r[idx] + test_inner_stack[fold_of[idx], q] - base_te[q]
    return T


def apply_corr(task_type, base, c, alpha):
    if task_type == "regression":
        return base + alpha * c
    return expit(logit(np.clip(base, EPS, 1 - EPS)) + alpha * c)


GRID = {
    "k": [5, 10, 20, 40],
    "hmult": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "kappa": [0.0, 0.1, 0.5, 2.0],
    "alpha_reg": [0.0, 0.25, 0.5, 0.75, 1.0],
    "alpha_clf": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
}


def select_correction(task_type, X_tr, y_tr, base_oof, spaces=("magpie", "frac"), grid=GRID, T_fn=None):
    """Choose (space, k, h, kappa, alpha) by the training-row score of the corrected OOF prediction,
    with leave-self-out neighbour sets. alpha = 0 (no correction) is always a candidate; ties go to it."""
    T_fn = T_fn or naive_T(np.asarray(y_tr, dtype=float) - base_oof)
    alphas = grid["alpha_reg"] if task_type == "regression" else grid["alpha_clf"]
    best = {"space": None, "k": 0, "hmult": 0.0, "kappa": 0.0, "alpha": 0.0, "h": 0.0,
            "score": score(task_type, y_tr, base_oof)}
    base_score = best["score"]
    kmax = max(grid["k"])
    for sp_name in spaces:
        sp = Space(sp_name).fit(X_tr)
        d, idx = neighbour_table(sp, X_tr, X_tr, kmax, exclude_self=True)
        T = T_fn(sp_name, idx)
        scale = float(np.median(d[:, 0][d[:, 0] > 0])) if np.any(d[:, 0] > 0) else 1.0
        for k, hm, kap in itertools.product(grid["k"], grid["hmult"], grid["kappa"]):
            c = correction(d, T, k, hm * scale, kap)
            for a in alphas:
                if a == 0:
                    continue
                s = score(task_type, y_tr, apply_corr(task_type, base_oof, c, a))
                if better(task_type, s, best["score"]):
                    best = {"space": sp_name, "k": k, "hmult": hm, "kappa": kap, "alpha": a,
                            "h": hm * scale, "score": s}
    best["base_score"] = base_score
    return best


def corrected_oof(task_type, X_tr, y_tr, base_oof, sel, T_fn=None):
    """Training-row (leave-self-out) corrected OOF predictions under a selected configuration."""
    if sel["alpha"] == 0 or sel["space"] is None:
        return base_oof.copy()
    T_fn = T_fn or naive_T(np.asarray(y_tr, dtype=float) - base_oof)
    sp = Space(sel["space"]).fit(X_tr)
    d, idx = neighbour_table(sp, X_tr, X_tr, sel["k"], exclude_self=True)
    return apply_corr(task_type, base_oof, correction(d, T_fn(sel["space"], idx), sel["k"], sel["h"], sel["kappa"]),
                      sel["alpha"])


def balanced_threshold(y, p):
    """Threshold maximising balanced accuracy on training OOF predictions (for hard-label metrics)."""
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y).astype(int)
    cands = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    scores = [balanced_accuracy_score(y, p >= t) for t in cands]
    best = max(scores)
    # among ties take the one closest to 0.5
    ties = [t for t, sc in zip(cands, scores) if sc == best]
    return float(min(ties, key=lambda t: abs(t - 0.5)))


def predict_correction(task_type, X_tr, y_tr, base_oof, X_te, base_te, sel, T_fn=None):
    if sel["alpha"] == 0 or sel["space"] is None:
        return base_te.copy()
    T_fn = T_fn or naive_T(np.asarray(y_tr, dtype=float) - base_oof)
    sp = Space(sel["space"]).fit(X_tr)
    d, idx = neighbour_table(sp, X_tr, X_te, sel["k"], exclude_self=False)
    c = correction(d, T_fn(sel["space"], idx), sel["k"], sel["h"], sel["kappa"])
    return apply_corr(task_type, base_te, c, sel["alpha"])


# ---------------------------------------------------------------- one split, all arms
def run_split(task_type, X_tr, y_tr, X_te, cache_dir, cfg=None, n_inner=5, seed=0, log=print):
    """Returns dict of test predictions per arm + selection metadata. Never receives y_test."""
    y_tr = np.asarray(y_tr, dtype=float)
    splits = inner_splits(task_type, y_tr, n_inner, seed)
    fm_on = bool((cfg or {}).get("fold_matched"))
    P_oof, P_te, names, fm = crossfit_experts(task_type, X_tr, y_tr, X_te, splits, pathlib.Path(cache_dir), cfg, seed, log,
                                              fold_matched=fm_on)
    out, meta = {}, {"experts": names}
    # single experts
    oof_scores = {}
    for j, n in enumerate(names):
        out[f"single:{n}"] = P_te[:, j]
        oof_scores[n] = score(task_type, y_tr, P_oof[:, j])
    meta["expert_oof_scores"] = oof_scores
    # best single expert selected by inner (OOF) score
    best_j = min(range(len(names)), key=lambda j: oof_scores[names[j]]) if task_type == "regression" \
        else max(range(len(names)), key=lambda j: oof_scores[names[j]])
    meta["best_single"] = names[best_j]
    out["best_single"] = P_te[:, best_j]
    # equal-weight mean (simple ensemble comparator)
    out["mean_ensemble"] = P_te.mean(1)
    # global simplex stack
    loss = (cfg or {}).get("stack_loss", "l2")
    w = simplex_weights(P_oof, y_tr, task_type, loss)
    meta["stack_loss"] = loss
    meta["stack_weights"] = dict(zip(names, map(float, w)))
    s_te = P_te @ w
    s_oof = crossfitted_stack(P_oof, y_tr, splits, task_type, loss)
    meta["stack_oof_score"] = score(task_type, y_tr, s_oof)
    out["stack"] = s_te
    # full method: stack + local correction (space chosen by inner CV)
    sel = select_correction(task_type, X_tr, y_tr, s_oof)
    meta["sel_stack_local"] = sel
    out["stack_local"] = predict_correction(task_type, X_tr, y_tr, s_oof, X_te, s_te, sel)
    # kernel-choice ablations (space fixed)
    for sp in ("magpie", "frac"):
        sel_sp = select_correction(task_type, X_tr, y_tr, s_oof, spaces=(sp,))
        meta[f"sel_stack_local_{sp}"] = sel_sp
        out[f"stack_local_{sp}"] = predict_correction(task_type, X_tr, y_tr, s_oof, X_te, s_te, sel_sp)
    # fold-matched correction (IdeaSpark candidate): global stack weights w for every stacked quantity
    if fm is not None:
        y_ = y_tr
        oof_w = P_oof @ w
        pair_w = np.einsum("e,eabn->abn", w, np.nan_to_num(fm["pair"]))
        ti_w = np.einsum("e,ejn->jn", w, fm["test_inner"])
        Ttr = fm_train_T(y_, oof_w, pair_w, fm["fold_of"])
        Tte = fm_test_T(y_, oof_w, ti_w, s_te, fm["fold_of"])
        sel_fm = select_correction(task_type, X_tr, y_tr, oof_w, T_fn=Ttr)
        meta["sel_stack_fm"] = sel_fm
        out["stack_fm"] = predict_correction(task_type, X_tr, y_tr, oof_w, X_te, s_te, sel_fm, T_fn=Tte)
        meta["fm_oof_corrected_score"] = sel_fm["score"]
    # label-permutation control (selection only; predictions untouched): permute the training labels
    # within each inner fold, recompute residuals against the SAME base predictions, and re-select.
    # A correction that exploits real local structure should be switched off (alpha = 0) here.
    rng = np.random.default_rng(seed + 991)
    y_perm = y_tr.copy()
    for _, va_i in splits:
        y_perm[va_i] = y_tr[rng.permutation(va_i)]
    meta["perm_control_naive"] = select_correction(task_type, X_tr, y_perm, s_oof)
    if fm is not None:
        meta["perm_control_fm"] = select_correction(task_type, X_tr, y_perm, oof_w,
                                                    T_fn=fm_train_T(y_perm, oof_w, pair_w, fm["fold_of"]))
    # correction without stacking: on the inner-CV-selected best single expert
    b_oof = P_oof[:, best_j]
    sel_b = select_correction(task_type, X_tr, y_tr, b_oof)
    meta["sel_single_local"] = sel_b
    out["single_local"] = predict_correction(task_type, X_tr, y_tr, b_oof, X_te, P_te[:, best_j], sel_b)
    if task_type != "regression":
        for k in out:
            out[k] = np.clip(out[k], 0.0, 1.0)
        # thresholds for hard-label metrics, chosen on training OOF predictions only
        meta["thresholds"] = {
            "stack": balanced_threshold(y_tr, s_oof),
            "stack_local": balanced_threshold(y_tr, corrected_oof(task_type, X_tr, y_tr, s_oof, sel)),
            "best_single": balanced_threshold(y_tr, b_oof),
            "single_local": balanced_threshold(y_tr, corrected_oof(task_type, X_tr, y_tr, b_oof, sel_b)),
        }
        if fm is not None:
            meta["thresholds"]["stack_fm"] = balanced_threshold(y_tr, corrected_oof(task_type, X_tr, y_tr, oof_w, sel_fm, T_fn=Ttr))
    return out, meta
