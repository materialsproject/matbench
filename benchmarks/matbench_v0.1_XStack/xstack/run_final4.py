"""AMENDMENT-4 FINAL EVALUATION on the official Matbench v0.1 folds (ANALYSIS_PLAN.md Section 14).

Amendment 3 (src/run_final3.py) with the two changes requested in the matbench PR #372 review of 2026-11-16:
no DFT block in any candidate, and median imputation computed from training rows only in every split, inner CV
included. Per task and official fold k, using the official training rows T_k only:
  1. for every candidate block set (src.frozen_config4.CAND_BLOCKS: F_min, F_all) the FULL expert pool is tuned and
     cross-fitted on T_k (pipeline2.tune_expert / crossfit2 with the experts4 pool; J = 5, seed 0, 20 Optuna trials);
  2. every candidate = (block set, expert subset) is scored by the cross-fitted OOF score of the uncorrected simplex
     stack on T_k; ties -> earlier (simpler) candidate;
  3. the chosen candidate is handed to pipeline4.run_split4 (stack, gated local correction, thresholds) and the test
     rows of fold k are predicted. Test TARGETS are never loaded here.

Every expert is fitted from scratch in this run: NO cache of an earlier amendment is read. Inside one fold, an
expert that reads exactly the same input columns under both block sets (the experts on F1, Magpie and
element-fraction columns) is fitted once, for F_min, and its cache file is copied to F_all (same name, same columns,
same rows, same code and seeds -> the same computation); recorded in <cache dir>/cache_provenance.json.

Usage:
  python -m src.run_final4 select <task> --fold k      # steps 1-3 for one fold
  python -m src.run_final4 smoke <task>                # dry run on fold-0 TRAINING rows only, tiny budget
"""
from __future__ import annotations
import json, pathlib, shutil, socket, sys, time
import numpy as np

from .data import load_task_frames, ROOT
from .features4 import features4
from .experts2 import group_cols
from .experts4 import make_experts4
from .pipeline import inner_splits, crossfitted_stack, score, better
from .pipeline2 import tune_expert, crossfit2
from .pipeline4 import run_split4
from .design import TASK_TYPE
from .frozen_config4 import SEED, N_INNER, BLOCKSETS, CAND_BLOCKS, SUBSETS, pool_cfg, final_cfg, candidate_name
from .run_final3 import cpu_model, host_info, cache_dir, open_log   # host/CPU description and log helpers only

OUT = ROOT / "results" / "final_a4"


def build_X(task, df, bs):
    X = features4(task, df, BLOCKSETS[bs])
    X["composition"] = df["composition"].values
    return X


def expert_cols(X, ex):
    return list(group_cols(X, ex.group))


def prepare_cache(k, bs, X_by_bs, experts, out_root, log, n_tr, n_te):
    """Copy an expert's cache from the OTHER candidate dir of this fold and this run if the expert has the same name
    and reads exactly the same input columns (names and order). Nothing from earlier amendments is read."""
    dst = cache_dir(out_root, k, bs)
    dst.mkdir(parents=True, exist_ok=True)
    pf = dst / "cache_provenance.json"
    prov = json.loads(pf.read_text()) if pf.exists() else {}
    for ex in experts:
        if (dst / f"{ex.name}.npz").exists():
            continue
        cols = expert_cols(X_by_bs[bs], ex)
        for b in X_by_bs:
            sdir = cache_dir(out_root, k, b)
            f = sdir / f"{ex.name}.npz"
            if b == bs or not f.exists() or expert_cols(X_by_bs[b], ex) != cols:
                continue
            z = np.load(f)
            assert len(z["oof"]) == n_tr and len(z["test"]) == n_te and bool(np.isfinite(z["oof"]).all())
            shutil.copy2(f, dst / f.name)
            p = sdir / f"{ex.name}_params.json"
            if p.exists():
                shutil.copy2(p, dst / p.name)
            prov[ex.name] = f"copied from A4:{b} of this fold: identical inputs ({len(cols)} columns)"
            log(f"    cache {bs}/{ex.name}: {prov[ex.name]}")
            break
    pf.write_text(json.dumps(prov, indent=1))
    return prov


def fit_candidate(task, k, bs, tr, te, y_tr, X_by_bs, out_root, log, n_trials=None):
    """Step 1 for one (fold, block set): tune + cross-fit the full expert pool on T_k (resumable, cached)."""
    tt = TASK_TYPE[task]
    cfg = pool_cfg(task)
    if n_trials is not None:
        cfg["n_trials"] = n_trials
    X = X_by_bs[bs]
    X_tr, X_te = X.loc[tr], X.loc[te]
    splits = inner_splits(tt, y_tr, N_INNER, SEED)
    experts = make_experts4(tt, len(y_tr), seed=SEED, cfg=cfg)
    dst = cache_dir(out_root, k, bs)
    prov = prepare_cache(k, bs, X_by_bs, experts, out_root, log, len(tr), len(te))
    t0 = time.time()
    fitted_here = [e.name for e in experts if not (dst / f"{e.name}.npz").exists()]
    log(f"  [{task}] fold {k} {bs}: ncols={X.shape[1] - 1}; NaN cells in train rows of the imputed blocks="
        f"{int(X_tr[[c for c in X.columns if c.startswith(('extra_', 'cbfv_oliynyk_'))]].isna().sum().sum())}; "
        f"to fit here: {fitted_here or 'nothing'}")
    tuned = {}
    for ex in experts:
        if ex.tunable:
            r = tune_expert(ex, tt, X_tr, y_tr, splits, cfg["n_trials"], dst, SEED, log)
            ex.params = r["params"]; tuned[ex.name] = r
    P_oof, P_te, names, secs = crossfit2(tt, experts, X_tr, y_tr, X_te, splits, dst, log)
    for n in fitted_here:
        prov[n] = f"fitted here ({socket.gethostname()}, {cpu_model()})"
    (dst / "cache_provenance.json").write_text(json.dumps(prov, indent=1))
    info_f = dst / "fit_info.json"
    info = json.loads(info_f.read_text()) if info_f.exists() else {"runs": []}
    info["runs"].append({**host_info(), "fitted_here": fitted_here, "wall_s": round(time.time() - t0, 1),
                         "n_trials": cfg["n_trials"], "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    info_f.write_text(json.dumps(info, indent=1))
    return P_oof, names, secs, splits, tuned


def select_and_predict(task, k, df, folds, out_root, log, n_trials=None):
    tt = TASK_TYPE[task]
    pf = out_root / f"fold{k}_predictions.npz"
    if pf.exists():
        log(f"[{task}] fold {k}: already done, skipping"); return
    t0 = time.time()
    tr, te = folds[k]["train"], folds[k]["test"]
    assert not (set(tr) & set(te))
    y_tr = df.loc[tr, "target"].to_numpy(dtype=float)        # training targets only
    cands = list(CAND_BLOCKS[task])
    X_by_bs = {b: build_X(task, df, b) for b in cands}
    assert not any(c.startswith("dft_") for X in X_by_bs.values() for c in X.columns)
    log(f"[{task}] fold {k}: n_train={len(tr)} n_test={len(te)}; block sets {cands}; subsets {len(SUBSETS[task])}")
    rows, best = [], None
    loss = "l1" if tt == "regression" else "l2"
    for bs in cands:
        P_oof, names, secs, splits, _ = fit_candidate(task, k, bs, tr, te, y_tr, X_by_bs, out_root, log, n_trials=n_trials)
        for sub in SUBSETS[task]:
            keep = [j for j, n in enumerate(names) if not (n.startswith("hurdle") and not sub["hurdle"])
                    and n not in sub["drop_experts"]]
            s_oof = crossfitted_stack(P_oof[:, keep], y_tr, splits, tt, loss)
            sc = float(score(tt, y_tr, s_oof))
            name = candidate_name(bs, sub)
            rows.append({"candidate": name, "blockset": bs, "subset": sub, "stack_oof_score": sc,
                         "experts": [names[j] for j in keep], "n_columns": int(X_by_bs[bs].shape[1] - 1)})
            log(f"    candidate {name}: stack OOF {sc:.5f}")
            if best is None or better(tt, sc, best[2]):
                best = (bs, sub, sc)
    bs, sub, sc = best
    cfg = final_cfg(task, sub)
    if n_trials is not None:
        cfg["n_trials"] = n_trials
    log(f"[{task}] fold {k}: selected {candidate_name(bs, sub)} (stack OOF {sc:.5f}) -> cfg={cfg}")
    X = X_by_bs[bs]
    preds, meta = run_split4(tt, X.loc[tr], y_tr, X.loc[te], cache_dir(out_root, k, bs), cfg=cfg,
                             n_inner=N_INNER, seed=SEED, log=log)
    assert abs(meta["stack_oof_score"] - sc) < 1e-9, (meta["stack_oof_score"], sc)
    fits = {}
    for b in cands:
        d = cache_dir(out_root, k, b)
        fits[b] = {"cache_provenance": json.loads((d / "cache_provenance.json").read_text()),
                   "fit_info": json.loads((d / "fit_info.json").read_text())}
    imp_cols = [c for c in X.columns if c.startswith(("extra_", "cbfv_oliynyk_"))]
    meta["imputation"] = {"rule": "column medians of the fit's training rows only (experts4.Expert4)",
                          "selected_blockset_nan_cells_train": int(X.loc[tr, imp_cols].isna().sum().sum()),
                          "selected_blockset_nan_cells_test": int(X.loc[te, imp_cols].isna().sum().sum())}
    meta["design_selection"] = {"selected": candidate_name(bs, sub), "blockset": bs, "subset": sub, "cfg": cfg,
                                "criterion": "cross-fitted OOF score of the uncorrected simplex stack on T_k "
                                             "(MAE / ROC-AUC); ties -> earlier candidate",
                                "candidates": rows, "dropped_for_budget": []}
    meta["compute"] = {"selection_job": {**host_info(), "wall_s": round(time.time() - t0, 1)}, "expert_fits": fits}
    np.savez(pf, ids=np.array(te), **{a.replace(":", "__"): v for a, v in preds.items()})
    json.dump(meta, open(out_root / f"fold{k}_meta.json", "w"), indent=1, default=float)
    log(f"[{task}] fold {k}: done in {time.time() - t0:.0f}s; selected={candidate_name(bs, sub)}; "
        f"weights={ {a: round(b, 3) for a, b in meta['stack_weights'].items() if b > 0} }")


def cmd_select(task, k):
    df, folds = load_task_frames(task)
    out_root = OUT / task
    log = open_log(out_root, f"log_fold{k}.txt")
    select_and_predict(task, k, df, folds, out_root, log)


def cmd_smoke(task, n_rows=300):
    """Dry run of the whole selection on a pseudo split INSIDE fold 0's training rows (no official test row is
    read): 2 Optuna trials, a small row subset, outputs in a throw-away folder outside results/."""
    import tempfile
    df, folds = load_task_frames(task)
    tr0 = folds[0]["train"][:n_rows]
    cut = int(0.8 * len(tr0))
    pseudo = [{"train": tr0[:cut], "test": tr0[cut:]}]
    out_root = pathlib.Path(tempfile.mkdtemp(prefix="a4smoke_")) / task
    log = open_log(out_root, "log.txt")
    log(f"SMOKE {task}: {len(tr0)} fold-0 training rows, pseudo split {cut}/{len(tr0) - cut}, out={out_root}")
    select_and_predict(task, 0, df, pseudo, out_root, log, n_trials=2)
    z = np.load(out_root / "fold0_predictions.npz"); m = json.load(open(out_root / "fold0_meta.json"))
    assert all(np.isfinite(z[a]).all() for a in z.files if a != "ids")
    log(f"SMOKE OK: arms={[a for a in z.files]}; selected={m['design_selection']['selected']}; "
        f"candidates={[(r['candidate'], round(r['stack_oof_score'], 4)) for r in m['design_selection']['candidates']]}; "
        f"imputation={m['imputation']}")
    shutil.rmtree(out_root.parent, ignore_errors=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    fold = a.pop(a.index("--fold") + 1) if "--fold" in a else None
    if "--fold" in a:
        a.remove("--fold")
    mode, task = a[0], a[1]
    if mode == "select":
        cmd_select(task, int(fold))
    elif mode == "smoke":
        cmd_smoke(task)
    else:
        raise SystemExit(__doc__)
