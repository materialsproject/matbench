"""AMENDMENT-3 FINAL EVALUATION on the official Matbench v0.1 folds (ANALYSIS_PLAN.md Section 13).

Per task and official fold k, using the official training rows T_k only:
  1. for every candidate block set (src.frozen_config3.CAND_BLOCKS) the FULL expert pool is tuned and cross-fitted
     on T_k with the unchanged Amendment-2 code (pipeline2.tune_expert / crossfit2; J = 5, seed 0, 20 Optuna
     trials). Expert caches are reused only where the expert has the same name and reads exactly the same columns
     (then the computation is identical: same code, seeds, deterministic LightGBM); the provenance of every cache
     file is recorded in <cache dir>/cache_provenance.json;
  2. every candidate = (block set, expert subset) is scored by the cross-fitted OOF score of the uncorrected simplex
     stack on T_k (the Amendment-1 criterion); ties -> earlier (simpler) candidate;
  3. the chosen candidate is handed to the unchanged Amendment-2 pipeline (pipeline2.run_split2: stack, gated local
     correction, thresholds) and the test rows of fold k are predicted. Test TARGETS are never loaded here.

Usage:
  python -m src.run_final3 select <task> --fold k [--drop-cand F_min]      # steps 1-3 for one fold (PC)
  python -m src.run_final3 fit <task> --fold k --cand F_red [--no-reuse] [--tag determinism]
                                                                           # step 1 only (pod / PC worker)
  python -m src.run_final3 smoke <task>                                    # dry run on T_0 rows only, tiny budget
"""
from __future__ import annotations
import json, os, pathlib, platform, shutil, socket, subprocess, sys, time
import numpy as np

from .data import load_task_frames, ROOT
from .features_all import features
from .experts2 import make_experts2, group_cols
from .pipeline import inner_splits, crossfitted_stack, score, better
from .pipeline2 import run_split2, tune_expert, crossfit2
from .design import TASK_TYPE
from .frozen_config3 import (SEED, N_INNER, BLOCKSETS, CAND_BLOCKS, SUBSETS, pool_cfg, final_cfg, candidate_name)

OUT = ROOT / "results" / "final_a3"
A2 = ROOT / "results" / "final_a2"
EXPERT_THREADS = 4


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            if out:
                return " ".join(out.split())
        except Exception:
            pass
    return platform.processor() or platform.machine()


def host_info():
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    return {"host": socket.gethostname(), "cpu_model": cpu_model(), "cpu_count": os.cpu_count(), "cpu_affinity": affinity,
            "threads_per_worker": EXPERT_THREADS, "n_workers": int(os.environ.get("A3_N_WORKERS", "1")),
            "platform": platform.platform(), "python": platform.python_version()}


def cache_dir(out_root, k, bs):
    return out_root / f"fold{k}_experts_{bs}"


def build_X(task, df, bs):
    X = features(task, df, BLOCKSETS[bs])
    X["composition"] = df["composition"].values
    return X


def expert_cols(X, ex):
    return list(group_cols(X, ex.group))


def prepare_cache(task, k, bs, X_by_bs, experts, out_root, log, n_tr, n_te, reuse=True, reuse_a2=True):
    """Copy identical expert caches into this candidate's cache dir: from the Amendment-2 run
    (results/final_a2/<task>/fold<k>_experts, block set F_full) and from the other candidate dirs of this fold.
    'Identical' = same expert name AND exactly the same input columns (same names, same order)."""
    dst = cache_dir(out_root, k, bs)
    dst.mkdir(parents=True, exist_ok=True)
    pf = dst / "cache_provenance.json"
    prov = json.loads(pf.read_text()) if pf.exists() else {}
    if not reuse:
        return prov
    sources = []
    a2dir = A2 / task / f"fold{k}_experts"
    if reuse_a2 and a2dir.exists():
        sources.append((a2dir, X_by_bs["F_full"], "A2:F_full"))
    for b in X_by_bs:
        if b != bs and cache_dir(out_root, k, b).exists():
            sources.append((cache_dir(out_root, k, b), X_by_bs[b], f"A3:{b}"))
    for ex in experts:
        if (dst / f"{ex.name}.npz").exists():
            continue
        cols = expert_cols(X_by_bs[bs], ex)
        for sdir, Xs, label in sources:
            f = sdir / f"{ex.name}.npz"
            if f.exists() and expert_cols(Xs, ex) == cols:
                try:                      # a concurrent fit job may still be writing the source file: validate first
                    z = np.load(f)
                    ok = len(z["oof"]) == n_tr and len(z["test"]) == n_te and bool(np.isfinite(z["oof"]).all())
                except Exception:
                    ok = False
                if not ok:
                    continue
                shutil.copy2(f, dst / f.name)
                p = sdir / f"{ex.name}_params.json"
                if p.exists():
                    shutil.copy2(p, dst / p.name)
                rel = sdir.relative_to(ROOT).as_posix() if sdir.is_relative_to(ROOT) else sdir.name
                prov[ex.name] = f"copied from {label} ({rel}): identical inputs ({len(cols)} columns)"
                log(f"    cache {bs}/{ex.name}: {prov[ex.name]}")
                break
    pf.write_text(json.dumps(prov, indent=1))
    return prov


def fit_candidate(task, k, bs, df, tr, te, y_tr, X_by_bs, out_root, log, reuse=True, n_trials=None, reuse_a2=True):
    """Step 1 for one (fold, block set): tune + cross-fit the full expert pool on T_k (resumable, cached)."""
    tt = TASK_TYPE[task]
    cfg = pool_cfg(task)
    if n_trials is not None:
        cfg["n_trials"] = n_trials
    X = X_by_bs[bs]
    X_tr, X_te = X.loc[tr], X.loc[te]
    splits = inner_splits(tt, y_tr, N_INNER, SEED)
    experts = make_experts2(tt, len(y_tr), seed=SEED, cfg=cfg)
    dst = cache_dir(out_root, k, bs)
    prov = prepare_cache(task, k, bs, X_by_bs, experts, out_root, log, len(tr), len(te), reuse=reuse, reuse_a2=reuse_a2)
    t0 = time.time()
    fitted_here = [e.name for e in experts if not (dst / f"{e.name}.npz").exists()]
    log(f"  [{task}] fold {k} {bs}: ncols={X.shape[1] - 1}; to fit here: {fitted_here or 'nothing'}")
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


def select_and_predict(task, k, df, folds, out_root, log, drop_cands=(), n_trials=None, reuse=True, reuse_a2=True):
    tt = TASK_TYPE[task]
    pf = out_root / f"fold{k}_predictions.npz"
    if pf.exists():
        log(f"[{task}] fold {k}: already done, skipping"); return
    t0 = time.time()
    tr, te = folds[k]["train"], folds[k]["test"]
    assert not (set(tr) & set(te))
    y_tr = df.loc[tr, "target"].to_numpy(dtype=float)        # training targets only
    cands = [b for b in CAND_BLOCKS[task] if b not in drop_cands]
    X_by_bs = {b: build_X(task, df, b) for b in set(cands) | {"F_full"}}
    log(f"[{task}] fold {k}: n_train={len(tr)} n_test={len(te)}; block sets {cands}"
        + (f" (dropped for budget: {list(drop_cands)})" if drop_cands else "") + f"; subsets {len(SUBSETS[task])}")
    rows, best = [], None
    loss = "l1" if tt == "regression" else "l2"
    for bs in cands:
        P_oof, names, secs, splits, _ = fit_candidate(task, k, bs, df, tr, te, y_tr, X_by_bs, out_root, log,
                                                     reuse=reuse, n_trials=n_trials, reuse_a2=reuse_a2)
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
    log(f"[{task}] fold {k}: selected {candidate_name(bs, sub)} (stack OOF {sc:.5f}) -> cfg={cfg}")
    X = X_by_bs[bs]
    preds, meta = run_split2(tt, X.loc[tr], y_tr, X.loc[te], cache_dir(out_root, k, bs), cfg=cfg,
                             n_inner=N_INNER, seed=SEED, log=log)
    assert abs(meta["stack_oof_score"] - sc) < 1e-9, (meta["stack_oof_score"], sc)
    fits = {}
    for b in cands:
        d = cache_dir(out_root, k, b)
        fits[b] = {"cache_provenance": json.loads((d / "cache_provenance.json").read_text()),
                   "fit_info": json.loads((d / "fit_info.json").read_text())}
    meta["design_selection"] = {"selected": candidate_name(bs, sub), "blockset": bs, "subset": sub, "cfg": cfg,
                                "criterion": "cross-fitted OOF score of the uncorrected simplex stack on T_k "
                                             "(MAE / ROC-AUC); ties -> earlier candidate",
                                "candidates": rows, "dropped_for_budget": list(drop_cands)}
    meta["compute"] = {"selection_job": {**host_info(), "wall_s": round(time.time() - t0, 1)}, "expert_fits": fits}
    np.savez(pf, ids=np.array(te), **{a.replace(":", "__"): v for a, v in preds.items()})
    json.dump(meta, open(out_root / f"fold{k}_meta.json", "w"), indent=1, default=float)
    log(f"[{task}] fold {k}: done in {time.time() - t0:.0f}s; selected={candidate_name(bs, sub)}; "
        f"weights={ {a: round(b, 3) for a, b in meta['stack_weights'].items() if b > 0} }")


def open_log(out_root, name):
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / name, "a", encoding="utf-8")
    return lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())


def cmd_select(task, k, drop_cands):
    df, folds = load_task_frames(task)
    out_root = OUT / task
    log = open_log(out_root, f"log_fold{k}.txt")
    select_and_predict(task, k, df, folds, out_root, log, drop_cands=drop_cands)


def cmd_fit(task, k, bs, reuse, tag):
    df, folds = load_task_frames(task)
    out_root = (OUT / tag / task) if tag else (OUT / task)
    log = open_log(out_root, f"log_fit_fold{k}_{bs}.txt")
    tr, te = folds[k]["train"], folds[k]["test"]
    assert not (set(tr) & set(te))
    y_tr = df.loc[tr, "target"].to_numpy(dtype=float)
    X_by_bs = {b: build_X(task, df, b) for b in {bs, "F_full"}}
    t0 = time.time()
    fit_candidate(task, k, bs, df, tr, te, y_tr, X_by_bs, out_root, log, reuse=reuse)
    log(f"[{task}] fold {k} {bs}: expert fits complete in {time.time() - t0:.0f}s ({'no reuse' if not reuse else 'with cache reuse'})")


def cmd_smoke(task, n_rows=300):
    """Dry run of the whole selection on a pseudo split INSIDE fold 0's training rows (no official test row is
    read): 2 Optuna trials, a small row subset, outputs in a throw-away folder outside results/."""
    import tempfile
    df, folds = load_task_frames(task)
    tr0 = folds[0]["train"][:n_rows]
    cut = int(0.8 * len(tr0))
    pseudo = [{"train": tr0[:cut], "test": tr0[cut:]}]
    out_root = pathlib.Path(tempfile.mkdtemp(prefix="a3smoke_")) / task
    log = open_log(out_root, "log.txt")
    log(f"SMOKE {task}: {len(tr0)} fold-0 training rows, pseudo split {cut}/{len(tr0) - cut}, out={out_root}")
    select_and_predict(task, 0, df, pseudo, out_root, log, n_trials=2, reuse=True, reuse_a2=False)  # no A2 cache: other split
    z = np.load(out_root / "fold0_predictions.npz"); m = json.load(open(out_root / "fold0_meta.json"))
    assert all(np.isfinite(z[a]).all() for a in z.files if a != "ids")
    log(f"SMOKE OK: arms={[a for a in z.files]}; selected={m['design_selection']['selected']}; "
        f"candidates={[(r['candidate'], round(r['stack_oof_score'], 4)) for r in m['design_selection']['candidates']]}")
    shutil.rmtree(out_root.parent, ignore_errors=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    def opt(flag, default=None, n=1):
        if flag in a:
            i = a.index(flag); v = a[i + 1] if n else True; del a[i:i + 1 + n]; return v
        return default
    fold = opt("--fold"); cand = opt("--cand"); tag = opt("--tag"); no_reuse = opt("--no-reuse", False, 0)
    drops = []
    while "--drop-cand" in a:
        drops.append(opt("--drop-cand"))
    mode, task = a[0], a[1]
    if mode == "select":
        cmd_select(task, int(fold), tuple(drops))
    elif mode == "fit":
        cmd_fit(task, int(fold), cand, reuse=not no_reuse, tag=tag)
    elif mode == "smoke":
        cmd_smoke(task)
    else:
        raise SystemExit(__doc__)
