"""DESIGN PHASE: inner CV on fold 0's TRAINING rows only.

The official fold-0 test rows and all targets outside fold-0 training are never loaded into
the arrays used here (asserted). Fold-0 training rows are split into 5 pseudo-outer folds
(seed 2026); each pseudo-outer training part runs the full pipeline (with its own 5 inner
folds), and arms are scored on the pseudo-held-out part.

Usage: python -m src.design <task> [tag] [cfg-json]
"""
from __future__ import annotations
import json, sys, pathlib, time
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .data import load_task_frames, featurise, featurise_extra, ROOT
from .pipeline import run_split, score

TASK_TYPE = {"matbench_steels": "regression", "matbench_expt_gap": "regression",
             "matbench_expt_is_metal": "classification", "matbench_glass": "classification"}


def main(task, tag="v1", cfg=None):
    tt = TASK_TYPE[task]
    df, folds = load_task_frames(task)
    X_all = featurise(task, df)
    if (cfg or {}).get("extra_features"):
        X_all = X_all.join(featurise_extra(task, df))
    tr_ids = folds[0]["train"]
    te_ids = set(folds[0]["test"])
    assert not (set(tr_ids) & te_ids)
    X = X_all.loc[tr_ids]
    y = df.loc[tr_ids, "target"].to_numpy(dtype=float)
    del df, X_all  # nothing outside fold-0 training is kept
    out_dir = ROOT / "results" / "design" / tag / task
    out_dir.mkdir(parents=True, exist_ok=True)
    if tt == "regression":
        po = list(KFold(5, shuffle=True, random_state=2026).split(X))
    else:
        po = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(X, y.astype(int)))
    rows, metas = [], []
    logf = open(out_dir / "log.txt", "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    log(f"== design {tag} {task} n_fold0_train={len(y)} cfg={cfg}")
    for i, (a, b) in enumerate(po):
        t0 = time.time()
        preds, meta = run_split(tt, X.iloc[a], y[a], X.iloc[b], out_dir / f"pseudo{i}", cfg=cfg, log=log)
        sc = {k: float(score(tt, y[b], v)) for k, v in preds.items()}
        rows.append(sc); metas.append(meta)
        log(f"  pseudo-fold {i} ({time.time()-t0:.0f}s): " + ", ".join(f"{k}={v:.4f}" for k, v in sc.items()
             if not k.startswith("single:")))
        log(f"    weights={ {k: round(v, 3) for k, v in meta['stack_weights'].items()} } sel={ {k: meta['sel_stack_local'][k] for k in ('space','k','hmult','kappa','alpha')} }")
    arms = rows[0].keys()
    summ = {k: {"mean": float(np.mean([r[k] for r in rows])), "folds": [r[k] for r in rows]} for k in arms}
    log("  MEAN: " + ", ".join(f"{k}={v['mean']:.4f}" for k, v in summ.items()))
    json.dump({"task": task, "tag": tag, "cfg": cfg, "summary": summ, "meta": metas}, open(out_dir / "summary.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    task = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "v1"
    cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    main(task, tag, cfg)
