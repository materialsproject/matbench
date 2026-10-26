"""One bounded fresh replay, starting from chemistry and official split files only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import run

ROOT=Path(__file__).resolve().parent

def fold_from_scratch(stage,seed,fold):
    run.ROOT=Path(stage);run.base.ROOT=run.ROOT/'inputs'
    # Rebuild all sixteen five-fold OOF libraries, using only training rows.
    run.base.run_fold(seed,fold)
    df,_,folds=run.base.load();tr,te=folds[fold]
    path=run.ROOT/'inputs'/f'seed_{seed:03d}'/fold/'final_components.npz'
    np.savez_compressed(path,components=np.full((len(te),16,2),np.nan),
        active_configs=np.array([],dtype=int),train_indices=tr,test_indices=te)
    return run.run_fold(seed,fold)

def reproduce(stage,seed=417,workers=3,compare_reference=True):
    stage=Path(stage).resolve();assert stage.is_relative_to(ROOT),stage
    assert not stage.exists(),'Fresh replay refuses an existing directory.'
    assert 1<=workers<=3
    start=time.monotonic();(stage/'inputs').mkdir(parents=True)
    for f in ['base_qrf.py','protocol.json','matbench_steels.json.gz','steels_splits.json']:
        shutil.copyfile(ROOT/'inputs'/f,stage/'inputs'/f)
    with ProcessPoolExecutor(workers) as pool:
        tasks=[pool.submit(fold_from_scratch,str(stage),seed,f'fold_{f}') for f in range(5)]
        results=[t.result() for t in tasks]
    # Full exact new-implementation replay; no previously fit arrays reused.
    maximum=0.
    for f in range(5) if compare_reference else []:
        fold=f'fold_{f}';original=ROOT/f'seed_{seed:03d}'/fold;fresh=stage/f'seed_{seed:03d}'/fold
        assert json.loads((fresh/'weights.json').read_text())==json.loads((original/'weights.json').read_text())
        oo=np.load(stage/'inputs'/f'seed_{seed:03d}'/fold/'inner_predictions.npz')
        cached=np.load(ROOT/'inputs'/f'seed_{seed:03d}'/fold/'inner_predictions.npz')
        np.testing.assert_array_equal(oo['oof'],cached['oof'])
        a=pd.read_csv(original/'predictions.csv',float_precision='round_trip')
        b=pd.read_csv(fresh/'predictions.csv',float_precision='round_trip')
        assert a.mbid.tolist()==b.mbid.tolist()
        for p in run.POLICIES:
            difference=float(np.max(abs(a[p]-b[p])));maximum=max(maximum,difference)
            np.testing.assert_array_equal(a[p],b[p])
    result=dict(passed=True,seed=seed,empty_fit_directory_at_start=True,reused_training_cache=False,
        inner_forests=400,auxiliary_old_selector_forests=5,
        active_final_forests=sum(len(r['active_configs']) for r in results),
        max_prediction_difference_mpa=maximum if compare_reference else None,seconds=time.monotonic()-start,
        exact_weights_and_inner_predictions=compare_reference,verified=False)
    run.dump(stage/'fresh_verification.json',result);print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='fresh_reference');ap.add_argument('--seed',type=int,default=417)
    ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    reproduce(ROOT/a.output,a.seed,a.workers)

if __name__=='__main__':main()
