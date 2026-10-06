"""Frozen affine QRF aggregator; fitting APIs receive only outer-training labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'inputs'))
import base_qrf as base
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
POLICIES=[PROTOCOL['primary'],*PROTOCOL['ablations']]

def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def affine_lp(prediction,target,row_weight,policy):
    """Solve one convex weighted MAE fit, with an explicit dual certificate."""
    n,k=prediction.shape
    center=float(np.median(target))
    design=sparse.csr_matrix(np.column_stack([(prediction-center)/100.,np.ones(n)]))
    yy=(target-center)/100.
    eye=sparse.eye(n,format='csr')
    au=sparse.vstack([sparse.hstack([design,-eye]),sparse.hstack([-design,-eye])],format='csr')
    bu=np.r_[yy,-yy]
    sum_row=sparse.csr_matrix(np.r_[np.ones(k),np.zeros(n+1)][None,:])
    ae=be=None
    if policy=='bayesian_affine':
        au=sparse.vstack([au,sum_row],format='csr');bu=np.r_[bu,PROTOCOL['coefficient_sum_bounds'][1]]
    elif policy=='bayesian_shift':
        ae=sum_row;be=np.array([1.])
    else:
        raise ValueError(policy)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected',category=OptimizeWarning)
        fit=linprog(np.r_[np.zeros(k+1),row_weight],A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,
            bounds=[(0,None)]*k+[(None,None)]+[(0,None)]*n,method='highs-ds',options={'threads':1})
    assert fit.success,fit.message
    w=fit.x[:k]
    intercept=float(center*(1-w.sum())+100*fit.x[k])
    objective=float(row_weight@abs(prediction@w+intercept-target)/100.)
    dual=float(bu@fit.ineqlin.marginals+(be@fit.eqlin.marginals if be is not None else 0.))
    assert w.min()>=0 and w.sum()<=2+1e-8
    if policy=='bayesian_shift':assert abs(w.sum()-1)<1e-8
    assert abs(objective-fit.fun)<1e-8 and abs(objective-dual)<1e-8
    return w,intercept,dict(primal=objective,dual=dual,iterations=fit.nit,slope=float(w.sum()),intercept_mpa=intercept)

def fit_weights(oof,target):
    p=oof.transpose(1,0,2).reshape(len(target),-1)
    rng=np.random.default_rng(PROTOCOL['bootstrap_seed'])
    rows=rng.exponential(size=(PROTOCOL['bootstrap_draws'],len(target)))
    rows/=rows.sum(axis=1,keepdims=True)
    weights={};draws={};certificates={}
    for policy in POLICIES:
        ws=[];offsets=[];cs=[]
        for row in rows:
            w,b,c=affine_lp(p,target,row,policy)
            ws.append(w);offsets.append(b);cs.append(c)
        weights[policy]=dict(coefficients=np.mean(ws,axis=0).tolist(),intercept_mpa=float(np.mean(offsets)))
        draws[policy]=dict(coefficients=np.array(ws),intercepts=np.array(offsets))
        certificates[policy]=cs
    return weights,draws,rows,certificates

def run_fold(seed,fold):
    start=time.monotonic();out=ROOT/f'seed_{seed:03d}'/fold;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    df,x,folds=base.load();tr,te=folds[fold]
    target=df['yield strength'].to_numpy()[tr].copy()
    inp=ROOT/'inputs'/f'seed_{seed:03d}'/fold
    oo=np.load(inp/'inner_predictions.npz');assert np.array_equal(oo['train_indices'],tr)
    weights,draws,rows,certificates=fit_weights(oo['oof'],target)
    # Persist the full frozen fit before any final prediction is accessed.
    dump(out/'weights.json',weights);dump(out/'lp_certificates.json',certificates)
    np.savez_compressed(out/'bootstrap.npz',row_weights=rows,
        **{f'{p}_{k}':v for p in POLICIES for k,v in draws[p].items()})
    comp=np.load(inp/'final_components.npz')
    assert np.array_equal(comp['train_indices'],tr) and np.array_equal(comp['test_indices'],te)
    components=comp['components'].copy()
    active=np.flatnonzero(np.array([w['coefficients'] for w in weights.values()]).reshape(2,16,2).sum(axis=(0,2))>0)
    refitted=[]
    for ci in active:
        if not np.isfinite(components[:,ci]).all():
            c=base.configs()[ci];xx=x[c['representation']]
            model=base.fit_model(xx[tr],target,c,seed,base.PROTOCOL['final_trees'])
            components[:,ci]=base.distribution_predictions(model,xx[tr],target,xx[te]);refitted.append(int(ci))
    flat=np.nan_to_num(components.reshape(len(te),32),nan=0.)
    result=pd.DataFrame(dict(mbid=df.index[te],fold=fold))
    for policy,w in weights.items():
        result[policy]=flat@w['coefficients']+w['intercept_mpa']
    result.to_csv(out/'predictions.csv',index=False)
    np.savez_compressed(out/'final_components.npz',components=components,active_configs=active,
        train_indices=tr,test_indices=te)
    info=dict(seed=seed,fold=fold,active_configs=active.tolist(),newly_refitted_configs=refitted,
        train_ids=df.index[tr].tolist(),test_ids=df.index[te].tolist(),seconds=time.monotonic()-start)
    dump(out/'complete.json',info);print(json.dumps(info),flush=True)
    return info

def source_check():
    for name,h in json.loads((ROOT/'inputs_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--fold',type=int)
    ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();assert 1<=a.workers<=3
    source_check()
    dump(ROOT/'environment.json',{n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','matbench','pymatgen','pymatgen-core']})
    seeds=[a.seed] if a.seed is not None else [417,*PROTOCOL['repeats']]
    folds=[a.fold] if a.fold is not None else range(5)
    with ProcessPoolExecutor(a.workers) as pool:
        tasks=[pool.submit(run_fold,s,f'fold_{f}') for s in seeds for f in folds]
        for t in tasks:t.result()

if __name__=='__main__':main()
