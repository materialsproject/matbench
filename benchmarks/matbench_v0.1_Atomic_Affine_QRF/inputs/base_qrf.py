"""Frozen nested quantile-forest experiment; writes only beside this script."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT/'protocol.json').read_text())
SOURCE_HASH = '473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252'

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')

def load():
    raw = (ROOT/'matbench_steels.json.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASH
    data = json.loads(gzip.decompress(raw))
    df = pd.DataFrame(data['data'], columns=data['columns'])
    df.index = [f'mb-steels-{i+1:03d}' for i in range(len(df))]
    atom = np.array([[Composition(f).get_atomic_fraction(e) for e in PROTOCOL['elements']] for f in df.composition])
    assert atom.shape == (312,14) and np.allclose(atom.sum(axis=1),1)
    x = dict(atomic=atom, sqrt_atomic=np.sqrt(atom), square_atomic=atom**2,
             log1p_100atomic=np.log1p(100*atom))
    splits = json.loads((ROOT/'steels_splits.json').read_text())['splits']['matbench_steels']
    folds = {name: tuple(np.array([df.index.get_loc(i) for i in split[part]]) for part in ('train','test'))
             for name,split in splits.items()}
    assert sorted(np.concatenate([v[1] for v in folds.values()]).tolist()) == list(range(len(df)))
    for tr,te in folds.values():
        assert not set(tr)&set(te) and sorted(np.r_[tr,te].tolist())==list(range(len(df)))
    return df,x,folds

def configs():
    return [dict(representation=r,min_samples_leaf=l,max_features=m)
            for r in PROTOCOL['representations'] for l in PROTOCOL['leaves']
            for m in PROTOCOL['max_features']]

def fit_model(x,y,c,seed,trees):
    return ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=c['min_samples_leaf'],
        max_features=c['max_features'],random_state=seed,n_jobs=1).fit(x,y)

def distribution_predictions(model,x_train,y_train,x_query):
    """Exact leaf-sample weights (also handles non-pure or duplicate leaves)."""
    training_leaves = model.apply(x_train)
    query_leaves = model.apply(x_query)
    weights = np.zeros((len(x_query),len(y_train)),dtype=np.float64)
    for i in range(len(model.estimators_)):
        same = query_leaves[:,i,None] == training_leaves[None,:,i]
        denominator = same.sum(axis=1)
        assert (denominator>0).all()
        weights += same/denominator[:,None]
    weights /= len(model.estimators_)
    assert np.allclose(weights.sum(axis=1),1)
    mean = weights@y_train
    np.testing.assert_allclose(mean,model.predict(x_query),rtol=0,atol=1e-8)
    order = np.argsort(y_train,kind='stable')
    cumulative = weights[:,order].cumsum(axis=1)
    median = y_train[order[np.argmax(cumulative >= 0.5 - 1e-12,axis=1)]]
    return np.column_stack([mean,median])

def run_fold(seed,fold):
    started=time.monotonic()
    out=ROOT/f'seed_{seed:03d}'/fold
    out.mkdir(parents=True,exist_ok=True)
    if (out/'prediction.csv').exists():
        return dict(seed=seed,fold=fold,existing=True)
    df,x,folds=load()
    tr,te=folds[fold]
    y=df['yield strength'].to_numpy()
    # The fitting API receives only this fold's training labels.
    yt=y[tr].copy()
    cv=list(KFold(PROTOCOL['inner_cv']['folds'],shuffle=True,
        random_state=PROTOCOL['inner_cv']['random_state']).split(tr))
    cfg=configs()
    oof=np.empty((len(cfg),len(tr),2))
    log=[]
    split_log=[dict(train_ids=df.index[tr[a]].tolist(),validation_ids=df.index[tr[b]].tolist()) for a,b in cv]
    dump(out/'inner_splits.json',split_log)
    for ci,c in enumerate(cfg):
        xx=x[c['representation']]
        for a,b in cv:
            m=fit_model(xx[tr[a]],yt[a],c,seed,PROTOCOL['inner_trees'])
            oof[ci,b]=distribution_predictions(m,xx[tr[a]],yt[a],xx[tr[b]])
        for weight in PROTOCOL['median_weights']:
            p=(1-weight)*oof[ci,:,0]+weight*oof[ci,:,1]
            log.append(dict(config_index=ci,**c,median_weight=weight,
                            inner_mae=float(np.mean(abs(p-yt)))))
        print(json.dumps(dict(stage='inner',seed=seed,fold=fold,config=ci,
            seconds=round(time.monotonic()-started,2))),flush=True)
    chosen=min(range(len(log)),key=lambda i: log[i]['inner_mae'])
    best=log[chosen]
    ci=best['config_index'];c=cfg[ci];xx=x[c['representation']]
    # No test response is passed to final training or selection.
    m=fit_model(xx[tr],yt,c,seed,PROTOCOL['final_trees'])
    components=distribution_predictions(m,xx[tr],yt,xx[te])
    weight=best['median_weight']
    predicted=(1-weight)*components[:,0]+weight*components[:,1]
    np.savez_compressed(out/'inner_predictions.npz',oof=oof,train_indices=tr)
    pd.DataFrame(log).to_csv(out/'candidate_scores.csv',index=False)
    dump(out/'selection.json',dict(seed=seed,fold=fold,chosen=best,
        chosen_row=chosen,train_ids=df.index[tr].tolist(),test_ids=df.index[te].tolist(),
        seconds=time.monotonic()-started))
    # Test labels only enter here, after predictions and selection exist.
    rows=pd.DataFrame(dict(mbid=df.index[te],fold=fold,actual_mpa=y[te],
        predicted_mpa=predicted,mean_mpa=components[:,0],median_mpa=components[:,1]))
    rows.to_csv(out/'prediction.csv',index=False)
    result=dict(seed=seed,fold=fold,mae=float(np.mean(abs(predicted-y[te]))),
        chosen=best,seconds=time.monotonic()-started)
    dump(out/'score.json',result)
    print(json.dumps(dict(stage='finished',**result)),flush=True)
    return result

def finalize(seed):
    folder=ROOT/f'seed_{seed:03d}'
    rows=pd.concat([pd.read_csv(folder/f'fold_{i}'/'prediction.csv') for i in range(5)])
    assert len(rows)==312 and rows.mbid.nunique()==312
    rows.to_csv(folder/'predictions.csv',index=False)
    scores=[json.loads((folder/f'fold_{i}'/'score.json').read_text()) for i in range(5)]
    summary=dict(seed=seed,mean_fold_mae=float(np.mean([s['mae'] for s in scores])),folds=scores)
    dump(folder/'summary.json',summary)
    print(json.dumps(dict(stage='summary',seed=seed,mae=summary['mean_fold_mae'])),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=417)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    assert 1<=args.workers<=3
    dump(ROOT/'environment.json',{n:importlib.metadata.version(n) for n in
        ['numpy','scikit-learn','pandas','pymatgen','pymatgen-core','matbench']})
    if args.fold is not None:
        run_fold(args.seed,f'fold_{args.fold}')
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures=[pool.submit(run_fold,args.seed,f'fold_{i}') for i in range(5)]
            for f in futures:f.result()
        finalize(args.seed)

if __name__=='__main__':main()
