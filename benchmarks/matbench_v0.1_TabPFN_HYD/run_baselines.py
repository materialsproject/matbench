"""Frozen 42-candidate nested selection, independently for each official fold.

Only each fold's training labels are loaded in its fit loop. No separate official test-target file is provided. Each fold reads only its
own training-label file; other folds legitimately reuse rows under official CV. All five models are frozen before any test prediction.
The candidate function is the exact source preregistered before the CPU probe.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, ast, hashlib, importlib.metadata, json, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    bundle, out=args.bundle, args.output
    out.mkdir(parents=True, exist_ok=True)
    started=time.monotonic()
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(bundle/name)==expected, name
    source=ast.parse((bundle/'preregistered_probe.py').read_text())
    candidate_ast=next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='candidates')
    exec(compile(ast.Module(body=[candidate_ast], type_ignores=[]), '<frozen-candidates>', 'exec'), globals())
    configs=candidates()
    assert len(configs)==42
    cache=np.load(bundle/'features.npz')
    ids=cache['ids'].tolist()
    index={name:i for i,name in enumerate(ids)}
    features={'fractions':cache['fractions'], 'magpie':cache['magpie'],
              'combined':np.concatenate([cache['fractions'],cache['magpie']],axis=1)}
    all_models=[]
    with threadpool_limits(limits=4):
        for fold in range(5):
            folder=out/f'fold_{fold}'
            folder.mkdir(exist_ok=False)
            training=json.loads((bundle/f'fold_{fold}_training.json').read_text())
            fold_spec=manifest['folds'][f'fold_{fold}']
            train_ids=training['ids']
            assert train_ids==fold_spec['train']
            assert set(train_ids).isdisjoint(fold_spec['test'])
            assert len(train_ids)==len(set(train_ids))
            positions=np.asarray([index[i] for i in train_ids])
            y=np.asarray(training['targets'])
            inner=list(KFold(3,shuffle=True,random_state=1701).split(train_ids))
            results=[]
            for name, feature_name, model in configs:
                x=features[feature_name][positions]
                scores=[]
                for fit_idx,val_idx in inner:
                    estimator=clone(model).fit(x[fit_idx],y[fit_idx])
                    scores.append(float(mean_absolute_error(y[val_idx],estimator.predict(x[val_idx]))))
                result={'name':name,'features':feature_name,'inner_fold_mae':scores,'inner_mae':float(np.mean(scores))}
                results.append(result)
                with (folder/'search.jsonl').open('a') as handle:
                    handle.write(json.dumps(result)+'\n')
            selected=min(results,key=lambda x:(x['inner_mae'],x['name']))
            estimator=next(clone(m) for name,f,m in configs if name==selected['name'])
            estimator.fit(features[selected['features']][positions],y)
            joblib.dump(estimator,folder/'model.joblib')
            frozen={'frozen_at_utc':now(),'fold':fold,'selected':selected,
                    'training_ids':train_ids,'excluded_test_ids':fold_spec['test'],
                    'model_sha256':sha(folder/'model.joblib'),
                    'training_input_sha256':sha(bundle/f'fold_{fold}_training.json'),
                    'manifest_sha256':sha(bundle/'manifest.json'),
                    'official_test_targets_accessed':False,'test_predictions_generated':False}
            write(folder/'selection-freeze.json',frozen)
            all_models.append((fold,selected['features'],estimator))
            print('FROZEN',fold,selected['name'],round(selected['inner_mae'],5),flush=True)
    # A global guard: test inference starts only after every fold is immutable.
    for fold,_,_ in all_models:
        folder=out/f'fold_{fold}'
        frozen=json.loads((folder/'selection-freeze.json').read_text())
        assert sha(folder/'model.joblib')==frozen['model_sha256']
    write(out/'all-folds-freeze.json',{'frozen_at_utc':now(),
        'folds':{str(f):sha(out/f'fold_{f}/selection-freeze.json') for f,_,_ in all_models},
        'separate_test_target_file_provided':False,'label_access':'Each fit reads only its own official fold training labels; other folds reuse rows by the official CV design','test_predictions_generated':False})
    predictions=[]
    for fold,feature_name,estimator in all_models:
        test_ids=manifest['folds'][f'fold_{fold}']['test']
        values=estimator.predict(features[feature_name][[index[i] for i in test_ids]])
        predictions.extend({'fold':fold,'id':i,'prediction':float(v)} for i,v in zip(test_ids,values))
    assert len(predictions)==312 and len({r['id'] for r in predictions})==312
    write(out/'test-predictions.json',predictions)
    write(out/'run-summary.json',{'completed_at_utc':now(),'experiment_succeeded':True,
        'training_seconds':time.monotonic()-started,'folds':5,'predictions':312,
        'test_targets_accessed':False,'test_scored':False,
        'model_execution':'CPU estimators on course A100 worker; no pretrained model, no GPU acceleration claimed',
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','scipy','catboost','joblib','threadpoolctl']}})
    outputs={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'artifact-sha256.json',outputs)

if __name__=='__main__':
    main()
