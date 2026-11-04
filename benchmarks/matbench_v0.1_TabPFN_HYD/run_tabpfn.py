"""Per-outer-fold inner selection for TabPFN-HYD. Built with PriorLabs-TabPFN.

The 48-config grid is identical to the preregistered discovery probe. Each
official fold independently selects against its previous 42-config training-CV
winner. No official test target file is read. All component states and mixtures
are frozen before any outer test inference. No foundation gradient updates.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,importlib.metadata,json,time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import save_fitted_tabpfn_model
from threadpoolctl import threadpool_limits

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    assert not path.exists(),path
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def regressor(path,count):
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2,
        model_path=str(path),device='cuda',n_estimators=count,random_state=1701,
        n_preprocessing_jobs=1,inference_precision=torch.float32,show_progress_bar=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('/trisol/input/models/model-0'))
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(a.bundle/name)==h,name
    weight=a.model/'tabpfn-v2-regressor-v2_default.ckpt'
    assert sha(weight)=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    baseline_root=a.baseline/'nested'
    baseline_manifest=json.loads((baseline_root/'artifact-sha256.json').read_text())
    data=np.load(a.bundle/'features.npz');ids=data['ids'].tolist();index={i:k for k,i in enumerate(ids)}
    features={'fractions':data['fractions'],'magpie':data['magpie'],
        'combined':np.concatenate([data['fractions'],data['magpie']],axis=1)}
    models=[]
    for fold in range(5):
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=False)
        training=json.loads((a.bundle/f'fold_{fold}_training.json').read_text())
        spec=manifest['folds'][f'fold_{fold}'];assert training['ids']==spec['train']
        assert set(training['ids']).isdisjoint(spec['test'])
        positions=np.array([index[i] for i in training['ids']]);y=np.array(training['targets'])
        old_folder=baseline_root/f'fold_{fold}'
        for name in ['selection-freeze.json','model.joblib','search.jsonl']:
            rel=f'fold_{fold}/{name}';assert sha(old_folder/name)==baseline_manifest[rel],rel
        previous=json.loads((old_folder/'selection-freeze.json').read_text())
        assert previous['training_ids']==training['ids']
        assert previous['training_input_sha256']==sha(a.bundle/f'fold_{fold}_training.json')
        baseline=joblib.load(old_folder/'model.joblib');baseline_features=previous['selected']['features']
        inner=list(KFold(3,shuffle=True,random_state=1701).split(y))
        base_oof=np.full(len(y),np.nan)
        with threadpool_limits(limits=4):
            for fit,val in inner:
                # Clone drops fitted state: validation predictions never use a model fitted on all training rows.
                fresh=clone(baseline).fit(features[baseline_features][positions[fit]],y[fit])
                base_oof[val]=fresh.predict(features[baseline_features][positions[val]])
        base_scores=[float(mean_absolute_error(y[v],base_oof[v])) for _,v in inner]
        records=[{'features':baseline_features,'n_estimators':0,'output_type':'baseline',
            'tabpfn_weight':0.0,'inner_mae':float(np.mean(base_scores)),'inner_fold_mae':base_scores}]
        for fset in ['fractions','magpie','combined']:
            for count in [4,8]:
                pred={t:np.full(len(y),np.nan) for t in ['mean','median']}
                for k,(fit,val) in enumerate(inner):
                    model=regressor(weight,count)
                    model.fit(features[fset][positions[fit]],y[fit])
                    output=model.predict(features[fset][positions[val]],output_type='main')
                    for t in pred:pred[t][val]=np.asarray(output[t])
                    del model;torch.cuda.empty_cache()
                np.savez_compressed(folder/f'oof-{fset}-{count}.npz',ids=training['ids'],**pred)
                for t,values in pred.items():
                    for w in [0.25,0.5,0.75,1.0]:
                        mixed=w*values+(1-w)*base_oof
                        scores=[float(mean_absolute_error(y[v],mixed[v])) for _,v in inner]
                        records.append({'features':fset,'n_estimators':count,'output_type':t,
                            'tabpfn_weight':w,'inner_mae':float(np.mean(scores)),'inner_fold_mae':scores})
                print(json.dumps({'fold':fold,'inner_candidate_features':fset,'n_estimators':count}),flush=True)
        write(folder/'search.json',records)
        chosen=min(records,key=lambda r:(r['inner_mae'],json.dumps(r,sort_keys=True)))
        final_model=None
        if chosen['tabpfn_weight']>0:
            final_model=regressor(weight,chosen['n_estimators'])
            final_model.fit(features[chosen['features']][positions],y)
            save_fitted_tabpfn_model(final_model,folder/'model.tabpfn_fit')
        freeze={'frozen_at_utc':now(),'fold':fold,'selected':chosen,
            'training_ids':training['ids'],'excluded_test_ids':spec['test'],
            'training_input_sha256':sha(a.bundle/f'fold_{fold}_training.json'),
            'manifest_sha256':sha(a.bundle/'manifest.json'),
            'baseline_model_ref':'hyd-matbench-steels-3042:3',
            'baseline_model_path':f'nested/fold_{fold}/model.joblib',
            'baseline_model_sha256':previous['model_sha256'],
            'baseline_cv_mae_current':float(np.mean(base_scores)),
            'baseline_cv_mae_previous':previous['selected']['inner_mae'],
            'foundation_weight_sha256':sha(weight),
            'fitted_context_sha256':sha(folder/'model.tabpfn_fit') if final_model is not None else None,
            'foundation_gradient_updates':0,'own_fold_test_targets_used':False,
            'test_predictions_generated':False}
        write(folder/'selection-freeze.json',freeze)
        models.append((fold,chosen,baseline_features,baseline,final_model))
        print('FROZEN',fold,json.dumps(chosen),flush=True)
    # All five independent selections are immutable before outer test inference.
    write(out/'all-folds-freeze.json',{'frozen_at_utc':now(),
        'fold_freeze_sha256':{str(f):sha(out/f'fold_{f}/selection-freeze.json') for f,_,_,_,_ in models},
        'official_test_scored':False,'test_predictions_generated':False})
    predictions=[]
    with threadpool_limits(limits=4):
        for fold,chosen,bfeatures,baseline,model in models:
            test_ids=manifest['folds'][f'fold_{fold}']['test'];positions=np.array([index[i] for i in test_ids]);w=chosen['tabpfn_weight']
            base=baseline.predict(features[bfeatures][positions]) if w<1 else np.zeros(len(test_ids))
            if w>0:
                pred=model.predict(features[chosen['features']][positions],output_type=chosen['output_type'])
                values=w*np.asarray(pred)+(1-w)*base
            else:values=base
            predictions.extend({'fold':fold,'id':i,'prediction':float(v)} for i,v in zip(test_ids,values))
    assert len(predictions)==312 and len({r['id'] for r in predictions})==312
    write(out/'test-predictions.json',predictions)
    write(out/'run-summary.json',{'completed_at_utc':now(),'experiment_succeeded':True,'seconds':time.monotonic()-started,
        'folds':5,'predictions':312,'official_test_scored':False,'foundation_gradient_updates':0,
        'attribution':'Built with PriorLabs-TabPFN','packages':{n:importlib.metadata.version(n) for n in ['torch','tabpfn','numpy','scikit-learn','scipy']}})
    write(out/'artifact-sha256.json',{str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()})

if __name__=='__main__':main()
