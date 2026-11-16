"""Fresh-process replay then all official held-out predictions from five frozen fits."""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main(bundle,out):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4);m=json.loads((bundle/'manifest.json').read_text())
    for name,h in m['files'].items():assert sha(bundle/name)==h,name
    assert m['all5_final_models_frozen'] and len(m['models'])==5
    out.mkdir();rows=[];replays=[]
    for f,model in enumerate(m['models']):
        root=Path(m['model_roots'][f])
        checkpoint=root/'probe'/model['checkpoint_relative']
        assert sha(checkpoint)==model['checkpoint_sha256']
        train=np.load(bundle/f'train-{f}.npz');test=np.load(bundle/f'test-{f}.npz')
        assert set(train['ids']).isdisjoint(test['ids'])
        assert train['ids'].tolist()==model['training_ids'] and test['ids'].tolist()==model['test_ids']
        assert set(test.files)=={'ids','x'},'Test labels are not admissible here'
        witness=np.load(root/'probe/final-training-witness.npz')
        estimator=TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=str(checkpoint),device='cuda',
            n_estimators=8,random_state=3042,n_preprocessing_jobs=1,show_progress_bar=False)
        estimator.fit(train['x'],train['y'])
        replay=np.asarray(estimator.predict(witness['x'],output_type='median'))
        difference=float(np.abs(replay-witness['prediction']).max())
        assert np.allclose(replay,witness['prediction'],atol=1e-3,rtol=1e-4),'Fresh-process frozen-model replay failed'
        pred=np.asarray(estimator.predict(test['x'],output_type='median'),dtype=float)
        assert pred.shape==(len(test['ids']),) and np.isfinite(pred).all()
        assert sha(checkpoint)==model['checkpoint_sha256']
        rows.extend({'fold':f,'id':str(sid),'prediction':float(p)} for sid,p in zip(test['ids'],pred,strict=True))
        replays.append({'fold':f,'fresh_process':True,'training_witness_rows':len(witness['ids']),
            'max_abs_difference':difference,'checkpoint_sha256':sha(checkpoint),'passed':True})
        save(out/'test-predictions-progress.json',rows)
        print('FROZEN_OUTER_PREDICTIONS_COMPLETE',f,len(pred),'fresh_replay_maxdiff',difference,flush=True)
        del estimator;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==len({r['id'] for r in rows})==636
    save(out/'test-predictions.json',rows)
    save(out/'result.json',{'experiment_succeeded':True,'all636_official_heldout_predictions':True,
        'n_predictions':636,'all5_final_models_frozen_before_this_job':True,'model_freeze_sha256':m['model_freeze_sha256'],
        'fresh_process_replays':replays,'official_test_labels_loaded':False,'gradient_updates_in_this_inference_job':0,
        'official_score_not_computed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path) # original input is mounted for lineage; frozen trained versions are auxiliary mounts
    a=p.parse_args();main(a.bundle,a.output)
