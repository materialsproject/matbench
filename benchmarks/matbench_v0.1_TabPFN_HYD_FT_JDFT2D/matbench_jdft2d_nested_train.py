"""One official outer fold: TRAIN-only model selection then genuine final refit."""
import argparse,gc,hashlib,json,logging,time
from pathlib import Path
import numpy as np
from matbench_jdft2d_training import fit_verified,save,sha,WEIGHTS_SHA


def main(bundle,model,out):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    m=json.loads((bundle/'manifest.json').read_text())
    for name,h in m['files'].items():assert sha(bundle/name)==h,name
    checkpoint=model/'tabpfn-v2-regressor-v2_default.ckpt'
    assert sha(checkpoint)==WEIGHTS_SHA and torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    d=np.load(bundle/'development.npz');x,y,ids=d['x'],d['y'],d['ids'];n=len(ids)
    assert n==m['training_row_count'] and x.shape==(n,370)
    assert ids.tolist()==m['development_ids'] and set(ids).isdisjoint(m['excluded_official_test_ids'])
    out.mkdir();offset=0
    if (bundle/'reused-inner-result.json').exists():
        prior=json.loads((bundle/'reused-inner-result.json').read_text())
        assert m['outer_fold']==0 and [v['configuration'] for v in prior['reports']]==m['configurations']
        assert m['inner_reuse_evidence']['same_training_feature_labels_ids_and_CV']
        reports=prior['reports'];save(out/'reused-inner-evidence.json',m['inner_reuse_evidence'])
        print('REUSED_VERIFIED_TRAIN_ONLY_INNER_SELECTION',m['outer_fold'],flush=True)
    else:
        baseline=np.full(n,np.nan)
        for f,cv in enumerate(m['inner_folds']):
            a,b=np.array(cv['train']),np.array(cv['validation'])
            reg=TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=str(checkpoint),device='cuda',
                n_estimators=8,random_state=3042,n_preprocessing_jobs=1,show_progress_bar=False)
            reg.fit(x[a],y[a]);baseline[b]=reg.predict(x[b],output_type='median')
            del reg;gc.collect();torch.cuda.empty_cache()
        assert np.isfinite(baseline).all()
        reports=[];oof=[]
        for k,cfg in enumerate(m['configurations']):
            predicted=np.full(n,np.nan);folds=[]
            for f,cv in enumerate(m['inner_folds']):
                a,b=np.array(cv['train']),np.array(cv['validation'])
                pred,proof=fit_verified(x,y,ids,a,b,cfg,out/f'candidate-{k}-inner-{f}',checkpoint,offset)
                offset+=proof['gradient_steps'];predicted[b]=pred
                folds.append({'fold':f,'mae':float(np.abs(pred-y[b]).mean()),
                    'paired_frozen_baseline_mae':float(np.abs(baseline[b]-y[b]).mean()),**proof})
            assert np.isfinite(predicted).all();oof.append(predicted)
            reports.append({'configuration':cfg,'folds':folds,'mean_fold_mae':float(np.mean([v['mae'] for v in folds]))})
            save(out/'inner-progress.json',{'reports':reports})
        np.savez_compressed(out/'inner-oof-predictions.npz',ids=ids,predictions=np.asarray(oof),baseline=baseline)
    choice=min(range(len(reports)),key=lambda i:(reports[i]['mean_fold_mae'],i))
    cfg=m['configurations'][choice]
    assert cfg['epochs']>0
    selection={'outer_fold':m['outer_fold'],'configuration_index':choice,'configuration':cfg,
        'inner_reports':reports,'selection_uses_only_this_outer_training_fold':True,
        'official_outer_test_features_or_labels_loaded':False,'manifest_sha256':sha(bundle/'manifest.json')}
    save(out/'selection-freeze-before-final-refit.json',selection)
    a=np.arange(n);witness=np.arange(min(16,n))
    # Witness inputs are TRAIN rows used only for prediction replay; no witness score is computed.
    pred,proof=fit_verified(x,y,ids,a,witness,cfg,out/'final-fit',checkpoint,offset,probe_role='training_witness_replay_only')
    np.savez_compressed(out/'final-training-witness.npz',ids=ids[witness],x=x[witness],prediction=pred)
    final=out/'final-fit'/proof['checkpoint_file'];assert sha(checkpoint)==WEIGHTS_SHA
    report={'experiment_succeeded':True,'outer_fold':m['outer_fold'],'training_rows':n,
        'configuration':cfg,'selection_sha256':sha(out/'selection-freeze-before-final-refit.json'),
        'final_checkpoint':'final-fit/'+proof['checkpoint_file'],'final_checkpoint_sha256':sha(final),
        'final_refit_proof':proof,'all_steps_in_this_job':offset+proof['gradient_steps'],
        'official_outer_test_labels_loaded':False,'official_outer_test_predictions_made':False,
        'all_models_genuinely_trained':True,'not_an_official_score':True}
    save(out/'result.json',report)
    print('OUTER_TRAINING_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['bundle','model','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();main(a.bundle,a.model,a.output)
