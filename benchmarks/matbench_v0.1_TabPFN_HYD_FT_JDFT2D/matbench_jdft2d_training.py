"""Shared audited training primitive; training receives only its declared rows."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNRegressor
WEIGHTS_SHA='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
class AuditedFinetuner(FinetunedTabPFNRegressor):
    def _forward_with_loss(self,batch):
        if not hasattr(self,'initial_weights_'):
            if self.trainable_scope_=='readout':
                for name,param in self.finetuned_estimator_.model_.named_parameters():
                    param.requires_grad_(name.startswith('output_projection.'))
                assert any(param.requires_grad for param in self.finetuned_estimator_.model_.parameters())
            self.initial_weights_={n:p.detach().cpu().clone() for n,p in self.finetuned_estimator_.model_.named_parameters()}
        return super()._forward_with_loss(batch)

class LocalLogger:
    def __init__(self,path,offset):self.path=path;self.steps=[];self.epochs=[];self.offset=offset
    def record(self,d):
        with self.path.open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n')
    def setup(self,config):self.record({'event':'setup','learning_rate':config['learning_rate'],'model_version':'v2 explicitly pinned'})
    def log_step(self,metrics,step):
        clean={k:float(v) for k,v in metrics.items() if np.isfinite(v)}
        self.steps.append(step);self.record({'event':'gradient_step','step':step,'metrics':clean})
        print('TRISOL_PROGRESS '+json.dumps({'v':1,'step':self.offset+step,'metrics':clean}),flush=True)
    def log_epoch(self,metrics,step):
        self.epochs.append(int(metrics['train/epoch']));self.record({'event':'epoch','step':step,'metrics':metrics})
    def finish(self):self.record({'event':'training_finished','actual_gradient_steps':max(self.steps,default=0)})


def fit_verified(x,y,ids,a,b,cfg,run,checkpoint,offset=0,probe_role='inner_validation'):
    run.mkdir()
    logger=LocalLogger(run/'training-metrics.jsonl',offset)
    started=time.monotonic()
    ft=AuditedFinetuner(device='cuda',model_version=ModelVersion.V2,
        epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],time_limit=120,
        validation_split_ratio=0,early_stopping=False,use_lr_scheduler=True,lr_warmup_only=True,
        n_finetune_ctx_plus_query_samples=128,finetune_ctx_query_split_ratio=.25,
        n_estimators_finetune=2,n_estimators_validation=2,n_estimators_final_inference=8,
        use_activation_checkpointing=True,save_checkpoint_interval=cfg['epochs'],
        random_state=3042,experiment_logger=logger,
        extra_regressor_kwargs={'model_path':str(checkpoint),'n_preprocessing_jobs':1,'show_progress_bar':False},
        ce_loss_weight=0.,crps_loss_weight=.25,mse_loss_weight=0.,mae_loss_weight=1.)
    # No held-out development rows or targets are passed into fit or checkpoint selection.
    ft.trainable_scope_=cfg['trainable_scope']
    ft.fit(x[a],y[a],output_dir=run)
    assert logger.epochs==list(range(cfg['epochs'])),'Incomplete predeclared training run; retain diagnostics, no automatic GPU retry'
    steps=max(logger.steps);assert steps>0;
    final={n:p.detach().cpu() for n,p in ft.finetuned_estimator_.model_.named_parameters()}
    initial=ft.initial_weights_;assert initial.keys()==final.keys()
    changes=[{'name':n,'initial_sha256':tensor_hash(initial[n]),'final_sha256':tensor_hash(final[n]),
        'max_abs_change':float((final[n]-initial[n]).abs().max()),'elements':final[n].numel()} for n in final]
    changed=[v for v in changes if v['initial_sha256']!=v['final_sha256']]
    assert len(changed)>0 and max(v['max_abs_change'] for v in changed)>0
    if cfg['trainable_scope']=='readout':
        assert all(v['name'].startswith('output_projection.') for v in changed),'Frozen backbone unexpectedly changed'
    files=list(run.glob('*.pth'));assert len(files)==1
    saved=torch.load(files[0],map_location='cpu',weights_only=False)
    assert saved['epoch']==cfg['epochs'] and saved['optimizer']['state']
    prediction=np.asarray(ft.predict(x[b],output_type='median'),dtype=float)
    assert prediction.shape==(len(b),) and np.isfinite(prediction).all()
    save(run/'weight-update-evidence.json',{'actual_gradient_steps':steps,'actual_epochs':len(logger.epochs),
        'changed_parameter_tensors':len(changed),'updated_parameter_elements':sum(v['elements'] for v in changed),
        'parameters':changes,'original_checkpoint_sha256':WEIGHTS_SHA,'fine_tuned_checkpoint_sha256':sha(files[0]),
        'optimizer_saved':True,'trainable_scope':cfg['trainable_scope'],'frozen_backbone_unchanged':cfg['trainable_scope']!='readout' or all(v['name'].startswith('output_projection.') for v in changed),'train_ids':ids[a].tolist(),'probe_ids':ids[b].tolist(),'probe_role':probe_role,
        'validation_rows_passed_to_fit':0,'initial_untrained_fallback_disabled':True})
    del ft,saved,final,initial;gc.collect();torch.cuda.empty_cache()
    replay=TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=str(files[0]),device='cuda',
        n_estimators=8,random_state=3042,n_preprocessing_jobs=1,show_progress_bar=False)
    replay.fit(x[a],y[a]);replayed=np.asarray(replay.predict(x[b],output_type='median'))
    maxdiff=float(np.max(np.abs(replayed-prediction)))
    assert np.allclose(replayed,prediction,atol=1e-3,rtol=1e-4),'Saved checkpoint replay mismatch'
    np.savez_compressed(run/'probe-predictions.npz',ids=ids[b],prediction=prediction,replay=replayed)


    record={'gradient_steps':steps,'epochs':len(logger.epochs),'changed_parameter_tensors':len(changed),
        'checkpoint_sha256':sha(files[0]),'checkpoint_file':files[0].name,
        'checkpoint_reload_max_abs_difference':maxdiff,'same_process_checkpoint_reload_not_fresh_process':True,
        'seconds':time.monotonic()-started,'probe_role':probe_role}
    del replay
    gc.collect();torch.cuda.empty_cache()
    return prediction,record
