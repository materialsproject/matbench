"""Export all frozen primary seeds to the official Matbench benchmark format."""
import gzip
import json
import numpy as np
import pandas as pd
from run import ROOT,PROTOCOL,base,dump
from matbench.bench import MatbenchBenchmark

def export_seed(seed,root=ROOT):
    df,_,folds=base.load();folder=root/f'seed_{seed:03d}'
    prediction=pd.concat([pd.read_csv(folder/f/'predictions.csv',float_precision='round_trip') for f in folds]).set_index('mbid')
    mb=MatbenchBenchmark(autoload=False,subset=['matbench_steels']);task=mb.matbench_steels
    task.df=df[['composition','yield strength']]
    for fi in task.folds:
        f=f'fold_{fi}';test=task.get_test_data(fi,include_target=False)
        assert test.index.tolist()==df.index[folds[f][1]].tolist()
        fit=json.loads((folder/f/'weights.json').read_text())[PROTOCOL['primary']]
        task.record(fi,prediction.loc[test.index,PROTOCOL['primary']].to_numpy(),params=dict(
            seed=seed,**fit,inner_trees=600,final_trees=900,inner_folds=5,bootstrap_draws=128,
            bootstrap_seed=19072026,coefficient_sum_bounds=[0.,2.],selection='Nonnegative affine MAE fits on 128 Bayesian bootstrap samples of outer-training OOF predictions'))
    mb.user_metadata=dict(algorithm=PROTOCOL['name'],protocol=PROTOCOL,verified=False,
        application_scope='Retrospective composition benchmark; no prospective or physical qualification claim.')
    mb.validate();path=folder/'results.json.gz'
    path.write_bytes(gzip.compress(json.dumps(mb.as_dict()).encode(),mtime=0))
    reread=MatbenchBenchmark.from_dict(json.loads(gzip.decompress(path.read_bytes())));reread.validate()
    expected=np.mean([np.mean(abs(prediction.loc[df.index[te],PROTOCOL['primary']].to_numpy()-df.iloc[te]['yield strength'].to_numpy())) for _,te in folds.values()])
    assert abs(expected-reread.matbench_steels.scores.mae.mean)<1e-10
    return dict(seed=seed,mean_fold_mae=float(expected),passed=True)

def main():
    records=[export_seed(s) for s in [417,*PROTOCOL['repeats']]]
    dump(ROOT/'official_exports.json',records);print(json.dumps(records,indent=2))

if __name__=='__main__':main()
