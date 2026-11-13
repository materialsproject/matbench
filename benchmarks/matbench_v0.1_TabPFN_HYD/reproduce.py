"""Reproduce the two-stage, train-only nested TabPFN-HYD benchmark.

Built with PriorLabs-TabPFN. Run in a fresh work directory on an A100 CUDA
machine. The foundation checkpoint is unchanged. Official test scores are
computed only after all final per-fold selections have been frozen.
"""
import argparse,functools,gzip,hashlib,json,shutil,subprocess,sys,urllib.request
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
for parent in HERE.parents:
    if (parent/'matbench/__init__.py').is_file():
        sys.path.insert(0,str(parent));break

DATA_SHA='473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252'
WEIGHT_SHA='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
WEIGHT_FILE='tabpfn-v2-regressor-v2_default.ckpt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(work):
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty,Stoichiometry,ValenceOrbital
    from pymatgen.core import Composition
    import matbench
    from matbench import MatbenchBenchmark
    source=Path(matbench.__file__).resolve().parent
    lock=json.loads((HERE/'source-lock.json').read_text())
    for name,expected in lock['official_matbench_files'].items():
        if sha(source/name)!=expected:
            raise RuntimeError('Use the pinned Matbench commit in README; source differs: '+name)
    bundle=work/'inputs';bundle.mkdir(parents=True)
    raw_dir=work/'raw';raw_dir.mkdir()
    request=urllib.request.Request('https://ml.materialsproject.org/projects/matbench_steels.json.gz?download=1',headers={'User-Agent':'Mozilla/5.0'})
    raw=urllib.request.urlopen(request,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=DATA_SHA:raise ValueError('Raw dataset checksum mismatch')
    (raw_dir/'matbench_steels.json.gz').write_bytes(raw)
    data=json.loads(gzip.decompress(raw));assert data['index']==list(range(312))
    assert data['columns']==['composition','yield strength']
    ids=[f'mb-steels-{i+1:03d}' for i in data['index']]
    lookup=dict(zip(ids,data['data']))
    # Each feature row depends only on its own composition, with no target or
    # full-dataset-fitted statistics. Estimator preprocessing is fitted inside CV.
    compositions=[Composition(row[0]) for row in data['data']]
    fractions=np.zeros((312,118))
    for row,composition in zip(fractions,compositions):
        for element,value in composition.fractional_composition.items():row[element.Z-1]=value
    featurizer=MultipleFeaturizer([Stoichiometry(),ElementProperty.from_preset('magpie',impute_nan=False),ValenceOrbital(props=['frac'],impute_nan=False)])
    magpie=np.asarray([featurizer.featurize(c) for c in compositions],dtype=float)
    assert magpie.shape==(312,142) and not np.isinf(magpie).any()
    np.savez_compressed(bundle/'features.npz',ids=ids,fractions=fractions,magpie=magpie)
    mb=MatbenchBenchmark(subset=['matbench_steels'],autoload=False)
    folds=json.loads(json.dumps(mb.matbench_steels.validation))
    for name,split in folds.items():
        assert set(split['train']).isdisjoint(split['test'])
        # These are the only labels admitted to the corresponding fold's fit.
        (bundle/f'{name}_training.json').write_text(json.dumps({'ids':split['train'],'targets':[lookup[i][1] for i in split['train']]},indent=2)+'\n')
    for original,renamed in [('model_candidates.py','preregistered_probe.py'),('run_baselines.py','run_baselines.py'),('run_tabpfn.py','run_tabpfn.py')]:shutil.copyfile(HERE/original,bundle/renamed)
    manifest={'folds':folds,'dataset_sha256':DATA_SHA,
        'candidate_grid':{'features':['fractions','magpie','combined'],'n_estimators':[4,8],'output_type':['mean','median'],'tabpfn_weight':[0.25,0.5,0.75,1.0]},
        'selection':'Independent per-outer-fold threefold inner CV, seed1701; all final models frozen before test scoring',
        'files':{p.name:sha(p) for p in bundle.iterdir()}}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return bundle,raw_dir

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--weights',type=Path,help='Optional local original v2 checkpoint; otherwise download pinned public weights')
    parser.add_argument('--prepare-only',action='store_true',help='Only verify sources and prepare features/per-fold training files; no fitting or scoring')
    args=parser.parse_args();work=args.workdir.resolve()
    if work.exists():raise FileExistsError('Use a fresh directory; existing experiments are never overwritten.')
    work.mkdir(parents=True);bundle,raw_dir=prepare(work)
    if args.prepare_only:
        print('Prepared per-row features and official per-fold training files. No fitting/scoring.');return
    weights_dir=work/'foundation';weights_dir.mkdir()
    weights=args.weights
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights=Path(hf_hub_download('Prior-Labs/TabPFN-v2-reg',WEIGHT_FILE,revision='02a44705ff7790b50cded404bc59adb94da456a1',token=False))
    if sha(weights)!=WEIGHT_SHA:raise ValueError('Foundation checkpoint checksum mismatch')
    shutil.copyfile(weights,weights_dir/WEIGHT_FILE);shutil.copyfile(HERE/'LICENSE-TabPFN.txt',weights_dir/'LICENSE.txt')
    baselines=work/'baselines';candidate=work/'candidate'
    subprocess.run([sys.executable,str(bundle/'run_baselines.py'),'--bundle',str(bundle),'--output',str(baselines/'nested')],check=True)
    # The first-stage test predictions are never read or scored to make choices.
    subprocess.run([sys.executable,str(bundle/'run_tabpfn.py'),'--bundle',str(bundle),'--model',str(weights_dir),'--baseline',str(baselines),'--output',str(candidate/'nested')],check=True)
    frozen=json.loads((candidate/'nested/all-folds-freeze.json').read_text())
    for fold in range(5):assert sha(candidate/f'nested/fold_{fold}/selection-freeze.json')==frozen['fold_freeze_sha256'][str(fold)]
    # Only now admit official test targets to native scoring.
    from matminer.datasets import load_dataset
    import matbench.data_ops
    from matbench import MatbenchBenchmark
    matbench.data_ops.load_dataset=functools.partial(load_dataset,data_home=str(raw_dir),download_if_missing=False)
    mb=MatbenchBenchmark(subset=['matbench_steels'],autoload=True)
    rows=json.loads((candidate/'nested/test-predictions.json').read_text());assert len(rows)==312
    for fold in range(5):
        predictions=[r for r in rows if r['fold']==fold]
        expected=mb.matbench_steels.get_test_data(fold,include_target=False).index.tolist()
        assert [r['id'] for r in predictions]==expected
        parameters=json.loads((candidate/f'nested/fold_{fold}/selection-freeze.json').read_text())['selected']
        mb.matbench_steels.record(fold,[r['prediction'] for r in predictions],params=parameters)
    mb.add_metadata({'algorithm':'TabPFN-HYD','authors':'HYD','attribution':'Built with PriorLabs-TabPFN','foundation_gradient_updates':0,'selection':'Independent train-only nested CV; all final selections frozen before outer test scoring'})
    assert not mb.validate();mb.to_file(str(work/'results.json.gz'))
    (work/'metrics.json').write_text(json.dumps(mb.scores,indent=2)+'\n');print(json.dumps(mb.scores,indent=2))
if __name__=='__main__':main()
