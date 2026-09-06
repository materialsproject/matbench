#!/usr/bin/env python3
"""Reproduce the release offline, or run its versioned hosted verifier.

python submission_client.py
python submission_client.py --live
python submission_client.py --live --record rerun.json.gz

The service evaluates fitted correction models using its private evidence view.
It accepts benchmark identities, not target labels. Full retraining requires
the service-side assets; this public driver covers inference and scoring.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent


def verify_materials(endpoint, fold, candidates, variant=None):
    manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
    variant=variant or manifest['variant']
    url=os.environ.get('SCIENTIA_VERIFIER_API_URL',manifest['service_url']).rstrip('/')+'/predict'
    payload=dict(release=manifest['release'],endpoint=endpoint,fold=fold,variant=variant,
                 candidates=[{'id':str(row['id'])} for row in candidates])
    req=Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'SciKG-Benchmark/2.0'},method='POST')
    for attempt in range(3):
        try:
            with urlopen(req,timeout=60) as response: value=json.load(response)
            break
        except (HTTPError,URLError) as error:
            if attempt==2 or isinstance(error,HTTPError) and error.code<500 and error.code!=429: raise
            time.sleep(2**attempt)
    if value.get('release')!=manifest['release'] or value.get('variant')!=variant or value.get('execution')!='model_inference':
        raise ValueError('unexpected service version or execution mode')
    out=value.get('predictions',[])
    if [r['id'] for r in out]!=[r['id'] for r in payload['candidates']]: raise ValueError('service changed row identities/order')
    if any(not math.isfinite(r['prediction']) for r in out): raise ValueError('nonfinite response')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--record',type=Path)
    args=parser.parse_args()
    if args.record and not args.live: parser.error('--record requires --live')
    manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
    for name,expected in manifest['hashes'].items():
        path=ROOT/name
        if path.parent!=ROOT or hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('artifact hash mismatch: '+name)
    with gzip.open(ROOT/'reference.json.gz','rt') as handle: reference=json.load(handle)
    with gzip.open(ROOT/'results.json.gz','rt') as handle: recording=json.load(handle)
    stored=recording['tasks'][manifest['task']]['results']
    if set(reference)!=set(stored) or set(stored)!={'fold_'+str(i) for i in range(5)}: raise ValueError('fold coverage differs')
    benchmark=None
    if args.record:
        from matbench.bench import MatbenchBenchmark
        benchmark=MatbenchBenchmark(benchmark='matbench_v0.1',autoload=True,subset=[manifest['task']])
    metrics={}; maximum=0.; total=0
    for fold in range(5):
        key='fold_'+str(fold); rows=reference[key]; table=stored[key]['data']; ids=[r['id'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(table): raise ValueError('row coverage differs')
        predictions=table
        if args.live:
            batches=[rows[i:i+32] for i in range(0,len(rows),32)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses=pool.map(lambda batch:verify_materials(manifest['task'],fold,batch),batches)
                predictions={r['id']:r['prediction'] for response in responses for r in response}
            difference=max(abs(predictions[i]-table[i]) for i in ids)
            maximum=max(maximum,difference)
            if difference>1e-12: raise ValueError('hosted prediction differs: '+str(difference))
        score=statistics.mean(abs(predictions[r['id']]-r['target']) for r in rows)
        if not math.isclose(score,manifest['folds'][str(fold)],rel_tol=0,abs_tol=1e-12): raise ValueError('fold MAE differs')
        if not math.isclose(score,stored[key]['scores']['mae'],rel_tol=0,abs_tol=1e-12): raise ValueError('native score differs')
        metrics[str(fold)]=score; total+=len(rows)
        if benchmark:
            task=getattr(benchmark,manifest['task']); _,targets=task.get_test_data(fold,as_type='tuple',include_target=True)
            official=list(map(str,targets.index))
            if official!=ids or any(float(t)!=r['target'] for t,r in zip(targets.to_numpy(),rows)): raise ValueError('official test set differs')
            task.record(fold,[predictions[i] for i in official],params={'release':manifest['release'],'variant':manifest['variant']})
    mean=statistics.mean(metrics.values())
    if total!=manifest['rows'] or not math.isclose(mean,manifest['mae'],rel_tol=0,abs_tol=1e-12): raise ValueError('aggregate differs')
    if benchmark:
        if args.record.exists(): raise FileExistsError(args.record)
        benchmark.add_metadata({'release':manifest['release'],'variant':manifest['variant']})
        if benchmark.validate(): raise ValueError('MatBench validation failed')
        benchmark.to_file(str(args.record))
        if args.record.suffix == '.gz' and not args.record.read_bytes().startswith(b'\x1f\x8b'):
            args.record.write_bytes(gzip.compress(args.record.read_bytes(),mtime=0))
    print(json.dumps(dict(release=manifest['release'],variant=manifest['variant'],rows=total,folds=metrics,
        mean_mae=mean,reduction_percent=manifest['reduction_percent'],hosted_inference_checked=args.live,
        maximum_hosted_prediction_difference=maximum if args.live else None),indent=2))


if __name__=='__main__': main()
