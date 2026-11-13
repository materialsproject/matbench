"""Fresh official native replay; frozen fitting implementation is unchanged."""
import argparse
import gzip
import json
from pathlib import Path
from preflight import verify

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', default='fresh_script_review')
    ap.add_argument('--workers', type=int, choices=(1,2,3), default=3)
    args=ap.parse_args()
    stage=verify(args.output)
    from fresh import reproduce
    from export import export_seed
    fitting=reproduce(stage, seed=417, workers=args.workers, compare_reference=False)
    score=export_seed(417, root=stage)
    root=Path(__file__).resolve().parent
    expected=json.loads(gzip.decompress((root/'results.json.gz').read_bytes()))
    actual=json.loads(gzip.decompress((stage/'seed_417/results.json.gz').read_bytes()))
    passed=(actual['tasks']['matbench_steels']['results'] == expected['tasks']['matbench_steels']['results'])
    record={'passed':passed, 'complete_fold_dictionaries_exact':passed,
            'native_dependencies':True, 'fitting':fitting, 'score':score, 'verified':False}
    (stage/'strict_entry_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    if not passed:
        raise RuntimeError('Exact official fold-dictionary replay failed; no tolerance widened.')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':
    main()
