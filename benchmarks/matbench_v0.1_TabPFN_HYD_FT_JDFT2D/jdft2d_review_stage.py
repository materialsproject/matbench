"""Scientific stage entrypoint, identical in local and bounded external jobs."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('bundle', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    phase = json.loads((args.bundle / 'manifest.json').read_text())['phase']
    if phase == 'candidates':
        from jdft2d_review_candidates import candidate_stage
        candidate_stage(args.bundle, args.model, args.output)
    elif phase == 'final-fit':
        from jdft2d_review_workflow import final_fit
        final_fit(args.bundle, args.model, args.output)
    elif phase == 'predict':
        from jdft2d_review_workflow import predict
        predict(args.bundle, args.model, args.output)
    else:
        raise ValueError('Unknown phase')
