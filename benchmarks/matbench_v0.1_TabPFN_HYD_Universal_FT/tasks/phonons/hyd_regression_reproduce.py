"""Complete portable review-corrected reproduction; stage executors are replaceable.

Default executor is a local CUDA Python subprocess. --executor supplies only
bounded-job transport, not a scientific alternative. All scientific stages and
their input/output contracts are included beside this file and SHA-pinned.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from hyd_regression_workflow import MLPS, SCHEDULES, prepare, select, sha, save, now

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ['hyd_regression_features.py', 'hyd_regression_training.py',
                'hyd_regression_candidates.py', 'hyd_regression_workflow.py',
                'hyd_regression_stage.py']


def checkpoint_directory(checkpoint, directory, expected_sha256):
    """Preserve the expected filename even for HF snapshots -> blobs symlinks."""
    checkpoint = Path(checkpoint)
    assert checkpoint.is_file() and sha(checkpoint) == expected_sha256
    directory = Path(directory)
    directory.mkdir()
    target = directory / 'tabpfn-v2-regressor-v2_default.ckpt'
    shutil.copyfile(checkpoint, target)
    assert sha(target) == expected_sha256
    return directory.resolve()


def make_bundle(directory, scientific, data_files, protocol):
    directory.mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(HERE / name, directory / name)
    for name, source in data_files.items():
        shutil.copyfile(source, directory / name)
    scientific.update(protocol_sha256=sha(protocol), files={p.name: sha(p) for p in directory.iterdir()})
    save(directory / 'manifest.json', scientific)


def run_stage(run, protocol, args, auxiliary_models=()):
    """No hidden fit fallback and no automatic replay of a failed execution."""
    intent = run / 'execution-intent.json'
    assert not intent.exists(), 'Existing stage intent: inspect it; do not automatically retry'
    manifest = json.loads((run / 'inputs/manifest.json').read_text())
    request = dict(bundle=str((run / 'inputs').resolve()), output=str((run / 'results').resolve()),
                   protocol_sha256=sha(protocol), original_checkpoint_sha256=manifest['original_checkpoint_sha256'],
                   auxiliary_model_refs=list(auxiliary_models), scientific_phase=manifest['phase'])
    save(intent, request)
    if args.executor:
        subprocess.run([args.executor_python, str(args.executor.resolve()), '--request', str(intent.resolve())], check=True)
    else:
        checkpoint = args.checkpoint
        if checkpoint is None:
            from huggingface_hub import hf_hub_download
            checkpoint = Path(hf_hub_download('Prior-Labs/TabPFN-v2-reg',
                'tabpfn-v2-regressor-v2_default.ckpt', revision='02a44705ff7790b50cded404bc59adb94da456a1'))
        model_directory = checkpoint_directory(checkpoint, run / 'local-checkpoint',
            request['original_checkpoint_sha256'])
        command = [sys.executable, str(run / 'inputs/hyd_regression_stage.py'), '--bundle',
            str(run / 'inputs'), '--model', str(model_directory), '--output', str(run / 'results/probe')]
        (run / 'results').mkdir()
        env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            DO_NOT_TRACK='1', TABPFN_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (run / 'results/stage.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        save(run / 'execution-receipt.json', dict(executor='local', succeeded=True, model_ref=None))
    report = json.loads((run / 'results/probe/result.json').read_text())
    assert report['experiment_succeeded']
    receipt = json.loads((run / 'execution-receipt.json').read_text())
    assert receipt['succeeded']
    return report, receipt


def record_score(prepared, inference, freeze, protocol, data_path, output):
    task_config = json.loads(protocol.read_text())['task']
    """Only the corrected run is recorded; never compare it to choose a model."""
    assert not output.exists()
    report = json.loads((inference / 'result.json').read_text())
    assert report['n_predictions'] == task_config['rows'] and all(v['passed'] for v in report['fresh_process_replays'])
    assert report['model_freeze_sha256'] == sha(freeze)
    rows = json.loads((inference / 'test-predictions.json').read_text())
    assert len(rows) == len({v['id'] for v in rows}) == task_config['rows']
    frozen = json.loads(freeze.read_text())
    assert len(frozen['models']) == 5
    import matbench.data_ops
    from matminer.datasets import load_dataset
    matbench.data_ops.load_dataset = functools_partial_load(load_dataset, data_path)
    from matbench import MatbenchBenchmark
    benchmark = MatbenchBenchmark(subset=[task_config['name']], autoload=False)
    task = getattr(benchmark, task_config['name'])
    task.load()
    for f in range(5):
        expected = task.get_test_data(f, include_target=False).index.tolist()
        values = {v['id']: v['prediction'] for v in rows if v['fold'] == f}
        assert set(values) == set(expected)
        task.record(f, [values[sid] for sid in expected], params=dict(
            selected=frozen['models'][f]['selected']['candidate'], protocol_sha256=sha(protocol),
            training_card_hours=frozen['models'][f].get('compute', {}).get('training_card_hours'),
            wall_seconds=frozen['models'][f].get('compute', {}).get('wall_seconds'),
            real_gradient_training=True))
    benchmark.add_metadata(dict(algorithm='TabPFN-HYD-Universal-FT', attribution='Built with PriorLabs-TabPFN',
        unified_regression_protocol=True, task=task_config['name'],
        limitation='Method transferred from previously evaluated steel/JDFT2D tasks. Search rules fixed before new phonon targets; each own TRAIN selects independently. Foundation pretraining is third-party and not claimed unseen.',
        protocol_sha256=sha(protocol), model_freeze_sha256=sha(freeze)))
    assert not benchmark.validate()
    benchmark.to_file(str(output))
    scores = json.loads(json.dumps(task.scores))
    save(output.parent / 'official-score.json', dict(scored_at_utc=now(), native_five_fold_scores=scores,
        n_predictions=task_config['rows'], result_sha256=sha(output), protocol_sha256=sha(protocol),
        old_score_used_for_model_selection=False, replacement_independent_of_score_direction=True,
        method_transfer_from_other_tasks_disclosed=True, public_first_confirmed=False))


def functools_partial_load(load_dataset, data_path):
    import functools
    return functools.partial(load_dataset, data_home=str(data_path.resolve().parent), download_if_missing=False)


def main(args):
    root = args.output.resolve()
    assert not root.exists(), 'Never overwrite a prior reproduction or restart without reconciliation'
    root.mkdir(parents=True)
    protocol = args.protocol.resolve()
    frozen_protocol = json.loads(protocol.read_text())
    for name, digest in frozen_protocol['scientific_source_sha256'].items():
        assert sha(HERE / name) == digest, name
    prepared = root / 'prepared'
    prepare(args.data.resolve(), args.splits.resolve(), prepared, protocol)
    models, model_roots = [], []
    save(root / 'state.json', dict(stage='prepared', active_run=None, completed_folds=0))
    for f in range(5):
        source = prepared / f'fold-{f}'
        manifest = json.loads((source / 'training-manifest.json').read_text())
        candidates = []
        stage_times = []
        stages = [('baselines-mlp', None, None)] + [(feature + '-block-' + str(block), feature, list(range(block * 3, block * 3 + 3))) for feature in ('composition260', 'all370') for block in (0, 1)]
        for stage, feature, schedule_indices in stages:
            run = root / f'fold-{f}-{stage}'
            run.mkdir()
            science = dict(manifest, phase='candidates', stage='baselines-mlp' if stage == 'baselines-mlp' else 'finetune',
                           original_checkpoint_sha256=frozen_protocol['original_checkpoint_sha256'])
            if stage != 'baselines-mlp':
                science['features'] = feature
                science['schedule_indices'] = schedule_indices
            make_bundle(run / 'inputs', science, {'training.npz': source / 'training.npz'}, protocol)
            save(root / 'state.json', dict(stage='training_candidates', active_run=str(run), completed_folds=f))
            _, receipt = run_stage(run, protocol, args)
            candidates.append(run / 'results/probe')
            stage_times.append(receipt.get('compute', {}))
        selection = source / 'selection.json'
        select(source / 'training.npz', source / 'training-manifest.json', candidates, selection)
        run = root / f'fold-{f}-final'
        run.mkdir()
        make_bundle(run / 'inputs', dict(task=frozen_protocol['task'], phase='final-fit', original_checkpoint_sha256=frozen_protocol['original_checkpoint_sha256']),
                    {'training.npz': source / 'training.npz', 'selection.json': selection}, protocol)
        save(root / 'state.json', dict(stage='final_refit', active_run=str(run), completed_folds=f))
        report, receipt = run_stage(run, protocol, args)
        stage_times.append(receipt.get('compute', {}))
        hours = [v.get('training_card_hours') for v in stage_times]
        seconds = [v.get('wall_seconds') for v in stage_times]
        compute = dict(training_card_hours=sum(hours) if all(v is not None for v in hours) else None,
                       wall_seconds=sum(seconds) if all(v is not None for v in seconds) else None,
                       scope='candidate selection plus actual final refit; measured executor intervals, not queue time')
        models.append(dict(outer_fold=f, checkpoint_relative=report['checkpoint_relative'],
            checkpoint_sha256=report['checkpoint_sha256'], selected=report['selected'],
            selection_sha256=sha(selection), actual_training_proof=report['true_training_proof'],
            model_ref=receipt.get('model_ref'), compute=compute))
        model_roots.append(str((run / 'results').resolve()))
        save(root / 'state.json', dict(stage='fold_frozen', active_run=None, completed_folds=f + 1, models=models))
    freeze = root / 'all-five-models-freeze.json'
    save(freeze, dict(frozen_at_utc=now(), protocol_sha256=sha(protocol), models=models,
                     all_five_genuinely_trained=True, corrected_test_predictions_made=False))
    run = root / 'fresh-inference'
    run.mkdir()
    data_files = {}
    for f in range(5):
        source = prepared / f'fold-{f}'
        data_files[f'train-{f}.npz'] = source / 'training.npz'
        data_files[f'test-{f}.npz'] = source / 'test-geometry-only.npz'
        data_files[f'selection-{f}.json'] = source / 'selection.json'
    make_bundle(run / 'inputs', dict(task=frozen_protocol['task'], phase='predict', original_checkpoint_sha256=frozen_protocol['original_checkpoint_sha256'],
        all5_final_models_frozen_before_inference=True, model_freeze_sha256=sha(freeze),
        models=models, model_roots=model_roots), data_files, protocol)
    save(root / 'state.json', dict(stage='fresh_process_replay_and_inference', active_run=str(run), completed_folds=5, models=models))
    _, receipt = run_stage(run, protocol, args, [m['model_ref'] for m in models if m.get('model_ref')])
    record_score(prepared, run / 'results/probe', freeze, protocol, args.data, root / 'results.json.gz')
    save(root / 'state.json', dict(stage='official_result_complete_not_public_first', active_run=None,
         completed_folds=5, models=models, public_first_confirmed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'splits', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--executor', type=Path)
    parser.add_argument('--executor-python', default=sys.executable)
    main(parser.parse_args())
