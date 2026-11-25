"""Replay all 4764 submitted predictions from the exact five frozen models.

No training, selection, target loading, or benchmark scoring is performed.
Only SHA-pinned public assets are loaded. Requires NVIDIA A100 and requirements.txt.
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_assets(manifest, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, record in manifest['downloads'].items():
        assert Path(name).name == name
        target = directory / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(record['url'], timeout=120) as incoming, temporary.open('wb') as output:
                shutil.copyfileobj(incoming, output)
            assert temporary.stat().st_size == record['bytes'] and sha(temporary) == record['sha256'], name
            temporary.rename(target)
        assert target.stat().st_size == record['bytes'] and sha(target) == record['sha256'], name
    with zipfile.ZipFile(directory / 'replay-context.zip') as archive:
        assert set(archive.namelist()) == set(manifest['context_files'])
        for name in archive.namelist():
            assert Path(name).name == name
            target = directory / name
            content = archive.read(name)
            record = manifest['files'][name]
            assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
            if target.exists():
                assert target.read_bytes() == content
            else:
                target.write_bytes(content)


def verify(args):
    manifest = json.loads((HERE / 'frozen-model-assets.json').read_text())
    directory = args.assets.resolve()
    if args.download:
        download_assets(manifest, directory)
    for name, record in manifest['files'].items():
        assert Path(name).name == name
        file = directory / name
        assert file.stat().st_size == record['bytes'] and sha(file) == record['sha256'], name
    for name, digest in manifest['inference_source_sha256'].items():
        assert sha(HERE / name) == digest, name
    freeze_path = directory / 'all-five-models-freeze.json'
    assert sha(freeze_path) == manifest['original_freeze_sha256']
    freeze = json.loads(freeze_path.read_text())
    assert len(freeze['models']) == 5 and freeze['all_five_genuinely_trained']
    expected = json.loads((directory / 'test-predictions.json').read_text())
    assert len(expected) == len({row['id'] for row in expected}) == 4764
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download('Prior-Labs/TabPFN-v2-reg',
            'tabpfn-v2-regressor-v2_default.ckpt', revision='02a44705ff7790b50cded404bc59adb94da456a1'))
    assert sha(checkpoint) == manifest['foundation_sha256']
    if args.check_only:
        return dict(all_asset_SHA_verified=True, predictions_replayed=False)
    import torch
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    from hyd_regression_workflow import mixed_prediction
    reports, rows = [], []
    for fold, record in enumerate(freeze['models']):
        assert record['outer_fold'] == fold
        model = directory / f'fold-{fold}.pth'
        assert sha(model) == record['checkpoint_sha256']
        selection_file = directory / f'selection-{fold}.json'
        assert sha(selection_file) == record['selection_sha256']
        selection = json.loads(selection_file.read_text())
        assert selection['selected'] == record['selected']
        with np.load(directory / f'train-{fold}.npz', allow_pickle=False) as train, np.load(directory / f'test-{fold}.npz', allow_pickle=False) as test:
            assert set(test.files) == {'ids', 'x'}
            assert train['ids'].tolist() == selection['training_ids']
            assert test['ids'].tolist() == selection['excluded_test_ids']
            assert set(train['ids']).isdisjoint(test['ids'])
            values = mixed_prediction(model, checkpoint, selection, train['x'], train['y'], test['x'])
            targets = {r['id']: r['prediction'] for r in expected if r['fold'] == fold}
            assert set(targets) == set(test['ids'])
            reference = np.asarray([targets[str(sid)] for sid in test['ids']])
            np.testing.assert_allclose(values, reference, atol=1e-3, rtol=1e-4)
            reports.append(dict(fold=fold, rows=len(values), passed=True,
                maximum_absolute_difference=float(np.max(np.abs(values - reference)))))
            rows.extend(dict(fold=fold, id=str(sid), prediction=float(v)) for sid, v in zip(test['ids'], values, strict=True))
        assert sha(model) == record['checkpoint_sha256']
        print(json.dumps(reports[-1]), flush=True)
    assert len(rows) == len({r['id'] for r in rows}) == 4764
    report = dict(experiment_succeeded=True, all4764_replayed=True, n_predictions=4764,
        all_asset_SHA_verified=True, original_freeze_sha256=sha(freeze_path), folds=reports,
        atol=1e-3, rtol=1e-4, test_targets_loaded=False, gradient_updates=0,
        benchmark_score_computed=False, models_or_recipes_changed=False)
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('full-replay-report.json'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    print(json.dumps(verify(parser.parse_args()), indent=2), flush=True)
