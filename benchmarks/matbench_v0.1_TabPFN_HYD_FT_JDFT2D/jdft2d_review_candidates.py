"""Review correction: all historical choices are evaluated inside one outer TRAIN.

This file never receives outer-test features, targets, scores or old predictions.
Candidate optimizer tensors are hashed and removed after their full-checkpoint
replay; learned weights and all update evidence remain. Final fits retain AdamW.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from matbench_jdft2d_training import fit_verified, tensor_hash

WEIGHTS_SHA = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def view(x, feature_set):
    assert feature_set in ('composition260', 'all370')
    return x[:, :260] if feature_set == 'composition260' else x


def estimator(checkpoint, n):
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=str(checkpoint), device='cuda', n_estimators=n,
        random_state=3042, n_preprocessing_jobs=1, show_progress_bar=False)


def infer_variants(checkpoint, x, y, a, b):
    variants = {}
    for count in (4, 8):
        reg = estimator(checkpoint, count)
        reg.fit(x[a], y[a])
        for summary in ('mean', 'median'):
            p = np.asarray(reg.predict(x[b], output_type=summary), dtype=float)
            assert p.shape == (len(b),) and np.isfinite(p).all()
            variants[f'e{count}-{summary}'] = p
        del reg
        cleanup()
    return variants


def optimizer_digest(value):
    """Hash each actual saved optimizer tensor/scalar without exposing targets."""
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_hash(value))
    if isinstance(value, dict):
        return {str(k): optimizer_digest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [optimizer_digest(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value))


def retain_candidate_weights(checkpoint, evidence):
    original_sha = sha(checkpoint)
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert data.get('optimizer', {}).get('state')
    digest = optimizer_digest(data.pop('optimizer'))
    save(evidence / 'optimizer-state-digests.json', digest)
    temporary = checkpoint.with_suffix('.weight-only.tmp')
    torch.save(data, temporary)
    retained_sha = sha(temporary)
    temporary.replace(checkpoint)
    save(evidence / 'checkpoint-retention.json', dict(
        original_full_checkpoint_sha256=original_sha, retained_weight_checkpoint_sha256=retained_sha,
        optimizer_tensor_digests_retained=True, optimizer_tensors_retained=False,
        learned_model_fields_unmodified=True, resumable_training_claimed=False))
    del data
    cleanup()
    return retained_sha


def make_mlp(width, hidden):
    return torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(), torch.nn.Linear(hidden // 2, 1)).cuda()


def mlp_fit(x, y, ids, a, b, config, directory, seed):
    directory.mkdir()
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    xx = view(x, config['features'])
    imputer = SimpleImputer(keep_empty_features=True)
    scaler = StandardScaler()
    xa = scaler.fit_transform(imputer.fit_transform(xx[a]))
    xb = scaler.transform(imputer.transform(xx[b]))
    xa = torch.tensor(np.clip(xa, -10, 10), dtype=torch.float32, device='cuda')
    xb = torch.tensor(np.clip(xb, -10, 10), dtype=torch.float32, device='cuda')
    center = float(np.median(y[a]))
    scale = max(float(np.subtract(*np.percentile(y[a], [75, 25]))), 1.)
    ya = torch.tensor((y[a] - center) / scale, dtype=torch.float32, device='cuda')
    model = make_mlp(xx.shape[1], config['width'])
    initial = {n: tensor_hash(p) for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=config['decay'])
    with (directory / 'training-metrics.jsonl').open('w') as log:
        for step in range(600):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(xa).squeeze(-1), ya)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
            if (step + 1) % 200 == 0:
                metric = dict(v=1, step=step + 1, metrics={'train/loss': float(loss.detach().cpu())})
                log.write(json.dumps(metric) + '\n')
                print('TRISOL_PROGRESS ' + json.dumps(metric), flush=True)
    model.eval()
    with torch.no_grad():
        prediction = model(xb).squeeze(-1).cpu().numpy() * scale + center
    state = {n: p.detach().cpu() for n, p in model.state_dict().items()}
    changed = [n for n, p in model.named_parameters() if tensor_hash(p) != initial[n]]
    assert changed
    checkpoint = directory / 'model.pt'
    torch.save(dict(state_dict=state, optimizer=opt.state_dict(), configuration=config,
        train_ids=ids[a].tolist(), seed=seed, imputer_statistics=imputer.statistics_,
        feature_mean=scaler.mean_, feature_scale=scaler.scale_, target_center=center,
        target_scale=scale, actual_gradient_steps=600), checkpoint)
    check = make_mlp(xx.shape[1], config['width'])
    check.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    check.eval()
    with torch.no_grad():
        replay = check(xb).squeeze(-1).cpu().numpy() * scale + center
    assert np.allclose(replay, prediction, atol=1e-5, rtol=1e-6)
    np.savez_compressed(directory / 'prediction.npz', ids=ids[b], prediction=prediction)
    proof = dict(actual_gradient_steps=600, changed_parameter_tensors=len(changed),
        initial_parameter_sha256=initial, final_parameter_sha256={n: tensor_hash(p) for n, p in model.named_parameters()},
        training_rows=len(a), probe_rows=len(b), checkpoint_sha256=sha(checkpoint),
        optimizer_saved=True, seed=seed, seconds=time.monotonic() - started,
        max_checkpoint_replay_difference=float(np.max(np.abs(prediction - replay))))
    save(directory / 'update-evidence.json', proof)
    del model, check, opt, xa, xb, ya, state
    cleanup()
    return prediction, proof


def candidate_stage(bundle, model_root, output):
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(bundle / name) == digest, name
    data = np.load(bundle / 'training.npz')
    x, y, ids, groups = data['x'], data['y'], data['ids'], data['groups']
    assert x.shape == (len(ids), 370) and len(ids) in (508, 509)
    assert ids.tolist() == manifest['training_ids']
    assert set(ids).isdisjoint(manifest['excluded_test_ids'])
    cv = manifest['inner_folds']
    for fold in cv:
        assert set(groups[fold['train']]).isdisjoint(groups[fold['validation']])
    checkpoint = model_root / 'tabpfn-v2-regressor-v2_default.ckpt'
    assert sha(checkpoint) == WEIGHTS_SHA
    output.mkdir()
    arrays, evidence, keys = [], [], []
    if manifest['stage'] == 'baselines-mlp':
        for features in ('composition260', 'all370'):
            pending = {key: np.full(len(ids), np.nan) for key in ('e4-mean', 'e4-median', 'e8-mean', 'e8-median')}
            for f, split in enumerate(cv):
                a, b = np.asarray(split['train']), np.asarray(split['validation'])
                values = infer_variants(checkpoint, view(x, features), y, a, b)
                for key, value in values.items():
                    pending[key][b] = value
            for key, value in pending.items():
                keys.append('frozen-' + features + '-' + key)
                arrays.append(value)
        for k, config in enumerate(manifest['mlp_configurations']):
            predicted = np.full(len(ids), np.nan)
            for f, split in enumerate(cv):
                a, b = np.asarray(split['train']), np.asarray(split['validation'])
                p, proof = mlp_fit(x, y, ids, a, b, config, output / f'mlp-{k}-inner-{f}', 3042 + f)
                predicted[b] = p
                evidence.append(dict(kind='mlp', configuration_index=k, inner_fold=f, **proof))
            keys.append(f'mlp-{k}')
            arrays.append(predicted)
    else:
        features = manifest['features']
        xx = view(x, features)
        for k, config in enumerate(manifest['finetune_schedules']):
            pending = {key: np.full(len(ids), np.nan) for key in ('e4-mean', 'e4-median', 'e8-mean', 'e8-median')}
            for f, split in enumerate(cv):
                a, b = np.asarray(split['train']), np.asarray(split['validation'])
                directory = output / f'ft-{k}-inner-{f}'
                reference, proof = fit_verified(xx, y, ids, a, b, config, directory, checkpoint)
                fitted_checkpoint = directory / proof['checkpoint_file']
                retained_sha = retain_candidate_weights(fitted_checkpoint, directory)
                values = infer_variants(fitted_checkpoint, xx, y, a, b)
                assert np.allclose(values['e8-median'], reference, atol=1e-3, rtol=1e-4)
                for key, value in values.items():
                    pending[key][b] = value
                evidence.append(dict(kind='finetune', configuration_index=k, inner_fold=f,
                    retained_weight_sha256=retained_sha, optimizer_removed_after_full_replay=True, **proof))
            for key, value in pending.items():
                keys.append(f'ft-{features}-s{k}-{key}')
                arrays.append(value)
            save(output / 'progress.json', dict(completed_schedules=k + 1, evidence=evidence))
    predictions = np.asarray(arrays)
    assert predictions.shape == (len(keys), len(ids)) and np.isfinite(predictions).all()
    assert sha(checkpoint) == WEIGHTS_SHA
    np.savez_compressed(output / 'inner-predictions.npz', ids=ids, keys=np.asarray(keys), predictions=predictions)
    save(output / 'result.json', dict(experiment_succeeded=True, stage=manifest['stage'],
        outer_fold=manifest['outer_fold'], training_rows=len(ids), keys=keys,
        outer_test_features_or_targets_loaded=False, old_official_scores_loaded=False,
        configurations=manifest.get('finetune_schedules', manifest.get('mlp_configurations')),
        retained_candidate_weights_without_optimizer=manifest['stage'] != 'baselines-mlp', evidence=evidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('bundle', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    arguments = parser.parse_args()
    candidate_stage(arguments.bundle, arguments.model, arguments.output)
