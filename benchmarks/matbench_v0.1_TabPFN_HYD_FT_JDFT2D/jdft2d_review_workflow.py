"""Portable staged reproduction of the review-corrected JDFT2D protocol.

Preparation/select/scoring run on CPU; candidate/refit/predict stages need CUDA.
Every stage refuses overwrites. Failed GPU stages are not retried automatically.
"""
import argparse
import datetime
import functools
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_SHA = '26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf'
SPLITS_SHA = '94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d'
WEIGHTS_SHA = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
SCHEDULES = [
    dict(family='finetuned_tabpfn_v2', learning_rate=5e-6, epochs=20, features='selected', trainable_scope='full'),
    dict(family='finetuned_tabpfn_v2', learning_rate=2e-5, epochs=20, features='selected', trainable_scope='full'),
    dict(family='finetuned_tabpfn_v2', learning_rate=1e-6, epochs=5, features='selected', trainable_scope='full'),
    dict(family='finetuned_tabpfn_v2', learning_rate=1e-6, epochs=10, features='selected', trainable_scope='full'),
    dict(family='finetuned_tabpfn_v2', learning_rate=5e-6, epochs=5, features='selected', trainable_scope='full'),
    dict(family='finetuned_tabpfn_v2', learning_rate=5e-5, epochs=10, features='selected', trainable_scope='readout'),
]
MLPS = [dict(family='mlp', features=f, width=w, decay=d, steps=600)
    for f, w, d in [('all370', 64, .001), ('all370', 128, .001), ('all370', 128, .01),
                    ('all370', 64, .01), ('composition260', 128, .001), ('composition260', 128, .01)]]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def construct_features(raw):
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
    from pymatgen.core import Structure
    from jdft2d_geometry_features import geometry_features
    fs = [Stoichiometry(), ElementProperty.from_preset('magpie', impute_nan=True),
          ValenceOrbital(props=['frac'], impute_nan=True)]
    xs, ids, groups = [], [], []
    for index, row in zip(raw['index'], raw['data'], strict=True):
        structure = Structure.from_dict(row[0])
        composition = structure.composition
        fractions = np.zeros(118)
        for element, fraction in composition.fractional_composition.items():
            fractions[element.Z - 1] = fraction
        values = list(fractions)
        for transform in fs:
            try:
                values.extend(transform.featurize(composition))
            except Exception:
                values.extend([np.nan] * len(transform.feature_labels()))
        xs.append(np.r_[values, geometry_features(structure)])
        ids.append(f'mb-jdft2d-{index + 1:03d}')
        groups.append(composition.reduced_formula)
    x = np.asarray(xs, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (636, 370)
    return np.asarray(ids), x, np.asarray(groups)


def prepare(data_path, splits_path, output, protocol):
    from sklearn.model_selection import GroupKFold
    assert sha(data_path) == DATA_SHA and sha(splits_path) == SPLITS_SHA
    assert not output.exists()
    output.mkdir(parents=True)
    raw = json.loads(gzip.decompress(data_path.read_bytes()))
    assert raw['columns'] == ['structure', 'exfoliation_en']
    ids, x, groups = construct_features(raw)
    target = {f'mb-jdft2d-{i + 1:03d}': row[1] for i, row in zip(raw['index'], raw['data'], strict=True)}
    lookup = {sid: i for i, sid in enumerate(ids)}
    splits = json.loads(splits_path.read_text())['splits']['matbench_jdft2d']
    np.savez_compressed(output / 'geometry-only.npz', ids=ids, x=x, groups=groups)
    for f in range(5):
        fold = output / f'fold-{f}'
        fold.mkdir()
        split = splits[f'fold_{f}']
        a = np.array([lookup[s] for s in split['train']])
        b = np.array([lookup[s] for s in split['test']])
        assert set(ids[a]).isdisjoint(ids[b])
        np.savez_compressed(fold / 'training.npz', ids=ids[a], x=x[a], groups=groups[a],
                            y=np.asarray([target[s] for s in ids[a]]))
        np.savez_compressed(fold / 'test-geometry-only.npz', ids=ids[b], x=x[b])
        inner = [dict(train=fit.tolist(), validation=val.tolist())
                 for fit, val in GroupKFold(3).split(x[a], groups=groups[a])]
        for split2 in inner:
            assert set(groups[a][split2['train']]).isdisjoint(groups[a][split2['validation']])
        save(fold / 'training-manifest.json', dict(outer_fold=f, training_ids=ids[a].tolist(),
             excluded_test_ids=ids[b].tolist(), inner_folds=inner, protocol_sha256=sha(protocol),
             finetune_schedules=SCHEDULES, mlp_configurations=MLPS,
             no_outer_test_in_training_bundle=True))
    save(output / 'preparation.json', dict(prepared_at_utc=now(), rows=636, features=370,
         pure_geometric_transforms_only=True, outer_test_scored=False, protocol_sha256=sha(protocol)))


def candidate_summaries(ft_keys, frozen_keys, mlp_keys):
    """A complete deterministic Cartesian closure of previously tried axes."""
    candidates = []
    for key in sorted(mlp_keys):
        candidates.append(dict(kind='mlp', key=key))
    for key in sorted(ft_keys):
        candidates.append(dict(kind='finetune_blend', key=key, baseline_key=None, weight=1.))
        for base in sorted(frozen_keys):
            for weight in (.25, .5, .75):
                candidates.append(dict(kind='finetune_blend', key=key, baseline_key=base, weight=weight))
    return candidates


def select(training, manifests, candidate_results, output):
    assert not output.exists()
    d = np.load(training)
    ids, y = d['ids'], d['y']
    m = json.loads(manifests.read_text())
    assert ids.tolist() == m['training_ids'] and set(ids).isdisjoint(m['excluded_test_ids'])
    predictions = {}
    for path in candidate_results:
        result = json.loads((path / 'result.json').read_text())
        assert result['outer_fold'] == m['outer_fold'] and result['experiment_succeeded']
        assert not result['outer_test_features_or_targets_loaded']
        saved = np.load(path / 'inner-predictions.npz')
        assert np.array_equal(saved['ids'], ids)
        for key, values in zip(saved['keys'], saved['predictions'], strict=True):
            key = str(key)
            assert key not in predictions and np.isfinite(values).all()
            predictions[key] = values
    keys = list(predictions)
    ft = [k for k in keys if k.startswith('ft-')]
    frozen = [k for k in keys if k.startswith('frozen-')]
    mlp = [k for k in keys if k.startswith('mlp-')]
    assert (len(ft), len(frozen), len(mlp)) == (48, 8, 6)
    candidates = candidate_summaries(ft, frozen, mlp)
    assert len(candidates) == 1206
    scores = []
    for index, candidate in enumerate(candidates):
        value = predictions[candidate['key']]
        if candidate.get('baseline_key'):
            weight = candidate['weight']
            value = weight * value + (1 - weight) * predictions[candidate['baseline_key']]
        per_fold = [float(np.mean(np.abs(value[fold['validation']] - y[fold['validation']])))
                    for fold in m['inner_folds']]
        scores.append(dict(index=index, candidate=candidate, mean_fold_mae=float(np.mean(per_fold)), folds=per_fold))
    choice = min(range(len(scores)), key=lambda i: (scores[i]['mean_fold_mae'], i))
    save(output, dict(frozen_at_utc=now(), outer_fold=m['outer_fold'], selected=scores[choice],
         all_inner_scores=scores, selection_uses_only_this_outer_TRAIN=True,
         no_old_or_new_outer_test_score_loaded=True, protocol_sha256=m['protocol_sha256'],
         training_ids=m['training_ids'], excluded_test_ids=m['excluded_test_ids']))


def parse_key(key):
    parts = key.split('-')
    if parts[0] == 'ft':
        return parts[1], int(parts[2][1:]), int(parts[3][1:]), parts[4]
    assert parts[0] == 'frozen'
    return parts[1], None, int(parts[2][1:]), parts[3]


def predict_selected(checkpoint, selected, x, y, probe_x, device='cuda'):
    from jdft2d_review_candidates import cleanup, estimator, make_mlp, view
    import torch
    recipe = selected['candidate']
    if recipe['kind'] == 'mlp':
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = stored['configuration']
        xx, px = view(x, config['features']), view(probe_x, config['features'])
        imputer, scaler = SimpleImputer(keep_empty_features=True), StandardScaler()
        scaler.fit(imputer.fit_transform(xx))
        assert np.allclose(imputer.statistics_, stored['imputer_statistics'], equal_nan=True)
        assert np.allclose(scaler.mean_, stored['feature_mean']) and np.allclose(scaler.scale_, stored['feature_scale'])
        tx = torch.tensor(np.clip(scaler.transform(imputer.transform(px)), -10, 10), dtype=torch.float32, device=device)
        model = make_mlp(xx.shape[1], config['width'])
        model.load_state_dict(stored['state_dict'])
        model.eval()
        with torch.no_grad():
            prediction = model(tx).squeeze(-1).cpu().numpy() * stored['target_scale'] + stored['target_center']
        del stored, model, tx
        cleanup()
        return np.asarray(prediction, dtype=float)
    features, _, count, summary = parse_key(recipe['key'])
    model = estimator(checkpoint, count)
    model.fit(view(x, features), y)
    prediction = np.asarray(model.predict(view(probe_x, features), output_type=summary), dtype=float)
    del model
    cleanup()
    return prediction


def mixed_prediction(checkpoint, original_checkpoint, selection, x, y, probe_x):
    from jdft2d_review_candidates import cleanup, estimator, view
    selected = selection['selected']
    prediction = predict_selected(checkpoint, selected, x, y, probe_x)
    recipe = selected['candidate']
    if recipe.get('baseline_key'):
        features, _, count, summary = parse_key(recipe['baseline_key'])
        model = estimator(original_checkpoint, count)
        model.fit(view(x, features), y)
        baseline = np.asarray(model.predict(view(probe_x, features), output_type=summary))
        prediction = recipe['weight'] * prediction + (1 - recipe['weight']) * baseline
        del model
        cleanup()
    assert prediction.shape == (len(probe_x),) and np.isfinite(prediction).all()
    return prediction


def final_fit(bundle, model_root, output):
    from jdft2d_review_candidates import mlp_fit, view
    from matbench_jdft2d_training import fit_verified
    import torch
    torch.set_num_threads(4)
    m = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in m['files'].items():
        assert sha(bundle / name) == digest
    selection = json.loads((bundle / 'selection.json').read_text())
    assert selection['selection_uses_only_this_outer_TRAIN']
    d = np.load(bundle / 'training.npz')
    x, y, ids = d['x'], d['y'], d['ids']
    assert ids.tolist() == selection['training_ids']
    assert set(ids).isdisjoint(selection['excluded_test_ids'])
    recipe = selection['selected']['candidate']
    a, b = np.arange(len(ids)), np.arange(min(16, len(ids)))
    output.mkdir()
    original = model_root / 'tabpfn-v2-regressor-v2_default.ckpt'
    assert sha(original) == WEIGHTS_SHA
    if recipe['kind'] == 'mlp':
        config = MLPS[int(recipe['key'].split('-')[1])]
        _, proof = mlp_fit(x, y, ids, a, b, config, output / 'model', 3042)
        checkpoint = output / 'model/model.pt'
    else:
        features, schedule, _, _ = parse_key(recipe['key'])
        _, proof = fit_verified(view(x, features), y, ids, a, b, SCHEDULES[schedule],
            output / 'model', original, probe_role='TRAIN_witness_only')
        checkpoint = output / 'model' / proof['checkpoint_file']
    assert (proof.get('actual_gradient_steps') or proof.get('gradient_steps')) > 0
    witness = mixed_prediction(checkpoint, original, selection, x, y, x[b])
    np.savez_compressed(output / 'training-witness.npz', ids=ids[b], x=x[b], prediction=witness)
    save(output / 'result.json', dict(experiment_succeeded=True, stage='final-fit', outer_fold=selection['outer_fold'],
         selected=selection['selected'], checkpoint_relative=str(checkpoint.relative_to(output)),
         checkpoint_sha256=sha(checkpoint), selection_sha256=sha(bundle / 'selection.json'),
         original_checkpoint_sha256=WEIGHTS_SHA, true_training_proof=proof,
         official_test_features_or_targets_loaded=False))


def predict(bundle, original_root, output):
    import torch
    torch.set_num_threads(4)
    m = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in m['files'].items():
        assert sha(bundle / name) == digest, name
    assert m['all5_final_models_frozen_before_inference']
    original = original_root / 'tabpfn-v2-regressor-v2_default.ckpt'
    assert sha(original) == WEIGHTS_SHA
    output.mkdir()
    rows, replays = [], []
    for f, record in enumerate(m['models']):
        root = Path(m['model_roots'][f]) / 'probe'
        checkpoint = root / record['checkpoint_relative']
        assert sha(checkpoint) == record['checkpoint_sha256']
        selection = json.loads((bundle / f'selection-{f}.json').read_text())
        train, test = np.load(bundle / f'train-{f}.npz'), np.load(bundle / f'test-{f}.npz')
        assert set(test.files) == {'ids', 'x'}
        assert set(train['ids']).isdisjoint(test['ids'])
        assert train['ids'].tolist() == selection['training_ids'] and test['ids'].tolist() == selection['excluded_test_ids']
        witness = np.load(root / 'training-witness.npz')
        replay = mixed_prediction(checkpoint, original, selection, train['x'], train['y'], witness['x'])
        assert np.allclose(replay, witness['prediction'], atol=1e-3, rtol=1e-4)
        values = mixed_prediction(checkpoint, original, selection, train['x'], train['y'], test['x'])
        rows.extend(dict(fold=f, id=str(sid), prediction=float(v)) for sid, v in zip(test['ids'], values, strict=True))
        replays.append(dict(fold=f, passed=True, fresh_process=True, witness_rows=len(witness['ids']),
                            maximum_absolute_difference=float(np.max(np.abs(replay - witness['prediction'])))))
        assert sha(checkpoint) == record['checkpoint_sha256']
    assert len(rows) == len({row['id'] for row in rows}) == 636
    save(output / 'test-predictions.json', rows)
    save(output / 'result.json', dict(experiment_succeeded=True, stage='predict', n_predictions=636,
         model_freeze_sha256=m['model_freeze_sha256'], fresh_process_replays=replays,
         official_test_targets_loaded=False, gradient_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['candidates', 'final-fit', 'predict'], required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'candidates':
        from jdft2d_review_candidates import candidate_stage
        candidate_stage(args.bundle, args.model, args.output)
    elif args.phase == 'final-fit':
        final_fit(args.bundle, args.model, args.output)
    else:
        predict(args.bundle, args.model, args.output)
