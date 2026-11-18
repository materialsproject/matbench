"""Reproduce TabPFN-HYD-FT: prepare -> TRAIN-only fits -> freeze -> predict -> score.

The optional --prepare-only path never trains or evaluates test targets.
No stored benchmark predictions are imported by this reproduction program.
"""
import argparse
import datetime
import gc
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from pymatgen.core import Structure
from sklearn.model_selection import GroupKFold

from geometry_features import geometry_features

HERE = Path(__file__).resolve().parent
DATA_SHA = "26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf"
WEIGHTS_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
REVISION = "02a44705ff7790b50cded404bc59adb94da456a1"
CONFIGURATIONS = [
    dict(family="finetuned_tabpfn_v2", learning_rate=1e-6, epochs=5, features="all370", trainable_scope="full"),
    dict(family="finetuned_tabpfn_v2", learning_rate=1e-6, epochs=10, features="all370", trainable_scope="full"),
    dict(family="finetuned_tabpfn_v2", learning_rate=5e-6, epochs=5, features="all370", trainable_scope="full"),
    dict(family="finetuned_tabpfn_v2", learning_rate=5e-5, epochs=10, features="all370", trainable_scope="readout"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(data, output):
    """Only analytic feature transforms see all geometries; fits use their TRAIN."""
    assert sha(data) == DATA_SHA
    raw = json.loads(gzip.decompress(data.read_bytes()))
    assert raw["columns"] == ["structure", "exfoliation_en"]
    transforms = [Stoichiometry(), ElementProperty.from_preset("magpie", impute_nan=True),
                  ValenceOrbital(props=["frac"], impute_nan=True)]
    ids, features, groups, targets = [], [], [], {}
    for index, row in zip(raw["index"], raw["data"], strict=True):
        sid = f"mb-jdft2d-{index + 1:03d}"
        structure = Structure.from_dict(row[0])
        composition = structure.composition
        fractions = np.zeros(118)
        for element, amount in composition.fractional_composition.items():
            fractions[element.Z - 1] = amount
        vector = list(fractions)
        for transform in transforms:
            try:
                vector.extend(transform.featurize(composition))
            except Exception:
                vector.extend([np.nan] * len(transform.feature_labels()))
        features.append(np.r_[vector, geometry_features(structure)])
        ids.append(sid)
        groups.append(composition.reduced_formula)
        targets[sid] = row[1]
    ids, groups = np.asarray(ids), np.asarray(groups)
    x = np.asarray(features, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (636, 370) and len(set(ids)) == 636
    np.savez_compressed(output / "all-geometric-features.npz", ids=ids, x=x, groups=groups)
    lookup = {sid: index for index, sid in enumerate(ids)}
    # Canonical official fold assignments, independent of our archived predictions.
    from matbench.constants import MBV01_VALIDATION_DATA_PATH
    assert sha(MBV01_VALIDATION_DATA_PATH) == "94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d"
    splits = json.loads(Path(MBV01_VALIDATION_DATA_PATH).read_text())["splits"]["matbench_jdft2d"]
    manifests = []
    for fold in range(5):
        split = splits[f"fold_{fold}"]
        train_ids, test_ids = split["train"], split["test"]
        assert set(train_ids).isdisjoint(test_ids)
        a = np.asarray([lookup[sid] for sid in train_ids])
        b = np.asarray([lookup[sid] for sid in test_ids])
        train = output / f"train-input-{fold}"
        train.mkdir()
        np.savez_compressed(train / "development.npz", ids=ids[a], x=x[a], groups=groups[a],
                            y=np.asarray([targets[sid] for sid in train_ids]))
        np.savez_compressed(output / f"test-{fold}.npz", ids=ids[b], x=x[b])
        inner = []
        for fit, validation in GroupKFold(3).split(x[a], groups=groups[a]):
            assert set(groups[a][fit]).isdisjoint(groups[a][validation])
            inner.append({"train": fit.tolist(), "validation": validation.tolist()})
        manifest = dict(outer_fold=fold, training_row_count=len(a), development_ids=train_ids,
                        excluded_official_test_ids=test_ids, configurations=CONFIGURATIONS,
                        inner_folds=inner, files={"development.npz": sha(train / "development.npz")})
        save(train / "manifest.json", manifest)
        manifests.append(manifest)
    return manifests


def checked_child(arguments, log):
    """A failure stops; there are no automatic training retries or fallbacks."""
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", DO_NOT_TRACK="1",
               TABPFN_DISABLE_TELEMETRY="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    with log.open("w") as stream:
        subprocess.run([sys.executable, "-u", *map(str, arguments)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT, env=env)


def main(args):
    output = args.output.resolve()
    output.mkdir()  # Refuse to overwrite a previous training/evaluation run.
    manifests = prepare(args.data.resolve(), output)
    save(output / "preparation.json", dict(data_sha256=DATA_SHA, rows=636, features=370,
         official_test_scored=False, per_outer_training_rows=[m["training_row_count"] for m in manifests]))
    if args.prepare_only:
        return
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download("Prior-Labs/TabPFN-v2-reg",
                          "tabpfn-v2-regressor-v2_default.ckpt", revision=REVISION))
    checkpoint = checkpoint.resolve()
    assert sha(checkpoint) == WEIGHTS_SHA
    # The actual worker expects this filename in its supplied model directory.
    assert checkpoint.name == "tabpfn-v2-regressor-v2_default.ckpt"
    models, model_roots = [], []
    for fold, manifest in enumerate(manifests):
        root = output / f"outer-{fold}"
        root.mkdir()
        checked_child([HERE / "matbench_jdft2d_nested_train.py", "--bundle", output / f"train-input-{fold}",
                       "--model", checkpoint.parent, "--output", root / "probe"], root / "training.log")
        report = json.loads((root / "probe/result.json").read_text())
        assert report["all_models_genuinely_trained"] and report["final_refit_proof"]["gradient_steps"] > 0
        assert not report["official_outer_test_labels_loaded"]
        models.append(dict(outer_fold=fold, configuration=report["configuration"],
                           checkpoint_relative=report["final_checkpoint"],
                           checkpoint_sha256=report["final_checkpoint_sha256"],
                           training_ids=manifest["development_ids"], test_ids=manifest["excluded_official_test_ids"]))
        model_roots.append(str(root))
    freeze = output / "all-five-models-freeze.json"
    save(freeze, dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), models=models,
                     all5_genuinely_trained=True, outer_test_predictions_made=False, official_score_computed=False))
    bundle = output / "prediction-input"
    bundle.mkdir()
    import shutil
    for fold in range(5):
        shutil.copyfile(output / f"train-input-{fold}/development.npz", bundle / f"train-{fold}.npz")
        shutil.copyfile(output / f"test-{fold}.npz", bundle / f"test-{fold}.npz")
    save(bundle / "manifest.json", dict(all5_final_models_frozen=True, models=models, model_roots=model_roots,
         model_freeze_sha256=sha(freeze), files={p.name: sha(p) for p in bundle.iterdir()}))
    # The original inference algorithm, with only private mount roots replaced.
    checked_child([HERE / "predict_frozen.py", "--bundle", bundle, "--output", output / "predictions"],
                   output / "inference.log")
    report = json.loads((output / "predictions/result.json").read_text())
    assert report["n_predictions"] == 636 and all(p["passed"] for p in report["fresh_process_replays"])
    assert report["model_freeze_sha256"] == sha(freeze)
    rows = json.loads((output / "predictions/test-predictions.json").read_text())
    assert len(rows) == len({row["id"] for row in rows}) == 636
    # Only now load test targets through the official scorer; no feedback path to fits.
    import functools
    from matminer.datasets import load_dataset
    import matbench.data_ops
    matbench.data_ops.load_dataset = functools.partial(load_dataset, data_home=str(args.data.resolve().parent),
                                                     download_if_missing=False)
    from matbench import MatbenchBenchmark
    benchmark = MatbenchBenchmark(subset=["matbench_jdft2d"], autoload=False)
    task = benchmark.matbench_jdft2d
    task.load()
    for fold in range(5):
        predictions = {row["id"]: row["prediction"] for row in rows if row["fold"] == fold}
        expected = task.get_test_data(fold, include_target=False).index.tolist()
        assert set(predictions) == set(expected)
        task.record(fold, [predictions[sid] for sid in expected], params=models[fold]["configuration"])
    benchmark.add_metadata(dict(algorithm="TabPFN-HYD-FT", attribution="Built with PriorLabs-TabPFN",
                                real_gradient_training=True, official_test_used_for_selection=False))
    assert not benchmark.validate()
    benchmark.to_file(str(output / "results.json.gz"))
    save(output / "scores.json", dict(task.scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Official checksum-pinned matbench_jdft2d.json.gz")
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing runs are never overwritten")
    parser.add_argument("--checkpoint", type=Path, help="Optional pinned public V2 checkpoint; otherwise download exact HF revision")
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
