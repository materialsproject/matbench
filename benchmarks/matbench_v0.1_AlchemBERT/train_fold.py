"""Corrected prospective AlchemBERT training, never a historical reproduction claim.

Train one official outer fold with ``train --config run.json``. Each input is
JSONL with exactly ``sample_id,instruction,input``; labels are obtained only
from official outer training data. The config requires immutable Hugging Face
model/tokenizer commit revisions, input SHA256s and an input provenance object.
Defaults reflect the archived BERT scheduler; protocol repairs are new choices.

``predict --run-dir RUN --output NEW.csv`` reloads the saved best checkpoint,
tokenizer and target transformer without loading outer test labels. Combine all
five fold CSVs for a task before using record_predictions.py. Dependencies are
Matbench, NumPy, scikit-learn, PyTorch and Transformers; no Lightning is needed.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import time
from pathlib import Path

import numpy as np

TRANSFORM_TASKS = {
    "matbench_jdft2d",
    "matbench_phonons",
    "matbench_dielectric",
    "matbench_steels",
}
DEFAULTS = {
    "seed": 42,
    "validation_fraction": 0.1,
    "max_length": 512,
    "batch_size": 32,
    "test_batch_size": 1,
    "max_epochs": 5000,
    "patience": 200,
    "fine_tune_encoder": True,
    "device": "cpu",
}
REQUIRED = {
    "task",
    "fold",
    "model_name",
    "model_revision",
    "tokenizer_revision",
    "train_file",
    "train_sha256",
    "test_file",
    "test_sha256",
    "input_provenance",
    "output_dir",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_config(path):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("Config must be a JSON object")
    allowed = REQUIRED | set(DEFAULTS) | {"tokenizer_name", "learning_rate"}
    if REQUIRED - set(supplied) or set(supplied) - allowed:
        raise ValueError(
            f"Config keys: missing={sorted(REQUIRED - set(supplied))}, "
            f"unknown={sorted(set(supplied) - allowed)}"
        )
    config = {**DEFAULTS, **supplied}
    for name in ("model_name", "task"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(config[name])):
            raise ValueError(f"{name} must be an immutable 40-character commit SHA")
    for name in ("train_sha256", "test_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(config[name])):
            raise ValueError(f"{name} must be a SHA256 digest")
    for name in (
        "seed",
        "fold",
        "max_length",
        "batch_size",
        "test_batch_size",
        "max_epochs",
        "patience",
    ):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if config["fold"] not in range(5):
        raise ValueError("fold must be 0 through 4")
    if any(
        config[name] == 0
        for name in (
            "max_length",
            "batch_size",
            "test_batch_size",
            "max_epochs",
            "patience",
        )
    ):
        raise ValueError(
            "Lengths, batch sizes, epochs and patience must be positive"
        )
    if type(config["fine_tune_encoder"]) is not bool:
        raise ValueError("fine_tune_encoder must be a JSON boolean")
    if (
        type(config["validation_fraction"]) not in (int, float)
        or not 0 < config["validation_fraction"] < 1
    ):
        raise ValueError("validation_fraction must be between zero and one")
    if (
        not isinstance(config["input_provenance"], dict)
        or not config["input_provenance"]
    ):
        raise ValueError(
            "input_provenance must describe the input generation/mapping"
        )
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", str(config["device"])):
        raise ValueError("device must be cpu, cuda or cuda:N")
    config.setdefault("tokenizer_name", config["model_name"])
    if (
        not isinstance(config["tokenizer_name"], str)
        or not config["tokenizer_name"].strip()
    ):
        raise ValueError("tokenizer_name must be a nonempty string")
    for name in ("model_name", "tokenizer_name"):
        if (
            not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?",
                config[name],
            )
            or Path(config[name]).exists()
        ):
            raise ValueError(
                f"{name} must identify a Hugging Face repository, not a local "
                "directory where revision pins would be ignored"
            )
    config.setdefault("learning_rate", 1e-5 if config["fine_tune_encoder"] else 1e-4)
    if (
        type(config["learning_rate"]) not in (int, float)
        or not math.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    for name in ("train_file", "test_file", "output_dir"):
        config[name] = str((path.parent / config[name]).resolve())
    return config


def load_texts(path, expected_sha256, expected_ids):
    """Validate IDs and reject target fields; text still needs a semantic audit."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Input hash mismatch: {path}")
    rows = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {
            "sample_id",
            "instruction",
            "input",
        }:
            raise ValueError(f"Unexpected fields at {path}:{line_number}")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in row):
            raise ValueError(
                f"Input fields must be nonempty strings at line {line_number}"
            )
        sample_id = row["sample_id"]
        if sample_id != sample_id.strip() or sample_id in rows:
            raise ValueError(f"Duplicate or padded sample ID: {sample_id}")
        rows[sample_id] = f"{row['instruction']} {row['input']}"
    expected_ids = [str(value) for value in expected_ids]
    if len(set(expected_ids)) != len(expected_ids) or set(rows) != set(expected_ids):
        raise ValueError("Input sample IDs do not exactly match the official fold")
    return [rows[sample_id] for sample_id in expected_ids]


def split_and_transform(labels, seed, validation_fraction, transform):
    """Split first; fitting a target transform never sees validation labels."""
    from sklearn.preprocessing import PowerTransformer

    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 1 or not np.isfinite(labels).all():
        raise ValueError("Training labels must be a finite one-dimensional array")
    n_train = int((1 - validation_fraction) * len(labels))
    if not 1 < n_train < len(labels):
        raise ValueError(
            "Need at least two inner-training and one validation sample"
        )
    permutation = np.random.RandomState(seed).permutation(len(labels))
    train_indices, val_indices = permutation[:n_train], permutation[n_train:]
    transformed = labels.copy()
    state = {"method": "identity"}
    if transform:
        fitted = PowerTransformer(method="yeo-johnson")
        fitted.fit(labels[train_indices, None])
        transformed = fitted.transform(labels[:, None]).ravel()
        state = {
            "method": "yeo-johnson",
            "lambda": float(fitted.lambdas_[0]),
            "mean": float(fitted._scaler.mean_[0]),
            "scale": float(fitted._scaler.scale_[0]),
            "fit_sample_count": len(train_indices),
        }
    if not np.isfinite(transformed).all():
        raise ValueError("Target transformation produced nonfinite values")
    return train_indices, val_indices, transformed, state


def inverse_targets(values, state):
    """Invert the saved scalar Yeo-Johnson parameters without pickle loading."""
    values = np.asarray(values, dtype=float)
    if state["method"] == "identity":
        return values
    if state["method"] != "yeo-johnson":
        raise ValueError("Unknown target transformation")
    x = values * state["scale"] + state["mean"]
    power = state["lambda"]
    positive = x >= 0
    result = np.empty_like(x)
    with np.errstate(invalid="raise", over="raise", divide="raise"):
        if abs(power) < np.spacing(1.0):
            result[positive] = np.expm1(x[positive])
        else:
            result[positive] = np.expm1(np.log1p(power * x[positive]) / power)
        if abs(power - 2) < np.spacing(1.0):
            result[~positive] = -np.expm1(-x[~positive])
        else:
            result[~positive] = -np.expm1(
                np.log1p(-(2 - power) * x[~positive]) / (2 - power)
            )
    if not np.isfinite(result).all():
        raise ValueError("Inverse target transform produced nonfinite predictions")
    return result


def prepare_fold(config, task):
    train_inputs, targets = task.get_train_and_val_data(config["fold"])
    test_inputs = task.get_test_data(config["fold"], include_target=False)
    train_ids = [str(value) for value in train_inputs.index]
    test_ids = [str(value) for value in test_inputs.index]
    if set(train_ids) & set(test_ids):
        raise ValueError("Official outer train/test IDs overlap")
    if list(targets.index) != list(train_inputs.index):
        raise ValueError("Official training inputs and labels are not aligned")
    classification = task.metadata["task_type"] == "classification"
    labels = np.asarray(targets, dtype=float)
    if classification and not np.isin(labels, [0, 1]).all():
        raise ValueError("Classification training labels must be zero or one")
    train_idx, val_idx, transformed, state = split_and_transform(
        labels,
        config["seed"],
        config["validation_fraction"],
        config["task"] in TRANSFORM_TASKS,
    )
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_texts": load_texts(
            config["train_file"], config["train_sha256"], train_ids
        ),
        "test_texts": load_texts(
            config["test_file"], config["test_sha256"], test_ids
        ),
        "train_indices": train_idx,
        "validation_indices": val_idx,
        "labels": transformed,
        "target_transform": state,
        "classification": classification,
    }


def load_task(config):
    from matbench.task import MatbenchTask

    return MatbenchTask(config["task"], autoload=True)


def make_model(encoder, fine_tune, classification):
    import torch

    class BertPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            for parameter in encoder.parameters():
                parameter.requires_grad = fine_tune
            self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            values = self.head(hidden.last_hidden_state[:, 0, :]).squeeze(-1)
            return torch.sigmoid(values) if classification else values

    return BertPredictor()


def batches(texts, indices, batch_size, tokenizer, max_length, device):
    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        selected = indices[start:end]
        encoded = tokenizer(
            [texts[int(i)] for i in selected],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # The archived BERT path uses input IDs and attention mask only.
        yield selected, {
            key: encoded[key].to(device) for key in ("input_ids", "attention_mask")
        }


def predict_values(model, tokenizer, texts, config, transform):
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for _, encoded in batches(
            texts,
            np.arange(len(texts)),
            config["test_batch_size"],
            tokenizer,
            config["max_length"],
            config["device"],
        ):
            chunks.extend(model(**encoded).cpu().numpy().tolist())
    values = inverse_targets(chunks, transform)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite test predictions")
    return values


def write_predictions(path, config, sample_ids, values, classification):
    if len(sample_ids) != len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid prediction length or values")
    if (
        classification
        and not np.logical_and(
            np.asarray(values) >= 0, np.asarray(values) <= 1
        ).all()
    ):
        raise ValueError("Classification outputs must be probabilities")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "fold", "sample_id", "prediction"])
        writer.writerows(
            (config["task"], config["fold"], sample_id, float(value))
            for sample_id, value in zip(sample_ids, values)
        )


def train(config):
    # Required by deterministic CUDA matrix multiplication on supported drivers.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModel, AutoTokenizer

    # All validation precedes model download and creation of output files.
    output = Path(config["output_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to reuse an output directory: {output}")
    prepared = prepare_fold(config, load_task(config))
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_name"],
        revision=config["tokenizer_revision"],
        trust_remote_code=False,
    )
    encoder = AutoModel.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
        trust_remote_code=False,
    )
    model = make_model(
        encoder, config["fine_tune_encoder"], prepared["classification"]
    ).to(config["device"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    save_json(output / "target_transform.json", prepared["target_transform"])
    save_json(
        output / "split.json",
        {
            "inner_train_ids": [
                prepared["train_ids"][i] for i in prepared["train_indices"]
            ],
            "inner_validation_ids": [
                prepared["train_ids"][i] for i in prepared["validation_indices"]
            ],
            "outer_test_ids": prepared["test_ids"],
        },
    )
    tokenizer.save_pretrained(output / "tokenizer")
    encoder.config.save_pretrained(output / "encoder_config")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    labels = torch.tensor(
        prepared["labels"], dtype=torch.float32, device=config["device"]
    )
    shuffle_rng = np.random.RandomState(config["seed"])
    best_loss, best_epoch, stale_epochs = math.inf, None, 0
    started = time.monotonic()
    log = []
    for epoch in range(config["max_epochs"]):
        model.train()
        order = shuffle_rng.permutation(prepared["train_indices"])
        train_total = 0.0
        for indices, encoded in batches(
            prepared["train_texts"],
            order,
            config["batch_size"],
            tokenizer,
            config["max_length"],
            config["device"],
        ):
            optimizer.zero_grad()
            predictions = model(**encoded)
            truth = labels[torch.as_tensor(indices, device=config["device"])]
            loss_fn = (
                torch.nn.functional.binary_cross_entropy
                if prepared["classification"]
                else torch.nn.functional.mse_loss
            )
            loss = loss_fn(predictions, truth)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for indices, encoded in batches(
                prepared["train_texts"],
                prepared["validation_indices"],
                config["batch_size"],
                tokenizer,
                config["max_length"],
                config["device"],
            ):
                predictions = model(**encoded)
                truth = labels[torch.as_tensor(indices, device=config["device"])]
                loss_fn = (
                    torch.nn.functional.binary_cross_entropy
                    if prepared["classification"]
                    else torch.nn.functional.l1_loss
                )
                val_total += float(loss_fn(predictions, truth)) * len(indices)
        val_loss = val_total / len(prepared["validation_indices"])
        if not math.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        log.append(
            {
                "epoch": epoch,
                "train_loss": train_total / len(order),
                "validation_loss": val_loss,
            }
        )
        print(json.dumps(log[-1]), flush=True)
        with (output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log[-1], allow_nan=False) + "\n")
        if val_loss < best_loss:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            # Only this newly-created run directory is ever overwritten.
            torch.save(model.state_dict(), output / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break
    model.load_state_dict(
        torch.load(
            output / "best_model.pt",
            map_location=config["device"],
            weights_only=True,
        )
    )
    # Complete training evidence before inference: an OOM or out-of-domain
    # inverse transform must not make a valid trained checkpoint unreplayable.
    save_json(output / "training_log.json", log)
    save_json(
        output / "run.json",
        {
            "purpose": "new corrected experiment; not historical reproduction",
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_elapsed_seconds": time.monotonic() - started,
            "classification": prepared["classification"],
            "resolved_model_revision": getattr(encoder.config, "_commit_hash", None),
            "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "numpy",
                    "scikit-learn",
                    "matbench",
                )
            },
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
            "device": str(next(model.parameters()).device),
            "gpu_name": (
                torch.cuda.get_device_name(config["device"])
                if config["device"].startswith("cuda")
                else None
            ),
            "runner_sha256": digest(__file__),
        },
    )
    save_json(
        output / "artifacts.json",
        {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    values = predict_values(
        model,
        tokenizer,
        prepared["test_texts"],
        config,
        prepared["target_transform"],
    )
    write_predictions(
        output / "predictions.csv",
        config,
        prepared["test_ids"],
        values,
        prepared["classification"],
    )
    save_json(
        output / "prediction.json",
        {
            "prediction_sha256": digest(output / "predictions.csv"),
            "sample_count": len(prepared["test_ids"]),
            "task": config["task"],
            "fold": config["fold"],
        },
    )
    return output


def replay(run_dir, output, device):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    run_dir = Path(run_dir)
    artifacts = json.loads((run_dir / "artifacts.json").read_text())
    for relative, expected in artifacts.items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()) or digest(path) != expected:
            raise ValueError(f"Saved artifact hash mismatch: {relative}")
    config = json.loads((run_dir / "config.json").read_text())
    config["device"] = device
    run = json.loads((run_dir / "run.json").read_text())
    transform = json.loads((run_dir / "target_transform.json").read_text())
    task = load_task(config)
    ids = [
        str(value)
        for value in task.get_test_data(config["fold"], include_target=False).index
    ]
    texts = load_texts(config["test_file"], config["test_sha256"], ids)
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "tokenizer", local_files_only=True
    )
    encoder = AutoModel.from_config(
        AutoConfig.from_pretrained(
            run_dir / "encoder_config",
            local_files_only=True,
            trust_remote_code=False,
        ),
        trust_remote_code=False,
    )
    model = make_model(
        encoder, config["fine_tune_encoder"], run["classification"]
    ).to(device)
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    values = predict_values(model, tokenizer, texts, config, transform)
    write_predictions(output, config, ids, values, run["classification"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Run one new official fold")
    training.add_argument("--config", required=True)
    predicting = subparsers.add_parser(
        "predict", help="Replay a saved new checkpoint"
    )
    predicting.add_argument("--run-dir", required=True)
    predicting.add_argument("--output", required=True)
    predicting.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        train(read_config(args.config))
    else:
        replay(args.run_dir, args.output, args.device)


if __name__ == "__main__":
    main()
