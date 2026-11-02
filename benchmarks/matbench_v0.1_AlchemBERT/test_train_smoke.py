"""Optional offline integration tests; synthetic data, tiny random BERT only.

These exercise actual training and checkpoint replay without pretrained weights,
official dataset downloads, GPUs, or claims about historical experiment results.
Install requirements-rerun-cpu.txt to run them; otherwise they are skipped.
"""

import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_fold as runner

HAS_TRAINING_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
)


class SyntheticTask:
    """Task double whose outer test labels cannot be retrieved."""

    def __init__(self, classification):
        self.metadata = {
            "task_type": "classification" if classification else "regression"
        }
        self.train_ids = [f"synthetic-train-{index}" for index in range(20)]
        self.test_ids = [f"synthetic-test-{index}" for index in range(3)]
        self.labels = [
            index % 2 if classification else index * 0.7 + 1 for index in range(20)
        ]

    def get_train_and_val_data(self, fold):
        if fold != 0:
            raise AssertionError("Unexpected fold")
        return pd.Series("Fe O", index=self.train_ids), pd.Series(
            self.labels, index=self.train_ids
        )

    def get_test_data(self, fold, include_target):
        if fold != 0 or include_target is not False:
            raise AssertionError("Outer test labels must never be requested")
        return pd.Series("Fe O", index=self.test_ids)


@unittest.skipUnless(
    HAS_TRAINING_DEPS, "Optional torch/transformers are not installed"
)
class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Limit tiny CPU workloads without leaving process-wide state changed.
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.set_num_threads(1)

    def exercise(self, task_name, classification, inference_failure=False):
        from transformers import BertConfig, BertModel, BertTokenizerFast

        task = SyntheticTask(classification)
        vocab = self.root / "vocab.txt"
        vocab.write_text(
            "\n".join(
                [
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]",
                    "[MASK]",
                    "predict",
                    "property",
                    "fe",
                    "o",
                    "material",
                    "composition",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
        for partition, sample_ids in (
            ("train", task.train_ids),
            ("test", task.test_ids),
        ):
            with (self.root / f"{partition}.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for sample_id in reversed(sample_ids):
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "instruction": "Predict property",
                                "input": "Material composition Fe O",
                            }
                        )
                        + "\n"
                    )
        supplied = {
            "task": task_name,
            "fold": 0,
            "model_name": "SYNTHETIC-RANDOM-BERT-NO-PRETRAINING",
            # Sentinels used only by mocked pretrained-loading calls below.
            "model_revision": "0" * 40,
            "tokenizer_revision": "0" * 40,
            "train_file": "train.jsonl",
            "train_sha256": runner.digest(self.root / "train.jsonl"),
            "test_file": "test.jsonl",
            "test_sha256": runner.digest(self.root / "test.jsonl"),
            "input_provenance": {"purpose": "synthetic integration test"},
            "output_dir": "run",
            "max_epochs": 1 if inference_failure else 2,
            "patience": 1,
            "max_length": 16,
            "batch_size": 4,
            "test_batch_size": 2,
            "device": "cpu",
            "validation_fraction": 0.2,
        }
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(supplied), encoding="utf-8")
        config = runner.read_config(config_path)
        model_config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
        initial = {}

        def new_encoder(*args, **kwargs):
            encoder = BertModel(model_config)
            initial["embeddings"] = (
                encoder.embeddings.word_embeddings.weight.detach().clone()
            )
            return encoder

        with patch.object(runner, "load_task", return_value=task):
            with patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
            ), patch(
                "transformers.AutoModel.from_pretrained", side_effect=new_encoder
            ):
                if inference_failure:
                    with patch.object(
                        runner,
                        "predict_values",
                        side_effect=RuntimeError("synthetic inference failure"),
                    ), self.assertRaisesRegex(
                        RuntimeError, "synthetic inference failure"
                    ):
                        runner.train(config)
                    output = Path(config["output_dir"])
                    for name in (
                        "config.json",
                        "run.json",
                        "artifacts.json",
                        "best_model.pt",
                        "training_log.json",
                        "target_transform.json",
                        "split.json",
                    ):
                        self.assertTrue((output / name).is_file(), name)
                    self.assertFalse((output / "predictions.csv").exists())
                else:
                    output = runner.train(config)
            replay_path = self.root / "replay.csv"
            # Replay uses real local tokenizer/config/checkpoint loading.
            runner.replay(output, replay_path, "cpu")
            if not inference_failure:
                self.assertEqual(
                    (output / "predictions.csv").read_bytes(),
                    replay_path.read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                runner.train(config)
            with self.assertRaises(FileExistsError):
                runner.replay(output, replay_path, "cpu")
        checkpoint = self.torch.load(
            output / "best_model.pt", map_location="cpu", weights_only=True
        )
        self.assertFalse(
            self.torch.equal(
                checkpoint["encoder.embeddings.word_embeddings.weight"],
                initial["embeddings"],
            ),
            "Fine-tuning must actually update encoder weights",
        )
        predictions = pd.read_csv(replay_path)
        self.assertEqual(list(predictions.sample_id), task.test_ids)
        self.assertTrue(np.isfinite(predictions.prediction).all())
        if classification:
            self.assertTrue(
                ((predictions.prediction > 0) & (predictions.prediction < 1)).all()
            )
        transform = json.loads((output / "target_transform.json").read_text())
        if task_name == "matbench_steels":
            self.assertEqual(transform["method"], "yeo-johnson")
            self.assertEqual(transform["fit_sample_count"], 16)
        split = json.loads((output / "split.json").read_text())
        self.assertEqual(len(split["inner_train_ids"]), 16)
        self.assertTrue(
            set(split["inner_train_ids"]).isdisjoint(split["inner_validation_ids"])
        )

    def test_classification_train_and_replay(self):
        self.exercise("matbench_glass", classification=True)

    def test_transformed_regression_train_and_replay(self):
        self.exercise("matbench_steels", classification=False)

    def test_saved_checkpoint_survives_initial_prediction_failure(self):
        self.exercise(
            "matbench_steels", classification=False, inference_failure=True
        )


if __name__ == "__main__":
    unittest.main()
